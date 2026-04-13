# Copyright (c) OpenMMLab. All rights reserved.
import tilelang
import tilelang.language as T
import tilelang.layout
import torch
from fla.ops.utils import prepare_chunk_offsets


@T.macro
def _load_state_const(dst: T.Buffer,
                      State: T.Buffer,
                      seq_id,
                      hv_id,
                      v_base,
                      K: int,
                      BV: int,
                      V: int) -> None:
    for k_local, v_inner in T.Parallel(K, BV):
        v_idx = v_base + v_inner
        if v_idx < V:
            dst[k_local, v_inner] = T.cast(State[seq_id, hv_id, k_local, v_idx], T.float32)
        else:
            dst[k_local, v_inner] = 0.0


@T.macro
def _store_h_const(dst: T.Buffer,
                   src: T.Buffer,
                   batch_id,
                   chunk_id,
                   hv_id,
                   v_base,
                   K: int,
                   BV: int,
                   V: int,
                   out_dtype: torch.dtype) -> None:
    for k_local, v_inner in T.Parallel(K, BV):
        v_idx = v_base + v_inner
        if v_idx < V:
            dst[batch_id, chunk_id, hv_id, k_local, v_idx] = T.cast(src[k_local, v_inner], out_dtype)


@T.macro
def _store_final_const(dst: T.Buffer,
                       src: T.Buffer,
                       seq_id,
                       hv_id,
                       v_base,
                       K: int,
                       BV: int,
                       V: int) -> None:
    for k_local, v_inner in T.Parallel(K, BV):
        v_idx = v_base + v_inner
        if v_idx < V:
            dst[seq_id, hv_id, k_local, v_idx] = src[k_local, v_inner]


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}, )
def chunk_gated_delta_rule_fwd_h_kernel_unsplit(
    H: int,
    HV: int,
    K: int,
    V: int,
    BT: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    g_dtype: torch.dtype,
    cu_seqlen_dtype: torch.dtype,
    use_g: bool,
    use_gk: bool,
    use_initial_state: bool,
    store_final_state: bool,
    save_new_value: bool,
    use_exp2: bool,
    is_varlen: bool,
    num_warps: int = 4,
):
    B = 1 if is_varlen else T.dynamic('B')
    N_seq = T.dynamic('N_seq') if is_varlen else B
    if is_varlen:
        N_state = T.dynamic('N_state') if (use_initial_state or store_final_state) else 1
    else:
        N_state = B if (use_initial_state or store_final_state) else 1
    TT = T.dynamic('TT')
    NT = T.dynamic('NT')
    seq_count = N_seq if is_varlen else B
    state_shape = (N_state, HV, K, V)

    BV = 32
    assert K in (64, 128) and V in (64, 128) and BT == 64

    num_threads = num_warps * 32

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_main(
        K_in: T.Tensor((B, TT, H, K), dtype=dtype),
        W_in: T.Tensor((B, TT, HV, K), dtype=dtype),
        U_in: T.Tensor((B, TT, HV, V), dtype=dtype),
        G: T.Tensor((B, TT, HV), dtype=g_dtype) = None,
        GK: T.Tensor((B, TT, HV, K), dtype=g_dtype) = None,
        InitialState: T.Tensor(state_shape, dtype=state_dtype) = None,
        CuSeqlens: T.Tensor((N_seq + 1,), dtype=cu_seqlen_dtype) = None,
        ChunkOffsets: T.Tensor((N_seq + 1,), dtype=cu_seqlen_dtype) = None,
        H_out: T.Tensor((B, NT, HV, K, V), dtype=dtype) = None,
        VNew_out: T.Tensor((B, TT, HV, V), dtype=dtype) = None,
        FinalState_out: T.Tensor(state_shape, dtype=torch.float32) = None,
    ):
        with T.Kernel(T.ceildiv(V, BV), seq_count * HV, threads=num_threads) as (i_v, i_nh):
            seq_id = T.cast(i_nh // HV, T.int32)
            hv_id = T.cast(i_nh % HV, T.int32)
            h_id = T.cast(hv_id // (HV // H), T.int32)

            if is_varlen:
                bos = T.cast(CuSeqlens[seq_id], T.int32)
                eos = T.cast(CuSeqlens[seq_id + 1], T.int32)
                seqlen = eos - bos
                chunk_base = T.cast(ChunkOffsets[seq_id], T.int32)
                batch_id = T.cast(0, T.int32)
                num_chunks = T.ceildiv(seqlen, BT)
            else:
                bos = T.cast(0, T.int32)
                seqlen = T.cast(TT, T.int32)
                chunk_base = T.cast(0, T.int32)
                batch_id = seq_id
                num_chunks = T.cast(NT, T.int32)

            v_base = T.cast(i_v * BV, T.int32)

            b_h_shared = T.alloc_shared([K, BV], dtype)
            b_h_frag = T.alloc_fragment([K, BV], T.float32)
            b_w = T.alloc_shared([BT, K], dtype)
            b_k = T.alloc_shared([BT, K], dtype)
            b_u = T.alloc_shared([BT, BV], dtype)
            b_vnew = T.alloc_shared([BT, BV], dtype)
            u_frag = T.alloc_fragment([BT, BV], T.float32)
            vnew_frag = T.alloc_fragment([BT, BV], T.float32)
            b_row_scale = T.alloc_shared([BT], T.float32)
            b_gk = T.alloc_shared([K, 1], T.float32)
            T.annotate_layout({
                b_w: tilelang.layout.make_swizzled_layout(b_w),
                b_k: tilelang.layout.make_swizzled_layout(b_k),
                b_u: tilelang.layout.make_swizzled_layout(b_u),
                b_vnew: tilelang.layout.make_swizzled_layout(b_vnew),
            })

            if use_initial_state:
                _load_state_const(b_h_frag, InitialState, seq_id, hv_id, v_base, K, BV, V)
            else:
                T.clear(b_h_frag)

            for chunk_idx in T.Pipelined(num_chunks, num_stages=4):
                chunk_slot = T.cast(chunk_base + chunk_idx, T.int32) if is_varlen else T.cast(chunk_idx, T.int32)
                seq_base = T.cast(chunk_idx * BT, T.int32)
                chunk_end_candidate = T.cast((chunk_idx + 1) * BT, T.int32)
                chunk_end = T.if_then_else(chunk_end_candidate < seqlen, chunk_end_candidate, seqlen)
                last_idx = chunk_end - 1

                T.copy(b_h_frag, b_h_shared)
                _store_h_const(H_out, b_h_shared, batch_id, chunk_slot, hv_id, v_base, K, BV, V, dtype)

                for t, k_local in T.Parallel(BT, K):
                    row = seq_base + t
                    if row < seqlen:
                        b_w[t, k_local] = W_in[batch_id, bos + row, hv_id, k_local]
                        b_k[t, k_local] = K_in[batch_id, bos + row, h_id, k_local]
                    else:
                        b_w[t, k_local] = T.cast(0.0, dtype)
                        b_k[t, k_local] = T.cast(0.0, dtype)

                for t, v_inner in T.Parallel(BT, BV):
                    row = seq_base + t
                    v_idx = v_base + v_inner
                    if row < seqlen and v_idx < V:
                        b_u[t, v_inner] = U_in[batch_id, bos + row, hv_id, v_idx]
                    else:
                        b_u[t, v_inner] = T.cast(0.0, dtype)

                g_last = T.alloc_var(T.float32)
                g_last = 0.0
                if use_g:
                    g_last = T.cast(G[batch_id, bos + last_idx, hv_id], T.float32)
                    for t in T.Parallel(BT):
                        row = seq_base + t
                        if row < seqlen:
                            g_row = T.cast(G[batch_id, bos + row, hv_id], T.float32)
                            if use_exp2:
                                b_row_scale[t] = T.exp2(g_last - g_row)
                            else:
                                b_row_scale[t] = T.exp(g_last - g_row)

                if use_gk:
                    for k_local, _ in T.Parallel(K, 1):
                        gk_val = T.cast(GK[batch_id, bos + last_idx, hv_id, k_local], T.float32)
                        if use_exp2:
                            b_gk[k_local, 0] = T.exp2(gk_val)
                        else:
                            b_gk[k_local, 0] = T.exp(gk_val)

                T.copy(b_u, u_frag)
                T.gemm(b_w, b_h_shared, vnew_frag, clear_accum=True)
                for t, v_inner in T.Parallel(BT, BV):
                    vnew_frag[t, v_inner] = u_frag[t, v_inner] - vnew_frag[t, v_inner]
                if save_new_value:
                    T.copy(vnew_frag, b_vnew)
                    for t, v_inner in T.Parallel(BT, BV):
                        row = seq_base + t
                        v_idx = v_base + v_inner
                        if row < seqlen and v_idx < V:
                            VNew_out[batch_id, bos + row, hv_id, v_idx] = b_vnew[t, v_inner]
                if use_g:
                    for t, v_inner in T.Parallel(BT, BV):
                        vnew_frag[t, v_inner] = vnew_frag[t, v_inner] * b_row_scale[t]
                T.copy(vnew_frag, b_vnew)

                h_scale_base = T.alloc_var(T.float32)
                h_scale_base = 1.0
                if use_g:
                    if use_exp2:
                        h_scale_base = T.exp2(g_last)
                    else:
                        h_scale_base = T.exp(g_last)

                if use_g or use_gk:
                    for k_local, v_inner in T.Parallel(K, BV):
                        scale = T.alloc_var(T.float32)
                        scale = h_scale_base
                        if use_gk:
                            scale = scale * b_gk[k_local, 0]
                        b_h_frag[k_local, v_inner] = b_h_frag[k_local, v_inner] * scale

                T.gemm(b_k, b_vnew, b_h_frag, transpose_A=True)

            if store_final_state:
                _store_final_const(FinalState_out, b_h_frag, seq_id, hv_id, v_base, K, BV, V)

    return chunk_gated_delta_rule_fwd_h_main

def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute the forward chunk-state recurrence for chunk gated delta
    rule."""
    B, T, H, K = k.shape
    HV, V = u.shape[2], u.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        N = B
        NT = (T + BT - 1) // BT
        chunk_offsets = None
    else:
        assert chunk_indices is not None
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    assert transpose_state_layout is False, 'transpose_state_layout=True is not supported'
    assert K in (64, 128) and V in (64, 128) and BT == 64, 'kernel supports K/V in {64,128} with BT=64'

    h = k.new_empty(B, NT, HV, K, V)
    final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    kernel = chunk_gated_delta_rule_fwd_h_kernel_unsplit(
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        dtype=k.dtype,
        state_dtype=initial_state.dtype if initial_state is not None else torch.float32,
        g_dtype=g.dtype if g is not None else (gk.dtype if gk is not None else torch.float32),
        cu_seqlen_dtype=cu_seqlens.dtype if cu_seqlens is not None else torch.long,
        use_g=g is not None,
        use_gk=gk is not None,
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
        save_new_value=save_new_value,
        use_exp2=use_exp2,
        is_varlen=cu_seqlens is not None,
    )
    kernel(
        k,
        w,
        u,
        g,
        gk,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        h,
        v_new,
        final_state,
    )
    return h, v_new, final_state
