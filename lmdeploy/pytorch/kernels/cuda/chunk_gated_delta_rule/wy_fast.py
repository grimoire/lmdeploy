# Copyright (c) OpenMMLab. All rights reserved.
import tilelang
import tilelang.language as T
import tilelang.layout
import torch


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}, )
def recompute_w_u_fwd_kernel(H: int,
                             HV: int,
                             K: int,
                             V: int,
                             BT: int,
                             k_post_stride: tuple,
                             v_post_stride: tuple,
                             dtype: torch.dtype,
                             cu_seqlen_dtype: torch.dtype,
                             a_dtype: torch.dtype,
                             beta_dtype: torch.dtype,
                             g_dtype: torch.dtype,
                             use_g: bool,
                             use_exp2: bool,
                             is_varlen: bool):

    B = 1 if is_varlen else T.dynamic('B')
    N = T.dynamic('N')
    TT = T.dynamic('TT')
    k_stride0 = T.dynamic('k_stride0')
    v_stride0 = T.dynamic('v_stride0')
    k_stride = (k_stride0, *k_post_stride)
    v_stride = (v_stride0, *v_post_stride)

    NC = T.dynamic('NC')
    NT = T.ceildiv(TT, BT) if not is_varlen else NC

    BK = 64
    BV = 64
    NV = (V + BV - 1) // BV
    NK = (K + BK - 1) // BK
    num_stages_v = min(NV, 2) if NV > 2 else 1
    num_stages_k = min(NK, 2) if NK > 2 else 1

    @T.prim_func
    def recompute_w_u_fwd_main(
        K_in: T.StridedTensor((B, TT, H, K), dtype=dtype, strides=k_stride),
        V_in: T.StridedTensor((B, TT, HV, V), dtype=dtype, strides=v_stride),
        Beta: T.Tensor((B, TT, HV), dtype=beta_dtype),
        A: T.Tensor((B, TT, HV, BT), dtype=a_dtype),
        G: T.Tensor((B, TT, HV), dtype=g_dtype) = None,
        CuSeqlens: T.Tensor((N + 1,), dtype=cu_seqlen_dtype) = None,
        ChunkIndices: T.Tensor((NT, 2), dtype=torch.int32) = None,
        W_out: T.Tensor((B, TT, HV, K), dtype=dtype) = None,
        U_out: T.Tensor((B, TT, HV, V), dtype=dtype) = None,
    ):
        with T.Kernel(NT, B * HV, threads=256) as (i_t, i_bh):
            i_b = i_bh // HV
            i_h = i_bh % HV

            if is_varlen:
                i_n = ChunkIndices[i_t, 0]
                i_t = ChunkIndices[i_t, 1]
                bos = CuSeqlens[i_n]
                seqlen = CuSeqlens[i_n + 1] - bos
            else:
                bos = 0
                seqlen = TT

            b_A = T.alloc_shared((BT, BT), dtype=a_dtype)
            for i, j in T.Parallel(BT, BT):
                row_offset = i_t * BT + i
                if row_offset < seqlen:
                    b_A[i, j] = A[i_b, bos + row_offset, i_h, j]
                else:
                    b_A[i, j] = T.cast(0.0, a_dtype)

            b_bd = T.alloc_shared((BT,), dtype=beta_dtype)
            for i in T.Parallel(BT):
                idx = bos + i_t * BT + i
                b_bd[i] = T.if_then_else(
                    i_t * BT + i < seqlen,
                    Beta[i_b, idx, i_h],
                    T.cast(0.0, beta_dtype),
                )

            for i_v in T.Pipelined(NV, num_stages=num_stages_v):
                b_vb = T.alloc_shared((BT, BV), dtype=dtype)
                for i, j in T.Parallel(BT, BV):
                    row_offset = i_t * BT + i
                    v_idx = i_v * BV + j
                    if row_offset < seqlen and v_idx < V:
                        b_vb[i, j] = T.cast(V_in[i_b, bos + row_offset, i_h, v_idx] * b_bd[i], dtype)
                    else:
                        b_vb[i, j] = T.cast(0.0, dtype)

                u_frag = T.alloc_fragment((BT, BV), dtype=T.float32)
                b_u = T.alloc_shared((BT, BV), dtype=dtype)
                T.annotate_layout(
                    {b_u: tilelang.layout.make_swizzled_layout(b_u)}
                )
                T.clear(u_frag)
                T.gemm(b_A, b_vb, u_frag)
                T.copy(u_frag, b_u)

                for i, j in T.Parallel(BT, BV):
                    row_offset = i_t * BT + i
                    v_idx = i_v * BV + j
                    if row_offset < seqlen and v_idx < V:
                        U_out[i_b, bos + row_offset, i_h, v_idx] = b_u[i, j]

            if use_g:
                b_g = T.alloc_shared((BT,), dtype=T.float32)
                for i in T.Parallel(BT):
                    idx = bos + i_t * BT + i
                    if i_t * BT + i < seqlen:
                        g_val = T.cast(G[i_b, idx, i_h], T.float32)
                        if use_exp2:
                            b_g[i] = T.exp2(g_val)
                        else:
                            b_g[i] = T.exp(g_val)
                    else:
                        b_g[i] = 0.0

            for i_k in T.Pipelined(NK, num_stages=num_stages_k):
                b_kb = T.alloc_shared((BT, BK), dtype=dtype)
                for i, j in T.Parallel(BT, BK):
                    row_offset = i_t * BT + i
                    k_idx = i_k * BK + j
                    if row_offset < seqlen and k_idx < K:
                        kb_val = K_in[i_b, bos + row_offset, i_h // (HV // H), k_idx] * b_bd[i]
                        if use_g:
                            b_kb[i, j] = T.cast(T.cast(kb_val, T.float32) * b_g[i], dtype)
                        else:
                            b_kb[i, j] = kb_val
                    else:
                        b_kb[i, j] = T.cast(0.0, dtype)

                w_frag = T.alloc_fragment((BT, BK), dtype=T.float32)
                b_w = T.alloc_shared((BT, BK), dtype=dtype)
                T.annotate_layout(
                    {b_w: tilelang.layout.make_swizzled_layout(b_w)}
                )
                T.clear(w_frag)
                T.gemm(b_A, b_kb, w_frag)
                T.copy(w_frag, b_w)

                for i, j in T.Parallel(BT, BK):
                    row_offset = i_t * BT + i
                    k_idx = i_k * BK + j
                    if row_offset < seqlen and k_idx < K:
                        W_out[i_b, bos + row_offset, i_h, k_idx] = b_w[i, j]

    return recompute_w_u_fwd_main


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the chunk-local WY representation tensors ``w`` and ``u``.

    This kernel consumes the solved intra-chunk matrix ``A`` and produces two
    tensors used by the later gated-delta-rule forward kernels:

    - ``u = A @ (beta * v)``
    - ``w = A @ (beta * k * gate)``

    where ``gate`` is optional. If ``g`` is provided, the gate factor is
    ``exp(g)`` or ``exp2(g)`` depending on ``use_exp2``; otherwise the gate is
    treated as 1.

    The computation is performed independently for each chunk and value head.
    For grouped-value attention, key heads are shared across value heads, so the
    key head used for a value head ``hv`` is ``hv // (HV // H)``.

    Args:
        k: Key tensor of shape ``[B, T, H, K]``.
        v: Value tensor of shape ``[B, T, HV, V]``.
        beta: Per-token scaling tensor of shape ``[B, T, HV]``.
        A: Solved chunk-local matrix of shape ``[B, T, HV, BT]`` where ``BT`` is
            the chunk size.
        g: Optional cumulative gate tensor of shape ``[B, T, HV]``.
        cu_seqlens: Optional cumulative sequence lengths for varlen mode.
        chunk_indices: Optional precomputed ``(sequence_idx, chunk_idx)`` pairs
            for varlen mode.
        use_exp2: Whether to interpret the gate in base-2 exponent space.

    Returns:
        A tuple ``(w, u)`` where:

        - ``w`` has shape ``[B, T, HV, K]``
        - ``u`` has shape ``[B, T, HV, V]``
    """
    B, TT, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]

    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int32) if cu_seqlens.dtype != torch.int32 else cu_seqlens
        chunk_indices = chunk_indices.to(torch.int32) if chunk_indices.dtype != torch.int32 else chunk_indices
        assert chunk_indices is not None
        assert cu_seqlens.is_contiguous()
        assert chunk_indices.is_contiguous()

    w = k.new_empty(B, TT, HV, K)
    u = torch.empty_like(v)
    kernel = recompute_w_u_fwd_kernel(
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        k_post_stride=k.stride()[1:],
        v_post_stride=v.stride()[1:],
        dtype=k.dtype,
        cu_seqlen_dtype=cu_seqlens.dtype if cu_seqlens is not None else torch.int32,
        a_dtype=A.dtype,
        beta_dtype=beta.dtype,
        g_dtype=g.dtype if g is not None else torch.float,
        use_g=g is not None,
        use_exp2=use_exp2,
        is_varlen=cu_seqlens is not None,
    )
    kernel(
        k,
        v,
        beta,
        A,
        g,
        cu_seqlens,
        chunk_indices,
        w,
        u,
    )

    return w, u
