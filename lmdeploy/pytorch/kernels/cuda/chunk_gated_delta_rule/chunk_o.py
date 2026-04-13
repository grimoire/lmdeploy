# Copyright (c) OpenMMLab. All rights reserved.
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}, )
def chunk_fwd_kernel_o(H: int,
                       HV: int,
                       K: int,
                       V: int,
                       BT: int,
                       scale: float | None,
                       dtype: torch.dtype,
                       g_dtype: torch.dtype,
                       cu_seqlen_dtype: torch.dtype,
                       is_varlen: bool,
                       transpose_state: bool,
                       use_g: bool,
                       use_exp2: bool):
    B = 1 if is_varlen else T.dynamic('B')
    TT = T.dynamic('TT')
    N = T.dynamic('N')
    NC = T.dynamic('NC')
    NT = T.ceildiv(TT, BT) if not is_varlen else NC
    state_shape = (B, NT, HV, K, V)

    if scale is None:
        scale = K ** -0.5

    BK = 128
    BV = 128
    num_stages = 1
    num_warps = 8
    num_threads = 32 * num_warps

    @T.prim_func
    def chunk_fwd_o_main(
        Q_in: T.Tensor((B, TT, H, K), dtype=dtype),
        K_in: T.Tensor((B, TT, H, K), dtype=dtype),
        V_in: T.Tensor((B, TT, HV, V), dtype=dtype),
        State: T.Tensor(state_shape, dtype=dtype),
        Out: T.Tensor((B, TT, HV, V), dtype=dtype),
        G: T.Tensor((B, TT, HV), dtype=g_dtype) = None,
        CuSeqlens: T.Tensor((N + 1,), dtype=cu_seqlen_dtype) = None,
        ChunkIndices: T.Tensor((NT, 2), dtype=cu_seqlen_dtype) = None,
    ):
        with T.Kernel(T.ceildiv(V, BV), NT, B * HV, threads=num_threads) as (i_v, i_t, i_bh):
            i_b = i_bh // HV
            i_h = i_bh % HV
            i_qkh = i_h // (HV // H)
            if is_varlen:
                i_tg = i_t
                i_n = ChunkIndices[i_t, 0]
                i_t = ChunkIndices[i_t, 1]
                bos = CuSeqlens[i_n]
                seqlen = CuSeqlens[i_n + 1] - bos
            else:
                i_tg = i_t
                bos = 0
                seqlen = TT

            b_o = T.alloc_fragment((BT, BV), dtype=T.float32)
            b_A = T.alloc_fragment((BT, BT), dtype=T.float32)
            b_av = T.alloc_fragment((BT, BV), dtype=T.float32)
            b_q = T.alloc_shared((BT, BK), dtype=dtype)
            b_k = T.alloc_shared((BT, BK), dtype=dtype)
            b_v = T.alloc_shared((BT, BV), dtype=dtype)

            T.clear(b_o)
            T.clear(b_A)
            T.clear(b_av)

            offs_t  = i_t * BT
            offs_v = i_v * BV
            for i_k in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
                offs_k = i_k * BK

                # load q, k, h
                for i, j in T.Parallel(BT, BK):
                    if offs_t + i < seqlen and offs_k + j < K:
                        b_q[i, j] = Q_in[i_b, bos + offs_t + i, i_qkh, offs_k + j]
                    else:
                        b_q[i, j] = 0.0 
                
                for i, j in T.Parallel(BT, BK):
                    if offs_t + i < seqlen and offs_k + j < K:
                        b_k[i, j] = K_in[i_b, bos + offs_t + i, i_qkh, offs_k + j]
                    else:
                        b_k[i, j] = 0.0

                if transpose_state:
                    b_h = T.alloc_shared((BV, BK), dtype=dtype)
                    for i, j in T.Parallel(BV, BK):
                        if offs_v + i < V and offs_k + j < K:
                            b_h[i, j] = State[i_b, i_tg, i_h, offs_v + i, offs_k + j]
                        else:
                            b_h[i, j] = 0.0
                else:
                    b_h = T.alloc_shared((BK, BV), dtype=dtype)
                    for i, j in T.Parallel(BK, BV):
                        if offs_k + i < K and offs_v + j < V:
                            b_h[i, j] = State[i_b, i_tg, i_h, offs_k + i, offs_v + j]
                        else:
                            b_h[i, j] = 0.0

                T.gemm(b_q, b_h, b_o, transpose_B=transpose_state)
                T.gemm(b_q, b_k, b_A, transpose_B=True)

            if use_g:
                b_g = T.alloc_shared((BT,), dtype=T.float32)
                for i in T.Parallel(BT):
                    if offs_t + i < seqlen:
                        b_g[i] = T.cast(G[i_b, bos + offs_t + i, i_h], T.float32)
                    else:
                        b_g[i] = 0.0
                for i, j in T.Parallel(BT, BV):
                    b_o[i, j] = b_o[i, j] * (T.exp2(b_g[i]) if use_exp2 else T.exp(b_g[i]))
                for i, j in T.Parallel(BT, BT):
                    b_A[i, j] = b_A[i, j] * (
                        T.exp2(b_g[i] - b_g[j]) if use_exp2 else T.exp(b_g[i] - b_g[j]))

            for i, j in T.Parallel(BT, BT):
                o_t0 = i_t * BT + i
                o_t1 = i_t * BT + j
                if o_t0 < o_t1 or o_t0 >= seqlen:
                    b_A[i, j] = 0.0

            for i, j in T.Parallel(BT, BV):
                if offs_t + i < seqlen and offs_v + j < V:
                    b_v[i, j] = V_in[i_b, bos + offs_t + i, i_h, offs_v + j]
                else:
                    b_v[i, j] = 0.0

            b_sA = T.alloc_shared((BT, BT), dtype=dtype)
            T.copy(b_A, b_sA)

            T.gemm(b_sA, b_v, b_av, clear_accum=True)
            for i, j in T.Parallel(BT, BV):
                b_o[i, j] = b_o[i, j] * scale + b_av[i, j] * scale
            
            T.copy(b_o, b_v)
            
            for i, j in T.Parallel(BT, BV):
                if offs_t + i < seqlen and offs_v + j < V:
                    Out[i_b, bos + offs_t + i, i_h, offs_v + j] = b_v[i, j]

    return chunk_fwd_o_main

def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> torch.Tensor:
    assert g_gamma is None
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size

    o = torch.empty_like(v)
    kernel = chunk_fwd_kernel_o(
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        scale=scale,
        dtype=q.dtype,
        g_dtype=g.dtype if g is not None else torch.float32,
        cu_seqlen_dtype=cu_seqlens.dtype if cu_seqlens is not None else torch.int64,
        is_varlen=cu_seqlens is not None,
        transpose_state=transpose_state_layout,
        use_g=g is not None,
        use_exp2=use_exp2,
    )
    kernel(
        q,
        k,
        v,
        h,
        o,
        g,
        cu_seqlens,
        chunk_indices,
    )

    return o
