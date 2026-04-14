# Copyright (c) OpenMMLab. All rights reserved.

import torch
import tilelang
import tilelang.language as T



@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def l2norm_fwd_kernel(
    D: int,
    eps: float,
    dtype: torch.dtype,
    output_dtype: torch.dtype):
    """l2norm forward kernel"""
    TT = T.dynamic('TT')
    warp_size = 32
    data_num_bits = T.DataType(dtype).bits
    E_PER_T = 128 // data_num_bits
    E_PER_W = warp_size * E_PER_T

    # result could be wrong if D is not power of 2
    R_PER_W = max(1, E_PER_W // D)
    num_warps = max(1, 8 // R_PER_W)
    ROWS = R_PER_W * num_warps

    E_LOCAL = T.ceildiv(D, warp_size)
    num_threads = num_warps * warp_size

    @T.prim_func
    def l2norm_fwd_main(
        X: T.Tensor((TT, D), dtype=dtype),
        Out: T.Tensor((TT, D), dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(TT, ROWS), threads=num_threads) as i_t:
            tid = T.get_thread_binding(0)
            warp_id = tid // warp_size
            lane_id = tid % warp_size

            for r in range(R_PER_W):
                r_id = r + warp_id * R_PER_W
                valid = i_t * ROWS + r_id < TT
                x_local = T.alloc_local((E_LOCAL, ), dtype=T.float32)
                for i in T.Vectorized(E_LOCAL):
                    if valid:
                        x_local[i] = T.cast(X[i_t * ROWS + r_id, lane_id * E_LOCAL + i], T.float32)
                    else:
                        x_local[i] = T.cast(0, T.float32)

                p2sum = T.alloc_var(dtype=T.float32)
                p2sum = 0
                for i in T.Unroll(E_LOCAL):
                    p2sum = p2sum + x_local[i] * x_local[i]
                p2sum = T.warp_reduce_sum(p2sum)
                rstd = T.rsqrt(p2sum + eps)
                for i in T.Vectorized(E_LOCAL):
                    x_local[i] = x_local[i] * rstd
                
                for i in T.Vectorized(E_LOCAL):
                    if valid:
                        Out[i_t * ROWS + r_id, lane_id * E_LOCAL + i] = T.cast(x_local[i], output_dtype)

    return l2norm_fwd_main



def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    """l2norm"""
    D = x.size(-1)
    if not x.is_contiguous():
        return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps).to(output_dtype or x.dtype)

    x_shape = x.shape
    x = x.view(-1, x.shape[-1])

    out = torch.empty_like(x, dtype=output_dtype or x.dtype)

    kernel = l2norm_fwd_kernel(D=D, eps=eps, dtype=x.dtype, output_dtype=out.dtype)
    kernel(x, out)

    out = out.view(x_shape)
    return out
