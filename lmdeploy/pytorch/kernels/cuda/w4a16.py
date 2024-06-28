# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from .triton_utils import get_kernel_meta, wrap_jit_func


@triton.jit
def get_rpack_order():
    order = tl.arange(0, 2)[None, :] * 4 + tl.arange(0, 4)[:, None]
    order = tl.ravel(order)
    return order


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 8
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=4),
    ],
    key=['N', 'K'],
)
@wrap_jit_func(
    type_hint=dict(
    A=Tensor,
    B=Tensor,
    C=Tensor,
    Zeros=Tensor,
    Scales=Tensor,
    M=torch.int32,
    N=torch.int32,
    K=torch.int32,
    bits=torch.int32,
    group_size=torch.int32,
    stride_am=int, stride_ak=int,
    stride_bk=int, stride_bn=int,
    stride_cm=int, stride_cn=int,
    stride_sk=int, stride_sn=int,
    stride_zk=int, stride_zn=int,
    BLOCK_SIZE_M=torch.int32, BLOCK_SIZE_N=torch.int32,
    BLOCK_SIZE_K=torch.int32, GROUP_SIZE_M=torch.int32
)
)
@triton.jit
def _w4a16_mm(A, B, C, Zeros, Scales,
              M, N, K, bits: tl.constexpr,
              group_size: tl.constexpr,
              stride_am, stride_ak, stride_bk, stride_bn,
              stride_cm, stride_cn, stride_sk, stride_sn,
              stride_zk, stride_zn,
              BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
              BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, K) float16
        B is of shape (K, N//8) int32
        C is of shape (M, N) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N//8) int32
    """
    elem_per_val: tl.constexpr = 32 // bits
    maxq: tl.constexpr = (1 << bits) - 1

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    # offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] // elem_per_val * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    scales_ptrs = Scales + offs_bn[None, :] * stride_sn
    zeros_ptrs = Zeros + (offs_bn[None, :] // elem_per_val * stride_zn)

    num_shift: tl.constexpr = BLOCK_SIZE_N // elem_per_val
    shifter = get_rpack_order()[None, :] * bits + tl.zeros((num_shift,), dtype=tl.int32)[:, None]
    shifter = tl.ravel(shifter)
    zeros_shifter = shifter
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        g_idx = (k * BLOCK_SIZE_K + offs_k) // group_size
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_sk)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zk)
        zeros = (zeros >> zeros_shifter[None, :]) & maxq

        a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[None, :]) & maxq  # Extract the N-bit values
        b = (b - zeros) * scales  # Scale and shift

        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = C + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def w4a16_linear(input: Tensor, weight: Tensor, scales: Tensor, zeros: Tensor):
    kernel_meta = get_kernel_meta(input)

    bits = 4
    M, K = input.size()
    N = scales.size(1)
    group_size = K // scales.size(0)

    def _grid_fn(META):
        grid = (
            triton.cdiv(M, META['BLOCK_SIZE_M']) *
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        return grid
    output = input.new_empty((M, N))
    _w4a16_mm[_grid_fn](
        input,
        weight,
        output,
        zeros,
        scales,
        M, N, K,
        bits,
        group_size,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        scales.stride(0),
        scales.stride(1),
        zeros.stride(0),
        zeros.stride(1),
        **kernel_meta
    )
    return output
