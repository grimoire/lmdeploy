# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import torch
import triton
import triton.language as tl

from lmdeploy.pytorch.kernels.cuda.utils import get_device_props

from .activation import silu_and_mul


@triton.jit
def _get_exp_mask_kernel(
    a_ptr,
    o_mask_ptr,
    o_k_ptr,
    stride_a_token: tl.constexpr,
    stride_a_exp: tl.constexpr,
    stride_o_exp,
    stride_o_token: tl.constexpr,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_NA: tl.constexpr,
    BLOCK_NO: tl.constexpr,
):
    token_id = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_NA)
    mask_n = offs_n < topk
    a_ptrs = a_ptr + token_id * stride_a_token + offs_n * stride_a_exp
    a = tl.load(a_ptrs, mask=mask_n)

    # fill zeros
    offs_no = tl.arange(0, BLOCK_NO)
    mask_no = offs_no < num_experts
    o_ptrs = o_mask_ptr + token_id * stride_o_token + offs_no * stride_o_exp
    tl.store(o_ptrs, 0, mask=mask_no)

    # fill a
    o_ptrs = o_mask_ptr + token_id * stride_o_token + a * stride_o_exp
    tl.store(o_ptrs, 1, mask=mask_n)

    # fill kid
    ok_ptrs = o_k_ptr + token_id * stride_o_token + a * stride_o_exp
    tl.store(ok_ptrs, offs_n, mask=mask_n)


def _get_exp_mask(topk_ids: torch.Tensor, num_experts: int):
    """get exp mask."""
    assert topk_ids.dim() == 2
    M, topk = topk_ids.shape
    assert topk <= num_experts

    out_mask = topk_ids.new_empty((num_experts, M))
    out_k = topk_ids.new_empty((num_experts, M))
    BLOCK_NA = triton.next_power_of_2(topk)
    BLOCK_NO = triton.next_power_of_2(num_experts)

    grid = (M, )
    _get_exp_mask_kernel[grid](
        topk_ids,
        out_mask,
        out_k,
        stride_a_token=topk_ids.stride(0),
        stride_a_exp=topk_ids.stride(1),
        stride_o_exp=out_mask.stride(0),
        stride_o_token=out_mask.stride(1),
        topk=topk,
        num_experts=num_experts,
        BLOCK_NA=BLOCK_NA,
        BLOCK_NO=BLOCK_NO,
        num_warps=1,
    )
    return out_mask, out_k


@triton.jit
def _get_padded_shapes_kernel(lens_ptr, out_ptr, num_elem: tl.constexpr, stride_len, stride_out: tl.constexpr,
                              aligned_size: tl.constexpr, BLOCK_N: tl.constexpr):
    """get padded shapes kernel."""
    offs = tl.arange(0, BLOCK_N)
    mask = offs < num_elem
    lens_ptrs = lens_ptr + offs * stride_len
    lens = tl.load(lens_ptrs, mask=mask)

    num_blocks = (lens + aligned_size - 1) // aligned_size
    padded_lens = num_blocks * aligned_size
    cum_lens = tl.cumsum(padded_lens, axis=0)

    out_ptrs = out_ptr + offs * stride_out
    tl.store(out_ptrs, cum_lens, mask=mask)


def _get_padded_shapes(lens: torch.Tensor, aligned_size: int):
    """get padded shape.

    >>> padded_size = (lens + aligned_size - 1) // aligned_size * aligned_size
    >>> cum_padded_size = padded_size.cumsum(0)
    """
    out = torch.empty_like(lens)
    num_elem = out.size(0)
    BLOCK_N = triton.next_power_of_2(num_elem)

    _get_padded_shapes_kernel[(1, )](
        lens,
        out,
        num_elem=num_elem,
        stride_len=lens.stride(0),
        stride_out=out.stride(0),
        aligned_size=aligned_size,
        BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit
def _get_m_indices_kernel(
    pos_ptr,
    out_ptr,
    m_sum,
    pos_stride,
    out_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_experts: tl.constexpr,
    aligned_size: tl.constexpr,
):
    """get m indices kernel."""
    exp_id = tl.program_id(0)

    end_pos_ptr = pos_ptr + exp_id * pos_stride

    if exp_id < num_experts:
        end_pos = tl.load(end_pos_ptr)
        if exp_id == 0:
            start_pos = tl.zeros_like(end_pos)
        else:
            start_pos = tl.load(end_pos_ptr - pos_stride)
        out_val = exp_id
    else:
        start_pos = tl.load(end_pos_ptr - pos_stride)
        end_pos = tl.zeros_like(start_pos) + m_sum
        out_val = -1

    start_pos = start_pos.cast(tl.int32)
    end_pos = end_pos.cast(tl.int32)
    offs = tl.arange(0, BLOCK_N)
    for idx in tl.range(start_pos, end_pos, BLOCK_N):
        idx = tl.multiple_of(idx, aligned_size)
        offs_n = idx + offs
        mask_n = offs_n < end_pos
        out_ptrs = out_ptr + offs_n * out_stride
        tl.store(out_ptrs, out_val, mask=mask_n)


def _get_m_indices(exp_pos: torch.Tensor, m_sum: int, aligned_size: int = 128, neg_empty: bool = False):
    """get m indices."""
    num_experts = exp_pos.size(0)
    out = exp_pos.new_empty(m_sum)
    if neg_empty:
        grid = (num_experts + 1, )
    else:
        grid = (num_experts, )
    _get_m_indices_kernel[grid](
        exp_pos,
        out,
        m_sum=m_sum,
        pos_stride=exp_pos.stride(0),
        out_stride=out.stride(0),
        BLOCK_N=128,
        aligned_size=aligned_size,
        num_experts=num_experts,
        num_warps=1,
    )
    return out


@triton.jit
def _get_permute_map_kernel(
    pos_ptr,
    mask_ptr,
    topk_map_ptr,
    out_ptr,
    stride_pos0,
    stride_pos1: tl.constexpr,
    stride_mask0: tl.constexpr,
    stride_mask1: tl.constexpr,
    stride_out0: tl.constexpr,
    stride_out1: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """get permute map kernel."""
    token_id = tl.program_id(0)
    offs_exp = tl.arange(0, BLOCK_N)
    pos_ptrs = pos_ptr + offs_exp * stride_pos0 + token_id * stride_pos1
    mask_ptrs = mask_ptr + offs_exp * stride_mask0 + token_id * stride_mask1
    topk_map_ptrs = topk_map_ptr + offs_exp * stride_mask0 + token_id * stride_mask1

    mask_exp = offs_exp < num_experts
    mask_pos = tl.load(mask_ptrs, mask=mask_exp)
    pos = tl.load(pos_ptrs, mask=mask_exp)
    topk_map = tl.load(topk_map_ptrs, mask=mask_exp)

    mask_out = mask_exp & (mask_pos > 0)
    out = pos - 1
    out_ptrs = out_ptr + token_id * stride_out0 + topk_map * stride_out1
    tl.store(out_ptrs, out, mask=mask_out)


def _get_permute_map(exp_token_pos: torch.Tensor, exp_token_mask: torch.Tensor, token_topk_map: torch.Tensor,
                     topk: int):
    """get permute map."""
    num_experts, M = exp_token_pos.shape
    out = exp_token_pos.new_empty((M, topk))

    BLOCK_N = triton.next_power_of_2(num_experts)
    grid = (M, )
    _get_permute_map_kernel[grid](
        exp_token_pos,
        exp_token_mask,
        token_topk_map,
        out,
        stride_pos0=exp_token_pos.stride(0),
        stride_pos1=exp_token_pos.stride(1),
        stride_mask0=exp_token_mask.stride(0),
        stride_mask1=exp_token_mask.stride(1),
        stride_out0=out.stride(0),
        stride_out1=out.stride(1),
        num_experts=num_experts,
        BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit
def _permute_inputs_kernel(
    inputs_ptr,
    permute_map_ptr,
    out_ptr,
    stride_in0,
    stride_in1: tl.constexpr,
    stride_map0,
    stride_map1: tl.constexpr,
    stride_out0,
    stride_out1: tl.constexpr,
    M,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """permute inputs kernel."""
    n_id = tl.program_id(0)
    m_id = tl.program_id(1)
    stride_m = tl.num_programs(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offs_k < topk
    mask_n = offs_n < hidden_size
    mask_out = mask_k[:, None] & mask_n[None, :]

    in_ptrs = inputs_ptr + m_id * stride_in0 + offs_n * stride_in1
    map_ptrs = permute_map_ptr + m_id * stride_map0 + offs_k * stride_map1
    out_base_ptrs = out_ptr + offs_n[None, :] * stride_out1

    for _ in tl.range(m_id, M, stride_m):
        offs_out_m = tl.load(map_ptrs, mask=mask_k)
        inputs = tl.load(in_ptrs, mask=mask_n)

        out_ptrs = out_base_ptrs + offs_out_m[:, None] * stride_out0
        tl.store(out_ptrs, inputs[None, :], mask=mask_out)
        in_ptrs += stride_m * stride_in0
        map_ptrs += stride_m * stride_map0


def _permute_inputs(inputs: torch.Tensor, permute_map: torch.Tensor, m_sum: int):
    """permute inputs kernel launcher."""
    M, hidden_size = inputs.shape
    topk = permute_map.size(1)
    out = inputs.new_empty((m_sum, hidden_size))

    BLOCK_N = 128
    BLOCK_K = triton.next_power_of_2(topk)

    num_warps = 1
    props = get_device_props(inputs.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    max_ctas = num_sm * warps_per_sm // num_warps

    grid0 = triton.cdiv(hidden_size, BLOCK_N)
    grid1 = min(M, max_ctas // grid0)
    grid = (
        grid0,
        grid1,
    )

    _permute_inputs_kernel[grid](
        inputs,
        permute_map,
        out,
        stride_in0=inputs.stride(0),
        stride_in1=inputs.stride(1),
        stride_map0=permute_map.stride(0),
        stride_map1=permute_map.stride(1),
        stride_out0=out.stride(0),
        stride_out1=out.stride(1),
        M=M,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out


def permute_inputs(inputs: torch.Tensor, topk_ids: torch.Tensor, num_experts: int, aligned_size: int = 128):
    """permute inputs."""
    assert inputs.dim() == 2
    assert topk_ids.dim() == 2
    M, _ = inputs.shape
    _, topk = topk_ids.shape
    m_blocks = triton.cdiv(M, aligned_size) * topk + num_experts - topk
    m_sum = m_blocks * aligned_size

    # get valid expert-token map.
    # exp_token_mask[i, j]: inputs[j] needs experts[i]
    # token_topk_map[i, j]: topk_ids[j, token_topk_map[i, j]] = i
    exp_token_mask, token_topk_map = _get_exp_mask(topk_ids, num_experts)

    # get token pos per experts
    # exp_token_pos[i, j]: inputs[j] pos in permuted experts[j] inputs.
    exp_token_pos = exp_token_mask.cumsum(1)
    exp_token_count = exp_token_pos[:, -1]

    # get padded shape, update exp_token_pos
    # padded_shapes[i]: aligned num tokens for expert i
    padded_shapes = _get_padded_shapes(exp_token_count, aligned_size)
    exp_token_pos[1:, :] += padded_shapes[:-1, None]

    # get m_indices
    # (m_sum,)
    m_indices = _get_m_indices(padded_shapes, m_sum, aligned_size, neg_empty=True)

    # permute map
    # (M, topk)
    permute_map = _get_permute_map(exp_token_pos, exp_token_mask, token_topk_map, topk)

    # permute input
    # (m_sum, hidden_dim)
    permuted_inputs = _permute_inputs(inputs, permute_map, m_sum)

    return permuted_inputs, m_indices, permute_map


@triton.jit
def _unpermute_outputs_kernel(
    in_ptr,
    weight_ptr,
    pmap_ptr,
    out_ptr,
    stride_in0: tl.constexpr,
    stride_in1: tl.constexpr,
    stride_w0: tl.constexpr,
    stride_w1: tl.constexpr,
    stride_pmap0: tl.constexpr,
    stride_pmap1: tl.constexpr,
    stride_out0: tl.constexpr,
    stride_out1: tl.constexpr,
    M,
    N: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """unpermute outputs kernel."""
    n_block_id = tl.program_id(0)
    m_id = tl.program_id(1)
    stride_m = tl.num_programs(1)

    offs_n = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < topk

    in_ptrs = in_ptr + offs_n[None, :] * stride_in1
    weight_ptrs = weight_ptr + m_id * stride_w0 + offs_k * stride_w1
    pmap_ptrs = pmap_ptr + m_id * stride_pmap0 + offs_k * stride_pmap1
    out_ptrs = out_ptr + m_id * stride_out0 + offs_n * stride_out1

    for _ in tl.range(m_id, M, stride_m):
        pmap = tl.load(pmap_ptrs, mask=mask_k)
        weights = tl.load(weight_ptrs, mask=mask_k)

        select_in_ptrs = in_ptrs + pmap[:, None] * stride_in0
        inputs = tl.load(select_in_ptrs, mask=mask_k[:, None] & mask_n[None, :])
        inputs = inputs * weights[:, None]
        outputs = tl.sum(inputs, 0)
        tl.store(out_ptrs, outputs, mask=mask_n)

        weight_ptrs += stride_m * stride_w0
        pmap_ptrs += stride_m * stride_pmap0
        out_ptrs += stride_m * stride_out0


def unpermute_outputs(permuted_output: torch.Tensor, weights: torch.Tensor, permuted_map: torch.Tensor):
    """unpermute outputs."""
    assert permuted_output.dim() == 2
    assert weights.dim() == 2
    assert permuted_map.dim() == 2

    M, topk = weights.shape
    hidden_size = permuted_output.size(1)

    assert permuted_map.size(0) == M
    assert permuted_map.size(1) == topk

    outs = permuted_output.new_empty(M, hidden_size)

    BLOCK_N = 256
    BLOCK_K = triton.next_power_of_2(topk)
    num_warps = 1

    props = get_device_props(permuted_output.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    max_ctas = num_sm * warps_per_sm // num_warps

    grid0 = triton.cdiv(hidden_size, BLOCK_N)
    grid1 = min(M, max_ctas // grid0)
    grid = (
        grid0,
        grid1,
    )
    _unpermute_outputs_kernel[grid](
        permuted_output,
        weights,
        permuted_map,
        outs,
        stride_in0=permuted_output.stride(0),
        stride_in1=permuted_output.stride(1),
        stride_w0=weights.stride(0),
        stride_w1=weights.stride(1),
        stride_pmap0=permuted_map.stride(0),
        stride_pmap1=permuted_map.stride(1),
        stride_out0=outs.stride(0),
        stride_out1=outs.stride(1),
        M=M,
        N=hidden_size,
        topk=topk,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    return outs


def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1,
        }, num_stages=4, num_warps=4),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K', 'BLOCK_SIZE_M'],
    warmup=10,
    rep=25,
)
@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_ids_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """grouped gemm kernel."""
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        # pid_m = pid % num_pid_m
        # pid_n = pid // num_pid_m
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_n = offs_n < N
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

    exp_id = tl.load(m_ids_ptr + pid_m * BLOCK_SIZE_M)
    if exp_id < 0:
        return

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + exp_id * stride_be + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        mask_k = offs_k < K - k
        a = tl.load(a_ptrs, mask=mask_k[None, :])
        b = tl.load(b_ptrs, mask=mask_k[:, None])

        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(a_ptr.dtype.element_ty)

    tl.store(c_ptrs, c, mask=mask_n[None, :])


def grouped_gemm(A: torch.Tensor, B: torch.Tensor, m_indices: torch.Tensor, aligned_size: int):
    """grouped gemm."""

    assert A.dim() == 2
    assert B.dim() == 3
    assert m_indices.dim() == 1
    assert m_indices.is_contiguous()

    M, K = A.shape
    _, N = B.shape[:2]
    K = B.size(2)

    assert m_indices.size(0) == M

    outs = A.new_empty(M, N)

    BLOCK_SIZE_M = aligned_size

    def _grid_fn(META):
        grid = (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        return grid

    _grouped_gemm_kernel[_grid_fn](
        A,
        B,
        outs,
        m_indices,
        M,
        K,
        N,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_cm=outs.stride(0),
        stride_cn=outs.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return outs


def _renormalize(topk_weights: torch.Tensor, renormalize: bool):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    return topk_weights


def fused_moe(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              topk: int,
              expert_offset: int = 0,
              num_experts: int = None,
              renormalize: bool = False) -> torch.Tensor:
    """fused moe."""
    E, _, _ = w1.shape
    if num_experts is None:
        num_experts = E

    topk_weights = _renormalize(topk_weights, renormalize)

    aligned_size = 64
    permuted_inputs, m_indices, permuted_map = permute_inputs(hidden_states,
                                                              topk_ids,
                                                              num_experts,
                                                              aligned_size=aligned_size)

    gate_up = grouped_gemm(permuted_inputs, w1, m_indices, aligned_size=aligned_size)
    activation = silu_and_mul(gate_up)
    down = grouped_gemm(activation, w2, m_indices, aligned_size=aligned_size)

    outputs = unpermute_outputs(down, weights=topk_weights, permuted_map=permuted_map)
    return outputs


def _make_intermediate(shape: tuple, dtype: torch.dtype, device: torch.device, zeros: bool):
    """make intermediate."""
    if zeros:
        return torch.zeros(shape, dtype=dtype, device=device)
    else:
        return torch.empty(shape, dtype=dtype, device=device)


@triton.jit
def _start_end_kernel(TopkIdx, SortedIdx, ExpStart, ExpEnd, len_sorted_idx: int, num_experts: tl.constexpr,
                      BLOCK: tl.constexpr):
    """start end kernel."""
    exp_id = tl.program_id(0)
    exp_start = -1
    cnt = 0

    s_off = tl.arange(0, BLOCK)

    # find start
    for sidx_start in range(0, len_sorted_idx, BLOCK):
        sidx_off = sidx_start + s_off
        sidx_mask = sidx_off < len_sorted_idx
        sidx = tl.load(SortedIdx + sidx_off, mask=sidx_mask, other=0)
        tidx = tl.load(TopkIdx + sidx, mask=sidx_mask, other=num_experts)
        tidx_mask = tidx == exp_id
        cnt += tl.sum(tidx_mask.to(tl.int32))
        if cnt > 0 and exp_start < 0:
            exp_start = sidx_start + tl.argmax(tidx_mask, axis=0)

    if exp_start < 0:
        exp_start *= 0
    exp_end = exp_start + cnt
    tl.store(ExpStart + exp_id, exp_start)
    tl.store(ExpEnd + exp_id, exp_end)


def get_start_end(topk_idx: torch.Tensor, sorted_idx: torch.Tensor, num_experts: int):
    """get start and end.

    same process as:
    >>> exp_tok_cnt = F.one_hot(flatten_topk_ids, num_classes=E).sum(0)
    >>> exp_end = exp_tok_cnt.cumsum(0)
    >>> exp_start = exp_end - exp_tok_cnt
    """
    start_end = sorted_idx.new_empty(2, num_experts)
    exp_start = start_end[0, :]
    exp_end = start_end[1, :]

    BLOCK = 128
    _start_end_kernel[(num_experts, )](
        topk_idx,
        sorted_idx,
        exp_start,
        exp_end,
        len_sorted_idx=sorted_idx.numel(),
        num_experts=num_experts,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
    )

    return exp_start, exp_end


def _get_sorted_idx(topk_ids: torch.Tensor, num_experts: int):
    """get sorted idx."""
    flatten_topk_ids = topk_ids.flatten()
    sorted_idx = flatten_topk_ids.argsort()

    exp_start, exp_end = get_start_end(flatten_topk_ids, sorted_idx, num_experts)
    return sorted_idx, exp_start, exp_end
