# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from fla.ops.utils.index import prepare_chunk_indices

from .chunk_fwd import chunk_gated_delta_rule_fwd_intra
from .utils import chunk_local_cumsum

RCP_LN2 = 1.4426950216


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: Any | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
):
    assert use_gate_in_kernel is False, 'use_gate_in_kernel=True is not supported in the current implementation'
    g = chunk_local_cumsum(
        g,
        chunk_size=64,
        scale=RCP_LN2 if use_exp2 else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )

    return [None] * 6


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: Any | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
):
    # Validate head dimensions
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f'q and k must have the same number of heads, '
            f'but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}'
        )
    H, HV = q.shape[2], v.shape[2]
    if HV % H != 0:
        raise ValueError(
            f'For GVA, num_v_heads (HV={HV}) must be evenly divisible by '
            f'num_heads (H={H}), but got HV % H = {HV % H}'
        )

    if cp_context is not None:
        raise NotImplementedError('CUDA Gated Delta Rule does not support CP context')


    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f'The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.'
                f'Please flatten variable-length inputs before processing.',
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f'The number of initial states is expected to be equal to the number of input sequences, '
                f'i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.',
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    use_gate_in_kernel = False
    A_log = None
    dt_bias = None

    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

    chunk_indices = prepare_chunk_indices(
        cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
    _, o, _, final_state, _, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    return o.to(q.dtype), final_state
