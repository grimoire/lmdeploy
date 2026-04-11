# Copyright (c) OpenMMLab. All rights reserved.
import torch


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass
