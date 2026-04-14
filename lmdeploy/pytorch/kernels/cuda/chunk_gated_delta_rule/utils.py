# Copyright (c) OpenMMLab. All rights reserved.
import tilelang
import tilelang.language as T
import torch


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}, )
def chunk_local_cumsum_scalar_kernel(H: int,
                                     chunk_size: int,
                                     dtype: torch.dtype,
                                     reverse: bool,
                                     scale: float | None,
                                     head_first: bool,
                                     output_dtype: torch.dtype,
                                     cu_seqlen_dtype: torch.dtype,
                                     is_varlen: bool):

    has_scale = scale is not None and scale != 1.0
    BT = chunk_size
    num_threads = min(BT, 32)

    B = T.dynamic('B')
    N = T.dynamic('N')
    QT = T.dynamic('QT')
    if is_varlen:
        g_shape = (1, QT, H) if not head_first else (1, H, QT)
    else:
        g_shape = (B, H, QT) if head_first else (B, QT, H)

    NC = T.dynamic('NC')
    NT = T.ceildiv(QT, BT) if not is_varlen else NC
    seq_count = 1 if is_varlen else B

    @T.prim_func
    def chunk_local_cumsum_scalar_main(
        G: T.Tensor(g_shape, dtype=dtype),
        Out: T.Tensor(g_shape, dtype=output_dtype),
        CuSeqlens: T.Tensor((N + 1,), dtype=cu_seqlen_dtype) = None,
        ChunkIndices: T.Tensor((NT, 2), dtype=torch.int32) = None,
    ):
        with T.Kernel(NT, seq_count * H, threads=num_threads) as (i_t, i_bh):
            i_b = 0 if is_varlen else i_bh // H
            i_h = i_bh % H
            if is_varlen:
                i_n = ChunkIndices[i_t, 0]
                i_t = ChunkIndices[i_t, 1]
                bos = CuSeqlens[i_n]
                seqlen = CuSeqlens[i_n + 1] - bos
            else:
                bos = 0
                seqlen = QT

            # Use shared memory directly for cumsum to avoid the
            # fragment→shared→cumsum→shared→fragment roundtrip that
            # T.cumsum(fragment) generates via the cumsum_fragment macro.
            s_buf = T.alloc_shared((BT,), dtype=T.float32)
            if head_first:
                for i in T.Parallel(BT):
                    offset = i_t * BT + i
                    if offset >= seqlen:
                        s_buf[i] = 0.0
                    else:
                        s_buf[i] = G[i_b, i_h, bos + offset]
            else:
                for i in T.Parallel(BT):
                    offset = i_t * BT + i
                    if offset >= seqlen:
                        s_buf[i] = 0.0
                    else:
                        s_buf[i] = G[i_b, bos + offset, i_h]

            T.cumsum(s_buf, s_buf, dim=0, reverse=reverse)

            if has_scale:
                for i in T.Parallel(BT):
                    s_buf[i] = s_buf[i] * scale

            if head_first:
                for i in T.Parallel(BT):
                    offset = i_t * BT + i
                    if offset < seqlen:
                        Out[i_b, i_h, bos + offset] = s_buf[i]
            else:
                for i in T.Parallel(BT):
                    offset = i_t * BT + i
                    if offset < seqlen:
                        Out[i_b, bos + offset, i_h] = s_buf[i]

    return chunk_local_cumsum_scalar_main


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    assert g.is_contiguous(), 'Input tensor must be contiguous'
    assert chunk_size == 2**(chunk_size.bit_length() - 1), 'chunk_size must be a power of 2'

    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    if cu_seqlens is not None:
        assert B == 1, 'Only batch size 1 is supported when cu_seqlens are provided'
        assert chunk_indices is not None, 'chunk_indices must be provided when cu_seqlens are provided'
        cu_seqlens = cu_seqlens.to(torch.int32) if cu_seqlens.dtype != torch.int32 else cu_seqlens
        chunk_indices = chunk_indices.to(torch.int32) if chunk_indices.dtype != torch.int32 else chunk_indices
        assert cu_seqlens.is_contiguous(), 'cu_seqlens tensor must be contiguous'
        assert chunk_indices.is_contiguous(), 'chunk_indices tensor must be contiguous'
    out = torch.empty_like(g, dtype=output_dtype or g.dtype)

    kernel = chunk_local_cumsum_scalar_kernel(
        H=H,
        chunk_size=chunk_size,
        dtype=g.dtype,
        reverse=reverse,
        scale=scale,
        head_first=head_first,
        output_dtype=output_dtype or g.dtype,
        cu_seqlen_dtype=cu_seqlens.dtype if cu_seqlens is not None else torch.int32,
        is_varlen=cu_seqlens is not None,
    )
    kernel(
        g,
        out,
        cu_seqlens,
        chunk_indices,
    )

    return out


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, 'Only batch size 1 is supported when cu_seqlens are provided'

    if len(g.shape) != 3:
        raise ValueError(
            f'Unsupported input shape {g.shape}, '
            f'which should be (B, T, H) if `head_first=False` '
            f'or (B, H, T) otherwise',
        )

    return chunk_local_cumsum_scalar(
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        output_dtype=output_dtype,
        chunk_indices=chunk_indices,
    )
