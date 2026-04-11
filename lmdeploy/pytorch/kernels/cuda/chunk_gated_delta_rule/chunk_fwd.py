# Copyright (c) OpenMMLab. All rights reserved.
import tilelang
import tilelang.language as T
import tilelang.layout
import torch
from fla.ops.utils.index import prepare_chunk_indices

from .wy_fast import recompute_w_u_fwd


@T.macro
def _clear_block(dst: T.Buffer, BC: int) -> None:
    for i, j in T.Parallel(BC, BC):
        dst[i, j] = 0.0


@T.macro
def _load_k_block(dst: T.Buffer,
                  K_in: T.Buffer,
                  batch_id,
                  bos,
                  seq_base,
                  head_id,
                  k_base,
                  seqlen,
                  BC: int,
                  BK: int,
                  K: int,
                  dtype: torch.dtype) -> None:
    for i, j in T.Parallel(BC, BK):
        row = seq_base + i
        col = k_base + j
        if row < seqlen and col < K:
            dst[i, j] = K_in[batch_id, bos + row, head_id, col]
        else:
            dst[i, j] = T.cast(0.0, dtype)


@T.macro
def _load_scalar_block(dst: T.Buffer,
                       X: T.Buffer,
                       batch_id,
                       bos,
                       seq_base,
                       hv_id,
                       seqlen,
                       BC: int,
                       out_dtype) -> None:
    for i in T.Parallel(BC):
        row = seq_base + i
        if row < seqlen:
            dst[i] = T.cast(X[batch_id, bos + row, hv_id], out_dtype)
        else:
            dst[i] = 0.0


@T.macro
def _apply_diag_block_scale(dst: T.Buffer,
                            beta_blk: T.Buffer,
                            g_blk: T.Buffer,
                            seq_base,
                            seqlen,
                            BC: int,
                            use_g: bool,
                            use_exp2: bool) -> None:
    for i, j in T.Parallel(BC, BC):
        row = seq_base + i
        col = seq_base + j
        if row < seqlen and col < seqlen and i > j:
            val = T.alloc_var(T.float32)
            val = dst[i, j]
            if use_g:
                if use_exp2:
                    val = val * T.exp2(g_blk[i] - g_blk[j])
                else:
                    val = val * T.exp(g_blk[i] - g_blk[j])
            dst[i, j] = val * beta_blk[i]
        else:
            dst[i, j] = 0.0


@T.macro
def _apply_offdiag_block_scale(dst: T.Buffer,
                               beta_row: T.Buffer,
                               g_row: T.Buffer,
                               g_col: T.Buffer,
                               row_base,
                               col_base,
                               seqlen,
                               BC: int,
                               use_g: bool,
                               use_exp2: bool) -> None:
    for i, j in T.Parallel(BC, BC):
        row = row_base + i
        col = col_base + j
        if row < seqlen and col < seqlen:
            val = T.alloc_var(T.float32)
            val = dst[i, j]
            if use_g:
                if use_exp2:
                    val = val * T.exp2(g_row[i] - g_col[j])
                else:
                    val = val * T.exp(g_row[i] - g_col[j])
            dst[i, j] = val * beta_row[i]
        else:
            dst[i, j] = 0.0


@T.macro
def _solve_unit_lower_block(A_blk: T.Buffer,
                            Ai_blk: T.Buffer,
                            seq_base,
                            seqlen,
                            BC: int) -> None:
    # Initialize: identity on diagonal, -A on strict lower triangle, 0 elsewhere.
    for i, j in T.Parallel(BC, BC):
        if seq_base + i < seqlen and seq_base + j < seqlen and i > j:
            Ai_blk[i, j] = -A_blk[i, j]
        elif i == j and seq_base + i < seqlen:
            Ai_blk[i, j] = 1.0
        else:
            Ai_blk[i, j] = 0.0

    # Forward substitution: sequential across rows, parallel across columns.
    # Row i depends on rows 0..i-1. T.Parallel provides implicit barrier
    # between iterations so row i's writes are visible before row i+1 reads.
    for i in range(2, BC):
        for j in T.Parallel(BC):
            if j < i and seq_base + i < seqlen:
                acc = T.alloc_var(T.float32)
                acc = 0.0
                for t in T.Unroll(BC):
                    if t < i:
                        acc = acc + (-A_blk[i, t]) * Ai_blk[t, j]
                Ai_blk[i, j] = acc


@T.macro
def _matmul_block(lhs: T.Buffer,
                  rhs: T.Buffer,
                  out: T.Buffer,
                  BC: int) -> None:
    frag = T.alloc_fragment((BC, BC), dtype=T.float32)
    T.clear(frag)
    T.gemm(lhs, rhs, frag)
    T.copy(frag, out)


@T.macro
def _store_block(dst: T.Buffer,
                 src: T.Buffer,
                 batch_id,
                 bos,
                 row_base,
                 hv_id,
                 col_base,
                 seqlen,
                 BC: int,
                 out_dtype: torch.dtype) -> None:
    for i, j in T.Parallel(BC, BC):
        row = row_base + i
        col = col_base + j
        if row < seqlen:
            dst[batch_id, bos + row, hv_id, col] = T.cast(src[i, j], out_dtype)


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}, )
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    H: int,
    K: int,
    HV: int,
    BT: int,
    dtype: torch.dtype,
    a_dtype: torch.dtype,
    g_dtype: torch.dtype,
    beta_dtype: torch.dtype,
    cu_seqlen_dtype: torch.dtype,
    use_g: bool,
    is_varlen: bool,
    use_exp2: bool,
):
    """TileLang scaffold for the Triton intra-chunk KKT solve kernel.

    This kernel is intentionally specialized to the same block structure as the
    Triton reference:

    - one CTA handles one `(chunk, hv)` pair
    - `BT == 64`
    - the chunk is split into four `BC == 16` sub-blocks
    - the K dimension is tiled by `BK == 64`

    The current implementation wires the index decode, shared-memory layout, and
    block accumulator structure that the final port will use. The algebraic
    solve steps are still pending.
    """

    if BT != 64:
        raise ValueError(f'chunk_gated_delta_rule_fwd_kkt_solve_kernel currently expects BT=64, got {BT}')

    B = 1 if is_varlen else T.dynamic('B')
    N = T.dynamic('N')
    TT = T.dynamic('TT')

    NC = T.dynamic('NC')
    NT = T.ceildiv(TT, BT) if not is_varlen else NC
    seq_count = 1 if is_varlen else B

    BC = 16
    BK = 64
    num_k_tiles = (K + BK - 1) // BK

    @T.prim_func
    def chunk_gated_delta_rule_fwd_kkt_solve_main(
        K_in: T.Tensor((B, TT, H, K), dtype=dtype),
        A: T.Tensor((B, TT, HV, BT), dtype=a_dtype),
        G: T.Tensor((B, TT, HV), dtype=g_dtype) = None,
        Beta: T.Tensor((B, TT, HV), dtype=beta_dtype) = None,
        CuSeqlens: T.Tensor((N + 1,), dtype=cu_seqlen_dtype) = None,
        ChunkIndices: T.Tensor((NT, 2), dtype=torch.long) = None,
    ):
        with T.Kernel(NT, seq_count * HV, threads=64) as (i_t, i_bh):
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

            if i_t * BT >= seqlen:
                T.evaluate(0)

            # Sub-chunk bases inside the current 64-token chunk.
            i_tc0 = i_t * BT
            i_tc1 = i_tc0 + BC
            i_tc2 = i_tc0 + 2 * BC
            i_tc3 = i_tc0 + 3 * BC

            kv_head = i_h // (HV // H)

            # -----------------------------------------------------------------
            # Step 0: CTA-shared metadata.
            # -----------------------------------------------------------------
            b_b0 = T.alloc_shared((BC,), dtype=T.float32)
            b_b1 = T.alloc_shared((BC,), dtype=T.float32)
            b_b2 = T.alloc_shared((BC,), dtype=T.float32)
            b_b3 = T.alloc_shared((BC,), dtype=T.float32)
            _load_scalar_block(b_b0, Beta, i_b, bos, i_tc0, i_h, seqlen, BC, T.float32)
            _load_scalar_block(b_b1, Beta, i_b, bos, i_tc1, i_h, seqlen, BC, T.float32)
            _load_scalar_block(b_b2, Beta, i_b, bos, i_tc2, i_h, seqlen, BC, T.float32)
            _load_scalar_block(b_b3, Beta, i_b, bos, i_tc3, i_h, seqlen, BC, T.float32)

            if use_g:
                b_g0 = T.alloc_shared((BC,), dtype=T.float32)
                b_g1 = T.alloc_shared((BC,), dtype=T.float32)
                b_g2 = T.alloc_shared((BC,), dtype=T.float32)
                b_g3 = T.alloc_shared((BC,), dtype=T.float32)
                _load_scalar_block(b_g0, G, i_b, bos, i_tc0, i_h, seqlen, BC, T.float32)
                _load_scalar_block(b_g1, G, i_b, bos, i_tc1, i_h, seqlen, BC, T.float32)
                _load_scalar_block(b_g2, G, i_b, bos, i_tc2, i_h, seqlen, BC, T.float32)
                _load_scalar_block(b_g3, G, i_b, bos, i_tc3, i_h, seqlen, BC, T.float32)

            # -----------------------------------------------------------------
            # Step 1: shared K tiles and the 10 lower-triangular block
            # accumulators that match the Triton implementation.
            # -----------------------------------------------------------------
            k0 = T.alloc_shared((BC, BK), dtype=dtype)
            k1 = T.alloc_shared((BC, BK), dtype=dtype)
            k2 = T.alloc_shared((BC, BK), dtype=dtype)
            k3 = T.alloc_shared((BC, BK), dtype=dtype)
            kt0 = T.alloc_shared((BK, BC), dtype=dtype)
            kt1 = T.alloc_shared((BK, BC), dtype=dtype)
            kt2 = T.alloc_shared((BK, BC), dtype=dtype)
            kt3 = T.alloc_shared((BK, BC), dtype=dtype)
            T.annotate_layout({
                k0: tilelang.layout.make_swizzled_layout(k0),
                k1: tilelang.layout.make_swizzled_layout(k1),
                k2: tilelang.layout.make_swizzled_layout(k2),
                k3: tilelang.layout.make_swizzled_layout(k3),
                kt0: tilelang.layout.make_swizzled_layout(kt0),
                kt1: tilelang.layout.make_swizzled_layout(kt1),
                kt2: tilelang.layout.make_swizzled_layout(kt2),
                kt3: tilelang.layout.make_swizzled_layout(kt3),
            })

            A00_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A11_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A22_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A33_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A10_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A20_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A21_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A30_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A31_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            A32_frag = T.alloc_fragment((BC, BC), dtype=T.float32)
            T.clear(A00_frag)
            T.clear(A11_frag)
            T.clear(A22_frag)
            T.clear(A33_frag)
            T.clear(A10_frag)
            T.clear(A20_frag)
            T.clear(A21_frag)
            T.clear(A30_frag)
            T.clear(A31_frag)
            T.clear(A32_frag)

            for i_k in range(num_k_tiles):
                k_base = i_k * BK
                _load_k_block(k0, K_in, i_b, bos, i_tc0, kv_head, k_base, seqlen, BC, BK, K, dtype)
                for i, j in T.Parallel(BC, BK):
                    kt0[j, i] = k0[i, j]
                T.gemm(k0, kt0, A00_frag)

                if i_tc1 < seqlen:
                    _load_k_block(k1, K_in, i_b, bos, i_tc1, kv_head, k_base, seqlen, BC, BK, K, dtype)
                    for i, j in T.Parallel(BC, BK):
                        kt1[j, i] = k1[i, j]
                    T.gemm(k1, kt1, A11_frag)
                    T.gemm(k1, kt0, A10_frag)

                    if i_tc2 < seqlen:
                        _load_k_block(k2, K_in, i_b, bos, i_tc2, kv_head, k_base, seqlen, BC, BK, K, dtype)
                        for i, j in T.Parallel(BC, BK):
                            kt2[j, i] = k2[i, j]
                        T.gemm(k2, kt2, A22_frag)
                        T.gemm(k2, kt0, A20_frag)
                        T.gemm(k2, kt1, A21_frag)

                        if i_tc3 < seqlen:
                            _load_k_block(k3, K_in, i_b, bos, i_tc3, kv_head, k_base, seqlen, BC, BK, K, dtype)
                            for i, j in T.Parallel(BC, BK):
                                kt3[j, i] = k3[i, j]
                            T.gemm(k3, kt3, A33_frag)
                            T.gemm(k3, kt0, A30_frag)
                            T.gemm(k3, kt1, A31_frag)
                            T.gemm(k3, kt2, A32_frag)

            # -----------------------------------------------------------------
            # Step 2+: materialize the block accumulators to shared so the later
            # gate/mask/solve/merge stages can operate with scalar indexing.
            # -----------------------------------------------------------------
            A00 = T.alloc_shared((BC, BC), dtype=T.float32)
            A11 = T.alloc_shared((BC, BC), dtype=T.float32)
            A22 = T.alloc_shared((BC, BC), dtype=T.float32)
            A33 = T.alloc_shared((BC, BC), dtype=T.float32)
            A10 = T.alloc_shared((BC, BC), dtype=T.float32)
            A20 = T.alloc_shared((BC, BC), dtype=T.float32)
            A21 = T.alloc_shared((BC, BC), dtype=T.float32)
            A30 = T.alloc_shared((BC, BC), dtype=T.float32)
            A31 = T.alloc_shared((BC, BC), dtype=T.float32)
            A32 = T.alloc_shared((BC, BC), dtype=T.float32)
            T.copy(A00_frag, A00)
            T.copy(A11_frag, A11)
            T.copy(A22_frag, A22)
            T.copy(A33_frag, A33)
            T.copy(A10_frag, A10)
            T.copy(A20_frag, A20)
            T.copy(A21_frag, A21)
            T.copy(A30_frag, A30)
            T.copy(A31_frag, A31)
            T.copy(A32_frag, A32)

            # -----------------------------------------------------------------
            # Step 2: apply gate, triangular masking, and beta scaling.
            # -----------------------------------------------------------------
            if use_g:
                _apply_diag_block_scale(A00, b_b0, b_g0, i_tc0, seqlen, BC, True, use_exp2)
                _apply_diag_block_scale(A11, b_b1, b_g1, i_tc1, seqlen, BC, True, use_exp2)
                _apply_diag_block_scale(A22, b_b2, b_g2, i_tc2, seqlen, BC, True, use_exp2)
                _apply_diag_block_scale(A33, b_b3, b_g3, i_tc3, seqlen, BC, True, use_exp2)
                _apply_offdiag_block_scale(A10, b_b1, b_g1, b_g0, i_tc1, i_tc0, seqlen, BC, True, use_exp2)
                _apply_offdiag_block_scale(A20, b_b2, b_g2, b_g0, i_tc2, i_tc0, seqlen, BC, True, use_exp2)
                _apply_offdiag_block_scale(A21, b_b2, b_g2, b_g1, i_tc2, i_tc1, seqlen, BC, True, use_exp2)
                _apply_offdiag_block_scale(A30, b_b3, b_g3, b_g0, i_tc3, i_tc0, seqlen, BC, True, use_exp2)
                _apply_offdiag_block_scale(A31, b_b3, b_g3, b_g1, i_tc3, i_tc1, seqlen, BC, True, use_exp2)
                _apply_offdiag_block_scale(A32, b_b3, b_g3, b_g2, i_tc3, i_tc2, seqlen, BC, True, use_exp2)
            else:
                _apply_diag_block_scale(A00, b_b0, b_b0, i_tc0, seqlen, BC, False, use_exp2)
                _apply_diag_block_scale(A11, b_b1, b_b1, i_tc1, seqlen, BC, False, use_exp2)
                _apply_diag_block_scale(A22, b_b2, b_b2, i_tc2, seqlen, BC, False, use_exp2)
                _apply_diag_block_scale(A33, b_b3, b_b3, i_tc3, seqlen, BC, False, use_exp2)
                _apply_offdiag_block_scale(A10, b_b1, b_b1, b_b0, i_tc1, i_tc0, seqlen, BC, False, use_exp2)
                _apply_offdiag_block_scale(A20, b_b2, b_b2, b_b0, i_tc2, i_tc0, seqlen, BC, False, use_exp2)
                _apply_offdiag_block_scale(A21, b_b2, b_b2, b_b1, i_tc2, i_tc1, seqlen, BC, False, use_exp2)
                _apply_offdiag_block_scale(A30, b_b3, b_b3, b_b0, i_tc3, i_tc0, seqlen, BC, False, use_exp2)
                _apply_offdiag_block_scale(A31, b_b3, b_b3, b_b1, i_tc3, i_tc1, seqlen, BC, False, use_exp2)
                _apply_offdiag_block_scale(A32, b_b3, b_b3, b_b2, i_tc3, i_tc2, seqlen, BC, False, use_exp2)

            # Inverse block outputs are kept in shared because the triangular
            # solve and block merge are scalar/shared heavy, not GEMM heavy.
            Ai00 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai11 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai22 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai33 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai10 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai20 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai21 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai30 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai31 = T.alloc_shared((BC, BC), dtype=T.float32)
            Ai32 = T.alloc_shared((BC, BC), dtype=T.float32)

            # -----------------------------------------------------------------
            # Step 3: forward substitution on the four diagonal blocks.
            # -----------------------------------------------------------------
            _solve_unit_lower_block(A00, Ai00, i_tc0, seqlen, BC)
            _solve_unit_lower_block(A11, Ai11, i_tc1, seqlen, BC)
            _solve_unit_lower_block(A22, Ai22, i_tc2, seqlen, BC)
            _solve_unit_lower_block(A33, Ai33, i_tc3, seqlen, BC)

            # -----------------------------------------------------------------
            # Step 4: block merge for the full lower-triangular inverse.
            # Use fragment accumulation to fuse sums, reducing barriers.
            # -----------------------------------------------------------------
            tmp0 = T.alloc_shared((BC, BC), dtype=T.float32)

            # Ai10 = -(Ai11 @ A10 @ Ai00)
            _matmul_block(Ai11, A10, tmp0, BC)
            _matmul_block(tmp0, Ai00, Ai10, BC)
            for i, j in T.Parallel(BC, BC):
                Ai10[i, j] = -Ai10[i, j]

            # Ai21 = -(Ai22 @ A21 @ Ai11)
            _matmul_block(Ai22, A21, tmp0, BC)
            _matmul_block(tmp0, Ai11, Ai21, BC)
            for i, j in T.Parallel(BC, BC):
                Ai21[i, j] = -Ai21[i, j]

            # Ai32 = -(Ai33 @ A32 @ Ai22)
            _matmul_block(Ai33, A32, tmp0, BC)
            _matmul_block(tmp0, Ai22, Ai32, BC)
            for i, j in T.Parallel(BC, BC):
                Ai32[i, j] = -Ai32[i, j]

            # Ai20 = -(Ai22 @ (A20 @ Ai00 + A21 @ Ai10))
            # Accumulate both matmuls into one fragment.
            frag_acc = T.alloc_fragment((BC, BC), dtype=T.float32)
            T.clear(frag_acc)
            T.gemm(A20, Ai00, frag_acc)
            T.gemm(A21, Ai10, frag_acc)
            T.copy(frag_acc, tmp0)
            _matmul_block(Ai22, tmp0, Ai20, BC)
            for i, j in T.Parallel(BC, BC):
                Ai20[i, j] = -Ai20[i, j]

            # Ai31 = -(Ai33 @ (A31 @ Ai11 + A32 @ Ai21))
            T.clear(frag_acc)
            T.gemm(A31, Ai11, frag_acc)
            T.gemm(A32, Ai21, frag_acc)
            T.copy(frag_acc, tmp0)
            _matmul_block(Ai33, tmp0, Ai31, BC)
            for i, j in T.Parallel(BC, BC):
                Ai31[i, j] = -Ai31[i, j]

            # Ai30 = -(Ai33 @ (A30 @ Ai00 + A31 @ Ai10 + A32 @ Ai20))
            T.clear(frag_acc)
            T.gemm(A30, Ai00, frag_acc)
            T.gemm(A31, Ai10, frag_acc)
            T.gemm(A32, Ai20, frag_acc)
            T.copy(frag_acc, tmp0)
            _matmul_block(Ai33, tmp0, Ai30, BC)
            for i, j in T.Parallel(BC, BC):
                Ai30[i, j] = -Ai30[i, j]

            # -----------------------------------------------------------------
            # Step 5: store the lower-triangular block set to A.
            # -----------------------------------------------------------------
            _store_block(A, Ai00, i_b, bos, i_tc0, i_h, 0, seqlen, BC, a_dtype)
            _store_block(A, Ai10, i_b, bos, i_tc1, i_h, 0, seqlen, BC, a_dtype)
            _store_block(A, Ai11, i_b, bos, i_tc1, i_h, BC, seqlen, BC, a_dtype)
            _store_block(A, Ai20, i_b, bos, i_tc2, i_h, 0, seqlen, BC, a_dtype)
            _store_block(A, Ai21, i_b, bos, i_tc2, i_h, BC, seqlen, BC, a_dtype)
            _store_block(A, Ai22, i_b, bos, i_tc2, i_h, 2 * BC, seqlen, BC, a_dtype)
            _store_block(A, Ai30, i_b, bos, i_tc3, i_h, 0, seqlen, BC, a_dtype)
            _store_block(A, Ai31, i_b, bos, i_tc3, i_h, BC, seqlen, BC, a_dtype)
            _store_block(A, Ai32, i_b, bos, i_tc3, i_h, 2 * BC, seqlen, BC, a_dtype)
            _store_block(A, Ai33, i_b, bos, i_tc3, i_h, 3 * BC, seqlen, BC, a_dtype)

    return chunk_gated_delta_rule_fwd_kkt_solve_main


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
    """TileLang port of the FLA intra-chunk GDR forward path."""
    if beta is None:
        raise ValueError('beta must not be None')
    if chunk_size != 64:
        raise ValueError(f'chunk_gated_delta_rule_fwd_intra currently expects chunk_size=64, got {chunk_size}')

    B, TT, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is not None:
        assert chunk_indices is not None
        assert cu_seqlens.is_contiguous()
        assert chunk_indices.is_contiguous()

    A = torch.zeros(B, TT, HV, BT, device=k.device, dtype=k.dtype)
    kernel = chunk_gated_delta_rule_fwd_kkt_solve_kernel(
        H=H,
        K=K,
        HV=HV,
        BT=BT,
        dtype=k.dtype,
        a_dtype=A.dtype,
        g_dtype=g.dtype if g is not None else torch.float32,
        beta_dtype=beta.dtype,
        cu_seqlen_dtype=cu_seqlens.dtype if cu_seqlens is not None else torch.long,
        use_g=g is not None,
        is_varlen=cu_seqlens is not None,
        use_exp2=use_exp2,
    )
    kernel(
        k,
        A,
        g,
        beta,
        cu_seqlens,
        chunk_indices,
    )

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
    )
    return w, u, A
