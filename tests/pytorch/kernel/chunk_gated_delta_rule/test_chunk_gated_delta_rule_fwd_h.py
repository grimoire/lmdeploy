import pytest
import torch


def do_test():
    try:
        import tilelang  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_indices = []
    cu = cu_seqlens.tolist()
    for seq_idx in range(len(cu) - 1):
        seqlen = cu[seq_idx + 1] - cu[seq_idx]
        num_chunks = (seqlen + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_indices.append((seq_idx, chunk_idx))
    return torch.tensor(chunk_indices, device=cu_seqlens.device, dtype=torch.long)


def make_initial_state(num_seqs: int, num_v_heads: int, head_dim: int, value_dim: int,
                       dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(num_seqs, num_v_heads, head_dim, value_dim, device='cuda', dtype=dtype) * 0.05


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-4, 2e-4
    return 1e-2, 1e-2


def _chunk_entries(k: torch.Tensor, chunk_size: int, cu_seqlens: torch.Tensor | None = None,
                   chunk_indices: torch.Tensor | None = None):
    batch, total = k.shape[:2]
    if cu_seqlens is None:
        nt = (total + chunk_size - 1) // chunk_size
        return [
            (b, chunk_idx, b, chunk_idx * chunk_size, min((chunk_idx + 1) * chunk_size, total))
            for b in range(batch)
            for chunk_idx in range(nt)
        ]

    entries = []
    cu = cu_seqlens.tolist()
    for slot_t, (seq_idx, chunk_idx) in enumerate(chunk_indices.tolist()):
        bos, eos = cu[seq_idx], cu[seq_idx + 1]
        start = bos + chunk_idx * chunk_size
        end = min(start + chunk_size, eos)
        entries.append((0, slot_t, seq_idx, start, end))
    return entries


def torch_ref(k: torch.Tensor,
              w: torch.Tensor,
              u: torch.Tensor,
              g: torch.Tensor | None = None,
              gk: torch.Tensor | None = None,
              initial_state: torch.Tensor | None = None,
              output_final_state: bool = False,
              chunk_size: int = 64,
              save_new_value: bool = True,
              cu_seqlens: torch.LongTensor | None = None,
              chunk_indices: torch.LongTensor | None = None,
              use_exp2: bool = False,
              **_) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    batch, _, num_heads, head_dim = k.shape
    num_v_heads, value_dim = u.shape[2], u.shape[-1]
    group_size = num_v_heads // num_heads
    exp_fn = torch.exp2 if use_exp2 else torch.exp

    if cu_seqlens is None:
        nt = (k.shape[1] + chunk_size - 1) // chunk_size
        h = torch.zeros(batch, nt, num_v_heads, head_dim, value_dim, device=k.device, dtype=torch.float32)
        num_seqs = batch
    else:
        nt = len(chunk_indices)
        h = torch.zeros(1, nt, num_v_heads, head_dim, value_dim, device=k.device, dtype=torch.float32)
        num_seqs = len(cu_seqlens) - 1

    v_new = torch.zeros_like(u, dtype=torch.float32) if save_new_value else None
    if initial_state is None:
        states = torch.zeros(num_seqs, num_v_heads, head_dim, value_dim, device=k.device, dtype=torch.float32)
    else:
        states = initial_state.to(torch.float32).clone()

    entries = _chunk_entries(k, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    for slot_b, slot_t, seq_idx, start, end in entries:
        for hv in range(num_v_heads):
            h_idx = hv // group_size
            state = states[seq_idx, hv]
            h[slot_b, slot_t, hv] = state

            k_chunk = k[slot_b, start:end, h_idx].to(torch.float32)
            w_chunk = w[slot_b, start:end, hv].to(torch.float32)
            u_chunk = u[slot_b, start:end, hv].to(torch.float32)
            residual = u_chunk - w_chunk @ state

            if save_new_value:
                v_new[slot_b, start:end, hv] = residual

            update_value = residual
            state_work = state
            if g is not None:
                g_chunk = g[slot_b, start:end, hv].to(torch.float32)
                g_last = g_chunk[-1]
                update_value = update_value * exp_fn(g_last - g_chunk)[:, None]
                state_work = state_work * exp_fn(g_last)
            if gk is not None:
                gk_last = gk[slot_b, end - 1, hv].to(torch.float32)
                state_work = state_work * exp_fn(gk_last)[:, None]

            states[seq_idx, hv] = state_work + k_chunk.transpose(0, 1) @ update_value

    final_state = states if output_final_state else None
    return h, v_new, final_state


@pytest.mark.skipif(not do_test(), reason='tilelang is not available')
class TestChunkGatedDeltaRuleForwardH:

    @pytest.fixture(autouse=True)
    def auto_context(self):
        origin_dtype = torch.get_default_dtype()
        origin_device = torch.get_default_device()
        with torch.inference_mode():
            torch.set_default_dtype(torch.bfloat16)
            torch.set_default_device('cuda')
            try:
                yield
            finally:
                torch.set_default_dtype(origin_dtype)
                torch.set_default_device(origin_device)

    @pytest.mark.parametrize(
        'cu_seqlens,heads,v_heads,head_dim,value_dim,dtype,use_g,use_gk,use_initial_state,output_final_state,'
        'save_new_value,use_exp2',
        [
            ([0, 127, 257], 16, 32, 128, 128, torch.bfloat16, True, False, True, True, True, True),
        ],
    )
    def test_varlen(self, cu_seqlens, heads, v_heads, head_dim, value_dim, dtype, use_g, use_gk,
                    use_initial_state, output_final_state, save_new_value, use_exp2):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.chunk_delta_h import chunk_gated_delta_rule_fwd_h

        total = cu_seqlens[-1]
        num_seqs = len(cu_seqlens) - 1
        k = torch.randn(1, total, heads, head_dim, dtype=dtype) * 0.1
        w = torch.randn(1, total, v_heads, head_dim, dtype=dtype) * 0.1
        u = torch.randn(1, total, v_heads, value_dim, dtype=dtype) * 0.1
        g = -2.0 * torch.rand(1, total, v_heads, dtype=torch.float32) if use_g else None
        gk = -2.0 * torch.rand(1, total, v_heads, head_dim, dtype=torch.float32) if use_gk else None
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device='cuda')
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        initial_state = None
        if use_initial_state:
            initial_state = make_initial_state(num_seqs, v_heads, head_dim, value_dim, torch.float32)

        out_h, out_v_new, out_final_state = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=64,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
        )
        ref_h, ref_v_new, ref_final_state = torch_ref(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=64,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
        )

        atol, rtol = get_tolerances(dtype)
        torch.testing.assert_close(out_h, ref_h.to(out_h.dtype), atol=atol, rtol=rtol)
        if save_new_value:
            torch.testing.assert_close(out_v_new, ref_v_new.to(out_v_new.dtype), atol=atol, rtol=rtol)
        else:
            assert out_v_new is None
        if output_final_state:
            torch.testing.assert_close(out_final_state, ref_final_state, atol=atol, rtol=rtol)
        else:
            assert out_final_state is None

    def test_rejects_transpose_state_layout(self):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.chunk_delta_h import chunk_gated_delta_rule_fwd_h

        k = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16) * 0.1
        w = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16) * 0.1
        u = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16) * 0.1

        with pytest.raises(AssertionError, match='transpose_state_layout=True is not supported'):
            chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                chunk_size=64,
                transpose_state_layout=True,
            )
