import pytest
import torch

try:
    from fla.ops.common.chunk_o import chunk_fwd_o as fla_func
except Exception:
    fla_func = None


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


def torch_ref(q, k, v, h, g=None, g_gamma=None, scale=None, cu_seqlens=None, chunk_size=64, chunk_indices=None,
              use_exp2=False, transpose_state_layout=False):
    batch, total, num_heads, head_dim = q.shape
    _, _, num_v_heads, value_dim = v.shape
    group_size = num_v_heads // num_heads
    exp_fn = torch.exp2 if use_exp2 else torch.exp

    if scale is None:
        scale = head_dim ** -0.5

    o = torch.zeros_like(v, dtype=torch.float32)

    if cu_seqlens is None:
        entries = []
        nt = (total + chunk_size - 1) // chunk_size
        for b in range(batch):
            for chunk_idx in range(nt):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total)
                entries.append((b, chunk_idx, start, end))
    else:
        assert chunk_indices is not None
        entries = []
        cu = cu_seqlens.tolist()
        for slot_t, row in enumerate(chunk_indices.tolist()):
            seq_idx, chunk_idx = row
            bos, eos = cu[seq_idx], cu[seq_idx + 1]
            start = bos + chunk_idx * chunk_size
            end = min(start + chunk_size, eos)
            entries.append((0, slot_t, start, end))

    for b, slot_t, start, end in entries:
        t_len = end - start
        for hv in range(num_v_heads):
            h_idx = hv // group_size
            q_chunk = q[b, start:end, h_idx].to(torch.float32)
            k_chunk = k[b, start:end, h_idx].to(torch.float32)
            v_chunk = v[b, start:end, hv].to(torch.float32)

            state = h[b, slot_t, hv]
            if transpose_state_layout:
                state = state.transpose(-1, -2)
            state = state.to(torch.float32)

            b_o = q_chunk @ state
            b_A = q_chunk @ k_chunk.transpose(0, 1)

            if g is not None:
                g_chunk = g[b, start:end, hv].to(torch.float32)
                b_o = b_o * exp_fn(g_chunk)[:, None]
                b_A = b_A * exp_fn(g_chunk[:, None] - g_chunk[None, :])

            if g_gamma is not None:
                idx = torch.arange(1, t_len + 1, device=q.device, dtype=torch.float32)
                g_base = g_gamma[hv].to(torch.float32) * idx
                b_o = b_o * exp_fn(g_base)[:, None]
                b_A = b_A * exp_fn(g_base[:, None] - g_base[None, :])

            mask = torch.tril(torch.ones(t_len, t_len, device=q.device, dtype=torch.bool))
            b_A = torch.where(mask, b_A, torch.zeros_like(b_A))

            o[b, start:end, hv] = (b_o + b_A @ v_chunk) * scale

    return o


def ref_impl(**kwargs):
    if fla_func is not None:
        return fla_func(**kwargs)
    return torch_ref(**kwargs)


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-4, 2e-4
    return 2e-2, 2e-2


@pytest.mark.skipif(not do_test(), reason='tilelang is not available')
class TestChunkFwdO:

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
        'cu_seqlens,heads,v_heads,head_dim,value_dim,chunk_size,dtype,use_g,use_exp2,transpose_state_layout',
        [
            ([0, 127, 2051], 16, 32, 128, 128, 64, torch.bfloat16, True, False, False),
        ],
    )
    def test_varlen(self, cu_seqlens, heads, v_heads, head_dim, value_dim, chunk_size, dtype, use_g, use_exp2,
                    transpose_state_layout):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.chunk_o import chunk_fwd_o

        total = cu_seqlens[-1]
        nt = sum((cu_seqlens[i + 1] - cu_seqlens[i] + chunk_size - 1) // chunk_size for i in range(len(cu_seqlens) - 1))
        q = torch.rand(1, total, heads, head_dim, dtype=dtype) - 0.5
        k = torch.rand(1, total, heads, head_dim, dtype=dtype) - 0.5
        v = torch.rand(1, total, v_heads, value_dim, dtype=dtype) - 0.5
        h = torch.rand(1, nt, v_heads, head_dim, value_dim, dtype=dtype) - 0.5
        g = -2.0 * torch.rand(1, total, v_heads, dtype=torch.float32) if use_g else None
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device='cuda')
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        out = chunk_fwd_o(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
        )
        ref = ref_impl(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
            transpose_state_layout=transpose_state_layout,
        )
        atol, rtol = get_tolerances(dtype)
        torch.testing.assert_close(out, ref.to(out.dtype), atol=atol, rtol=rtol)
