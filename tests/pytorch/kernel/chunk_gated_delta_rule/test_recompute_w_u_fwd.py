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


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-3, 2e-3
    return 1e-2, 1e-2


def make_varlen_view(total: int,
                     heads: int,
                     dim: int,
                     dtype: torch.dtype,
                     stride0_scale: int = 2) -> torch.Tensor:
    base = torch.rand(stride0_scale, total, heads, dim, dtype=dtype, device='cuda') - 0.5
    return base[:1]


def make_fully_strided_varlen_view(total: int, heads: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.rand(2, total, heads, dim * 2, dtype=dtype, device='cuda') - 0.5
    return base[:1, :, :, ::2]


def torch_ref(k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None, use_exp2=False):
    batch, total, num_heads, head_dim = k.shape
    _, _, num_v_heads, value_dim = v.shape
    chunk_size = A.shape[-1]
    group_size = num_v_heads // num_heads

    w = torch.empty(batch, total, num_v_heads, head_dim, device=k.device, dtype=torch.float32)
    u = torch.empty(batch, total, num_v_heads, value_dim, device=v.device, dtype=torch.float32)

    if cu_seqlens is None:
        chunks = []
        for b in range(batch):
            num_chunks = (total + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total)
                chunks.append((b, start, end))
    else:
        chunks = []
        cu = cu_seqlens.tolist()
        for seq_idx, chunk_idx in chunk_indices.tolist():
            bos, eos = cu[seq_idx], cu[seq_idx + 1]
            start = bos + chunk_idx * chunk_size
            end = min(start + chunk_size, eos)
            chunks.append((0, start, end))

    for b, start, end in chunks:
        length = end - start
        for hv in range(num_v_heads):
            h = hv // group_size
            a = A[b, start:end, hv, :length].to(torch.float32)
            b_h = beta[b, start:end, hv].to(torch.float32)
            vb = v[b, start:end, hv].to(torch.float32) * b_h[:, None]
            kb = k[b, start:end, h].to(torch.float32) * b_h[:, None]
            if g is not None:
                gate = g[b, start:end, hv].to(torch.float32)
                gate = torch.exp2(gate) if use_exp2 else torch.exp(gate)
                kb = kb * gate[:, None]
            u[b, start:end, hv] = a @ vb
            w[b, start:end, hv] = a @ kb
    return w, u


@pytest.mark.skipif(not do_test(), reason='tilelang is not available')
class TestRecomputeWUForward:

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
        'cu_seqlens,heads,v_heads,head_dim,value_dim,chunk_size,dtype,beta_dtype,a_dtype,use_g,use_exp2,non_contiguous_kv,fully_strided_kv',
        [
            ([0, 127, 257], 16, 32, 128, 128, 64, torch.bfloat16, torch.float32,
             torch.bfloat16, True, True, True, False),
        ],
    )
    def test_varlen(self, cu_seqlens, heads, v_heads, head_dim, value_dim, chunk_size, dtype, beta_dtype, a_dtype,
                    use_g, use_exp2, non_contiguous_kv, fully_strided_kv):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.wy_fast import recompute_w_u_fwd

        total = cu_seqlens[-1]
        if fully_strided_kv:
            k = make_fully_strided_varlen_view(total, heads, head_dim, dtype)
            v = make_fully_strided_varlen_view(total, v_heads, value_dim, dtype)
        elif non_contiguous_kv:
            k = make_varlen_view(total, heads, head_dim, dtype)
            v = make_varlen_view(total, v_heads, value_dim, dtype)
        else:
            k = torch.rand(1, total, heads, head_dim, dtype=dtype) - 0.5
            v = torch.rand(1, total, v_heads, value_dim, dtype=dtype) - 0.5
        beta = torch.rand(1, total, v_heads, dtype=beta_dtype)
        A = torch.rand(1, total, v_heads, chunk_size, dtype=a_dtype) - 0.5
        g = None
        if use_g:
            g = -2 * torch.rand(1, total, v_heads, dtype=torch.float32)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device='cuda')
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        out_w, out_u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
        )
        ref_w, ref_u = torch_ref(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=use_exp2,
        )
        atol, rtol = get_tolerances(dtype)
        torch.testing.assert_close(out_w, ref_w.to(out_w.dtype), atol=atol, rtol=rtol)
        torch.testing.assert_close(out_u, ref_u.to(out_u.dtype), atol=atol, rtol=rtol)
