import pytest
import torch

try:
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra as fla_func
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


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-3, 2e-3
    if fla_func is not None:
        return 4e-2, 4e-2
    return 1e-1, 1e-1


def torch_ref(k, v, beta, g=None, cu_seqlens=None, chunk_indices=None, chunk_size=64, use_exp2=False):
    batch, total, num_heads, head_dim = k.shape
    _, _, num_v_heads, value_dim = v.shape
    group_size = num_v_heads // num_heads

    w = torch.zeros(batch, total, num_v_heads, head_dim, device=k.device, dtype=torch.float32)
    u = torch.zeros(batch, total, num_v_heads, value_dim, device=v.device, dtype=torch.float32)
    A = torch.zeros(batch, total, num_v_heads, chunk_size, device=k.device, dtype=torch.float32)

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
        eye = torch.eye(length, device=k.device, dtype=torch.float32)
        for hv in range(num_v_heads):
            h = hv // group_size
            k_chunk = k[b, start:end, h].to(torch.float32)
            v_chunk = v[b, start:end, hv].to(torch.float32)
            beta_chunk = beta[b, start:end, hv].to(torch.float32)

            gram = k_chunk @ k_chunk.T
            tril = torch.tril(gram, diagonal=-1)
            if g is not None:
                g_chunk = g[b, start:end, hv].to(torch.float32)
                gate = torch.exp2(g_chunk[:, None] - g_chunk[None, :]) if use_exp2 else torch.exp(
                    g_chunk[:, None] - g_chunk[None, :])
                tril = tril * torch.tril(gate, diagonal=-1)

            tril = tril * beta_chunk[:, None]
            a_chunk = torch.linalg.solve_triangular(eye + tril, eye, upper=False)
            A[b, start:end, hv, :length] = a_chunk

            vb = v_chunk * beta_chunk[:, None]
            kb = k_chunk * beta_chunk[:, None]
            if g is not None:
                gate_row = torch.exp2(g_chunk) if use_exp2 else torch.exp(g_chunk)
                kb = kb * gate_row[:, None]

            u[b, start:end, hv] = a_chunk @ vb
            w[b, start:end, hv] = a_chunk @ kb

    return w, u, A


def ref_impl(k, v, beta, g=None, cu_seqlens=None, chunk_indices=None, chunk_size=64, use_exp2=False):
    if fla_func is not None:
        return fla_func(
            k=k,
            v=v,
            beta=beta,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
        )
    return torch_ref(
        k=k,
        v=v,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        use_exp2=use_exp2,
    )


@pytest.mark.skipif(not do_test(), reason='tilelang is not available')
class TestChunkGatedDeltaRuleForwardIntra:

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
        'batch,seqlen,heads,v_heads,head_dim,value_dim,chunk_size,dtype,beta_dtype,g_dtype,use_g,use_exp2',
        [
            (2, 64, 4, 4, 64, 64, 64, torch.bfloat16, torch.bfloat16, torch.float32, False, False),
            (1, 65, 4, 4, 64, 64, 64, torch.bfloat16, torch.bfloat16, torch.float32, True, True),
        ],
    )
    def test_fixed(self, batch, seqlen, heads, v_heads, head_dim, value_dim, chunk_size, dtype, beta_dtype,
                   g_dtype, use_g, use_exp2):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra

        k = torch.rand(batch, seqlen, heads, head_dim, dtype=dtype) - 0.5
        v = torch.rand(batch, seqlen, v_heads, value_dim, dtype=dtype) - 0.5
        beta = torch.rand(batch, seqlen, v_heads, dtype=beta_dtype)
        g = None
        if use_g:
            g = -2 * torch.rand(batch, seqlen, v_heads, dtype=g_dtype)

        out_w, out_u, out_A = chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
        )
        ref_w, ref_u, ref_A = ref_impl(
            k=k,
            v=v,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
        )
        atol, rtol = get_tolerances(dtype)
        torch.testing.assert_close(out_w, ref_w.to(out_w.dtype), atol=atol, rtol=rtol)
        torch.testing.assert_close(out_u, ref_u.to(out_u.dtype), atol=atol, rtol=rtol)
        torch.testing.assert_close(out_A, ref_A.to(out_A.dtype), atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        'cu_seqlens,heads,v_heads,head_dim,value_dim,chunk_size,dtype,beta_dtype,g_dtype,use_g,use_exp2',
        [
            ([0, 65, 129], 4, 4, 64, 64, 64, torch.bfloat16, torch.bfloat16, torch.float32, False, False),
            ([0, 127, 2051], 4, 4, 64, 64, 64, torch.bfloat16, torch.bfloat16, torch.float32, True, False),
        ],
    )
    def test_varlen(self, cu_seqlens, heads, v_heads, head_dim, value_dim, chunk_size, dtype, beta_dtype, g_dtype,
                    use_g, use_exp2):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra

        total = cu_seqlens[-1]
        k = torch.rand(1, total, heads, head_dim, dtype=dtype) - 0.5
        v = torch.rand(1, total, v_heads, value_dim, dtype=dtype) - 0.5
        beta = torch.rand(1, total, v_heads, dtype=beta_dtype)
        g = None
        if use_g:
            g = -2 * torch.rand(1, total, v_heads, dtype=g_dtype)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device='cuda')
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        out_w, out_u, out_A = chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
        )
        ref_w, ref_u, ref_A = ref_impl(
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            use_exp2=use_exp2,
        )
        atol, rtol = get_tolerances(dtype)
        torch.testing.assert_close(out_w, ref_w.to(out_w.dtype), atol=atol, rtol=rtol)
        torch.testing.assert_close(out_u, ref_u.to(out_u.dtype), atol=atol, rtol=rtol)
        torch.testing.assert_close(out_A, ref_A.to(out_A.dtype), atol=atol, rtol=rtol)
