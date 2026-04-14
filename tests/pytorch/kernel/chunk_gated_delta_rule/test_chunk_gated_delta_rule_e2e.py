import pytest
import torch

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_func
except Exception:
    fla_func = None


def do_test():
    try:
        import tilelang  # noqa: F401
        return torch.cuda.is_available() and fla_func is not None
    except Exception:
        return False


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 2e-4, 2e-4
    return 2e-2, 2e-2


def make_varlen_view(total: int,
                     heads: int,
                     dim: int,
                     dtype: torch.dtype,
                     stride0_scale: int = 2) -> torch.Tensor:
    base = torch.rand(stride0_scale, total, heads, dim, dtype=dtype, device='cuda') - 0.5
    return base[:1]


@pytest.mark.skipif(not do_test(), reason='tilelang/fla/cuda is not available')
class TestChunkGatedDeltaRuleE2E:

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
        'cu_seqlens,heads,v_heads,head_dim,value_dim,dtype,output_final_state,non_contiguous_kv',
        [
            ([0, 127, 257], 16, 32, 128, 128, torch.bfloat16, True, True),
        ],
    )
    def test_varlen(self, cu_seqlens, heads, v_heads, head_dim, value_dim, dtype, output_final_state,
                    non_contiguous_kv):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.chunk import chunk_gated_delta_rule

        total = cu_seqlens[-1]
        num_seqs = len(cu_seqlens) - 1
        q = torch.rand(1, total, heads, head_dim, dtype=dtype) - 0.5
        if non_contiguous_kv:
            k = make_varlen_view(total, heads, head_dim, dtype)
            v = make_varlen_view(total, v_heads, value_dim, dtype)
        else:
            k = torch.rand(1, total, heads, head_dim, dtype=dtype) - 0.5
            v = torch.rand(1, total, v_heads, value_dim, dtype=dtype) - 0.5
        g = -2.0 * torch.rand(1, total, v_heads, dtype=torch.float32)
        beta = torch.rand(1, total, v_heads, dtype=dtype)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device='cuda')
        initial_state = torch.randn(num_seqs, v_heads, head_dim, value_dim, device='cuda',
                                    dtype=torch.float32) * 0.05

        out, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ref_out, ref_final_state = fla_func(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        atol, rtol = get_tolerances(dtype)
        torch.testing.assert_close(out, ref_out.to(out.dtype), atol=atol, rtol=rtol)
        if output_final_state:
            torch.testing.assert_close(final_state, ref_final_state, atol=atol, rtol=rtol)
        else:
            assert final_state is None
