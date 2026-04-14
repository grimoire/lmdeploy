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


def torch_ref(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False, output_dtype=torch.float):
    out = torch.empty_like(g, dtype=output_dtype or g.dtype)

    if cu_seqlens is None:
        if head_first:
            batch, heads, seqlen = g.shape
            for b in range(batch):
                for h in range(heads):
                    for start in range(0, seqlen, chunk_size):
                        end = min(start + chunk_size, seqlen)
                        chunk = g[b, h, start:end].to(torch.float32)
                        if reverse:
                            chunk = torch.flip(torch.cumsum(torch.flip(chunk, dims=[0]), dim=0), dims=[0])
                        else:
                            chunk = torch.cumsum(chunk, dim=0)
                        if scale is not None:
                            chunk = chunk * scale
                        out[b, h, start:end] = chunk.to(out.dtype)
        else:
            batch, seqlen, heads = g.shape
            for b in range(batch):
                for h in range(heads):
                    for start in range(0, seqlen, chunk_size):
                        end = min(start + chunk_size, seqlen)
                        chunk = g[b, start:end, h].to(torch.float32)
                        if reverse:
                            chunk = torch.flip(torch.cumsum(torch.flip(chunk, dims=[0]), dim=0), dims=[0])
                        else:
                            chunk = torch.cumsum(chunk, dim=0)
                        if scale is not None:
                            chunk = chunk * scale
                        out[b, start:end, h] = chunk.to(out.dtype)
    else:
        assert g.shape[0] == 1
        assert not head_first
        cu = cu_seqlens.tolist()
        _, _, heads = g.shape
        for seq_idx in range(len(cu) - 1):
            bos, eos = cu[seq_idx], cu[seq_idx + 1]
            seqlen = eos - bos
            for h in range(heads):
                for start in range(0, seqlen, chunk_size):
                    end = min(start + chunk_size, seqlen)
                    chunk = g[0, bos + start:bos + end, h].to(torch.float32)
                    if reverse:
                        chunk = torch.flip(torch.cumsum(torch.flip(chunk, dims=[0]), dim=0), dims=[0])
                    else:
                        chunk = torch.cumsum(chunk, dim=0)
                    if scale is not None:
                        chunk = chunk * scale
                    out[0, bos + start:bos + end, h] = chunk.to(out.dtype)

    return out


@pytest.mark.skipif(not do_test(), reason='tilelang is not available')
class TestChunkLocalCumsumScalar:

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
        'cu_seqlens,heads,chunk_size,reverse,scale,input_dtype',
        [
            ([0, 127, 2051], 4, 64, False, 0.5, torch.bfloat16),
        ],
    )
    def test_varlen(self, cu_seqlens, heads, chunk_size, reverse, scale, input_dtype):
        from lmdeploy.pytorch.kernels.cuda.chunk_gated_delta_rule.utils import chunk_local_cumsum_scalar

        total = cu_seqlens[-1]
        g = torch.rand(1, total, heads, dtype=input_dtype) - 0.5
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device='cuda')
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        out = chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=False,
            output_dtype=torch.float32,
            chunk_indices=chunk_indices,
        )
        ref = torch_ref(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=False,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
