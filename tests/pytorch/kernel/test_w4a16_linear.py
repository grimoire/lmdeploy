import pytest
import torch



def _to_w4(w: torch.Tensor):
    """to w4."""
    K = w.size(0)
    w_bit: int = 4
    pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
    elem_per_val = 32 // w_bit
    # reorder
    w = w.to(torch.int32)
    w = w.reshape(-1, elem_per_val)

    out = w.new_zeros(w.size(0))
    for i in range(8):
        ws = w[:, pack_order[i]]
        out |= ws << (i * w_bit)

    return out.reshape(K, -1)

class TestW4A16Linear:
    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def M(self, request):
        yield request.param

    @pytest.fixture
    def N(self, request):
        yield request.param

    @pytest.fixture
    def K(self, request):
        yield request.param

    @pytest.fixture
    def group_size(self, request):
        yield request.param

    @pytest.fixture
    def input(self, M, K, dtype, device):
        yield torch.rand(M, K, dtype=dtype, device=device)

    @pytest.fixture
    def weight(self, K, N, dtype, device):
        yield torch.rand(K, N, dtype=dtype, device=device)

    @pytest.fixture
    def fake_quant(self, weight, group_size):
        from lmdeploy.lite.quantization.awq import pseudo_quantize_tensor
        yield pseudo_quantize_tensor(
            weight.t(), 4, group_size, return_scale_zeros=True)

    @pytest.fixture
    def w4weight(self, fake_quant):
        val = fake_quant[0].t()
        yield _to_w4(val)
    
    @pytest.fixture
    def scales(self, fake_quant):
        yield fake_quant[1][..., 0].t()

    @pytest.fixture
    def w4zeros(self, fake_quant):
        val = fake_quant[2][..., 0].t()
        yield _to_w4(val)

    @pytest.fixture
    def fake_weight(self, fake_quant):
        qweight, scales, zeros = fake_quant
        group_size = qweight.size(1)//scales.size(1)
        pweight = (qweight.unflatten(-1, (-1, group_size)) - zeros) * scales
        yield pweight.flatten(1, 2).t()
    
    @pytest.fixture
    def gt(self, input, fake_weight):
        yield torch.matmul(input, fake_weight)

    @pytest.mark.parametrize(('M', 'N', 'K', 'group_size'), [(32, 64, 128, 16)],
                             indirect=True)
    def test_w4a16_linear(self, input, w4weight, scales, w4zeros, gt):
        from lmdeploy.pytorch.kernels.cuda.w4a16 import w4a16_linear
        out = w4a16_linear(input, w4weight, scales, w4zeros)
        torch.testing.assert_close(out, gt)