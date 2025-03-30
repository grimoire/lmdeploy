import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.kernels.fused_moe import fused_moe


class TestPermuteInputs:

    @pytest.fixture
    def M(self):
        yield 4096

    @pytest.fixture
    def hidden_size(self):
        yield 7168

    @pytest.fixture
    def topk(self):
        yield 6

    @pytest.fixture
    def num_experts(self):
        yield 64

    @pytest.fixture
    def aligned_size(self):
        yield 32

    @pytest.fixture
    def dtype(self):
        yield torch.bfloat16

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def inputs(self, M, hidden_size, dtype, device):
        yield torch.rand((M, hidden_size), dtype=dtype, device=device)

    @pytest.fixture
    def topk_ids(self, M, topk, num_experts, device):
        val = torch.rand((M, num_experts), device=device)
        yield val.topk(topk, -1)[1]

    def _align_up(self, m, aligned_size):
        return (m + aligned_size - 1) // aligned_size * aligned_size

    @pytest.fixture
    def gt(self, inputs, topk_ids, num_experts, aligned_size):
        device = inputs.device
        hidden_size = inputs.size(1)
        topk = topk_ids.size(1)
        flatten_topk_ids = topk_ids.flatten()
        sorted_topk_ids, sorted_ids = flatten_topk_ids.sort()
        cum_token = 0
        permuted_inputs = []
        m_indices = []
        permuted_map = torch.empty_like(topk_ids).flatten()
        for exp_id in range(num_experts):
            exp_mask = sorted_topk_ids == exp_id
            exp_sorted_ids = sorted_ids[exp_mask]
            num_ids = exp_sorted_ids.size(0)
            permuted_map[exp_sorted_ids] = torch.arange(cum_token, num_ids + cum_token, device=device)
            exp_inputs = inputs[exp_sorted_ids // topk]
            m_size = exp_inputs.size(0)
            aligned_m_size = self._align_up(m_size, aligned_size)
            cum_token += aligned_m_size
            aligned_inputs = inputs.new_zeros((aligned_m_size, hidden_size))
            aligned_inputs[:m_size] = exp_inputs
            exp_m_indices = topk_ids.new_full((aligned_m_size, ), exp_id)
            permuted_inputs.append(aligned_inputs)
            m_indices.append(exp_m_indices)
        permuted_inputs = torch.cat(permuted_inputs, dim=0)
        m_indices = torch.cat(m_indices, dim=0)
        yield permuted_inputs, m_indices, permuted_map.reshape(topk_ids.shape)

    @torch.inference_mode()
    def test_permute_inputs(self, inputs, topk_ids, num_experts, aligned_size, gt):
        from lmdeploy.pytorch.kernels.cuda.fused_moe import permute_inputs
        permuted_inputs, m_indices, permuted_map = permute_inputs(inputs,
                                                                  topk_ids,
                                                                  num_experts,
                                                                  aligned_size=aligned_size)
        gt_inputs, gt_indices, gt_map = gt

        gt_size = gt_inputs.size(0)
        permuted_inputs = permuted_inputs[:gt_size]
        m_indices = m_indices[:gt_size]
        torch.testing.assert_close(permuted_inputs, gt_inputs)
        torch.testing.assert_close(m_indices, gt_indices)
        torch.testing.assert_close(permuted_map, gt_map)


class TestUnpermuteOutputs:

    @pytest.fixture
    def M(self):
        yield 4096

    @pytest.fixture
    def topk(self):
        yield 6

    @pytest.fixture
    def num_experts(self):
        yield 64

    @pytest.fixture
    def EM(self, M, topk):
        yield M * topk + 128

    @pytest.fixture
    def hidden_size(self):
        yield 7168

    @pytest.fixture
    def dtype(self):
        yield torch.bfloat16

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def permuted_output(self, EM, hidden_size, dtype, device):
        yield torch.rand(EM, hidden_size, dtype=dtype, device=device)

    @pytest.fixture
    def weights(self, M, topk, dtype, device):
        w = torch.rand(M, topk, dtype=dtype, device=device)
        yield w - 0.5

    @pytest.fixture
    def permuted_map(self, M, topk, EM, dtype, device):
        indices = torch.randperm(EM, device=device)[:M * topk]
        yield indices.reshape(M, topk)

    @pytest.fixture
    def gt(self, permuted_output, weights, permuted_map):
        outputs = permuted_output[permuted_map.flatten()]
        outputs = outputs.unflatten(0, weights.shape)
        outputs *= weights[..., None]
        yield outputs.sum(1)

    def test_unpermute_outputs(self, permuted_output, weights, permuted_map, gt):
        from lmdeploy.pytorch.kernels.cuda.fused_moe import unpermute_outputs
        output = unpermute_outputs(permuted_output, weights, permuted_map)

        torch.testing.assert_close(output, gt)


class TestGroupedGemm:

    @pytest.fixture
    def aligned_size(self):
        yield 128

    @pytest.fixture
    def M(self, aligned_size):
        yield aligned_size * 200

    @pytest.fixture
    def K(self):
        yield 7168

    @pytest.fixture
    def N(self):
        yield 2560

    @pytest.fixture
    def num_experts(self):
        yield 64

    @pytest.fixture
    def dtype(self):
        yield torch.bfloat16

    @pytest.fixture
    def device(self):
        yield 'cuda'

    @pytest.fixture
    def inputs(self, M, K, dtype, device):
        a = torch.rand(M, K, dtype=dtype, device=device)
        a = a - 0.5
        a /= 10
        yield a

    @pytest.fixture
    def weights(self, num_experts, N, K, dtype, device):
        w = torch.rand(num_experts, N, K, dtype=dtype, device=device)
        w = w - 0.5
        w /= 10
        yield w

    @pytest.fixture
    def m_indices(self, M, aligned_size, num_experts, device):
        assert M % aligned_size == 0
        num_blocks = M // aligned_size
        block_exp_ids = torch.randint(0, num_experts, (num_blocks, ), device=device)
        indices = block_exp_ids[:, None].repeat(1, aligned_size).flatten()
        yield indices

    @pytest.fixture
    def gt(self, inputs, weights, m_indices):
        M = inputs.size(0)
        N = weights.size(1)
        E = weights.size(0)

        outs = inputs.new_empty(M, N)

        for exp_id in range(E):
            idx_mask = m_indices == exp_id
            exp_inputs = inputs[idx_mask]
            exp_weights = weights[exp_id]
            exp_outs = torch.nn.functional.linear(exp_inputs, exp_weights)
            outs[idx_mask] = exp_outs

        yield outs

    def test_unpermute_outputs(self, inputs, weights, m_indices, aligned_size, gt):
        from lmdeploy.pytorch.kernels.cuda.fused_moe import grouped_gemm
        outs = grouped_gemm(inputs, weights, m_indices, aligned_size)

        torch.testing.assert_close(outs, gt, atol=4e-3, rtol=1e-4)


def _mlp_forward(hidden_states, gate_proj, up_proj, down_proj):
    gate = F.linear(hidden_states, gate_proj)
    up = F.linear(hidden_states, up_proj)
    return F.linear(F.silu(gate) * up, down_proj)


class TestFusedMoe:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def in_size(self):
        yield 128

    @pytest.fixture
    def seq_len(seq_len):
        yield 128

    @pytest.fixture
    def hidden_size(self):
        yield 256

    @pytest.fixture
    def out_size(self):
        yield 128

    @pytest.fixture
    def num_experts(self):
        yield 4

    @pytest.fixture
    def top_k(self):
        yield 2

    @pytest.fixture
    def renormalize(self):
        yield True

    @pytest.fixture
    def hidden_states(self, seq_len, in_size, dtype, device):
        ret = torch.rand(seq_len, in_size, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w1(self, num_experts, hidden_size, in_size, dtype, device):
        ret = torch.rand(num_experts, hidden_size, in_size, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w2(self, num_experts, out_size, hidden_size, dtype, device):
        ret = torch.rand(num_experts, out_size, hidden_size // 2, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def router_logits(self, seq_len, num_experts, dtype, device):
        yield torch.rand(seq_len, num_experts, dtype=dtype, device=device)

    @pytest.fixture
    def topk_logits(self, router_logits, top_k):
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        yield torch.topk(routing_weights, top_k, dim=-1)

    @pytest.fixture
    def topk_weights(self, topk_logits):
        yield topk_logits[0]

    @pytest.fixture
    def topk_idx(self, topk_logits):
        yield topk_logits[1]

    @pytest.fixture
    def gt(self, hidden_states, w1, w2, topk_weights, topk_idx, renormalize):
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        seq_len = hidden_states.size(0)
        out_size = w2.size(1)
        output = hidden_states.new_zeros(seq_len, out_size)
        num_experts = w1.size(0)
        for eid in range(num_experts):
            token_idx, k_idx = torch.where(topk_idx == eid)
            gate_proj, up_proj = w1[eid].chunk(2, dim=0)
            down_proj = w2[eid]
            tmp_out = _mlp_forward(hidden_states[token_idx], gate_proj, up_proj, down_proj)
            tmp_out = tmp_out * topk_weights[token_idx, k_idx, None]
            output.index_add_(0, token_idx, tmp_out.to(output.dtype))
        yield output

    @torch.inference_mode()
    def test_fused_moe(self, hidden_states, w1, w2, topk_weights, topk_idx, top_k, renormalize, gt):
        output = fused_moe(hidden_states, w1, w2, topk_weights, topk_idx, topk=top_k, renormalize=renormalize)
        torch.testing.assert_close(output, gt, atol=1e-3, rtol=1e-3)


# class TestFusedMoeW8A8(TestFusedMoe):

#     @pytest.fixture
#     def quant_states(self, hidden_states):
#         from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_token_quant_int8
#         states_i8, states_scale = per_token_quant_int8(hidden_states, 1e-7)
#         yield states_i8, states_scale

#     def quant_weight(self, w):
#         from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_channel_quant
#         num_experts, num_outs, _ = w.shape
#         w = w.flatten(0, -2)
#         w_i8, w_scale = per_channel_quant(w, torch.int8)
#         w_i8 = w_i8.view(num_experts, num_outs, -1)
#         w_scale = w_scale.view(num_experts, num_outs, -1)
#         return w_i8, w_scale

#     @pytest.fixture
#     def quant_w1(self, w1):
#         w_i8, w_scale = self.quant_weight(w1)
#         yield w_i8, w_scale

#     @pytest.fixture
#     def quant_w2(self, w2):
#         w_i8, w_scale = self.quant_weight(w2)
#         yield w_i8, w_scale

#     @torch.inference_mode()
#     def test_fused_moe(self, quant_states, quant_w1, quant_w2, topk_weights, topk_idx, top_k, renormalize, gt):
#         from lmdeploy.pytorch.kernels.cuda.w8a8_fused_moe import fused_moe_w8a8
#         state_i8, state_scale = quant_states
#         w1_i8, w1_scale = quant_w1
#         w2_i8, w2_scale = quant_w2

#         output = fused_moe_w8a8(state_i8,
#                                 state_scale,
#                                 w1_i8,
#                                 w1_scale,
#                                 w2_i8,
#                                 w2_scale,
#                                 topk_weights=topk_weights,
#                                 topk_ids=topk_idx,
#                                 topk=top_k,
#                                 out_dtype=torch.float16,
#                                 renormalize=renormalize)
#         torch.testing.assert_close(output, gt, atol=5e-3, rtol=1e-3)
