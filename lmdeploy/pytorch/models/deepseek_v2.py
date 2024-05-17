# Copyright (c) OpenMMLab. All rights reserved.
import gc
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch.kernels.fused_moe import fused_moe

from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd
from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        default_load_linear, load_no_recursive,
                                        rowwise_parallelize_linear)


class PatchedDeepseekV2Attention(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['q_a_proj', 'kv_a_proj_with_mqa']:
            default_load_linear(getattr(self, mod_name),
                                loader,
                                rank=rank,
                                prefix=mod_name)

        for mod_name in ['q_a_layernorm', 'kv_a_layernorm']:
            load_no_recursive(getattr(self, mod_name),
                              loader,
                              rank=rank,
                              prefix=mod_name)

        for mod_name in ['q_b_proj', 'kv_b_proj']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['o_proj']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    def _update_model_fn(self):
        """update model."""
        qk_nope_head_dim = self.qk_nope_head_dim
        v_head_dim = self.v_head_dim

        kv_b_proj = self.kv_b_proj
        w_kc, w_vc = kv_b_proj.weight.unflatten(
            0, (-1, qk_nope_head_dim + v_head_dim)).split(
                [qk_nope_head_dim, v_head_dim], dim=1)

        self.register_parameter('w_kc',
                                torch.nn.Parameter(w_kc, requires_grad=False))
        w_vc = w_vc.transpose(1, 2).contiguous()
        self.register_parameter('w_vc',
                                torch.nn.Parameter(w_vc, requires_grad=False))

        delattr(self, 'kv_b_proj')
        gc.collect()

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        world_size: int = 1,
    ):
        """forward impl."""
        context = self.context.context
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        max_kv_seq_length = context.max_kv_seq_length
        num_heads = self.num_heads // world_size
        q_len = hidden_states.size(1)

        def __qkv_proj(hidden_states):
            """qkv proj."""
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
            q = q.view(q_len, num_heads, self.q_head_dim)
            # q_pe: (q_len, num_heads, qk_rope_head_dim)
            q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # q_nope: (q_len, num_heads, kv_lora_rank)
            q_nope = torch.bmm(q_nope.transpose(0, 1),
                               self.w_kc).transpose(0, 1)

            compressed_kv = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
            # compressed_kv: (q_len, 1, kv_lora_rank)
            # k_pe: (q_len, 1, qk_rope_head_dim)
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1)
            # kv_heads == 1
            compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

            return q_nope, q_pe, k_pe, compressed_kv

        def __rotary_emb_fn(q_pe, k_pe, out_q_pe, out_k_pe):
            """rope."""
            if not hasattr(context, '_cos'):
                cos, sin = self.rotary_emb(q_pe, seq_len=max_kv_seq_length)
                context._cos = cos
                context._sin = sin
            else:
                cos = context._cos
                sin = context._sin
            apply_rotary_pos_emb(q_pe,
                                 k_pe,
                                 cos,
                                 sin,
                                 position_ids,
                                 context.position_ids_1d,
                                 q_embed=out_q_pe,
                                 k_embed=out_k_pe)
            return out_q_pe, out_k_pe

        q_nope, q_pe, k_pe, compressed_kv = __qkv_proj(hidden_states)

        nope_size = q_nope.size(-1)
        pe_size = q_pe.size(-1)
        query_states = k_pe.new_empty(q_len, num_heads, nope_size + pe_size)
        key_states = k_pe.new_empty(q_len, 1, nope_size + pe_size)
        query_states[..., :nope_size] = q_nope
        key_states[..., :nope_size] = compressed_kv
        value_states = compressed_kv

        __rotary_emb_fn(q_pe, k_pe, query_states[..., nope_size:],
                        key_states[..., nope_size:])

        fill_kv_cache(
            key_states,
            value_states[..., :0],
            past_key_value[0],
            past_key_value[0][..., :0],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )

        attn_output = query_states[..., :nope_size]
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[0][..., :nope_size],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
        )

        # (num_heads, q_len, nope_size)
        attn_output = attn_output.transpose(0, 1)
        # (num_heads, q_len, v_head_dim)
        attn_output = torch.bmm(attn_output, self.w_vc)
        # (1, q_len, o_proj_input)
        attn_output = attn_output.permute(1, 0, 2).flatten(-2, -1)[None]

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            world_size=world_size,
        )


class PatchedDeepseekV2MoE(nn.Module):

    def _update_model_fn(self):
        """update model."""
        num_experts = len(self.experts)

        def __get_meta():
            exp = self.experts[0]
            ffn_dim = exp.gate_proj.weight.size(0)
            hidden_dim = exp.down_proj.weight.size(0)
            dtype = exp.gate_proj.weight.dtype
            device = exp.gate_proj.weight.device
            return ffn_dim, hidden_dim, dtype, device

        def __copy_assign_param(param, weight):
            """copy assign."""
            weight.copy_(param.data)
            param.data = weight

        ffn_dim, hidden_dim, dtype, device = __get_meta()

        gate_up_weights = torch.empty(num_experts,
                                      ffn_dim * 2,
                                      hidden_dim,
                                      device=device,
                                      dtype=dtype)
        down_weights = torch.empty(num_experts,
                                   hidden_dim,
                                   ffn_dim,
                                   device=device,
                                   dtype=dtype)

        for exp_id, exp in enumerate(self.experts):
            __copy_assign_param(exp.gate_proj.weight,
                                gate_up_weights[exp_id, :ffn_dim])
            __copy_assign_param(exp.up_proj.weight, gate_up_weights[exp_id,
                                                                    ffn_dim:])
            __copy_assign_param(exp.down_proj.weight, down_weights[exp_id])

        torch.cuda.empty_cache()

        self.register_buffer('gate_up_weights', gate_up_weights)
        self.register_buffer('down_weights', down_weights)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs

    def moe_infer(self, x, topk_ids, topk_weight):
        """moe infer."""
        ret = fused_moe(x,
                        self.gate_up_weights,
                        self.down_weights,
                        topk_weight,
                        topk_ids,
                        topk=self.num_experts_per_tok,
                        renormalize=False)
        return ret
