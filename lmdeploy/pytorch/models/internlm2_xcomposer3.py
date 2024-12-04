# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.distributed import get_world_rank
from lmdeploy.pytorch.engine.input_process import (BaseModelInputProcessor,
                                                   PreprocessInputResult)
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 build_rotary_embedding)
from lmdeploy.pytorch.weight_loader.model_weight_loader import (
    default_weight_loader, load_weight)

from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin


class PLoRA(nn.Linear):
    """plora."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        colwise: bool = True,
    ):
        world_size, _ = get_world_rank()
        if world_size > 1:
            if colwise:
                out_features //= world_size
            else:
                in_features //= world_size
        super().__init__(in_features, out_features, bias, device, dtype)
        self.colwise = colwise
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_scaling = self.lora_alpha / self.lora_r

        self.Plora_A = nn.Linear(in_features,
                                 self.lora_r,
                                 bias=False,
                                 device=device,
                                 dtype=dtype)
        self.Plora_B = nn.Linear(self.lora_r,
                                 out_features,
                                 bias=False,
                                 device=device,
                                 dtype=dtype)

        self.MPlora_A = nn.Linear(in_features,
                                  256,
                                  bias=False,
                                  device=device,
                                  dtype=dtype)
        self.MPlora_B = nn.Linear(256,
                                  out_features,
                                  bias=False,
                                  device=device,
                                  dtype=dtype)

        self.weight.weight_loader = self.weight_loader
        if self.colwise:
            self.Plora_B.weight.weight_loader = self.weight_loader
            self.MPlora_B.weight.weight_loader = self.weight_loader
        else:
            self.Plora_A.weight.weight_loader = self.weight_loader
            self.MPlora_A.weight.weight_loader = self.weight_loader

    def _weight_loader_tp_colwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for colwise linear."""
        weight = loaded_weight.chunk(world_size, dim=0)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, rank: int,
                                  world_size: int):
        """weight loader for rowwise linear."""
        if loaded_weight.dim() == 2:
            weight = loaded_weight.chunk(world_size, dim=1)[rank]
            return default_weight_loader(param, weight)
        else:
            # bias
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor):
        """weight loader."""
        world_size, rank = get_world_rank()
        if world_size == 1:
            return default_weight_loader(param, loaded_weight)

        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank,
                                                  world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank,
                                                  world_size)

    def forward(self,
                x,
                mem_idx: torch.Tensor = None,
                im_idx: torch.Tensor = None):
        """forward."""
        B, N, C = x.shape
        x = x.reshape(-1, C)
        res = super().forward(x)
        if im_idx is not None:
            part_x = x[im_idx]
            lora_x = self.Plora_B(self.Plora_A(part_x))
            res.index_add_(0, im_idx, lora_x, alpha=self.lora_scaling)

        if mem_idx is not None:
            mem_x = x[mem_idx]
            lora_x = self.MPlora_B(self.MPlora_A(mem_x))
            res.index_add_(0, mem_idx, lora_x, alpha=self.lora_scaling)

        ret = res.reshape(B, N, -1)
        if not self.colwise:
            world_size, _ = get_world_rank()
            if world_size > 1:
                dist.all_reduce(ret)

        return ret


class InternLM2Attention(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        self.wqkv = PLoRA(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
            device=device,
            dtype=dtype,
            lora_r=256,
            lora_alpha=256,
            colwise=True,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
        )

        self.wo = PLoRA(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.bias,
            device=device,
            dtype=dtype,
            lora_r=256,
            lora_alpha=256,
            colwise=False,
        )

    def _attn_compress(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        mem_idx: torch.Tensor,
        attn_seqlen: torch.Tensor,
        attn_start_loc: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        beacon_rotary_pos_emb: Tuple[torch.FloatTensor,
                                     torch.FloatTensor] = None,
        attn_metadata: Any = None,
    ):
        """compress attn."""
        from lmdeploy.pytorch.kernels.cuda import (fill_kv_cache,
                                                   flash_attention_fwd)

        q_shape = query_states.shape
        query_states = query_states.flatten(0, -3)
        key_states = key_states.flatten(0, -3)
        value_states = value_states.flatten(0, -3)
        max_seqlen = query_states.size(0)

        # slice beacon
        fill_key = key_states[mem_idx]
        fill_val = value_states[mem_idx]

        cos, sin = beacon_rotary_pos_emb
        fill_key, _ = self.apply_rotary_pos_emb(fill_key, fill_key, cos, sin)

        # fill kv seqlen
        k_caches, v_caches = past_key_value
        fill_start_loc = attn_metadata.q_start_loc
        fill_seqlen = attn_metadata.q_seqlens
        fill_kvlen = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        fill_kv_cache(
            fill_key,
            fill_val,
            k_caches=k_caches,
            v_caches=v_caches,
            q_start_loc=fill_start_loc,
            q_seq_length=fill_seqlen,
            kv_seq_length=fill_kvlen,
            max_q_seq_length=fill_key.size(0),
            block_offsets=block_offsets,
        )

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
        attn_out = torch.empty_like(query_states)
        flash_attention_fwd(q_states=query_states,
                            k_states=key_states,
                            v_states=value_states,
                            o_states=attn_out,
                            q_start_loc=attn_start_loc,
                            q_seqlens=attn_seqlen,
                            kv_start_loc=attn_start_loc,
                            kv_seqlens=attn_seqlen,
                            max_seqlen=max_seqlen,
                            causal=True,
                            kv_layout='shd')

        return attn_out.view(q_shape)

    def _attn_default(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        q_shape = query_states.shape
        query_states = query_states.flatten(0, -3)
        key_states = key_states.flatten(0, -3)
        value_states = value_states.flatten(0, -3)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(q_shape)

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mem_idx: torch.Tensor = None,
        im_idx: torch.Tensor = None,
        attn_metadata: Any = None,
        q_startloc_full: torch.Tensor = None,
        q_seqlen_full: torch.Tensor = None,
        beacon_rotary_pos_emb: Tuple[torch.FloatTensor,
                                     torch.FloatTensor] = None,
    ):
        qkv_states = self.wqkv(hidden_states, mem_idx=mem_idx, im_idx=im_idx)
        qkv_states = qkv_states.unflatten(
            -1, (-1, 2 + self.num_key_value_groups, self.head_dim))
        query_states = qkv_states[..., :self.num_key_value_groups, :]
        query_states = query_states.flatten(-3, -2)
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        if mem_idx is not None:
            attn_output = self._attn_compress(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                rotary_pos_emb=rotary_pos_emb,
                mem_idx=mem_idx,
                attn_seqlen=q_seqlen_full,
                attn_start_loc=q_startloc_full,
                past_key_value=past_key_value,
                beacon_rotary_pos_emb=beacon_rotary_pos_emb,
                attn_metadata=attn_metadata,
            )
        else:
            attn_output = self._attn_default(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
            )

        # o proj
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        attn_output = self.wo(attn_output, mem_idx=mem_idx, im_idx=im_idx)
        return attn_output


class InternLM2MLP(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        from transformers.activations import ACT2FN
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.w1 = PLoRA(self.hidden_size,
                        self.intermediate_size,
                        bias=False,
                        dtype=dtype,
                        device=device,
                        lora_r=256,
                        lora_alpha=256,
                        colwise=True)
        self.w3 = PLoRA(self.hidden_size,
                        self.intermediate_size,
                        bias=False,
                        dtype=dtype,
                        device=device,
                        lora_r=256,
                        lora_alpha=256,
                        colwise=True)
        self.w2 = PLoRA(self.intermediate_size,
                        self.hidden_size,
                        bias=False,
                        dtype=dtype,
                        device=device,
                        lora_r=256,
                        lora_alpha=256,
                        colwise=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self,
                x,
                mem_idx: torch.Tensor = None,
                im_idx: torch.Tensor = None):
        down_proj = self.w2(
            self.act_fn(self.w1(x, mem_idx=mem_idx, im_idx=im_idx)) *
            self.w3(x, mem_idx=mem_idx, im_idx=im_idx),
            mem_idx=mem_idx,
            im_idx=im_idx)

        return down_proj


class InternLM2DecoderLayer(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx

        # build attention layer
        self.attention = InternLM2Attention(config, dtype=dtype, device=device)

        # build MLP
        self.feed_forward = InternLM2MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.attention_norm = RMSNorm(config.hidden_size,
                                      config.rms_norm_eps,
                                      dtype=dtype,
                                      device=device)

        # build attention layer norm
        self.ffn_norm = RMSNorm(config.hidden_size,
                                config.rms_norm_eps,
                                dtype=dtype,
                                device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        mem_idx: torch.Tensor = None,
        im_idx: torch.Tensor = None,
        attn_metadata: Any = None,
        q_startloc_full: torch.Tensor = None,
        q_seqlen_full: torch.Tensor = None,
        beacon_rotary_pos_emb: Tuple[torch.FloatTensor,
                                     torch.FloatTensor] = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.attention(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            mem_idx=mem_idx,
            im_idx=im_idx,
            attn_metadata=attn_metadata,
            q_startloc_full=q_startloc_full,
            q_seqlen_full=q_seqlen_full,
            beacon_rotary_pos_emb=beacon_rotary_pos_emb,
        )

        # Fully Connected
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states,
                                          mem_idx=mem_idx,
                                          im_idx=im_idx)

        outputs = (hidden_states, residual)
        return outputs


class InternLM2Model(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size,
                                           config.hidden_size,
                                           self.padding_idx,
                                           dtype=dtype,
                                           device=device)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.Embedding(config.vocab_size,
                                           config.hidden_size,
                                           self.padding_idx,
                                           dtype=dtype,
                                           device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            InternLM2DecoderLayer(config,
                                  layer_idx,
                                  dtype=dtype,
                                  device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            dtype=dtype,
                            device=device)

        # build rotary embedding in Model
        rope_scaling = config.rope_scaling
        scaling_factor = 1.0
        emb_type = RopeType.LinearScaling
        if rope_scaling is not None:
            scaling_factor = rope_scaling.get('factor', scaling_factor)
            rope_type = rope_scaling['type']
            if rope_type == 'linear':
                emb_type = RopeType.LinearScaling
            if rope_type == 'dynamic':
                emb_type = RopeType.DynamicNTKScaling
            else:
                raise RuntimeError(f'Unsupported rope type: {rope_type}')
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        im_idx: torch.Tensor = None,
        mem_idx: torch.Tensor = None,
        attn_metadata: Any = None,
        q_startloc_full: torch.Tensor = None,
        q_seqlen_full: torch.Tensor = None,
        beacon_pos_ids: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        beacon_rotary_pos_emb = None
        if beacon_pos_ids is not None:
            cos, sin = self.rotary_emb(hidden_states, beacon_pos_ids)
            cos, sin = cos[0], sin[0]
            beacon_rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                mem_idx=mem_idx,
                im_idx=im_idx,
                attn_metadata=attn_metadata,
                q_startloc_full=q_startloc_full,
                q_seqlen_full=q_seqlen_full,
                beacon_rotary_pos_emb=beacon_rotary_pos_emb,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.tok_embeddings


class CLIPVisionTower(nn.Module):
    """vit."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        from .clip import build_vision_model
        self.select_layer = -1
        self.select_feature = 'patch'
        self.ctx_mgr = ctx_mgr
        self.device = device
        self.dtype = dtype

        self.vision_tower = build_vision_model(config,
                                               dtype=dtype,
                                               device=device)

    def forward(self, images, glb_GN, sub_GN):
        """forward."""
        shapes = []
        input_imgs = []
        for img in images:
            _, C, H, W = img.shape
            shapes.append([H // 560, W // 560])
            sub_img = img.reshape(1, 3, H // 560, 560, W // 560, 560)
            sub_img = sub_img.permute(0, 2, 4, 1, 3, 5)
            sub_img = sub_img.reshape(-1, 3, 560, 560).contiguous()
            glb_img = torch.nn.functional.interpolate(
                img.float(),
                size=(560, 560),
                mode='bicubic',
            ).to(sub_img.dtype)
            input_imgs.append(glb_img)
            input_imgs.append(sub_img)
        input_imgs = torch.cat(input_imgs, dim=0)
        image_features = self.vision_tower(
            input_imgs.to(device=self.device, dtype=self.dtype),
            vision_feature_layer=self.select_layer)
        image_features = image_features.last_hidden_state[:, 1:].to(
            input_imgs.dtype)
        _, N, C = image_features.shape
        H = int(math.sqrt(N))
        assert N == 40**2

        output_imgs = []
        output_len = []
        for [h, w] in shapes:
            B_ = h * w
            glb_img = image_features[:1]
            glb_img = glb_img.reshape(1, H // 2, 2, H // 2, 2, C)
            glb_img = glb_img.permute(0, 1, 3, 2, 4, 5)
            glb_img = glb_img.reshape(1, H // 2, H // 2, 4 * C)
            temp_glb_GN = sub_GN.repeat(1, H // 2, 1, 1)
            glb_img = torch.cat([glb_img, temp_glb_GN],
                                dim=2).reshape(1, -1, 4 * C)

            sub_img = image_features[1:1 + B_]
            sub_img = sub_img.reshape(B_, H // 2, 2, H // 2, 2, C)
            sub_img = sub_img.permute(0, 1, 3, 2, 4, 5)
            sub_img = sub_img.reshape(1, h, w, 20, 20, -1)
            sub_img = sub_img.permute(0, 1, 3, 2, 4, 5)
            sub_img = sub_img.reshape(1, h * 20, w * 20, 4 * C)
            temp_sub_GN = sub_GN.repeat(1, h * 20, 1, 1)
            sub_img = torch.cat([sub_img, temp_sub_GN],
                                dim=2).reshape(1, -1, 4 * C)

            output_imgs.append(torch.cat([glb_img, glb_GN, sub_img], dim=1))
            temp_len = int((h * w + 1) * 400 + 1 + (h + 1) * 20)
            assert temp_len == output_imgs[-1].shape[1]
            output_len.append(temp_len)

            image_features = image_features[1 + h * w:]

        output_imgs = torch.cat(output_imgs, dim=1)

        return output_imgs, output_len


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def build_vision_projector(dtype, device):
    projector_type = 'mlp2x_gelu'
    mm_hidden_size = 4096
    mid_hidden_size = 4096

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [
            nn.Linear(mm_hidden_size,
                      mid_hidden_size,
                      dtype=dtype,
                      device=device)
        ]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(
                nn.Linear(mid_hidden_size,
                          mid_hidden_size,
                          dtype=dtype,
                          device=device))

        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class InternLM2ForCausalLM(nn.Module, CudaGraphMixin, DeployModelMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.device = device
        self.dtype = dtype
        self.vocab_size = config.vocab_size

        self.model = InternLM2Model(config, dtype=dtype, device=device)

        self.output = nn.Linear(config.hidden_size,
                                config.vocab_size,
                                bias=False,
                                dtype=dtype,
                                device=device)

        self.tower_path = ctx_mgr.sub_model_paths[0]
        tower_config = AutoConfig.from_pretrained(
            self.tower_path).vision_config
        self.vit = CLIPVisionTower(tower_config,
                                   ctx_mgr,
                                   dtype=dtype,
                                   device=device)
        self.vit_loaded = False
        self.vision_proj = build_vision_projector(dtype, device)

        self.plora_glb_GN = nn.Parameter(
            torch.empty([1, 1, 4096], dtype=dtype, device=device))
        self.plora_sub_GN = nn.Parameter(
            torch.empty([1, 1, 1, 4096], dtype=dtype, device=device))
        self.mplora_mem = nn.Parameter(
            torch.empty([1, 1, 4096], dtype=dtype, device=device))

        self.input_processor = InternLM2XComposer3InputProcessor(config,
                                                                 dtype=dtype)

    def img2emb(self, image):
        img_embeds, img_split = self.vit(image, self.plora_glb_GN,
                                         self.plora_sub_GN)
        img_embeds = self.vision_proj(img_embeds)

        return img_embeds, img_split

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        images: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        q_startloc_full: torch.Tensor = None,
        q_seqlen_full: torch.Tensor = None,
        beacon_pos_ids: torch.Tensor = None,
        **kwargs,
    ):
        """forward."""
        inputs_embeds = None
        im_idx = None
        mem_idx = None
        if image_mask is not None:
            base_range = torch.arange(0,
                                      image_mask.size(0),
                                      device=image_mask.device)
            text_idx = base_range[image_mask == 0]
            im_idx = base_range[image_mask == 1]
            mem_idx = base_range[image_mask == 2]
            if len(text_idx) == 0:
                text_idx = None
            if len(im_idx) == 0:
                im_idx = None
            if len(mem_idx) == 0:
                mem_idx = None

            img_embeds, _ = self.img2emb(images)

            inputs_embeds = self.get_input_embeddings()(input_ids)

            if mem_idx is not None:
                inputs_embeds[0, mem_idx] = self.mplora_mem

            if im_idx is not None:
                inputs_embeds[0, im_idx] = img_embeds

        ret = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            im_idx=im_idx,
            mem_idx=mem_idx,
            q_startloc_full=q_startloc_full,
            q_seqlen_full=q_seqlen_full,
            beacon_pos_ids=beacon_pos_ids,
            inputs_embeds=inputs_embeds,
        )
        return ret

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.output(hidden_states)

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        q_start_loc = context.q_start_loc

        # vision inputs
        images = None
        image_mask = None
        q_seqlen_full = None
        q_startloc_full = None
        beacon_pos_ids = None
        if context.input_multimodals is not None:
            assert len(context.input_multimodals) == 1
            mm_img = context.input_multimodals[0].get('image', [])
            # flatten batch
            if len(mm_img) > 0:
                images = [data.data for data in mm_img]
                image_mask = [data.meta['im_mask'] for data in mm_img]
                q_seqlen_full = [mask.size(0) for mask in image_mask]

                # make tensors
                device = input_ids.device
                beacon_pos_ids = position_ids
                position_ids = [
                    torch.arange(0, seqlen, device=device)
                    for seqlen in q_seqlen_full
                ]
                position_ids = torch.stack(position_ids)
                q_seqlen_full = q_start_loc.new_tensor(q_seqlen_full)
                q_startloc_full = q_seqlen_full.cumsum(0) - q_seqlen_full
                image_mask = torch.cat(image_mask)
                valid_ids = input_ids[:, q_start_loc]
                new_input_ids = input_ids.new_zeros(1, image_mask.size(0))
                new_input_ids[:, q_startloc_full] = valid_ids
                input_ids = new_input_ids

        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            images=images,
            image_mask=image_mask,
            q_startloc_full=q_startloc_full,
            q_seqlen_full=q_seqlen_full,
            beacon_pos_ids=beacon_pos_ids,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""

        clip_prefix = 'vit.vision_tower.'
        clip_prefix_len = len(clip_prefix)
        clip_weights = dict()

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith(clip_prefix):
                new_name = name[clip_prefix_len:]
                clip_weights[new_name] = loaded_weight
                continue
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            param = params_dict[name]
            load_weight(param, loaded_weight)

        self.vit.vision_tower.load_weights(clip_weights.items())

    def get_input_processor(self) -> BaseModelInputProcessor:
        """get input processor."""
        return self.input_processor


def _div_up(a, b):
    return (a + b - 1) // b


class InternLM2XComposer3InputProcessor(BaseModelInputProcessor):
    """Phi3V input processor."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype) -> None:
        self.config = config
        self.dtype = dtype

    def _get_im_mask(self, pixel_values: torch.Tensor):
        """get im mask."""
        c_ratio = 4
        H, W = pixel_values.size()[-2:]
        h, w = H // 560, W // 560
        num_glb = 20 * 21
        num_sub = (h * 20) * (w * 20 + 1)
        num_all = num_glb + num_sub + 1
        num_beacon = _div_up(num_all, c_ratio)

        im_mask = torch.full((num_all + num_beacon + 1, ),
                             1,
                             dtype=torch.int64)
        im_mask[0] = 0
        im_mask[c_ratio + 1::c_ratio + 1] = 2
        im_mask[-1] = 2

        return im_mask

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        pad_ids = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            im_mask = self._get_im_mask(pixel_values)
            num_beacon = torch.count_nonzero(im_mask == 2)
            pad_ids += [0] * num_beacon

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_beacon,
                                       meta=dict(im_mask=im_mask))
            input_imgs.append(mm_data)
        pad_ids[0] = 1

        result = PreprocessInputResult(
            input_ids=pad_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result
