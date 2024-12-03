# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight


class CLIPVisionEmbeddings(nn.Module):
    """clip vision embedding."""

    def __init__(self,
                 config,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.empty(self.embed_dim, dtype=dtype, device=device))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(
            self.num_positions,
            self.embed_dim,
            dtype=dtype,
            device=device,
        )
        self.register_buffer('position_ids',
                             torch.arange(self.num_positions,
                                          device=device).expand((1, -1)),
                             persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int,
                                 width: int) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position
        encodings, to be able to use the model on higher resolution images.

        This method is also adapted to support torch.jit tracing.
        """

        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing
        # to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing(
        ) and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        from transformers.utils import torch_int

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions,
                                                  sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode='bicubic',
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self,
                pixel_values: torch.FloatTensor,
                interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size
                                             or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model"
                f' ({self.image_size}*{self.image_size}).')
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(
                self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    """clip attention."""

    def __init__(self,
                 config,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = build_qkv_proj(
            self.embed_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        self.scale = self.head_dim**-0.5

        # o_proj
        self.out_proj = build_rowwise_linear(self.embed_dim,
                                             self.embed_dim,
                                             bias=True,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ):
        """forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split_qkv(qkv_states)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        attn_output = F.scaled_dot_product_attention(q,
                                                     k,
                                                     v,
                                                     attn_mask=attn_mask,
                                                     scale=self.scale)

        # o proj
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(-2, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CLIPMLP(nn.Module):
    """clip mlp."""

    def __init__(self,
                 config,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        from transformers.activations import ACT2FN
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = build_colwise_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )
        self.fc2 = build_rowwise_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """clip encoder layer."""

    def __init__(self,
                 config,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config, dtype=dtype, device=device)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps,
                                        dtype=dtype,
                                        device=device)
        self.mlp = CLIPMLP(config, dtype=dtype, device=device)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps,
                                        dtype=dtype,
                                        device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ):
        """forward."""
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """clip encoder."""

    def __init__(self,
                 config,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config, dtype=dtype, device=device)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        vision_feature_layer: int = -1,
    ):
        """forward."""
        hidden_states = inputs_embeds
        num_vision_layers = len(self.layers) + vision_feature_layer + 1
        for _, encoder_layer in enumerate(self.layers[:num_vision_layers]):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask=causal_attention_mask,
            )

            hidden_states = layer_outputs

        return hidden_states


class CLIPVisionTransformer(nn.Module):
    """clip vision transformer."""

    def __init__(self,
                 config,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config,
                                               dtype=dtype,
                                               device=device)
        self.pre_layrnorm = nn.LayerNorm(embed_dim,
                                         eps=config.layer_norm_eps,
                                         dtype=dtype,
                                         device=device)
        self.encoder = CLIPEncoder(config, dtype=dtype, device=device)
        self.post_layernorm = nn.LayerNorm(embed_dim,
                                           eps=config.layer_norm_eps,
                                           dtype=dtype,
                                           device=device)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
        vision_feature_layer: int = -1,
    ) -> BaseModelOutputWithPooling:
        """forward."""
        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            vision_feature_layer=vision_feature_layer)

        last_hidden_state = encoder_outputs
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=None,
            attentions=None,
        )


class CLIPVisionModel(nn.Module):
    """clip vision model."""

    def __init__(self,
                 config,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config,
                                                  dtype=dtype,
                                                  device=device)

    def forward(self,
                pixel_values: torch.FloatTensor,
                interpolate_pos_encoding: bool = False,
                vision_feature_layer: int = -1,
                **kwargs):
        """forward."""
        return self.vision_model(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            vision_feature_layer=vision_feature_layer)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]

        # vis model
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)


def build_vision_model(vision_config,
                       dtype: torch.dtype = None,
                       device: torch.device = None):
    """build vision model."""
    model_type = vision_config.model_type

    if model_type == 'clip_vision_model':
        return CLIPVisionModel(vision_config, dtype, device)
    else:
        raise NotImplementedError(f'<{model_type}> is not implemented.')
