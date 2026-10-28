# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy
import inspect

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BiMambaWrapper(nn.Module):
    def __init__(
        self,
        mixer_cls,
        d_model,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
    ):
        super().__init__()
        if bidirectional_strategy not in ["add", "ew_multiply"]:
            raise ValueError(
                f"Invalid bidirectional_strategy: {bidirectional_strategy}; "
                "expected 'add' or 'ew_multiply'."
            )
        self.bidirectional_strategy = bidirectional_strategy
        self.forward_mixer = mixer_cls(d_model)
        self.reverse_mixer = mixer_cls(d_model)
        if bidirectional_weight_tie:
            self._tie_projection_weights()

    def _tie_projection_weights(self):
        for name in ["in_proj", "out_proj"]:
            forward_proj = getattr(self.forward_mixer, name, None)
            reverse_proj = getattr(self.reverse_mixer, name, None)
            if forward_proj is None or reverse_proj is None:
                continue
            reverse_proj.weight = forward_proj.weight
            if getattr(forward_proj, "bias", None) is not None and getattr(reverse_proj, "bias", None) is not None:
                reverse_proj.bias = forward_proj.bias

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        if inference_params is not None:
            raise ValueError("Bidirectional Mamba does not support inference cache generation.")
        forward_hidden = self.forward_mixer(hidden_states, inference_params=None, **mixer_kwargs)
        reverse_hidden = self.reverse_mixer(
            torch.flip(hidden_states, dims=(1,)), inference_params=None, **mixer_kwargs
        )
        reverse_hidden = torch.flip(reverse_hidden, dims=(1,))
        if self.bidirectional_strategy == "add":
            return forward_hidden + reverse_hidden
        return forward_hidden * reverse_hidden

    def allocate_inference_cache(self, *args, **kwargs):
        raise ValueError("Bidirectional Mamba does not support inference cache generation.")


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    is_attn_layer = layer_idx in attn_layer_idx
    if not is_attn_layer:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        bidirectional = bool(ssm_cfg.pop("bidirectional", False))
        bidirectional_strategy = ssm_cfg.pop("bidirectional_strategy", "add")
        bidirectional_weight_tie = bool(ssm_cfg.pop("bidirectional_weight_tie", True))
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        base_mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
        if bidirectional:
            mixer_cls = partial(
                BiMambaWrapper,
                base_mixer_cls,
                bidirectional_strategy=bidirectional_strategy,
                bidirectional_weight_tie=bidirectional_weight_tie,
            )
        else:
            mixer_cls = base_mixer_cls
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    # Only attention (Transformer) layers use intermediate MLP to control memory usage.
    layer_d_intermediate = d_intermediate if is_attn_layer else 0
    if layer_d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=layer_d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# class scDNAmMixerModel(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         n_layer: int,
#         d_intermediate: int,
#         vocab_size: int,
#         ssm_cfg=None,
#         attn_layer_idx=None,
#         attn_cfg=None,
#         norm_epsilon: float = 1e-5,
#         rms_norm: bool = False,
#         initializer_cfg=None,
#         fused_add_norm=False,
#         residual_in_fp32=False,
#         device=None,
#         dtype=None,
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32

#         self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

#         # We change the order of residual and layer norm:
#         # Instead of LN -> Attn / MLP -> Add, we do:
#         # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
#         # the main branch (output of MLP / Mixer). The model definition is unchanged.
#         # This is for performance reason: we can fuse add + layer_norm.
#         self.fused_add_norm = fused_add_norm
#         if self.fused_add_norm:
#             if layer_norm_fn is None or rms_norm_fn is None:
#                 raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

#         self.layers = nn.ModuleList(
#             [
#                 create_block(
#                     d_model,
#                     d_intermediate=d_intermediate,
#                     ssm_cfg=ssm_cfg,
#                     attn_layer_idx=attn_layer_idx,
#                     attn_cfg=attn_cfg,
#                     norm_epsilon=norm_epsilon,
#                     rms_norm=rms_norm,
#                     residual_in_fp32=residual_in_fp32,
#                     fused_add_norm=fused_add_norm,
#                     layer_idx=i,
#                     **factory_kwargs,
#                 )
#                 for i in range(n_layer)
#             ]
#         )

#         self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
#             d_model, eps=norm_epsilon, **factory_kwargs
#         )

#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=n_layer,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#                 n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
#             )
#         )

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return {
#             i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
#             for i, layer in enumerate(self.layers)
#         }

#     def forward(self, unmethy_input_ids, methy_input_ids, methy_ratios, inference_params=None, need_layer_hidden_states=False, **mixer_kwargs):
#         unmethy_hidden_states = self.embedding(unmethy_input_ids)
#         methy_states = self.embedding(methy_input_ids)
#         methy_ratios_expanded = methy_ratios.unsqueeze(-1).to(unmethy_hidden_states.dtype)  # Make sure the dtype is consistent
#         hidden_states = unmethy_hidden_states * (1.0 - methy_ratios_expanded) + methy_states * methy_ratios_expanded
#         hidden_states = hidden_states.to(unmethy_hidden_states.dtype)  # Explicitly cast hidden_states to same dtype

#         if need_layer_hidden_states:
#             layer_hidden_states = []  # List to store hidden states of each layer
#             # layer_hidden_states = []
        
#         residual = None
#         for layer in self.layers:
#             # Filter kwargs to only include those accepted by layer.forward
#             valid_args = inspect.signature(layer.forward).parameters.keys()
#             filtered_kwargs = {key: mixer_kwargs[key] for key in mixer_kwargs if key in valid_args}

#             if need_layer_hidden_states:
#                 layer_hidden_states.append(hidden_states)  # List to store hidden states of each layer
            
#             # Call layer.forward with filtered arguments
#             hidden_states, residual = layer(
#                 hidden_states, residual, inference_params=inference_params, **filtered_kwargs
#             )
#             # if need_layer_hidden_states:
#             #     layer_hidden_states.append(hidden_states)  # Store the hidden state after this layer
        
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
#         else:
#             # Set prenorm=False here since we don't need the residual
#             hidden_states = layer_norm_fn(
#                 hidden_states,
#                 self.norm_f.weight,
#                 self.norm_f.bias,
#                 eps=self.norm_f.eps,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 is_rms_norm=isinstance(self.norm_f, RMSNorm)
#             )
        
#         # Add the final hidden state after normalization
#         if need_layer_hidden_states:
#             layer_hidden_states.append(hidden_states)
#             return hidden_states, layer_hidden_states  # Return all layer hidden states, including the final one
#         else:
#             return hidden_states


class scDNAmMixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.bidirectional = bool((ssm_cfg or {}).get("bidirectional", False))
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if (d_intermediate == 0 or not attn_layer_idx) else 2,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if self.bidirectional:
            raise ValueError("Bidirectional Mamba does not support inference cache generation.")
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, unmethy_input_ids, methy_input_ids, methy_ratios, inference_params=None, need_layer_hidden_states=False, **mixer_kwargs):
        unmethy_hidden_states = self.embedding(unmethy_input_ids)
        methy_states = self.embedding(methy_input_ids)
        methy_ratios_expanded = methy_ratios.unsqueeze(-1).to(unmethy_hidden_states.dtype)  # Make sure the dtype is consistent
        hidden_states = unmethy_hidden_states * (1.0 - methy_ratios_expanded) + methy_states * methy_ratios_expanded
        hidden_states = hidden_states.to(unmethy_hidden_states.dtype)  # Explicitly cast hidden_states to same dtype

        if need_layer_hidden_states:
            layer_hidden_states = [hidden_states]  # List to store hidden states of each layer
            # layer_hidden_states = []

        residual = None
        for layer in self.layers:
            # Filter kwargs to only include those accepted by layer.forward
            valid_args = inspect.signature(layer.forward).parameters.keys()
            filtered_kwargs = {key: mixer_kwargs[key] for key in mixer_kwargs if key in valid_args}

            # Every backbone layer receives the complete sequence produced by the
            # dataset/collator. Transformer layers are not sampled or truncated here.
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **filtered_kwargs
            )

            if need_layer_hidden_states:
                layer_hidden_states.append(hidden_states)  # Store the hidden state after this layer
        
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        
        # Add the final hidden state after normalization
        if need_layer_hidden_states:
            layer_hidden_states.append(hidden_states)
            return hidden_states, layer_hidden_states  # Return all layer hidden states, including the final one
        else:
            return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = scDNAmMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            norm_epsilon=getattr(config, "norm_epsilon", 1e-5),
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.tie_embeddings = getattr(self.config, "tie_embeddings", False)
        if self.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
