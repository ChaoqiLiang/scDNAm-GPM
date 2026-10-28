from glm_src.model.scdnam_mixer_seq_simple import MambaLMHeadModel
# from glm_src.model.scdnam_mixer_seq_simple_moe import MoEMambaLMHeadModel
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional
import sys, torch
from torch import nn
import numpy as np
from glm_src.mambaconfig import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import types
from flash_attn.modules.mha import FlashCrossAttention
from flash_attn.flash_attn_interface import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
import torch.nn.functional as F
import torch.nn.init as init
import math

# === 初始化函数 ===
def apply_custom_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# === 正弦时间嵌入 ===
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half_dim, device=timesteps.device, dtype=timesteps.dtype
        ) / half_dim
        sinusoid = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)

# === 去噪网络 ===
class DiffusionDenoiseNetwork(nn.Module):
    def __init__(self, cond_dim, timestep_dim=64, hidden_dim=256):
        super().__init__()
        self.timestep_embedding = SinusoidalTimestepEmbedding(timestep_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim + timestep_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(apply_custom_init)

    def forward(self, x_t, cond, t):
        t_emb = self.timestep_embedding(t)
        x = torch.cat([x_t, cond, t_emb], dim=-1)
        x = x.to(dtype=next(self.parameters()).dtype)
        return self.mlp(x)

# === Diffusion Head (no EMA) ===
class DiffusionHead(nn.Module):
    def __init__(self, cond_dim, T=1000, timestep_dim=64, hidden_dim=256):
        super().__init__()
        self.T = T
        self.denoise = DiffusionDenoiseNetwork(cond_dim, timestep_dim, hidden_dim)

        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        self.apply(apply_custom_init)

    def forward(self, cond, target):
        x0 = target.unsqueeze(1)
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(dtype=x0.dtype)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t + 1e-8) * eps

        pred_eps = self.denoise(x_t, cond, t)
        loss = F.mse_loss(pred_eps, eps)
        return loss, pred_eps, eps

    @torch.no_grad()
    def ddim_sample(self, cond, n_steps=50, eta=0.0):
        batch_size = cond.shape[0]
        device, dtype = cond.device, cond.dtype

        x = torch.randn(batch_size, 1, device=device, dtype=dtype)
        step_indices = torch.linspace(self.T - 1, 0, n_steps, device=device)

        for i in range(n_steps):
            t = torch.full((batch_size,), int(step_indices[i].item()), device=device, dtype=torch.long)

            alpha_bar = self.alpha_bars[t].view(-1, 1).to(dtype=dtype)
            eps_theta = self.denoise(x, cond, t)

            x0_pred = (x - torch.sqrt(1 - alpha_bar + 1e-8) * eps_theta) / torch.sqrt(alpha_bar + 1e-8)

            if i < n_steps - 1:
                t_prev = torch.full_like(t, int(step_indices[i + 1].item()))
                alpha_bar_prev = self.alpha_bars[t_prev].view(-1, 1).to(dtype=dtype)

                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar + 1e-8) * (1 - alpha_bar / alpha_bar_prev + 1e-8))
                noise = sigma * torch.randn_like(x) if eta > 0 else 0

                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma**2 + 1e-8) * eps_theta + noise
            else:
                x = x0_pred

            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"NaN/Inf detected at step {i}")
                break

        return x.squeeze(1)



# 备份原 forward（可选）
original_forward = FlashCrossAttention.forward

# 新的 forward 方法
def custom_forward(
    self,
    q,
    kv,
    causal=None,
    cu_seqlens=None,
    max_seqlen=None,
    cu_seqlens_k=None,
    max_seqlen_k=None,
):
    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.is_cuda and kv.is_cuda
    causal = self.causal if causal is None else causal
    unpadded = cu_seqlens is not None
    if self.alibi_slopes is not None:
        self.alibi_slopes = self.alibi_slopes.to(torch.float32)
    if unpadded:
        assert cu_seqlens.dtype == torch.int32
        assert max_seqlen is not None
        assert isinstance(max_seqlen, int)
        assert cu_seqlens_k is not None
        assert cu_seqlens_k.dtype == torch.int32
        assert max_seqlen_k is not None
        assert isinstance(max_seqlen_k, int)
        return flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens,
            cu_seqlens_k,
            max_seqlen,
            max_seqlen_k,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.window_size,
            deterministic=self.deterministic,
            return_attn_probs=False  # 添加这里
        )
    else:
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
        return flash_attn_kvpacked_func(
            q,
            kv,
            self.drop.p if self.training else 0.0,
            causal=causal,
            softmax_scale=self.softmax_scale,
            alibi_slopes=self.alibi_slopes,
            window_size=self.window_size,
            deterministic=self.deterministic,
            return_attn_probs=True  # 添加这里
        )

# 替换 forward 方法
FlashCrossAttention.forward = custom_forward


sys.path.append("..")

import torch
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)

def random_mask(sequence, mask_token_id, mask_prob=0.1):
    """
    随机遮盖序列，基于 PyTorch 操作，支持 CUDA 张量。
    Args:
        sequence: 原始序列 (Tensor)
        mask_token_id: 遮盖标记ID (int)
        mask_prob: 遮盖概率 (float)
    Returns:
        遮盖后的序列 (Tensor)
    """
    # 生成随机掩码
    mask = torch.rand(sequence.shape, device=sequence.device) < mask_prob
    # 根据掩码替换值
    return torch.where(mask, torch.tensor(mask_token_id, device=sequence.device), sequence)

def random_replace(sequence, tokenizer, replace_prob=0.1, replace_condition=None):
    """
    随机将序列中的 token 替换为符合条件的随机 token。
    
    Args:
        sequence: 原始序列 (Tensor)，形状为 (batch_size, sequence_length)。
        tokenizer: 用于替换的 tokenizer 对象。
        replace_prob: 替换的概率 (float)。
        replace_condition: 自定义替换条件函数 (callable)，用于筛选 vocab 中的替换 token。
    
    Returns:
        替换后的序列 (Tensor)。
    """
    # 获取符合条件的 token
    vocab = torch.arange(tokenizer.vocab_size, device=sequence.device)
    if replace_condition:
        vocab = vocab[replace_condition(vocab)]
    
    # 生成伯努利分布掩码
    replace_mask = torch.bernoulli(torch.full(sequence.shape, replace_prob, device=sequence.device)).bool()

    # 从符合条件的 token 中随机采样
    random_tokens = vocab[torch.randint(len(vocab), sequence.shape, device=sequence.device)]

    # 根据掩码替换 token
    return torch.where(replace_mask, random_tokens, sequence)

def random_replace_with_shared_mask(sequence1, sequence2, tokenizer, replace_prob=0.1, condition1=None):
    """
    随机将 sequence1 和 sequence2 中的 token 替换为符合条件的随机 token，并确保替换的位置一致。
    
    Args:
        sequence1: 第一个序列 (Tensor)。
        sequence2: 第二个序列 (Tensor)。
        tokenizer: 用于替换的 tokenizer 对象。
        replace_prob: 替换的概率 (float)。
        condition1: 第一个序列的替换条件函数。
        condition2: 第二个序列的替换条件函数。
    
    Returns:
        (sequence1_replaced, sequence2_replaced): 替换后的两个序列。
    """
    # 生成共享的替换掩码
    replace_mask = torch.bernoulli(torch.full(sequence1.shape, replace_prob, device=sequence1.device)).bool()

    # 获取符合条件的 token
    vocab = torch.arange(tokenizer.vocab_size, device=sequence1.device)
    if condition1:
        vocab1 = vocab[condition1(vocab)]
    else:
        vocab1 = vocab
    # if condition2:
    #     vocab2 = vocab[condition2(vocab)]
    # else:
    #     vocab2 = vocab

    # 从 vocab1 和 vocab2 中随机采样 token
    random_tokens1 = vocab1[torch.randint(len(vocab1), sequence1.shape, device=sequence1.device)]
    random_tokens2 = random_tokens1 + 1 #vocab2[torch.randint(len(vocab2), sequence2.shape, device=sequence2.device)]

    # 根据共享的掩码替换 token
    sequence1_replaced = torch.where(replace_mask, random_tokens1, sequence1)
    sequence2_replaced = torch.where(replace_mask, random_tokens2, sequence2)

    return sequence1_replaced, sequence2_replaced

def random_noise(methy_ratios, scale=0.05):
    """
    在甲基化比例中添加随机噪声，基于 PyTorch 操作。
    Args:
        methy_ratios: 原始甲基化比例 (Tensor)
        scale: 噪声的幅度 (float)
    Returns:
        添加噪声后的甲基化比例 (Tensor)
    """
    # 生成正态分布的噪声
    noise = torch.normal(mean=0.0, std=scale, size=methy_ratios.shape, device=methy_ratios.device)
    # 添加噪声并裁剪到范围 [0, 1]
    return torch.clamp(methy_ratios + noise, min=0.0, max=1.0)

class scDNAmGLMLMHeadModelwithLoss(MambaLMHeadModel):
    """
    A Language Model Head with additional support for loss calculations related to methylation.

    Attributes:
        pad_token_id: Token ID for the padding token.
        start_token_id: Token ID for the start token (e.g., [BOS]).
        end_token_id: Token ID for the end token (e.g., [SEP]).
    """
    def __init__(self, config, tokenizer, initializer_cfg=None, device=None, dtype=None, use_dataug=False):
        super(scDNAmGLMLMHeadModelwithLoss, self).__init__(config, initializer_cfg, device, dtype)

        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.end_token_id = tokenizer._convert_token_to_id("[SEP]")
        self.mask_token_id = tokenizer._convert_token_to_id("[MASK]")
        self.tokenizer = tokenizer
        self.use_dataug = use_dataug

        # 定义替换条件
        self.condition_unmethy = lambda x: (x > 6) & (x % 2 == 1)  # ID > 6 且为奇数
        # self.condition_methy = lambda x: (x > 6) & (x % 2 == 0)    # ID > 6 且为偶数
        
        print("------------------------------------------\nShape of lm_head.weight: ", self.lm_head.weight.shape, "\n------------------------------------------")

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, **kwargs):
        """Load a pretrained model and configuration."""
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)
        if torch.cuda.device_count() == 1:
            device = "cuda:0"
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error: {e}")
            del state_dict['classify.weight']
            del state_dict['classify.bias']
            model.load_state_dict(state_dict, strict=False)
        return model

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary and handle unexpected or missing keys."""
        load_result = super().load_state_dict(state_dict, strict=strict)

        if strict:
            missing_keys = [key for key in self.state_dict().keys() if key not in state_dict]
            unexpected_keys = [key for key in state_dict.keys() if key not in self.state_dict().keys()]
            load_result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

        self.tie_weights()
        return load_result

    def forward(self, unmethy_input_ids, methy_input_ids, methy_ratios, chrs, positions, labels=None, inference_params=None, num_last_tokens=0, return_aux_loss=True, **mixer_kwargs):
        """Perform a forward pass of the model."""
        
        # Apply random replacement augmentation, ensuring consistent replacement positions for both sequences
        if self.training and self.use_dataug:
            unmethy_input_ids_aug, methy_input_ids_aug = random_replace_with_shared_mask(
                unmethy_input_ids, methy_input_ids, self.tokenizer, replace_prob=0.15, condition1=self.condition_unmethy
            )
            methy_ratios_aug = random_noise(methy_ratios, scale=0.05)
        else:
            unmethy_input_ids_aug = unmethy_input_ids
            methy_input_ids_aug = methy_input_ids
            methy_ratios_aug = methy_ratios
        
        # Forward pass through the backbone
        hidden_states = self.backbone(
            unmethy_input_ids_aug, methy_input_ids_aug, methy_ratios_aug, inference_params=inference_params, **mixer_kwargs
        )

        # if return_aux_loss:
        #     hidden_states, aux_loss = out
        # else:
        #     hidden_states = out
        #     aux_loss = None

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        # Compute logits for language modeling
        lm_logits = self.lm_head(hidden_states)

        # Shift logits and labels for causal language modeling loss
        shift_logits = lm_logits[..., :-1, :].contiguous()
        unmethy_shift_labels = unmethy_input_ids[..., 1:].contiguous()
        methy_shift_labels = methy_input_ids[..., 1:].contiguous()
        methy_ratios_shift = methy_ratios[..., 1:].contiguous()

        # Masks for methylated/unmethylated C and other tokens
        mask_other = (unmethy_shift_labels != self.pad_token_id)

        # CrossEntropy Loss for classification
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        all_loss = (
            loss_fct(shift_logits.view(-1, shift_logits.size(-1)), unmethy_shift_labels.view(-1)) * (1.0 - methy_ratios_shift).view(-1) +
            loss_fct(shift_logits.view(-1, shift_logits.size(-1)), methy_shift_labels.view(-1)) * methy_ratios_shift.view(-1)
        )

        # Calculate individual losses
        loss_meth = all_loss[mask_other.view(-1)].mean() if mask_other.any() else torch.tensor(0.0, device=unmethy_shift_labels.device)

        # Average classification loss
        classification_loss = loss_meth #+ aux_loss
        # print(loss_meth)

        # if not self.training:
        #     return {
        #     "loss": classification_loss,
        #     "labels": classification_loss,
        #     "logits": lm_logits
        #     }

        return {
            "loss": classification_loss,
            "logits": lm_logits,
            "loss_meth": loss_meth,
            # "aux_loss": aux_loss
        }


def masked_mean(hidden_states, attention_mask):
    # 输入说明:
    # hidden_states: [batch_size, sequence_length, hidden_dim]
    # attention_mask: [batch_size, sequence_length], 1表示有效token，0表示padding

    # 扩展mask的维度以匹配hidden_states
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()

    # 应用mask：将padding位置的hidden_states置零
    masked_hidden = hidden_states * mask_expanded

    # 计算每个序列的有效长度（沿序列长度维度求和）
    valid_lengths = mask_expanded.sum(dim=1)  # [batch_size, hidden_dim]

    # 对mask后的hidden_states求和，并除以有效长度
    summed_hidden = masked_hidden.sum(dim=1)  # [batch_size, hidden_dim]
    mean_hidden = summed_hidden / valid_lengths  # 自动广播除法

    # 处理可能存在的全零mask（防止除以零）
    mean_hidden = torch.where(valid_lengths > 0, mean_hidden, torch.zeros_like(mean_hidden))

    return mean_hidden  # 输出形状 [batch_size, hidden_dim]


class scDNAmGLMForSequenceClassification(scDNAmGLMLMHeadModelwithLoss):
    """
    A class for sequence classification based on methylation data.

    Attributes:
        device: The device where the model is deployed (CPU/GPU).
        num_labels: The number of classification labels.
        classify: Linear layer for classification.
        attention: Multihead attention mechanism.
        dropout: Dropout layer to prevent overfitting.
        norm: Layer normalization to stabilize training.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        need_layer_hidden_states: Optional[bool]=False,
        cross_attn_every_hidden_states: Optional[bool]=False,
        lambd=1.0,
        just_mamba: Optional[bool]=False,
        use_tumor_ratios: Optional[bool]=False
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, initializer_cfg, device, dtype)

        # Sequence-classification models never use the language-modeling head.
        # Remove the registered module itself (rather than only hiding its key
        # from state_dict) so lm_head.weight is not counted by parameters() or
        # named_parameters(). Pretraining checkpoints may still contain the key;
        # from_pretrained will report and skip it as an unexpected checkpoint key.
        if hasattr(self, "lm_head"):
            del self.lm_head

        self.device = device
        self.num_labels = num_labels
        self.batch_correction = batch_correction
        self._keys_to_ignore_on_save = ['lm_head.weight']
        self.lambd = lambd
        self.need_layer_hidden_states = need_layer_hidden_states
        self.cross_attn_every_hidden_states = cross_attn_every_hidden_states
        self.just_mamba = just_mamba
        self.use_tumor_ratios = use_tumor_ratios

        # Configure attention dimensions
        self.projection_dim = attention_mechanism.get("projection_dim", None)
        self.attention_embed_dim = (
            self.projection_dim
            if isinstance(self.projection_dim, int) and self.projection_dim > 0 and not self.just_mamba
            else config.d_model
        )
        self.attention_num_heads = attention_mechanism["attention_num_heads"]
        self.dropout_rate = attention_mechanism.get("dropout_rate", 0.0)

        if not self.just_mamba:
            # Optional projection layers
            if self.projection_dim:
                if cross_attn_every_hidden_states:
                    # Create a list of projection layers for each backbone layer
                    self.query_proj_layers = nn.ModuleList(
                        [
                            nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                            for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                        ]
                    )

                    self.key_proj_layers = nn.ModuleList(
                        [
                            nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                            for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                        ]
                    )

                    self.value_proj_layers = nn.ModuleList(
                        [
                            nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                            for _ in range(len(self.backbone.layers) + 1)  # One projection layer per backbone layer
                        ]
                    )
                else:
                    self.query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                    self.key_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
                    self.value_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

                if batch_correction:
                    self.batch_query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

            if cross_attn_every_hidden_states:
                # Create a list of attention layers with the same length as backbone layers
                self.attention_layers = nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=self.attention_embed_dim,
                            num_heads=self.attention_num_heads,
                            dropout=self.dropout_rate,
                            batch_first=True,
                        )
                        for _ in range(len(self.backbone.layers) + 1)  # Number of attention layers = number of backbone layers
                    ]
                )
            else:
                # Multihead attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.attention_embed_dim,
                    num_heads=self.attention_num_heads,
                    dropout=self.dropout_rate,
                    batch_first=True,
                )
                # self.flash_attn = FlashCrossAttention(
                #     causal=False,
                #     softmax_scale=None,
                #     attention_dropout=0,
                #     alibi_slopes=None,
                #     window_size=(-1, -1),
                #     deterministic=False
                # )

        # Regularization and classification layers
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm = nn.LayerNorm(self.attention_embed_dim)
        if not self.use_tumor_ratios:
            self.classify = nn.Linear(self.attention_embed_dim, num_labels, dtype=dtype).to(device)
            nn.init.xavier_uniform_(self.classify.weight)
            nn.init.zeros_(self.classify.bias)
        else:
            self.regression = nn.Linear(self.attention_embed_dim, 1, dtype=dtype).to(device)
            nn.init.xavier_uniform_(self.regression.weight)
            nn.init.zeros_(self.regression.bias)

        # Batch correction components
        if batch_correction:
            self.batch_norm = nn.LayerNorm(self.attention_embed_dim)
            self.batch_emd = nn.Embedding(1, config.d_model, dtype=dtype).to(device)
            self.batch_projection = nn.Linear(self.attention_embed_dim, num_batches, dtype=dtype).to(device)
            self.batch_attention = nn.MultiheadAttention(
                embed_dim=self.attention_embed_dim,
                num_heads=self.attention_num_heads,
                batch_first=True,
            )
            self.batch_classify = nn.Linear(self.attention_embed_dim, num_batches, dtype=dtype).to(device)
            nn.init.xavier_uniform_(self.batch_projection.weight)
            nn.init.zeros_(self.batch_projection.bias)

    def tie_weights(self):
        """Tie LM weights only while the temporary LM head still exists."""
        if hasattr(self, "lm_head"):
            super().tie_weights()

    @classmethod
    def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, strict: bool = False, **kwargs):
        """
        Load a pretrained model with optional strictness.
        
        Args:
            tokenizer: The tokenizer to use.
            pretrained_model_name: Path or HuggingFace model name.
            device: Target device (cuda/cpu).
            dtype: Data type.
            strict: Whether to enforce exact parameter matching. Default is False.
            **kwargs: Other model init args.
        """
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)

        if torch.cuda.device_count() == 1:
            device = "cuda:0"
        if not torch.cuda.is_available():
            device = torch.device("cpu")

        # Load raw state dict
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)

        if not strict:
            # Filter out mismatched or unknown keys
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"[Skip] Shape mismatch for key: {k}, checkpoint: {v.shape}, model: {model_state_dict[k].shape}")
                else:
                    print(f"[Skip] Unexpected key not in model: {k}")
            state_dict = filtered_state_dict

        # Load weights
        load_result = model.load_state_dict(state_dict, strict=strict)

        # Print warnings if not strict
        if not strict:
            if load_result.missing_keys:
                print("[Missing keys]:")
                for k in load_result.missing_keys:
                    print(f"  - {k}")
            if load_result.unexpected_keys:
                print("[Unexpected keys]:")
                for k in load_result.unexpected_keys:
                    print(f"  - {k}")

        return model

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Retrieve the state dictionary, excluding unnecessary keys."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state.pop(prefix + 'lm_head.weight', None)
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dictionary and handle unexpected or missing keys."""
        load_result = super().load_state_dict(state_dict, strict=strict)

        if strict:
            missing_keys = [key for key in self.state_dict().keys() if key not in state_dict]
            unexpected_keys = [key for key in state_dict.keys() if key not in self.state_dict().keys()]
            load_result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

        self.tie_weights()
        return load_result

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""

        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        if self.just_mamba:
            # print("just_mamba!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            norm_attn_outputs_for_save = cls_token_states
            norm_attn_outputs = self.norm(cls_token_states)
        
        else:
            if self.cross_attn_every_hidden_states:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = layer_hidden_states[0].size()
                # print("!!!!!!!!!!!!!!!!", len(layer_hidden_states), batch_size, seq_len, embed_dim)

                # Initialize list to store attention outputs for each layer
                norm_attn_outputs, attn_weights = [], []
                
                # Iterate through each layer's hidden state and apply attention separately
                layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
                for i, hidden_states in enumerate(layer_hidden_states):
                    # Detach gradients for all but the last layer
                    # if i < len(layer_hidden_states) - 1:
                    #     hidden_states = hidden_states.detach()
                    # if i == len(layer_hidden_states) - 2:
                    #     continue

                    # Extract the CLS token states for each sequence and apply attention
                    cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                    key_padding_mask = ~attention_mask.bool()
                    # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                    if hasattr(self, "query_proj") or hasattr(self, "query_proj_layers"):
                        q_proj = self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                        k_proj = self.key_proj_layers[i](hidden_states)
                        v_proj = self.value_proj_layers[i](hidden_states)
                    else:
                        q_proj = cls_token_states.unsqueeze(1)
                        k_proj = hidden_states
                        v_proj = hidden_states
                    
                    # print(q_proj.shape, "q_proj.shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # print(k_proj.shape, "k_proj.shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # print(v_proj.shape, "v_proj.shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # Apply attention independently for this layer
                    attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                    # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")

                    # Store the attention output for this layer
                    norm_attn_outputs.append(self.norm(attn_output))
                    attn_weights.append(attn_weight)
                
                if return_dict:
                    norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                    
                # Aggregate the attention outputs (e.g., sum or average across layers)
                norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
                attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
                # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
            
            else:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = hidden_states.size()
                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                key_padding_mask = ~attention_mask.bool()

                if hasattr(self, "query_proj"):
                    q_proj = self.query_proj(cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj(hidden_states)
                    v_proj = self.value_proj(hidden_states)
                else:
                    q_proj = cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states

                # kv = torch.stack([k_proj, v_proj], dim=2)
                # kv = kv.view(kv.size(0), kv.size(1), kv.size(2), self.attention_num_heads, int(q_proj.size(-1)/self.attention_num_heads))  # (B, Sk, 2, H, D/H)

                # q = q_proj.view(q_proj.size(0), q_proj.size(1), self.attention_num_heads, int(q_proj.size(-1)/self.attention_num_heads))

                attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                attn_output = attn_output.squeeze(1)

                # print(q.shape, "q_proj.shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print(kv.shape, "k_proj.shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                # attn_output = self.flash_attn(q=q, kv=kv)
                # print(attn_output)
                # attn_output = attn_output.squeeze(1).view(attn_output.size(0), -1)
                # print(attn_output)
                # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")
                norm_attn_outputs = self.norm(attn_output)
                if return_dict:
                    norm_attn_outputs_for_save = norm_attn_outputs
        
        # Classify using the normalized attention output
        if not self.use_tumor_ratios:
            logits = self.classify(norm_attn_outputs)
        # print(logits)
        # Compute loss
            loss = nn.CrossEntropyLoss()(logits, labels)

            # Return the result based on return_dict flag
            if not return_dict:
                return (loss, logits) if loss is not None else (logits,)
            # print("!!!!!!!!!!!!!!!!!!", logits.shape)
            return {
                "logits": logits,
                "labels": labels,
                "loss": loss,
            }

        else:
            logits = self.regression(norm_attn_outputs)
            # Apply sigmoid to ensure the output is between 0 and 1
            predicted_probs = torch.sigmoid(logits).squeeze(dim=1)

            # Preprocess labels: divide by 100 and convert to float
            processed_labels = (labels.to(predicted_probs.dtype) / 100.0)

            # print("predicted_probs:", predicted_probs, "; processed_labels:", processed_labels)

            # Compute MSE loss
            loss = nn.MSELoss()(predicted_probs, processed_labels)

            # Return the result based on return_dict flag
            if not return_dict:
                return (loss, logits) if loss is not None else (logits,)
            
            return {
                "logits": logits,
                "labels": labels,
                "loss": loss,
                "predicted_probs": predicted_probs,
            }


# class scDNAmGLMForSequenceClassification(scDNAmGLMLMHeadModelwithLoss):
#     """
#     A class for sequence classification based on methylation data.

#     Attributes:
#         device: The device where the model is deployed (CPU/GPU).
#         num_labels: The number of classification labels.
#         classify: Linear layer for classification.
#         attention: Multihead attention mechanism.
#         dropout: Dropout layer to prevent overfitting.
#         norm: Layer normalization to stabilize training.
#     """

#     def __init__(
#         self,
#         config,
#         tokenizer,
#         num_labels: int,
#         num_batches: int = None,
#         attention_mechanism: dict = None,
#         initializer_cfg=None,
#         device=None,
#         dtype=None,
#         batch_correction=False,
#         need_layer_hidden_states: Optional[bool]=False,
#         cross_attn_every_hidden_states: Optional[bool]=False,
#         lambd=1.0,
#         just_mamba: Optional[bool]=False,
#         use_tumor_ratios: Optional[bool]=False,
#         **kwargs,
#     ) -> None:
#         """Initialize the sequence classification model."""
#         super().__init__(config, tokenizer, initializer_cfg, device, dtype)

#         self.device = device
#         self.num_labels = num_labels
#         self.batch_correction = batch_correction
#         self._keys_to_ignore_on_save = ['lm_head.weight']
#         self.lambd = lambd
#         self.need_layer_hidden_states = need_layer_hidden_states
#         self.cross_attn_every_hidden_states = cross_attn_every_hidden_states
#         self.just_mamba = just_mamba
#         self.use_tumor_ratios = use_tumor_ratios

#         # Configure attention dimensions
#         self.projection_dim = attention_mechanism.get("projection_dim", None)
#         self.attention_embed_dim = (
#             self.projection_dim
#             if isinstance(self.projection_dim, int) and self.projection_dim > 0
#             else config.d_model
#         )
#         self.attention_num_heads = attention_mechanism["attention_num_heads"]
#         self.dropout_rate = attention_mechanism.get("dropout_rate", 0.0)

#         # Optional projection layers
#         if self.projection_dim:
#             self.query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
#             self.key_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)
#             self.value_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

#             if batch_correction:
#                 self.batch_query_proj = nn.Linear(config.d_model, self.projection_dim, dtype=dtype).to(device)

#         # Multihead attention
#         self.attention = nn.MultiheadAttention(
#             embed_dim=self.attention_embed_dim,
#             num_heads=self.attention_num_heads,
#             batch_first=True,
#         )

#         # Regularization and classification layers
#         self.dropout = nn.Dropout(self.dropout_rate)
#         self.norm = nn.LayerNorm(self.attention_embed_dim)
#         self.classify = nn.Linear(self.attention_embed_dim, num_labels, dtype=dtype).to(device)
#         nn.init.xavier_uniform_(self.classify.weight)
#         nn.init.zeros_(self.classify.bias)

#         # Batch correction components
#         if batch_correction:
#             self.batch_norm = nn.LayerNorm(self.attention_embed_dim)
#             self.batch_emd = nn.Embedding(1, config.d_model, dtype=dtype).to(device)
#             self.batch_projection = nn.Linear(self.attention_embed_dim, num_batches, dtype=dtype).to(device)
#             self.batch_attention = nn.MultiheadAttention(
#                 embed_dim=self.attention_embed_dim,
#                 num_heads=self.attention_num_heads,
#                 batch_first=True,
#             )
#             self.batch_classify = nn.Linear(self.attention_embed_dim, num_batches, dtype=dtype).to(device)
#             nn.init.xavier_uniform_(self.batch_projection.weight)
#             nn.init.zeros_(self.batch_projection.bias)

#     @classmethod
#     def from_pretrained(cls, tokenizer, pretrained_model_name, device=None, dtype=None, **kwargs):
#         """Load a pretrained model and configuration."""
#         config_data = load_config_hf(pretrained_model_name)
#         config = MambaConfig(**config_data)
#         model = cls(config, tokenizer, device=device, dtype=dtype, **kwargs)
#         if torch.cuda.device_count() == 1:
#             device = "cuda:0"
#         if not torch.cuda.is_available():
#             device = torch.device("cpu")
#         state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
#         try:
#             model.load_state_dict(state_dict, strict=False)
#         except Exception as e:
#             print(f"Error: {e}")
#             del state_dict['classify.weight']
#             del state_dict['classify.bias']
#             model.load_state_dict(state_dict, strict=False)
#         return model

#     def state_dict(self, destination=None, prefix='', keep_vars=False):
#         """Retrieve the state dictionary, excluding unnecessary keys."""
#         state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
#         state.pop(prefix + 'lm_head.weight', None)
#         return state

#     def load_state_dict(self, state_dict, strict=True):
#         """Load the state dictionary and handle unexpected or missing keys."""
#         load_result = super().load_state_dict(state_dict, strict=strict)

#         if strict:
#             missing_keys = [key for key in self.state_dict().keys() if key not in state_dict]
#             unexpected_keys = [key for key in state_dict.keys() if key not in self.state_dict().keys()]
#             load_result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

#         self.tie_weights()
#         return load_result

#     def forward(
#         self,
#         unmethy_input_ids,
#         methy_input_ids,
#         methy_ratios,
#         labels=None,
#         attention_mask=None,
#         inference_params=None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ):
#         """Perform a forward pass of the model."""
        
#         # Get layer hidden states from the backbone
#         hidden_states = self.backbone(
#             unmethy_input_ids, methy_input_ids, methy_ratios,
#             inference_params=inference_params,
#             need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
#             **kwargs,
#         )
        
#         if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
#             hidden_states, layer_hidden_states = hidden_states
        
#         # Handle attention mask
#         if attention_mask is None:
#             attention_mask = (methy_input_ids != self.pad_token_id).int()
        
#         if self.cross_attn_every_hidden_states:
#             last_non_padding_indices = attention_mask.sum(dim=1) - 1
#             batch_size, seq_len, embed_dim = layer_hidden_states[0].size()
#             # print("!!!!!!!!!!!!!!!!", len(layer_hidden_states), batch_size, seq_len, embed_dim)

#             # Initialize list to store attention outputs for each layer
#             norm_attn_outputs, attn_weights = [], []
            
#             # Iterate through each layer's hidden state and apply attention separately
#             for i, hidden_states in enumerate(layer_hidden_states):
#                 # Detach gradients for all but the last layer
#                 if i < len(layer_hidden_states) - 1:
#                     hidden_states = hidden_states.detach()
#                 # if i == len(layer_hidden_states) - 2:
#                 #     continue

#                 # Extract the CLS token states for each sequence and apply attention
#                 cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
#                 key_padding_mask = ~attention_mask.bool()
#                 # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

#                 if hasattr(self, "query_proj"):
#                     q_proj = self.query_proj(cls_token_states.unsqueeze(1))
#                     k_proj = self.key_proj(hidden_states)
#                     v_proj = self.value_proj(hidden_states)
#                 else:
#                     q_proj = cls_token_states.unsqueeze(1)
#                     k_proj = hidden_states
#                     v_proj = hidden_states
                
#                 # Apply attention independently for this layer
#                 attn_output, attn_weight = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
#                 # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")

#                 # Store the attention output for this layer
#                 norm_attn_outputs.append(self.norm(attn_output))
#                 attn_weights.append(attn_weight)
            
#             # Aggregate the attention outputs (e.g., sum or average across layers)
#             norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
#             attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
#             # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
        
#         else:
#             last_non_padding_indices = attention_mask.sum(dim=1) - 1
#             batch_size, seq_len, embed_dim = hidden_states.size()
#             cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
#             # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

#             key_padding_mask = ~attention_mask.bool()

#             if hasattr(self, "query_proj"):
#                 q_proj = self.query_proj(cls_token_states.unsqueeze(1))
#                 k_proj = self.key_proj(hidden_states)
#                 v_proj = self.value_proj(hidden_states)
#             else:
#                 q_proj = cls_token_states.unsqueeze(1)
#                 k_proj = hidden_states
#                 v_proj = hidden_states

#             attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
#             attn_output = attn_output.squeeze(1)
#             # print(attn_weights.shape, "attn_weights!!!!!!!!!!!!!!!!")
#             norm_attn_outputs = self.norm(attn_output)
        
#         # Classify using the normalized attention output
#         logits = self.classify(norm_attn_outputs)
        
#         # Compute loss
#         loss = nn.CrossEntropyLoss()(logits, labels)

#         # Return the result based on return_dict flag
#         if not return_dict:
#             return (loss, logits) if loss is not None else (logits,)
        
#         return {
#             "norm_attn_outputs": norm_attn_outputs,
#             "hidden_states": masked_mean(hidden_states, attention_mask),
#             "attn_weights": attn_weights,
#             "logits": logits,
#             "labels": labels,
#             "methy_ratios": methy_ratios
#         }


class scDNAmGLMForTumorDeconv(scDNAmGLMForSequenceClassification):
    """
    scDNAm-GLM variant that replaces the classifier-plus-sigmoid with a diffusion head
    for tumor-fraction deconvolution.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        need_layer_hidden_states: bool = False,
        cross_attn_every_hidden_states: bool = False,
        lambd: float = 1.0,
        just_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            tokenizer,
            num_labels,
            num_batches=num_batches,
            attention_mechanism=attention_mechanism,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            batch_correction=batch_correction,
            need_layer_hidden_states=need_layer_hidden_states,
            cross_attn_every_hidden_states=cross_attn_every_hidden_states,
            lambd=lambd,
            just_mamba=just_mamba,
            **kwargs,
        )

        # projection_dim is created in the parent’s __init__
        self.diffusion_head = DiffusionHead(cond_dim=self.projection_dim)

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""
        
        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()
        
        if self.cross_attn_every_hidden_states:
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = layer_hidden_states[0].size()
            # print("!!!!!!!!!!!!!!!!", len(layer_hidden_states), batch_size, seq_len, embed_dim)

            # Initialize list to store attention outputs for each layer
            norm_attn_outputs, attn_weights = [], []
            
            # Iterate through each layer's hidden state and apply attention separately
            layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
            for i, hidden_states in enumerate(layer_hidden_states):
                # Detach gradients for all but the last layer
                # if i < len(layer_hidden_states) - 1:
                #     hidden_states = hidden_states.detach()
                # if i == len(layer_hidden_states) - 2:
                #     continue

                # Extract the CLS token states for each sequence and apply attention
                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                key_padding_mask = ~attention_mask.bool()
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                if hasattr(self, "query_proj") or hasattr(self, "query_proj_layers"):
                    q_proj = self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj_layers[i](hidden_states)
                    v_proj = self.value_proj_layers[i](hidden_states)
                else:
                    q_proj = cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states
                
                # Apply attention independently for this layer
                attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")

                # Store the attention output for this layer
                norm_attn_outputs.append(self.norm(attn_output))
                attn_weights.append(attn_weight)
            
            if return_dict:
                norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                
            # Aggregate the attention outputs (e.g., sum or average across layers)
            norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
            attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
            # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
        
        else:
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

            key_padding_mask = ~attention_mask.bool()

            if hasattr(self, "query_proj"):
                q_proj = self.query_proj(cls_token_states.unsqueeze(1))
                k_proj = self.key_proj(hidden_states)
                v_proj = self.value_proj(hidden_states)
            else:
                q_proj = cls_token_states.unsqueeze(1)
                k_proj = hidden_states
                v_proj = hidden_states

            attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
            attn_output = attn_output.squeeze(1)
            # print(attn_weights.shape, "attn_weights!!!!!!!!!!!!!!!!")
            norm_attn_outputs = self.norm(attn_output)
            if return_dict:
                norm_attn_outputs_for_save = norm_attn_outputs

        # Classify using the normalized attention output
        if self.training:
            loss, pred_eps, eps = self.diffusion_head(norm_attn_outputs, labels)
            # return {"loss": loss, "pred_eps": pred_eps, "labels": labels}
        else:
            predicted_probs = self.diffusion_head.sample(norm_attn_outputs)
            # return {"predicted_expression": pred}
        
        # # Apply sigmoid to ensure the output is between 0 and 1
        # predicted_probs = torch.sigmoid(logits).squeeze(dim=1)

        # # print("predicted_probs:", predicted_probs, "; labels:", labels)

        # # Compute MSE loss
        # loss = nn.MSELoss()(predicted_probs, labels)

        # Return the result based on return_dict flag
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        
        return {
            "norm_attn_outputs": norm_attn_outputs_for_save,
            "predicted_probs": predicted_probs,
            "hidden_states": masked_mean(hidden_states, attention_mask),
            "attn_weights": attn_weights,
            "labels": labels,
            "methy_ratios": methy_ratios
        }

class scDNAmGLMForPredictscRNAseq(scDNAmGLMForSequenceClassification):

    """
    scDNAm-GLM variant that replaces the classifier-plus-sigmoid with a diffusion head
    for tumor-fraction deconvolution.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        need_layer_hidden_states: bool = False,
        cross_attn_every_hidden_states: bool = False,
        lambd: float = 1.0,
        just_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            tokenizer,
            num_labels,
            num_batches=num_batches,
            attention_mechanism=attention_mechanism,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            batch_correction=batch_correction,
            need_layer_hidden_states=need_layer_hidden_states,
            cross_attn_every_hidden_states=cross_attn_every_hidden_states,
            lambd=lambd,
            just_mamba=just_mamba,
            **kwargs,
        )

        # projection_dim is created in the parent’s __init__
        # self.diffusion_head = DiffusionHead(cond_dim=self.projection_dim)

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""
        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        if self.just_mamba:
            # print("just_mamba!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            norm_attn_outputs_for_save = cls_token_states
            norm_attn_outputs = self.norm(cls_token_states)
        
        else:
            if self.cross_attn_every_hidden_states:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = layer_hidden_states[0].size()

                # Initialize list to store attention outputs for each layer
                norm_attn_outputs, attn_weights = [], []
                
                # Iterate through each layer's hidden state and apply attention separately
                layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
                for i, hidden_states in enumerate(layer_hidden_states):

                    # Extract the CLS token states for each sequence and apply attention
                    cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                    key_padding_mask = ~attention_mask.bool()
                    # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                    if hasattr(self, "query_proj") or hasattr(self, "query_proj_layers"):
                        q_proj = self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                        k_proj = self.key_proj_layers[i](hidden_states)
                        v_proj = self.value_proj_layers[i](hidden_states)
                    else:
                        q_proj = cls_token_states.unsqueeze(1)
                        k_proj = hidden_states
                        v_proj = hidden_states
                    
                    # Apply attention independently for this layer
                    attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                    # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")

                    # Store the attention output for this layer
                    norm_attn_outputs.append(self.norm(attn_output))
                    attn_weights.append(attn_weight)
                
                if return_dict:
                    norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                    
                # Aggregate the attention outputs (e.g., sum or average across layers)
                norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
                attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
                # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
            
            else:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = hidden_states.size()
                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                key_padding_mask = ~attention_mask.bool()

                if hasattr(self, "query_proj"):
                    q_proj = self.query_proj(cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj(hidden_states)
                    v_proj = self.value_proj(hidden_states)
                else:
                    q_proj = cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states

                attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                attn_output = attn_output.squeeze(1)

                norm_attn_outputs = self.norm(attn_output)
                if return_dict:
                    norm_attn_outputs_for_save = norm_attn_outputs
        
        # Classify using the normalized attention output
        logits = self.classify(norm_attn_outputs)
        
        # Apply sigmoid to ensure the output is between 0 and 1
        predicted_probs = logits #.squeeze(dim=1) #torch.sigmoid(logits).squeeze(dim=1)

        # print("predicted_probs:", predicted_probs, "; labels:", labels)

        # Compute MSE loss
        
        # 假设 predicted_probs 和 labels 是相同 shape 的张量
        # 比如 shape: (batch_size, num_genes)
        # mask_labels = labels > 0  # 只保留 labels > 0 的位置

        # 取出需要计算的位置
        # masked_preds = predicted_probs[mask_labels]
        # masked_labels = labels[mask_labels]

        # 计算 MSELoss
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predicted_probs, labels)
        else:
            loss = None

        # # Classify using the normalized attention output
        # if self.training:
        #     loss, pred_eps, eps = self.diffusion_head(norm_attn_outputs, labels)
        #     return {"loss": loss, "pred_eps": pred_eps, "labels": labels}
        # else:
        #     predicted_probs = self.diffusion_head.ddim_sample(norm_attn_outputs, n_steps=50, eta=0.0)  # DDIM deterministic sampling
        #     # print(predicted_probs)
        #     loss = F.mse_loss(predicted_probs, labels)
        #     return {"loss": loss, "predicted_probs": predicted_probs}

        # Return the result based on return_dict flag
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        # print("!!!!!!!!!!!!!!!!!!", logits.shape)
        return {
            "predicted_probs": predicted_probs,
            "labels": labels,
            "loss": loss,
        }


class scDNAmGLMForLungTumor(scDNAmGLMForSequenceClassification):

    """
    scDNAm-GLM variant that replaces the classifier-plus-sigmoid with a diffusion head
    for tumor-fraction deconvolution.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        need_layer_hidden_states: bool = False,
        cross_attn_every_hidden_states: bool = False,
        lambd: float = 1.0,
        just_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            tokenizer,
            num_labels,
            num_batches=num_batches,
            attention_mechanism=attention_mechanism,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            batch_correction=batch_correction,
            need_layer_hidden_states=need_layer_hidden_states,
            cross_attn_every_hidden_states=cross_attn_every_hidden_states,
            lambd=lambd,
            just_mamba=just_mamba,
            **kwargs,
        )

        # projection_dim is created in the parent’s __init__
        # self.diffusion_head = DiffusionHead(cond_dim=self.projection_dim)

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""
        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        if self.just_mamba:
            # print("just_mamba!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            norm_attn_outputs_for_save = cls_token_states
            norm_attn_outputs = self.norm(cls_token_states)
        
        else:
            if self.cross_attn_every_hidden_states:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = layer_hidden_states[0].size()

                # Initialize list to store attention outputs for each layer
                norm_attn_outputs, attn_weights = [], []
                
                # Iterate through each layer's hidden state and apply attention separately
                layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
                for i, hidden_states in enumerate(layer_hidden_states):

                    # Extract the CLS token states for each sequence and apply attention
                    cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                    key_padding_mask = ~attention_mask.bool()
                    # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                    if hasattr(self, "query_proj") or hasattr(self, "query_proj_layers"):
                        q_proj = self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                        k_proj = self.key_proj_layers[i](hidden_states)
                        v_proj = self.value_proj_layers[i](hidden_states)
                    else:
                        q_proj = cls_token_states.unsqueeze(1)
                        k_proj = hidden_states
                        v_proj = hidden_states
                    
                    # Apply attention independently for this layer
                    attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                    # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")

                    # Store the attention output for this layer
                    norm_attn_outputs.append(self.norm(attn_output))
                    attn_weights.append(attn_weight)
                
                if return_dict:
                    norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                    
                # Aggregate the attention outputs (e.g., sum or average across layers)
                norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
                attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
                # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
            
            else:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = hidden_states.size()
                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                key_padding_mask = ~attention_mask.bool()

                if hasattr(self, "query_proj"):
                    q_proj = self.query_proj(cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj(hidden_states)
                    v_proj = self.value_proj(hidden_states)
                else:
                    q_proj = cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states

                attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                attn_output = attn_output.squeeze(1)

                norm_attn_outputs = self.norm(attn_output)
                if return_dict:
                    norm_attn_outputs_for_save = norm_attn_outputs
        
        # Classify using the normalized attention output
        logits = self.classify(norm_attn_outputs)
        
        # Apply sigmoid to ensure the output is between 0 and 1
        predicted_probs = torch.sigmoid(logits).squeeze(dim=1) #logits.squeeze(dim=-1) #.squeeze(dim=1) #torch.sigmoid(logits).squeeze(dim=1)

        # print("predicted_probs:", predicted_probs, "; labels:", labels)

        # Compute MSE loss
        
        # 假设 predicted_probs 和 labels 是相同 shape 的张量
        # 比如 shape: (batch_size, num_genes)
        # mask_labels = labels > 0  # 只保留 labels > 0 的位置

        # 取出需要计算的位置
        # masked_preds = predicted_probs[mask_labels]
        # masked_labels = labels[mask_labels]

        # 计算 MSELoss
        if labels is not None:
            labels = labels.float().half()
            loss_fn = nn.MSELoss()
            loss = loss_fn(predicted_probs, labels)
        else:
            loss = None

        # # Classify using the normalized attention output
        # if self.training:
        #     loss, pred_eps, eps = self.diffusion_head(norm_attn_outputs, labels)
        #     return {"loss": loss, "pred_eps": pred_eps, "labels": labels}
        # else:
        #     predicted_probs = self.diffusion_head.ddim_sample(norm_attn_outputs, n_steps=50, eta=0.0)  # DDIM deterministic sampling
        #     # print(predicted_probs)
        #     loss = F.mse_loss(predicted_probs, labels)
        #     return {"loss": loss, "predicted_probs": predicted_probs}

        # Return the result based on return_dict flag
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        # print("!!!!!!!!!!!!!!!!!!", logits.shape)
        return {
            "predicted_probs": predicted_probs,
            "labels": labels,
            "loss": loss,
        }


class scDNAmGLMForPerturbationscRNAseq(scDNAmGLMForSequenceClassification):

    """
    scDNAm-GLM variant that replaces the classifier-plus-sigmoid with a diffusion head
    for tumor-fraction deconvolution.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        need_layer_hidden_states: bool = False,
        cross_attn_every_hidden_states: bool = False,
        lambd: float = 1.0,
        just_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            tokenizer,
            num_labels,
            num_batches=num_batches,
            attention_mechanism=attention_mechanism,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            batch_correction=batch_correction,
            need_layer_hidden_states=need_layer_hidden_states,
            cross_attn_every_hidden_states=cross_attn_every_hidden_states,
            lambd=lambd,
            just_mamba=just_mamba,
            **kwargs,
        )


    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""

        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        if self.just_mamba:
            # print("just_mamba!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            norm_attn_outputs_for_save = cls_token_states
            norm_attn_outputs = self.norm(cls_token_states)
        
        else:
            if self.cross_attn_every_hidden_states:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = layer_hidden_states[0].size()

                # Initialize list to store attention outputs for each layer
                norm_attn_outputs, attn_weights = [], []
                
                # Iterate through each layer's hidden state and apply attention separately
                layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
                for i, hidden_states in enumerate(layer_hidden_states):

                    # Extract the CLS token states for each sequence and apply attention
                    cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                    key_padding_mask = ~attention_mask.bool()
                    # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                    if hasattr(self, "query_proj") or hasattr(self, "query_proj_layers"):
                        q_proj = self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                        k_proj = self.key_proj_layers[i](hidden_states)
                        v_proj = self.value_proj_layers[i](hidden_states)
                    else:
                        q_proj = cls_token_states.unsqueeze(1)
                        k_proj = hidden_states
                        v_proj = hidden_states
                    
                    # Apply attention independently for this layer
                    attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                    # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")

                    # Store the attention output for this layer
                    norm_attn_outputs.append(self.norm(attn_output))
                    attn_weights.append(attn_weight)
                
                if return_dict:
                    norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                    
                # Aggregate the attention outputs (e.g., sum or average across layers)
                norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
                attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
                # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
            
            else:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = hidden_states.size()
                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                key_padding_mask = ~attention_mask.bool()

                if hasattr(self, "query_proj"):
                    q_proj = self.query_proj(cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj(hidden_states)
                    v_proj = self.value_proj(hidden_states)
                else:
                    q_proj = cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states

                attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                attn_output = attn_output.squeeze(1)

                norm_attn_outputs = self.norm(attn_output)
                if return_dict:
                    norm_attn_outputs_for_save = norm_attn_outputs
        
        # Classify using the normalized attention output
        logits = self.classify(norm_attn_outputs)
        
        # Apply sigmoid to ensure the output is between 0 and 1
        predicted_probs = logits #.squeeze(dim=1) #torch.sigmoid(logits).squeeze(dim=1)

        # print("predicted_probs:", predicted_probs, "; labels:", labels)

        # Compute MSE loss
        
        # 假设 predicted_probs 和 labels 是相同 shape 的张量
        # 比如 shape: (batch_size, num_genes)
        # mask_labels = labels > 0  # 只保留 labels > 0 的位置

        # 取出需要计算的位置
        # masked_preds = predicted_probs[mask_labels]
        # masked_labels = labels[mask_labels]

        # 计算 MSELoss
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predicted_probs, labels)
        else:
            loss = None

        # # Classify using the normalized attention output
        # if self.training:
        #     loss, pred_eps, eps = self.diffusion_head(norm_attn_outputs, labels)
        #     return {"loss": loss, "pred_eps": pred_eps, "labels": labels}
        # else:
        #     predicted_probs = self.diffusion_head.ddim_sample(norm_attn_outputs, n_steps=50, eta=0.0)  # DDIM deterministic sampling
        #     # print(predicted_probs)
        #     loss = F.mse_loss(predicted_probs, labels)
        #     return {"loss": loss, "predicted_probs": predicted_probs}

        # Return the result based on return_dict flag
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        # print("!!!!!!!!!!!!!!!!!!", logits.shape)
        return {
            "predicted_probs": predicted_probs,
            "labels": labels,
            "loss": loss,
        }


# def sample_nonzero_indices(labels: torch.Tensor, k: int):
#     """
#     从每个样本中采样最多 k 个非零元素索引（不重复采样），不足 k 的不补足，只在最终用 -1 补齐成 max_len。
    
#     参数:
#         labels: Tensor of shape (B, G)，二值矩阵（0/1），表示哪些基因被表达
#         k: int，每个样本最大采样数量
#     返回:
#         padded_indices: LongTensor of shape (B, max_len)，采样后用 -1 补齐
#         valid_lengths: LongTensor of shape (B,)，每个样本实际采样数量
#     """
#     B, G = labels.shape
#     device = labels.device

#     mask = labels.bool()
#     batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, G)
#     flat_batch_indices = batch_indices[mask]
#     flat_gene_indices = torch.arange(G, device=device).expand(B, -1)[mask]

#     per_sample_nonzero = [flat_gene_indices[flat_batch_indices == b] for b in range(B)]

#     sampled_indices = []
#     valid_lengths = []

#     for gene_ids in per_sample_nonzero:
#         if len(gene_ids) > k:
#             sampled = gene_ids[torch.randperm(len(gene_ids), device=device)[:k]]
#         else:
#             sampled = gene_ids
#         sampled_indices.append(sampled)
#         valid_lengths.append(len(sampled))

#     max_len = max(len(s) for s in sampled_indices)

#     # 用 -1 补齐
#     padded_indices = torch.full((B, max_len), fill_value=-1, dtype=torch.long, device=device)
#     for i, s in enumerate(sampled_indices):
#         padded_indices[i, :len(s)] = s

#     return padded_indices, torch.tensor(valid_lengths, device=device)


def sample_random_indices(labels: torch.Tensor, k: int):
    """
    从每个样本中随机采样最多 k 个索引（不考虑 labels 是否为 0），不重复采样，不足 k 的直接返回原始数量，最终用 -1 补齐为 max_len。
    
    参数:
        labels: Tensor of shape (B, G)，原始标签（0/1），此处不再区分，只是为了获取形状
        k: int，每个样本最多采样数量
    返回:
        padded_indices: LongTensor of shape (B, max_len)，采样后用 -1 补齐
        valid_lengths: LongTensor of shape (B,)，每个样本实际采样数量
    """
    B, G = labels.shape
    device = labels.device

    sampled_indices = []
    valid_lengths = []

    for _ in range(B):
        if G > k:
            sampled = torch.randperm(G, device=device)[:k]
        else:
            sampled = torch.randperm(G, device=device)
        sampled_indices.append(sampled)
        valid_lengths.append(len(sampled))

    max_len = max(len(s) for s in sampled_indices)

    # 用 -1 补齐
    padded_indices = torch.full((B, max_len), fill_value=-1, dtype=torch.long, device=device)
    for i, s in enumerate(sampled_indices):
        padded_indices[i, :len(s)] = s

    return padded_indices, torch.tensor(valid_lengths, device=device)



class scDNAmGLMForPredictAllscRNAseq(scDNAmGLMForSequenceClassification):

    """
    scDNAm-GLM variant that replaces the classifier-plus-sigmoid with a diffusion head
    for tumor-fraction deconvolution.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        need_layer_hidden_states: bool = False,
        cross_attn_every_hidden_states: bool = False,
        lambd: float = 1.0,
        just_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(
            config,
            tokenizer,
            num_labels=1,
            num_batches=num_batches,
            attention_mechanism=attention_mechanism,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            batch_correction=batch_correction,
            need_layer_hidden_states=need_layer_hidden_states,
            cross_attn_every_hidden_states=cross_attn_every_hidden_states,
            lambd=lambd,
            just_mamba=just_mamba,
            **kwargs,
        )
        
        # self.embeddings = nn.Parameter(torch.randn(num_layers, num_embeddings, embedding_dim))
        # init.xavier_uniform_(self.embedding.weight)

        self.k = 16

        if cross_attn_every_hidden_states:
            # 每层一个 gene embedding（或 embedding 的投影）
            self.gene_embedding_layers = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=num_labels, embedding_dim=self.projection_dim).to(device)
                    for _ in range(len(self.backbone.layers) + 1)
                ]
            )
        else:
            # 通用的 gene embedding
            self.gene_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=self.projection_dim).to(device)

        # projection_dim is created in the parent’s __init__
        # self.diffusion_head = DiffusionHead(cond_dim=self.projection_dim)

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        # sampled_gene_indices=None,
        **kwargs,
    ):
        B, G = labels.shape
        device = labels.device

        if self.training:
            # 训练阶段：仅对非零位置进行采样
            sampled_gene_indices, _ = sample_random_indices(labels, self.k)  # shape: (B, ≤k)，补齐为 -1
        else:
            # 验证 / 测试阶段：直接使用所有位置（0 ~ G-1）
            sampled_gene_indices = torch.arange(G, device=device).expand(B, -1)  # shape: (B, G)
        # print("#####################################sampled_gene_indices:", sampled_gene_indices)

        """Perform a forward pass of the model."""

        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        if self.just_mamba:
            # print("just_mamba!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            norm_attn_outputs_for_save = cls_token_states
            norm_attn_outputs = self.norm(cls_token_states)
        
        else:
            if self.cross_attn_every_hidden_states:
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = layer_hidden_states[0].size()

                # Initialize list to store attention outputs for each layer
                norm_attn_outputs, attn_weights = [], []
                
                # Iterate through each layer's hidden state and apply attention separately
                layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
                for i, hidden_states in enumerate(layer_hidden_states):

                    # Extract the CLS token states for each sequence and apply attention
                    gene_emb = self.gene_embedding_layers[i](sampled_gene_indices)
                    cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                    key_padding_mask = ~attention_mask.bool()
                    # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                    if hasattr(self, "query_proj") or hasattr(self, "value_proj_layers"):
                        q_proj = gene_emb #+ self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                        k_proj = self.key_proj_layers[i](hidden_states)
                        v_proj = self.value_proj_layers[i](hidden_states)
                    else:
                        q_proj = gene_emb #+ cls_token_states.unsqueeze(1)
                        k_proj = hidden_states
                        v_proj = hidden_states
                    
                    # Apply attention independently for this layer
                    attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                    # print(attn_output.shape, "attn_output!!!!!!!!!!!!!!!!")

                    # Store the attention output for this layer
                    norm_attn_outputs.append(self.norm(attn_output))
                    attn_weights.append(attn_weight)
                
                if return_dict:
                    norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                    
                # Aggregate the attention outputs (e.g., sum or average across layers)
                # print("###################################norm_attn_outputs", norm_attn_outputs.shape)
                norm_attn_outputs = torch.sum(torch.stack(norm_attn_outputs, dim=1), dim=1)
                attn_weights = torch.stack(attn_weights, dim=1) #.permute(1, 0, 2)
                # print("###################################attn_weights", attn_weights.shape)
                # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
            
            else:
                gene_emb = self.gene_embedding(sampled_gene_indices)      # shape: (2, 128, embedding_dim)
                last_non_padding_indices = attention_mask.sum(dim=1) - 1
                batch_size, seq_len, embed_dim = hidden_states.size()
                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                key_padding_mask = ~attention_mask.bool()

                if hasattr(self, "value_proj"):
                    q_proj = gene_emb + self.query_proj(cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj(hidden_states)
                    v_proj = self.value_proj(hidden_states)
                else:
                    q_proj = gene_emb + cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states

                # print("#####################################", q_proj.shape)
                attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", attn_output.shape)
                attn_output = attn_output.squeeze(1)

                norm_attn_outputs = self.norm(attn_output)
                if return_dict:
                    norm_attn_outputs_for_save = norm_attn_outputs
        
        # Classify using the normalized attention output
        logits = self.classify(norm_attn_outputs).squeeze(dim=-1)
        
        # Apply sigmoid to ensure the output is between 0 and 1
        predicted_probs = torch.sigmoid(logits) #.squeeze(dim=1)

        # print("predicted_probs:", predicted_probs, "; labels:", labels)

        # Compute MSE loss
        sampled_labels = torch.gather(labels, dim=1, index=sampled_gene_indices)
        # print("!!!!!!!!!!!!!!!!!!!!!!!",labels[sampled_gene_indices].shape)
        # labels = labels.unsqueeze(dim=-1)
        loss = nn.MSELoss()(predicted_probs, sampled_labels)

        # # Classify using the normalized attention output
        # if self.training:
        #     loss, pred_eps, eps = self.diffusion_head(norm_attn_outputs, labels)
        #     return {"loss": loss, "pred_eps": pred_eps, "labels": labels}
        # else:
        #     predicted_probs = self.diffusion_head.ddim_sample(norm_attn_outputs, n_steps=50, eta=0.0)  # DDIM deterministic sampling
        #     # print(predicted_probs)
        #     loss = F.mse_loss(predicted_probs, labels)
        #     return {"loss": loss, "predicted_probs": predicted_probs}

        # Return the result based on return_dict flag
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        # print("!!!!!!!!!!!!!!!!!!", logits.shape)
        return {
            "predicted_probs": predicted_probs,
            "labels": labels,
            "loss": loss,
        }


class scDNAmGLMForSequenceClassificationWithBatchCorrection(scDNAmGLMForSequenceClassification):
    """
    A class for sequence classification based on methylation data.

    Attributes:
        device: The device where the model is deployed (CPU/GPU).
        num_labels: The number of classification labels.
        classify: Linear layer for classification.
        attention: Multihead attention mechanism.
        dropout: Dropout layer to prevent overfitting.
        norm: Layer normalization to stabilize training.
    """

    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        num_batches: int = None,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        batch_correction=False,
        lambd=1.0,
        *args,
        **kwargs
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, num_labels, attention_mechanism, initializer_cfg, device, dtype, batch_correction, lambd)

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        batch_labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            **kwargs,
        )

        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        last_non_padding_indices = attention_mask.sum(dim=1) - 1
        batch_size, seq_len, embed_dim = hidden_states.size()
        cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]

        batch_cls_token_states = self.batch_emd(torch.zeros((batch_size,), dtype=torch.int).to(cls_token_states.device))

        key_padding_mask = ~attention_mask.bool()

        if hasattr(self, "query_proj"):
            q_proj = self.query_proj(cls_token_states.unsqueeze(1))
            batch_q_proj = self.batch_query_proj(batch_cls_token_states.unsqueeze(1))
            k_proj = self.key_proj(hidden_states)
            v_proj = self.value_proj(hidden_states)
        else:
            q_proj = cls_token_states.unsqueeze(1)
            batch_q_proj = batch_cls_token_states.unsqueeze(1)
            k_proj = hidden_states
            v_proj = hidden_states

        attn_output, _ = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
        attn_output = attn_output.squeeze(1)

        batch_attn_output, _ = self.batch_attention(
            batch_q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask
        )
        batch_attn_output = batch_attn_output.squeeze(1)
        norm_batch_attn_outputs = self.batch_norm(batch_attn_output)
        batch_logits = self.batch_classify(norm_batch_attn_outputs)

        norm_attn_outputs = self.norm(attn_output) - norm_batch_attn_outputs

        logits = self.classify(norm_attn_outputs)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        batch_loss, batch_reversed_loss = None, None
        reversed_features = grad_reverse(norm_attn_outputs, lambd=1.0)
        batch_reversed_logits = self.batch_projection(reversed_features)
        batch_loss = nn.CrossEntropyLoss()(batch_logits, batch_labels)
        batch_reversed_loss = nn.CrossEntropyLoss()(batch_reversed_logits, batch_labels)

        loss += batch_loss + self.lambd * batch_reversed_loss

        logits = (logits, batch_logits, batch_reversed_logits)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutputWithPast(loss=loss, logits=logits)
