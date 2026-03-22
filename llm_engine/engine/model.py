"""
GPT-2 模型手写实现，支持 KV Cache 的增量推理。
权重直接从 HuggingFace transformers 加载，前向逻辑完全自己实现，
方便理解 Prefill 和 Decode 两个阶段的差异。
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024   # 最大序列长度
    n_embd: int = 768         # 隐藏层维度
    n_layer: int = 12         # transformer 层数
    n_head: int = 12          # 注意力头数
    n_inner: int = 3072       # FFN 内层维度（4 * n_embd）
    layer_norm_epsilon: float = 1e-5
    attn_pdrop: float = 0.0   # 推理阶段不 dropout
    resid_pdrop: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def gpt2_small(cls) -> "GPT2Config":
        return cls(n_embd=768, n_layer=12, n_head=12)

    @classmethod
    def gpt2_medium(cls) -> "GPT2Config":
        return cls(n_embd=1024, n_layer=24, n_head=16)

    @classmethod
    def gpt2_large(cls) -> "GPT2Config":
        return cls(n_embd=1280, n_layer=36, n_head=20)


class GPT2Attention(nn.Module):
    """
    Multi-Head Self-Attention，支持 KV Cache。

    Prefill 阶段: 输入完整 prompt (seq_len > 1)，计算并缓存所有 K/V
    Decode 阶段: 输入单个新 token (seq_len = 1)，只计算当前 K/V，
                 与缓存中的历史 K/V 拼接后做 attention
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        # QKV 合并投影，GPT-2 原始设计
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 用于 causal mask 的 buffer（最大 n_positions × n_positions）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions, dtype=torch.bool))
            .view(1, 1, config.n_positions, config.n_positions),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,          # [batch, seq_len, n_embd]
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        qkv = self.c_attn(hidden_states)   # [batch, seq_len, 3*n_embd]
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 拆分多头: [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_dim]
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # 拼接历史 KV Cache（Decode 阶段使用）
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)   # [batch, n_head, past_len+1, head_dim]
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # 计算注意力
        total_len = k.size(2)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))    # [batch, n_head, seq_len, total_len]
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Causal mask（只在 Prefill 阶段需要；Decode 阶段 seq_len=1 天然满足因果性）
        if seq_len > 1:
            # query 位置: [past_len, past_len+seq_len)，key 位置: [0, past_len+seq_len)
            past_len = total_len - seq_len
            mask = self.bias[:, :, past_len:past_len + seq_len, :total_len]
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)     # [batch, n_head, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        attn_output = self.resid_dropout(self.c_proj(attn_output))

        return attn_output, new_cache


class GPT2MLP(nn.Module):
    """Position-wise Feed-Forward Network，使用 GELU 激活函数。"""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    """单个 Transformer Block: LayerNorm → Attention → 残差 → LayerNorm → MLP → 残差"""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LayerNorm 结构（GPT-2 原始设计）
        residual = hidden_states
        hidden_states, new_cache = self.attn(self.ln_1(hidden_states), kv_cache, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = residual + self.mlp(self.ln_2(hidden_states))

        return hidden_states, new_cache


class GPT2Model(nn.Module):
    """
    完整 GPT-2 模型。

    关键设计：
      - past_key_values: 每层一个 (K, V) tuple，维度 [batch, n_head, seq_len, head_dim]
      - Prefill: 输入完整 prompt，past_key_values=None，返回所有层的 KV Cache
      - Decode: 输入单 token，past_key_values=上步缓存，更新并返回新 KV Cache
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # position embeddings
        self.drop = nn.Dropout(config.resid_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # GPT-2 中 lm_head 和 wte 共享权重
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,                                          # [batch, seq_len]
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:

        batch, seq_len = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0

        # 位置编码: 从 past_len 开始，对新 token 的位置编码
        position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0)
        token_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)
        hidden_states = self.drop(token_emb + pos_emb)

        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.h):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_cache = block(hidden_states, layer_cache, use_cache)
            if use_cache and new_cache is not None:
                new_kv_caches.append(new_cache)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)    # [batch, seq_len, vocab_size]

        return logits, (new_kv_caches if use_cache else None)

    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2", device: str = "cuda") -> "GPT2Model":
        """
        从 HuggingFace 加载权重到我们手写的模型结构。
        模型结构完全自定义，只复用 HF 的权重数据。
        """
        from transformers import GPT2LMHeadModel as HF_GPT2

        print(f"[GPT2Model] Loading weights from HuggingFace: {model_name}")
        hf_model = HF_GPT2.from_pretrained(model_name)
        hf_cfg = hf_model.config

        # 根据 HF 配置构建我们的配置
        config = GPT2Config(
            vocab_size=hf_cfg.vocab_size,
            n_positions=hf_cfg.n_positions,
            n_embd=hf_cfg.n_embd,
            n_layer=hf_cfg.n_layer,
            n_head=hf_cfg.n_head,
            n_inner=hf_cfg.n_inner if hf_cfg.n_inner else 4 * hf_cfg.n_embd,
            layer_norm_epsilon=hf_cfg.layer_norm_epsilon,
        )
        model = cls(config)

        # 逐层复制权重
        # GPT-2 原始实现用 Conv1D（转置存储），我们用 Linear，需要转置
        def copy_conv1d(dst_linear: nn.Linear, src_conv1d):
            dst_linear.weight.data.copy_(src_conv1d.weight.data.T)
            dst_linear.bias.data.copy_(src_conv1d.bias.data)

        hf_transformer = hf_model.transformer
        model.wte.weight.data.copy_(hf_transformer.wte.weight.data)
        model.wpe.weight.data.copy_(hf_transformer.wpe.weight.data)
        model.ln_f.weight.data.copy_(hf_transformer.ln_f.weight.data)
        model.ln_f.bias.data.copy_(hf_transformer.ln_f.bias.data)

        for i, (my_block, hf_block) in enumerate(zip(model.h, hf_transformer.h)):
            my_block.ln_1.weight.data.copy_(hf_block.ln_1.weight.data)
            my_block.ln_1.bias.data.copy_(hf_block.ln_1.bias.data)
            my_block.ln_2.weight.data.copy_(hf_block.ln_2.weight.data)
            my_block.ln_2.bias.data.copy_(hf_block.ln_2.bias.data)

            copy_conv1d(my_block.attn.c_attn, hf_block.attn.c_attn)
            copy_conv1d(my_block.attn.c_proj, hf_block.attn.c_proj)
            copy_conv1d(my_block.mlp.c_fc, hf_block.mlp.c_fc)
            copy_conv1d(my_block.mlp.c_proj, hf_block.mlp.c_proj)

        print(f"[GPT2Model] Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
        model.eval()
        return model.to(device)

    @torch.no_grad()
    def generate_simple(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """简单的贪心/top-k 采样生成（用于单独测试模型正确性）"""
        past_key_values = None
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if past_key_values is None:
                cur_input = generated
            else:
                cur_input = generated[:, -1:]     # 只送最后一个 token

            logits, past_key_values = self(cur_input, past_key_values)
            next_logits = logits[:, -1, :] / temperature    # [batch, vocab_size]

            if top_k > 0:
                # 只保留 top-k 个候选
                topk_vals, _ = torch.topk(next_logits, top_k)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # [batch, 1]
            generated = torch.cat([generated, next_token], dim=1)

        return generated
