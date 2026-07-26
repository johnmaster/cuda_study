"""KV Cache 实现与分析

手动实现带 KV Cache 的自回归 Attention，对比有/无 Cache 的性能差异。
运行方式: python kv_cache_demo.py
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """标准 Multi-Head Attention，支持可选 KV Cache"""

    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        Args:
            x: [batch, seq_len, hidden] (prefill) 或 [batch, 1, hidden] (decode)
            kv_cache: (cached_k, cached_v) 或 None
            use_cache: 是否返回更新后的 cache
        """
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [batch, n_heads, seq_len, head_dim]

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        T_q, T_k = q.size(2), k.size(2)
        mask = torch.triu(torch.ones(T_q, T_k, device=x.device), diagonal=T_k - T_q + 1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.wo(out)

        return out, new_cache


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ffn_dim):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x, kv_cache=None, use_cache=False):
        h = self.norm1(x)
        attn_out, new_cache = self.attn(h, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class SmallLM(nn.Module):
    """用于 KV Cache 对比的小型语言模型"""
    def __init__(self, vocab_size=1000, hidden_dim=256, n_heads=8,
                 n_layers=4, ffn_dim=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, kv_caches=None, use_cache=False):
        x = self.embed(input_ids)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, kv_cache=cache, use_cache=use_cache)
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_caches if use_cache else None


# ==================== 生成函数 ====================

@torch.no_grad()
def generate_no_cache(model, prompt_ids, max_new_tokens):
    """不用 KV Cache 的生成：每步重新计算所有 token 的 attention"""
    generated = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits, _ = model(generated, use_cache=False)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


@torch.no_grad()
def generate_with_cache(model, prompt_ids, max_new_tokens):
    """用 KV Cache 的生成：每步只处理新 token"""
    # Prefill: 处理整个 prompt
    logits, kv_caches = model(prompt_ids, use_cache=True)
    next_token = logits[:, -1:, :].argmax(dim=-1)
    generated = torch.cat([prompt_ids, next_token], dim=1)

    # Decode: 每次只输入 1 个 token
    for _ in range(max_new_tokens - 1):
        logits, kv_caches = model(next_token, kv_caches=kv_caches, use_cache=True)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    print("=" * 60)
    print(" KV Cache Demo")
    print("=" * 60)

    model = SmallLM(vocab_size=1000, hidden_dim=256, n_heads=8,
                    n_layers=4, ffn_dim=1024).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: {n_params:,} 参数, 4 层, hidden=256, heads=8")

    # ==================== 实验 1: 正确性验证 ====================
    print(f"\n[1] 正确性验证")
    prompt = torch.randint(0, 1000, (1, 32)).to(device)
    max_new = 20

    out_no_cache = generate_no_cache(model, prompt, max_new)
    out_cache = generate_with_cache(model, prompt, max_new)

    match = (out_no_cache == out_cache).all().item()
    print(f"  生成结果一致: {'✓' if match else '✗'}")

    # ==================== 实验 2: 速度对比 ====================
    print(f"\n[2] 速度对比")

    for prompt_len, gen_len in [(32, 64), (128, 128), (256, 256)]:
        prompt = torch.randint(0, 1000, (1, prompt_len)).to(device)

        # Warmup
        generate_with_cache(model, prompt, 5)
        generate_no_cache(model, prompt, 5)
        torch.cuda.synchronize() if device == "cuda" else None

        # No cache
        start = time.perf_counter()
        generate_no_cache(model, prompt, gen_len)
        torch.cuda.synchronize() if device == "cuda" else None
        t_no_cache = time.perf_counter() - start

        # With cache
        start = time.perf_counter()
        generate_with_cache(model, prompt, gen_len)
        torch.cuda.synchronize() if device == "cuda" else None
        t_cache = time.perf_counter() - start

        speedup = t_no_cache / t_cache
        tokens_per_sec = gen_len / t_cache

        print(f"  prompt={prompt_len:3d}, gen={gen_len:3d}: "
              f"无Cache {t_no_cache*1000:7.1f}ms, "
              f"有Cache {t_cache*1000:7.1f}ms, "
              f"加速 {speedup:.1f}x, "
              f"{tokens_per_sec:.0f} tok/s")

    # ==================== 实验 3: KV Cache 显存分析 ====================
    print(f"\n[3] KV Cache 显存计算")

    configs = [
        ("LLaMA-7B",  32, 32, 128, 2048),
        ("LLaMA-13B", 40, 40, 128, 2048),
        ("LLaMA-70B", 80, 64, 128, 2048),
        ("GPT-4级别", 120, 96, 128, 8192),
    ]

    print(f"  {'模型':<12} {'层数':>4} {'头数':>4} {'seq_len':>8} "
          f"{'KV Cache (FP16)':>16} {'KV Cache (INT8)':>16}")
    print(f"  {'-'*70}")

    for name, n_layers, n_heads, head_dim, seq_len in configs:
        # 2 for K and V, 2 bytes for FP16
        kv_bytes_fp16 = 2 * n_layers * 1 * seq_len * n_heads * head_dim * 2
        kv_bytes_int8 = 2 * n_layers * 1 * seq_len * n_heads * head_dim * 1
        print(f"  {name:<12} {n_layers:>4} {n_heads:>4} {seq_len:>8} "
              f"{kv_bytes_fp16 / 1024**3:>13.2f} GB "
              f"{kv_bytes_int8 / 1024**3:>13.2f} GB")

    # ==================== 实验 4: Prefill vs Decode 分析 ====================
    print(f"\n[4] Prefill vs Decode 阶段分析")

    prompt = torch.randint(0, 1000, (1, 256)).to(device)

    # Warmup
    generate_with_cache(model, prompt, 5)
    torch.cuda.synchronize() if device == "cuda" else None

    # Prefill
    start = time.perf_counter()
    logits, kv_caches = model(prompt, use_cache=True)
    torch.cuda.synchronize() if device == "cuda" else None
    prefill_time = time.perf_counter() - start

    # Decode (单 token)
    next_token = logits[:, -1:, :].argmax(dim=-1)
    decode_times = []
    for _ in range(50):
        start = time.perf_counter()
        logits, kv_caches = model(next_token, kv_caches=kv_caches, use_cache=True)
        torch.cuda.synchronize() if device == "cuda" else None
        decode_times.append(time.perf_counter() - start)
        next_token = logits[:, -1:, :].argmax(dim=-1)

    avg_decode = sum(decode_times) / len(decode_times)

    print(f"  Prefill (256 tokens):  {prefill_time * 1000:.2f} ms  "
          f"({256 / prefill_time:.0f} tok/s)  ← Compute-bound")
    print(f"  Decode  (每 token):    {avg_decode * 1000:.2f} ms  "
          f"({1 / avg_decode:.0f} tok/s)  ← Memory-bound")
    print(f"  Prefill 处理 256 个 token 仅是 Decode 一步的 "
          f"{prefill_time / avg_decode:.1f}x")

    print(f"\n{'=' * 60}")
    print(" 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
