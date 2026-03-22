# KV Cache

## 为什么需要 KV Cache？

LLM 自回归生成时，每生成一个 token 都需要 attention 计算：

```
没有 KV Cache:
  生成第 1 个 token: 计算 Q1, K1, V1
  生成第 2 个 token: 计算 Q2, K1, K2, V1, V2   ← K1, V1 重复计算了！
  生成第 3 个 token: 计算 Q3, K1, K2, K3, V1, V2, V3  ← 又重复！
  ...
  生成第 N 个 token: 重复计算了 O(N²) 次

有 KV Cache:
  生成第 1 个 token: 计算 Q1, K1, V1，缓存 K1, V1
  生成第 2 个 token: 只计算 Q2, K2, V2，缓存 K2, V2，用 Q2 和 [K1,K2] 做 attention
  生成第 3 个 token: 只计算 Q3, K3, V3，缓存 K3, V3
  ...
  每步只需 O(N) 计算，总共 O(N²) → 但每步是 O(N) 而不是重新算
```

## KV Cache 显存公式

```
每层 KV Cache = 2 × batch_size × seq_len × n_heads × head_dim × dtype_size

总 KV Cache = n_layers × 每层 KV Cache

示例 (LLaMA-7B, batch=1, seq=2048, FP16):
  = 32 × 2 × 1 × 2048 × 32 × 128 × 2 bytes
  = 32 × 2 × 1 × 2048 × 4096 × 2
  = 1,073,741,824 bytes ≈ 1 GB
```

KV Cache 显存随 batch_size × seq_len **线性增长**，这就是为什么长序列推理显存紧张。

## Prefill vs Decode 两阶段

| 阶段 | 输入 | 计算特征 | 瓶颈 |
|------|------|---------|------|
| **Prefill** | 整个 prompt (N tokens) | 大矩阵乘，可以并行 | Compute-bound |
| **Decode** | 每次 1 个 token | 小矩阵乘，GEMV | Memory-bound (读 KV Cache) |

Decode 阶段的 arithmetic intensity 很低（每读一次 KV 只做一次乘加），
因此 GPU 算力利用率很低，主要瓶颈是 **HBM 带宽**。

## 运行

```bash
python kv_cache_demo.py
```
