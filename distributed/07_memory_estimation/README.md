# 显存估算 — 面试计算题必备

## 训练显存 = 模型状态 + 激活值 + 临时缓冲

### 模型状态 (以 Mixed Precision + Adam 为例)

| 项目 | 每参数字节 | 公式 |
|------|----------|------|
| FP16 参数 | 2 | 2Φ |
| FP16 梯度 | 2 | 2Φ |
| FP32 master weight (Adam) | 4 | 4Φ |
| FP32 一阶动量 m (Adam) | 4 | 4Φ |
| FP32 二阶动量 v (Adam) | 4 | 4Φ |
| **总计** | **16** | **16Φ** |

Φ = 参数量。7B 模型: 16 × 7 × 10⁹ = 112 GB

### 激活值 (Transformer)

每层激活 ≈ `2 × batch × seq × hidden × (10 + 24/t)`  (Megatron 论文公式)
- t = TP 并行度
- 34sbh per layer (粗略估计, s=seq_len, b=batch, h=hidden)

### Gradient Checkpointing

不存中间激活，反向时重新计算 → 激活显存从 O(N_layers) 降到 O(√N_layers)
代价：约 33% 额外计算时间

## 运行

```bash
python memory_calculator.py
```
