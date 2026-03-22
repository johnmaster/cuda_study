# Per-Channel 量化

## 核心思想

Per-Tensor 量化对整个张量只用一组 scale/zero_point，但神经网络中不同通道的数值范围差异很大。Per-Channel 量化为**每个输出通道**分配独立的量化参数，显著提高精度。

## 为什么需要 Per-Channel？

假设一个卷积层有 3 个输出通道：
```
通道 0: 权重范围 [-0.1, 0.1]
通道 1: 权重范围 [-2.0, 2.0]
通道 2: 权重范围 [-0.01, 0.01]
```

Per-Tensor 量化时 scale = 2.0 / 127 = 0.0157，通道 2 的权重全部被量化为 0，信息完全丢失。

Per-Channel 量化时每个通道有自己的 scale，通道 2 的 scale = 0.01 / 127，能精确表示。

## 数学公式

对于权重张量 W（形状 [out_channels, ...]）：

```
对每个通道 c:
    scale[c] = max(|W[c, :]|) / 127
    Q[c, :] = clamp(round(W[c, :] / scale[c]), -127, 127)

反量化:
    Ŵ[c, :] = Q[c, :] × scale[c]
```

## 量化粒度对比

| 粒度 | scale 数量 | 精度 | 存储开销 |
|------|-----------|------|---------|
| Per-Tensor | 1 | 低 | 最小 |
| Per-Channel | out_channels | 中高 | 每通道多一个 float |
| Per-Group | N / group_size | 最高 | 每组多一个 float |

## 适用场景

- **卷积层权重**：不同卷积核的数值范围差异大
- **线性层权重**：不同输出神经元的权重分布不同
- PyTorch 官方的量化 API 默认使用 Per-Channel 量化权重
