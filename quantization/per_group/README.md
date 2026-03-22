# Per-Group 量化

## 核心思想

Per-Group 量化将张量按固定大小（group_size）分组，每组独立计算 scale。它是 Per-Channel 的进一步细化，在大语言模型量化中被广泛使用（GPTQ、AWQ 都基于此）。

## 为什么需要 Per-Group？

Per-Channel 给每个输出通道一个 scale，但 LLM 的线性层维度很大（如 4096×4096），一个通道内部仍然存在数值范围差异。Per-Group 进一步切分，常用 group_size = 128。

```
通道 0 的权重 (4096个值):
  [前128个] scale_0  [第129-256个] scale_1  ...  [最后128个] scale_31
```

## 数学公式

```
对于每组 g（包含 group_size 个连续元素）:
    scale[g] = max(|x[g*gs : (g+1)*gs]|) / 127
    Q[i] = clamp(round(x[i] / scale[g]), -127, 127)    其中 i ∈ 组 g

反量化:
    x̂[i] = Q[i] × scale[g]
```

## 存储开销分析

以 [4096, 4096] 的权重矩阵、group_size=128 为例：

| 方案 | 权重大小 | scale 大小 | 总计 | 压缩比 |
|------|---------|-----------|------|--------|
| FP32 (原始) | 64 MB | 0 | 64 MB | 1x |
| Per-Tensor INT8 | 16 MB | 4 B | ~16 MB | 4x |
| Per-Channel INT8 | 16 MB | 16 KB | ~16 MB | 4x |
| Per-Group INT8 (g=128) | 16 MB | 512 KB | ~16.5 MB | 3.9x |
| Per-Group INT4 (g=128) | 8 MB | 512 KB | ~8.5 MB | 7.5x |

Per-Group 的额外开销很小，但精度显著优于 Per-Tensor/Per-Channel。

## 适用场景

- **大语言模型量化**：GPT、LLaMA 等模型的权重量化
- **INT4 量化**：位宽越低越需要细粒度 scale 补偿精度
- group_size 常用值：32、64、128
