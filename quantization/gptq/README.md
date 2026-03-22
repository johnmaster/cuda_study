# GPTQ (Generative Pre-trained Transformer Quantization)

## 核心思想

GPTQ 是针对**大语言模型**的高效训练后量化方法。核心目标：逐层最小化量化后输出与原始输出的误差，利用近似二阶信息（Hessian 矩阵）精确补偿量化误差。

## 背景：为什么需要 GPTQ？

普通 PTQ 对 LLM 做 INT4 量化时精度损失很大。LLM 的参数量巨大，不可能用 QAT 重新训练。GPTQ 在 PTQ 框架下，通过更精细的逐列量化 + 误差补偿，实现了 INT4 级别的高精度量化。

## 算法原理

GPTQ 基于 **OBQ (Optimal Brain Quantization)** 改进而来，核心步骤：

### Step 1: 计算 Hessian 矩阵

对于线性层 Y = XW，量化 W 的目标是最小化输出误差：

```
min_Q ||XW - XQ||²
```

这等价于加权的逐行问题，Hessian 矩阵为：

```
H = 2 · X^T · X
```

H 反映了每个权重对输出的影响程度。

### Step 2: 逐列量化 + 误差补偿

按列处理权重矩阵，对每一列：

```
1. 量化当前列:  q_j = quantize(w_j)
2. 计算量化误差: δ_j = w_j - q_j
3. 补偿后续列:   W[:, j+1:] += δ_j · (H_jj^{-1} · H[j, j+1:])
                                ↑ 用 Hessian 信息把误差分摊到未量化的列
```

关键insight：量化一列产生的误差，可以通过调整尚未量化的列来部分补偿。

### Step 3: 分块处理 (Lazy Batch Update)

GPTQ 不是严格逐列处理，而是以 block_size（如 128）为单位：
- 块内逐列量化 + 块内补偿
- 块处理完后，一次性更新后续所有列

这大幅减少了对 GPU 显存的更新操作，提升效率。

## 算法伪代码

```
输入: 权重 W [out, in], Hessian H [in, in], block_size B
输出: 量化后的权重 Q

for i = 0, B, 2B, ...:
    # 处理第 i 到 i+B 列
    for j = i 到 i+B:
        q_j = quantize(W[:, j])              # 量化
        err = (W[:, j] - q_j) / H[j, j]      # 缩放误差
        Q[:, j] = q_j
        W[:, j:(i+B)] -= err * H[j, j:(i+B)] # 块内补偿

    # 块间更新
    W[:, (i+B):] -= (W[:, i:(i+B)] - Q[:, i:(i+B)]) @ H[i:(i+B), (i+B):]^{-1} @ H[i:(i+B), (i+B):]
```

## 特点

| 特点 | 说明 |
|------|------|
| 逐层量化 | 每层独立处理，不需要全模型前向传播 |
| 需要校准数据 | 少量（128 条左右）用于计算 Hessian |
| 速度快 | LLaMA-7B 量化只需几分钟 |
| 精度高 | INT4 量化精度接近 FP16 |
| Per-Group | 通常搭配 Per-Group 量化使用 |

## 与其他方法对比

| 方法 | 需要训练 | INT4 精度 | 速度 | 适用模型 |
|------|---------|----------|------|---------|
| PTQ (朴素) | 否 | 差 | 最快 | 小模型 |
| QAT | 是 | 好 | 最慢 | 中小模型 |
| GPTQ | 否 | 很好 | 快 | LLM |
| AWQ | 否 | 很好 | 快 | LLM |
