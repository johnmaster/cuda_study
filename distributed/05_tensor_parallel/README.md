# 张量并行 (Tensor Parallelism)

## 核心思想

单层的权重太大无法放进一张卡 → 把**一个线性层的权重矩阵切分到多张卡**上。

这是 Megatron-LM 提出的方法，需要卡间有**高带宽互联**（NVLink），
因为每一层前向/反向都需要通信。

## Megatron-style Column/Row Parallel

### Column Parallel Linear

将权重按**列**切分，每卡计算输出的一部分：

```
完整: Y = XW + b       W: [in, out]

切分: W = [W1 | W2]    W1: [in, out/2]  在 GPU 0
                        W2: [in, out/2]  在 GPU 1

GPU 0: Y1 = X @ W1     → 得到输出的前半部分
GPU 1: Y2 = X @ W2     → 得到输出的后半部分

拼接: Y = [Y1 | Y2]    (AllGather)
```

**特点**：输入 X 需要在两卡上都有（Broadcast），输出需要 AllGather。

### Row Parallel Linear

将权重按**行**切分，每卡处理输入的一部分：

```
完整: Y = XW + b       W: [in, out]

切分: W = [W1]          W1: [in/2, out]  在 GPU 0
          [W2]          W2: [in/2, out]  在 GPU 1

X = [X1 | X2]          X1: 输入的前半部分
                        X2: 输入的后半部分

GPU 0: Y1 = X1 @ W1    → 部分结果
GPU 1: Y2 = X2 @ W2    → 部分结果

求和: Y = Y1 + Y2      (AllReduce)
```

**特点**：输入需要切分（ReduceScatter），输出需要 AllReduce。

### MLP 中的组合

Transformer MLP 由两个线性层组成：`Y = GELU(X @ W1) @ W2`

Megatron 的巧妙设计：
```
W1 用 Column Parallel → 输出自然分到两卡，不需要额外通信
GELU 各卡独立执行
W2 用 Row Parallel    → 输入已经在各卡上了，只需要最后 AllReduce
```

**结果：两个线性层只需要 1 次 AllReduce（而不是 2 次）！**

## 通信量对比

| 方法 | 每层通信次数 | 通信量 | 适用互联 |
|------|------------|--------|---------|
| Column + AllGather | 1 | O(batch × out) | NVLink |
| Row + AllReduce | 1 | O(batch × out) | NVLink |
| Megatron MLP (Column + Row) | 1 | O(batch × hidden) | NVLink |

## 运行

```bash
torchrun --nproc_per_node=2 tensor_parallel.py
```
