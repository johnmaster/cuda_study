# 流水线并行 (Pipeline Parallelism)

## 核心思想

将模型的**不同层**分配到不同 GPU 上，数据像流水线一样依次通过各 GPU。

```
GPU 0: [Layer 0, 1, 2, 3]
GPU 1: [Layer 4, 5, 6, 7]

数据流: Input → GPU0 (前4层) → GPU1 (后4层) → Output
```

## 朴素 PP 的问题：Bubble

```
时间 →
GPU 0: [F0][F1][F2][F3][  ][  ][  ][  ][B3][B2][B1][B0]
GPU 1: [  ][F0][F1][F2][F3][  ][  ][B3][B2][B1][B0][  ]

F = 前向, B = 反向, 空白 = 气泡（GPU 空闲）
```

气泡率 = (P-1) / (P-1+M)，P=PP 并行度，M=micro-batch 数

## GPipe vs 1F1B

### GPipe
先完成所有前向，再执行所有反向。简单但**显存高**（需要存所有 micro-batch 的激活）。

### 1F1B (One Forward One Backward)
交替执行前向和反向，及时释放激活 → **显存更优**。

```
1F1B 调度 (2 stages, 4 micro-batches):
GPU 0: [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 1:     [F0][F1][B0][F2][B1][F3][B2][B3]
```

## 通信

PP 只需要 **P2P Send/Recv**（相邻 stage 间传激活/梯度），通信量远小于 TP。
因此 PP 适合**跨节点**（带宽低），TP 适合**节点内**（NVLink 高带宽）。

## 运行

```bash
torchrun --nproc_per_node=2 pipeline_parallel.py
```
