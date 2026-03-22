# FSDP / ZeRO — 显存优化的数据并行

## DDP 的问题

DDP 中每张卡都存：完整参数 + 完整梯度 + 完整优化器状态。
对于大模型，**优化器状态是最大的显存消耗**。

以 FP16 训练 + Adam 优化器为例，每个参数需要：

| 项目 | 每参数字节 | 7B 模型占用 |
|------|----------|------------|
| FP16 参数 | 2 | 14 GB |
| FP16 梯度 | 2 | 14 GB |
| FP32 参数副本 (Adam) | 4 | 28 GB |
| FP32 一阶动量 (Adam) | 4 | 28 GB |
| FP32 二阶动量 (Adam) | 4 | 28 GB |
| **总计** | **16** | **112 GB** |

DDP：每卡都存 112GB → 4 张 A100 (80GB) 都不够！

## ZeRO 三个阶段

DeepSpeed ZeRO 将上述内容**分散到 N 张卡**：

| 阶段 | 分片内容 | 每卡显存 (7B, N=4) | 通信量 |
|------|---------|-------------------|--------|
| DDP | 无分片 | 112 GB | 2M (AllReduce) |
| **ZeRO-1** | 优化器状态 | 14+14+28 = 56 GB | 2M (同 DDP) |
| **ZeRO-2** | + 梯度 | 14+28 = 42 GB | 2M (同 DDP) |
| **ZeRO-3** (FSDP) | + 参数 | 112/4 = 28 GB | 3M (多了 AllGather) |

## FSDP 工作流程 (= ZeRO-3)

```
前向传播:
  对每一层:
    1. AllGather 收集完整参数     ← 通信
    2. 前向计算
    3. 释放非本卡分片的参数       ← 省显存

反向传播:
  对每一层 (反序):
    1. AllGather 收集完整参数     ← 通信
    2. 反向计算得到梯度
    3. ReduceScatter 梯度         ← 通信：每卡只保留自己负责的梯度分片
    4. 释放非本卡分片的参数       ← 省显存

参数更新:
  每卡只更新自己负责的参数分片
```

## ZeRO vs FSDP 的对应关系

| DeepSpeed ZeRO | PyTorch FSDP |
|---------------|-------------|
| ZeRO-1 | 不直接对应 |
| ZeRO-2 | `ShardingStrategy.SHARD_GRAD_OP` |
| ZeRO-3 | `ShardingStrategy.FULL_SHARD` (默认) |

## 运行

```bash
torchrun --nproc_per_node=2 fsdp_training.py
```
