# NCCL 通信原语

NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 提供的多 GPU 集合通信库，
PyTorch 的 `torch.distributed` 默认使用 NCCL 后端进行 GPU 间通信。

## 七大通信原语

### 1. Broadcast（广播）

一张卡的数据复制到所有卡。

```
Before:  GPU0=[A]  GPU1=[_]  GPU2=[_]  GPU3=[_]
After:   GPU0=[A]  GPU1=[A]  GPU2=[A]  GPU3=[A]
```

**用途**：模型初始化时，将 rank 0 的参数广播到所有卡。

### 2. Reduce（归约）

所有卡的数据聚合到一张卡（如求和）。

```
Before:  GPU0=[A0]  GPU1=[A1]  GPU2=[A2]  GPU3=[A3]
After:   GPU0=[A0+A1+A2+A3]  GPU1=[_]  GPU2=[_]  GPU3=[_]
```

### 3. AllReduce（全归约） ⭐ 最重要

所有卡的数据聚合，结果**每张卡都有一份完整副本**。
等价于 Reduce + Broadcast，但 Ring 算法比朴素实现高效得多。

```
Before:  GPU0=[A0]  GPU1=[A1]  GPU2=[A2]  GPU3=[A3]
After:   GPU0=[S]   GPU1=[S]   GPU2=[S]   GPU3=[S]    (S = A0+A1+A2+A3)
```

**用途**：DDP 梯度同步 — 每张卡独立前向+反向后，AllReduce 梯度取平均。

### 4. AllGather（全收集）

每张卡贡献自己的一小块，所有卡都得到完整拼接结果。

```
Before:  GPU0=[A]  GPU1=[B]  GPU2=[C]  GPU3=[D]
After:   GPU0=[ABCD]  GPU1=[ABCD]  GPU2=[ABCD]  GPU3=[ABCD]
```

**用途**：
- FSDP 前向时：AllGather 收集完整的权重参数
- 张量并行：AllGather 收集分片结果

### 5. ReduceScatter（归约分散）

AllReduce 的"一半"：先 Reduce 再 Scatter，每张卡只得到聚合结果的一部分。

```
Before:  GPU0=[A0,A1,A2,A3]  GPU1=[B0,B1,B2,B3]  GPU2=[C0,C1,C2,C3]  GPU3=[D0,D1,D2,D3]
After:   GPU0=[A0+B0+C0+D0]  GPU1=[A1+B1+C1+D1]  GPU2=[A2+B2+C2+D2]  GPU3=[A3+B3+C3+D3]
```

**用途**：
- FSDP 反向时：ReduceScatter 梯度，每张卡只保留自己负责的那部分
- Ring AllReduce 的第一阶段就是 ReduceScatter

### 6. All-to-All（全交换）

每张卡向每张卡发送不同的数据。最灵活但通信模式最复杂。

```
Before:  GPU0=[A0,A1,A2,A3]  GPU1=[B0,B1,B2,B3]
After:   GPU0=[A0,B0]         GPU1=[A1,B1]     (每卡收到所有卡发给自己的那份)
```

**用途**：MoE (Mixture of Experts) 模型中，tokens 路由到不同 expert 所在的 GPU。

### 7. P2P Send/Recv（点对点）

两张卡之间直接传数据。

**用途**：流水线并行 (Pipeline Parallelism) 中，前一个 stage 的输出发给下一个 stage。

## 关键关系

```
AllReduce  =  ReduceScatter  +  AllGather
```

这不仅是概念上的等价，Ring AllReduce 实际就是按这两步实现的！

## NCCL vs Gloo vs MPI

| 后端 | 硬件 | 性能 | 使用场景 |
|------|------|------|---------|
| **NCCL** | GPU | 最快 | GPU 间通信（默认选择） |
| **Gloo** | CPU/GPU | 中等 | CPU 训练、或 NCCL 不支持的操作 |
| **MPI** | CPU | 取决于实现 | HPC 场景 |

## 运行 demo

```bash
# 在你的双卡机器上运行（2 个进程，各占 1 张 GPU）
torchrun --nproc_per_node=2 nccl_ops_demo.py
```
