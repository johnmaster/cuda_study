# 分布式训练与 NCCL 通信

## 为什么需要分布式？

单卡显存和算力有限。以 LLaMA-70B 为例：
- 模型参数：70B × 2 bytes (FP16) = **140GB**，单张 A100 (80GB) 放不下
- 训练还需要：梯度 (140GB) + 优化器状态 (280GB for Adam) = 总共 **~560GB**
- 即使推理，INT4 量化后仍需 ~35GB

因此必须将**数据、模型、计算**拆分到多张 GPU 上。

## 目录结构

```
distributed/
├── README.md                          # 本文档（总览）
│
├── 01_nccl_primitives/                # NCCL 通信原语
│   ├── README.md                      #   AllReduce/AllGather/ReduceScatter 等原理
│   └── nccl_ops_demo.py              #   双卡通信原语验证
│
├── 02_ring_allreduce/                 # Ring AllReduce 深入
│   ├── README.md                      #   Ring 算法图解
│   ├── ring_allreduce_sim.py          #   纯 Python 算法模拟
│   └── ring_allreduce_nccl.py         #   真实双卡 NCCL 实现
│
├── 03_ddp/                            # 分布式数据并行
│   ├── README.md                      #   DDP 原理、Bucketing、Gradient Accumulation
│   └── ddp_training.py               #   双卡 DDP 训练 + 梯度同步验证
│
├── 04_fsdp_zero/                      # ZeRO / FSDP
│   ├── README.md                      #   ZeRO 1/2/3 原理、通信量分析
│   └── fsdp_training.py              #   DDP vs FSDP 显存对比实验
│
├── 05_tensor_parallel/                # 张量并行
│   ├── README.md                      #   Column/Row Parallel、Megatron MLP
│   └── tensor_parallel.py            #   手动实现 TP，双卡切分线性层
│
├── 06_pipeline_parallel/              # 流水线并行
│   ├── README.md                      #   GPipe vs 1F1B、Bubble Rate
│   └── pipeline_parallel.py          #   手动实现 GPipe + 1F1B
│
└── 07_memory_estimation/              # 显存估算
    ├── README.md                      #   训练/推理显存公式
    └── memory_calculator.py           #   LLM 显存计算器
```

## 核心概念速查

### 并行策略

| 策略 | 切分什么 | 通信操作 | 适用场景 |
|------|---------|---------|---------|
| **数据并行 (DP/DDP)** | 数据 batch | AllReduce 梯度 | 最常用，模型能放进单卡 |
| **张量并行 (TP)** | 模型层内的权重矩阵 | AllReduce / AllGather | 单层太大，需要卡间高带宽 (NVLink) |
| **流水线并行 (PP)** | 模型的不同层 | P2P Send/Recv | 模型层数多，跨节点 |
| **ZeRO (FSDP)** | 优化器状态/梯度/参数 | AllGather + ReduceScatter | 降低显存占用 |
| **序列并行 (SP)** | 序列长度维度 | AllGather / ReduceScatter | 长序列，减少 activation 内存 |
| **专家并行 (EP)** | MoE 的不同 expert | All-to-All | MoE 模型 |

### NCCL 通信原语

| 原语 | 输入 | 输出 | 通信量 (N卡, 每卡数据M) |
|------|------|------|----------------------|
| **Broadcast** | 1张卡有数据 | 所有卡都有完整数据 | M |
| **Reduce** | 每卡有数据 | 1张卡有聚合结果 | M |
| **AllReduce** | 每卡有数据 | 每卡都有聚合结果 | 2M·(N-1)/N |
| **AllGather** | 每卡有部分数据 | 每卡都有完整数据 | M·(N-1)/N |
| **ReduceScatter** | 每卡有数据 | 每卡有部分聚合结果 | M·(N-1)/N |
| **All-to-All** | 每卡有N份数据 | 每卡收到来自所有卡的对应份 | M·(N-1)/N |

### Ring AllReduce 为什么重要？

AllReduce 是分布式训练中**最核心的通信操作**（DDP 梯度同步的基础）。

朴素实现（所有卡发给一张卡汇总再广播）：通信瓶颈在单卡，**不可扩展**。

Ring AllReduce：
- 将 N 张卡排成环
- 分两个阶段：ReduceScatter + AllGather
- 每张卡的通信量恒定为 `2M·(N-1)/N`，**与卡数 N 几乎无关**
- 这就是 NCCL 内部实现 AllReduce 的核心算法

## 硬件背景知识

```
GPU 间通信带宽（从快到慢）：

NVLink 4.0 (H100)    : 900 GB/s 双向
NVLink 3.0 (A100)    : 600 GB/s 双向
NVLink 2.0 (V100)    : 300 GB/s 双向
PCIe 4.0             : ~32 GB/s 双向
PCIe 3.0             : ~16 GB/s 双向
InfiniBand HDR       : 200 Gbps (~25 GB/s) 节点间
```

2 × RTX 2080 Ti 环境通过 PCIe 互联，不支持 NVLink。
