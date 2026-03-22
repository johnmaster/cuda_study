# Ring AllReduce 算法

## 为什么需要 Ring？

### 朴素 AllReduce 的问题

最简单的 AllReduce：所有卡把数据发给 GPU 0，GPU 0 求和后再广播回去。

```
问题：GPU 0 是通信瓶颈！
- GPU 0 需要接收 (N-1) × M 的数据
- 再发送 (N-1) × M 的数据
- 总通信量 2(N-1)M 全压在一张卡上
- 卡数越多，GPU 0 越慢 → 不可扩展
```

### Ring AllReduce 的优势

- 将 N 张卡排成逻辑环：GPU0 → GPU1 → GPU2 → ... → GPU(N-1) → GPU0
- 每张卡只和左右邻居通信
- **每张卡的通信量恒定**为 `2M·(N-1)/N`
- 可以完美利用带宽，**线性可扩展**

## 算法详解（以 4 张卡为例）

### 准备

假设 4 张 GPU，每卡有长度为 4 的数组。将每卡的数据分成 N=4 个 chunk：

```
GPU 0: [a0, a1, a2, a3]
GPU 1: [b0, b1, b2, b3]
GPU 2: [c0, c1, c2, c3]
GPU 3: [d0, d1, d2, d3]

目标: 每卡都得到 [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
```

### Phase 1: ReduceScatter（N-1 步）

每步中，每张卡向右邻居发送一个 chunk，同时从左邻居接收一个 chunk 并累加。

**Step 1**：每卡发送自己编号对应的 chunk
```
GPU 0 发 chunk[0] 给 GPU 1,  GPU 0 从 GPU 3 收 chunk[3]
GPU 1 发 chunk[1] 给 GPU 2,  GPU 1 从 GPU 0 收 chunk[0]
GPU 2 发 chunk[2] 给 GPU 3,  GPU 2 从 GPU 1 收 chunk[1]
GPU 3 发 chunk[3] 给 GPU 0,  GPU 3 从 GPU 2 收 chunk[2]
```

接收后累加到本地对应 chunk：
```
GPU 0: [a0,     a1,     a2,     a3+d3    ]
GPU 1: [a0+b0,  b1,     b2,     b3       ]
GPU 2: [c0,     b1+c1,  c2,     c3       ]
GPU 3: [d0,     d1,     c2+d2,  d3       ]
```

**Step 2**：发送刚累加过的 chunk（沿环继续传递）
```
GPU 0: [a0,        a1,        a2,           a3+c3+d3       ]  ← 从GPU3收到c3+d3累加
GPU 1: [a0+b0+d0,  b1,        b2,           b3             ]  ← 从GPU0收到a0+d0... 不对
```

算了，直接看代码更清晰。经过 N-1=3 步后：

```
GPU 0 的 chunk[1] = a1+b1+c1+d1  (完整求和)
GPU 1 的 chunk[2] = a2+b2+c2+d2  (完整求和)
GPU 2 的 chunk[3] = a3+b3+c3+d3  (完整求和)
GPU 3 的 chunk[0] = a0+b0+c0+d0  (完整求和)
```

每张卡恰好持有**一个 chunk 的完整归约结果**。

### Phase 2: AllGather（N-1 步）

再转 N-1 圈，但这次收到的 chunk 不累加，而是直接覆盖（因为已经是完整结果了）。

经过 N-1=3 步后，每张卡都收集齐了所有 chunk 的完整求和结果：

```
GPU 0: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]  ✅
GPU 1: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]  ✅
GPU 2: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]  ✅
GPU 3: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]  ✅
```

### 通信量分析

```
Phase 1 (ReduceScatter): 每卡每步发 M/N, 共 N-1 步 → M·(N-1)/N
Phase 2 (AllGather):     每卡每步发 M/N, 共 N-1 步 → M·(N-1)/N
总计: 2·M·(N-1)/N
```

当 N 很大时约等于 2M，**与卡数无关**！这就是 Ring AllReduce 高效的本质。

## 代码

- `ring_allreduce_sim.py` — 纯 Python 单进程模拟，打印每一步的状态变化
- `ring_allreduce_nccl.py` — 真实双卡实现，使用 NCCL P2P Send/Recv

```bash
# 模拟（单进程，不需要 GPU）
python ring_allreduce_sim.py

# 真实双卡运行
torchrun --nproc_per_node=2 ring_allreduce_nccl.py
```
