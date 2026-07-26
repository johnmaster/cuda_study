# LLM 推理优化

## 目录结构

```
inference/
├── README.md                          # 本文档
│
├── 01_kv_cache/                       # KV Cache
│   ├── README.md                      #   原理、显存公式、Prefill vs Decode
│   └── kv_cache_demo.py              #   手动实现 + 速度/显存分析
│
├── 02_paged_attention/                # PagedAttention (vLLM 核心)
│   ├── README.md                      #   虚拟内存思想、Block 管理
│   └── paged_attention.py            #   Block Manager + 显存利用率对比
│
└── 03_continuous_batching/            # 连续批处理
    ├── README.md                      #   Static vs Continuous Batching
    └── batching_simulator.py          #   调度模拟器 + 吞吐量对比
```

## 核心问题

1. **KV Cache 显存怎么算？** → 见 01 的公式
2. **Prefill 和 Decode 的瓶颈分别是什么？** → Compute-bound vs Memory-bound
3. **PagedAttention 解决了什么问题？** → 内部碎片 + 外部碎片
4. **Continuous Batching 比 Static 好在哪？** → GPU 利用率 + TTFT
5. **Speculative Decoding 原理？** → 小模型 draft + 大模型 verify
