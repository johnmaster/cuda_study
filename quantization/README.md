# 量化与反量化 (Quantization & Dequantization)

量化是将高精度数值（如 FP32）映射到低精度表示（如 INT8、INT4）的过程，核心目的是在尽量保持精度的前提下，减少内存占用、提高推理速度。

## 目录结构

```
quantization/
├── README.md                   # 本文档（总览）
│
├── symmetric/                  # 对称量化
│   ├── README.md               #   原理介绍
│   ├── symmetric_quant.cu      #   CUDA 实现
│   └── symmetric_quant.py      #   PyTorch 实现
│
├── asymmetric/                 # 非对称量化
│   ├── README.md               #   原理介绍
│   ├── asymmetric_quant.cu     #   CUDA 实现
│   └── asymmetric_quant.py     #   PyTorch 实现
│
├── per_channel/                # Per-Channel 量化
│   ├── README.md               #   原理介绍
│   └── per_channel_quant.py    #   PyTorch 实现 + Per-Tensor 对比
│
├── per_group/                  # Per-Group 量化
│   ├── README.md               #   原理介绍
│   ├── per_group_quant.cu      #   CUDA 实现
│   └── per_group_quant.py      #   PyTorch 实现
│
├── ptq/                        # 训练后量化 (Post-Training Quantization)
│   ├── README.md               #   动态/静态量化、校准方法
│   └── ptq_example.py          #   完整 PTQ 流程 + 校准方法对比
│
├── qat/                        # 量化感知训练 (Quantization-Aware Training)
│   ├── README.md               #   STE、伪量化原理
│   └── qat_example.py          #   QAT vs PTQ 精度对比实验
│
├── gptq/                       # GPTQ
│   ├── README.md               #   Hessian 逐列量化 + 误差补偿
│   └── gptq_example.py         #   GPTQ 简化实现 + 效果对比
│
└── awq/                        # AWQ (Activation-aware Weight Quantization)
    ├── README.md               #   激活值感知缩放原理
    └── awq_example.py          #   AWQ 实现 + 异常值通道分析
```

## 各方法速查对比

| 方法 | 位宽 | 需要训练 | 精度 | 速度 | 适用场景 |
|------|------|---------|------|------|---------|
| 对称量化 | INT8 | 否 | 良好 | 快 | 权重量化 |
| 非对称量化 | UINT8 | 否 | 较好 | 快 | 激活值量化 |
| Per-Channel | INT8 | 否 | 很好 | 快 | 卷积/线性层 |
| Per-Group | INT4/8 | 否 | 很好 | 快 | LLM 权重 |
| PTQ | INT8 | 否 | 好 | 最快 | 快速部署 |
| QAT | INT4/8 | 是 | 最好 | 慢 | 追求精度 |
| GPTQ | INT4 | 否 | 很好 | 快 | LLM |
| AWQ | INT4 | 否 | 很好 | 快 | LLM |
