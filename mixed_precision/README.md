# Mixed Precision Training (混合精度训练)

混合精度训练是 AI Infra 工程师的核心技能。本目录从原理到实践，全面覆盖混合精度相关知识。

## 目录结构

```
mixed_precision/
├── README.md                           ← 你在这里
├── mixed_precision_guide.md            ← 完整知识指南 (面试必读)
│
├── 01_amp_basics/                      ← PyTorch AMP 标准用法
│   ├── amp_training.py                 ← FP32 vs FP16 vs BF16 训练对比
│   └── amp_autocast_demo.py            ← autocast 内部行为可视化
│
├── 02_manual_mixed_precision/          ← 手动实现混合精度 (理解底层)
│   └── manual_amp.py                   ← 手写 FP32 master weights + loss scaling
│
└── 03_fp16_kernels/                    ← CUDA 层面的精度对比
    ├── precision_benchmark.py          ← PyTorch 各精度 GEMM 性能 benchmark
    ├── fp16_gemm_cuda.cu               ← CUDA FP32/FP16/WMMA kernel 对比
    └── Makefile
```

## 快速开始

### 1. AMP 基础 (最常用)

```bash
# 对比 FP32 / FP16 / BF16 训练速度和显存
python 01_amp_basics/amp_training.py

# 看 autocast 怎么给不同 op 选 dtype
python 01_amp_basics/amp_autocast_demo.py
```

### 2. 理解底层原理

```bash
# 手动实现混合精度 (不用 torch.amp)
# 包含: FP32 master weights + dynamic loss scaling + 梯度下溢演示
python 02_manual_mixed_precision/manual_amp.py
```

### 3. CUDA 内核层面

```bash
# PyTorch 各精度 GEMM benchmark (TFLOPS + 精度误差)
python 03_fp16_kernels/precision_benchmark.py

# CUDA kernel: FP32 vs FP16 vs WMMA(Tensor Core)
cd 03_fp16_kernels && make && ./fp16_gemm_cuda
```

## 核心知识点

| 概念 | 说明 |
|------|------|
| **FP32 Master Weights** | FP16 模型 + FP32 副本, 优化器在 FP32 上更新 |
| **Loss Scaling** | 放大 loss 防止 FP16 梯度下溢, 更新前缩回 |
| **autocast** | 自动给不同 op 选最优精度 (GEMM→FP16, softmax→FP32) |
| **GradScaler** | Dynamic loss scaling, 自动调整 scale factor |
| **BF16** | 和 FP32 相同的范围, 不需要 loss scaling |
| **TF32** | Ampere+ 默认的 FP32 加速模式, 用户代码无需改动 |
| **Tensor Core** | NVIDIA GPU 的矩阵运算单元, FP16 吞吐 8-16x FP32 |

## 学习路径

1. 先读 `mixed_precision_guide.md` 了解全貌
2. 运行 `01_amp_basics/` 看 AMP 怎么用
3. 运行 `02_manual_mixed_precision/` 理解底层机制
4. 运行 `03_fp16_kernels/` 看 Tensor Core 的真实性能差异
5. 回到 guide 复习面试问题

## 依赖

```
PyTorch >= 2.0 (with CUDA)
NVIDIA GPU (Ampere+ 推荐, 支持 BF16/TF32)
nvcc (编译 CUDA kernel)
```
