# PyTorch Custom Ops

从零开始学习如何将 CUDA kernel 包装成 PyTorch 算子，涵盖 pybind11 绑定、autograd 集成、
Dispatcher 机制、torch.compile 兼容性等 AI Infra 核心知识。

## 项目结构

```
pytorch_custom_ops/
├── 01_custom_gelu/                  # 入门：elementwise 算子
│   ├── csrc/
│   │   ├── custom_gelu_kernel.cu    #   CUDA kernel (forward + backward)
│   │   └── binding.cpp              #   pybind11 绑定
│   ├── custom_gelu.py               #   autograd.Function + nn.Module
│   ├── test_custom_gelu.py          #   正确性 & 性能测试
│   └── setup.py                     #   pip install 构建
│
├── 02_flash_attention/              # 进阶：Flash Attention 完整 custom op
│   ├── csrc/
│   │   ├── flash_attn_kernel.cu     #   forward (online softmax) + backward (tile-based recomputation)
│   │   └── binding.cpp              #   pybind11 绑定
│   ├── flash_attention_op.py        #   autograd.Function (保存 logsumexp, 不存 N×N 矩阵)
│   ├── test_flash_attention.py      #   正确性 / 梯度 / 性能测试
│   └── setup.py
│
├── pytorch_custom_ops_guide.md      # 完整教程：从 kernel 到 torch.compile 全链路
└── README.md
```

## 快速开始

```bash
# 前置：需要 PyTorch (CUDA), ninja
pip install ninja

# 示例 1：Custom GELU
cd 01_custom_gelu
python test_custom_gelu.py

# 示例 2：Flash Attention (forward + backward)
cd ../02_flash_attention
python test_flash_attention.py
```

首次运行会 JIT 编译 CUDA 扩展（约 1-2 分钟），之后自动缓存。

## 学习路线

| 阶段 | 内容 | 对应代码 |
|------|------|---------|
| **1. 基础** | 写 CUDA kernel → pybind11 绑定 → Python 调用 | `01_custom_gelu/` |
| **2. Autograd** | `autograd.Function` + `save_for_backward` | `01_custom_gelu/custom_gelu.py` |
| **3. 复杂算子** | tiling、online softmax、tile-based backward | `02_flash_attention/` |
| **4. 全链路理解** | Dispatcher、torch.library、torch.compile | `pytorch_custom_ops_guide.md` |

## 核心知识点

### 数据流

```
Python nn.Module
    │  .forward(x)
    ▼
autograd.Function
    │  ctx.save_for_backward(...)
    │  调用 C++ extension
    ▼
binding.cpp (pybind11)
    │  TORCH_CHECK 校验
    │  调用 CUDA host function
    ▼
kernel.cu (__global__)
    │  GPU 并行执行
    ▼
返回 torch::Tensor → Python
```

### Flash Attention 的 backward 策略

| | 标准 Attention | Flash Attention |
|---|---|---|
| forward 保存 | P (N×N 矩阵) | L (logsumexp, 长度 N) |
| backward 时 | 直接用 P | 从 Q, K, L 分 tile 重算 P |
| 内存 | O(N²) | O(N) |

详细讲解见 [`pytorch_custom_ops_guide.md`](pytorch_custom_ops_guide.md)。
