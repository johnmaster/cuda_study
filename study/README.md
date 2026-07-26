# CUDA 基础算子

本目录包含常见算子的 PyTorch 参考实现与 CUDA 实现，涵盖 elementwise、激活函数、归约、Softmax、RoPE、SGEMM 等主题。

每个子目录通常包含：

- `*.py`：算子语义与结果验证的 PyTorch 参考实现。
- `*.cu`：对应的 CUDA kernel 与测试代码。
- `*.md` / `README.md`：部分算子的实现说明。
