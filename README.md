# AI Infra Notes

CUDA、GPU 算子优化与大模型系统知识库，覆盖 CUDA 编程、手写 kernel、高性能计算库、Triton、PyTorch 扩展、量化、分布式训练与推理系统。

## 内容导航

### CUDA 基础与算子

| 目录 | 内容 |
| --- | --- |
| [`docs/`](docs/) | CUDA 编程、GPU 硬件、WMMA 与性能分析笔记 |
| [`examples/`](examples/) | 可独立编译运行的基础示例 |
| [`study/`](study/) | PyTorch 参考实现与 CUDA 基础算子 |
| [`solution_for_leetGPU/`](solution_for_leetGPU/) | LeetGPU 风格题目及 CMake 工程 |

### 高性能计算与框架扩展

| 目录 | 内容 |
| --- | --- |
| [`cublas_cutlass_learning/`](cublas_cutlass_learning/) | cuBLAS / cuBLASLt API、矩阵布局与 GEMM 笔记 |
| [`triton_kernels/`](triton_kernels/) | Elementwise、Reduction、GEMM、Flash Attention 等 Triton kernel |
| [`pytorch_custom_ops/`](pytorch_custom_ops/) | 使用 C++/CUDA 扩展实现 PyTorch 自定义算子 |
| [`mixed_precision/`](mixed_precision/) | AMP、手动混合精度与 FP16 CUDA kernel |
| [`quantization/`](quantization/) | 对称/非对称、per-channel、per-group、PTQ、QAT、GPTQ 与 AWQ |

### 大模型训练与推理

| 目录 | 内容 |
| --- | --- |
| [`distributed/`](distributed/) | NCCL、Ring AllReduce、DDP、FSDP/ZeRO 及模型并行 |
| [`inference/`](inference/) | KV Cache、Paged Attention 与 Continuous Batching |
| [`llm_engine/`](llm_engine/) | 轻量 LLM 推理引擎实现 |
| [`vllm_study/`](vllm_study/) | vLLM 架构与源码笔记 |

## 快速开始

先确认本机 CUDA 环境：

```bash
nvidia-smi
nvcc --version
```

编译并运行设备查询示例：

```bash
nvcc -O2 examples/device_query.cu -o device_query
./device_query
```

大多数 CUDA 题目使用独立 CMake 工程：

```bash
cmake -S solution_for_leetGPU/matrix_multiply \
      -B solution_for_leetGPU/matrix_multiply/build
cmake --build solution_for_leetGPU/matrix_multiply/build -j
```

Python 专题的依赖和运行方式以各目录 README 为准。

## 仓库约定

- 源码、说明文档和必要的小型测试数据可以提交。
- `build/`、`__pycache__/`、编译产物和 Nsight 报告不提交。
- 专题使用 `README.md` 说明范围、环境、运行方式与预期结果。
- 算子目录使用小写英文命名；相关内容可使用 `01_`、`02_` 前缀排序。

## 环境说明

仓库中的示例覆盖 CUDA C++、Python、PyTorch、Triton、NCCL 等不同技术栈，并非所有目录共享同一套依赖。具体 GPU 架构、CUDA 版本和第三方依赖要求，请查看对应专题文档。
