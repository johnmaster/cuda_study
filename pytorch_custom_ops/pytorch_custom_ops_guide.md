# PyTorch Custom Op 开发完全指南

> 本文档系统性地介绍如何将自己写的 CUDA kernel 接入 PyTorch，覆盖从最基础的 pybind11 绑定
> 到 torch.compile 兼容的全链路知识。每个概念都配有本项目中的对应代码位置。

---

## 目录

1. [总体架构：从 Python 到 GPU](#1-总体架构从-python-到-gpu)
2. [CUDA Kernel 侧：用 torch/extension.h 写 kernel](#2-cuda-kernel-侧用-torchextensionh-写-kernel)
3. [C++ 绑定层：pybind11 与 TORCH_EXTENSION_NAME](#3-c-绑定层pybind11-与-torch_extension_name)
4. [构建方式：JIT vs setup.py](#4-构建方式jit-vs-setuppy)
5. [Python 侧：torch.autograd.Function](#5-python-侧torchautogradunction)
6. [封装为 nn.Module](#6-封装为-nnmodule)
7. [PyTorch Dispatcher 与 ATen 算子注册](#7-pytorch-dispatcher-与-aten-算子注册)
8. [torch.compile / Inductor 兼容性](#8-torchcompile--inductor-兼容性)
9. [调试与性能分析](#9-调试与性能分析)
10. [最佳实践 Checklist](#10-最佳实践-checklist)

---

## 1. 总体架构：从 Python 到 GPU

```
用户 Python 代码
    │
    ▼
nn.Module.forward()
    │
    ▼
torch.autograd.Function        ← save_for_backward / 返回梯度
    │  (forward / backward)
    ▼
C++ binding (pybind11)          ← PYBIND11_MODULE, 类型检查, dispatch
    │
    ▼
CUDA kernel (*.cu)              ← __global__ 函数, shared memory, warp ops
    │
    ▼
GPU 硬件执行
```

**关键原则**：每一层只做一件事——

| 层 | 职责 |
|---|------|
| `nn.Module` | 对外接口，管理参数 |
| `autograd.Function` | 定义前向/反向的计算图节点 |
| `binding.cpp` | Python ↔ C++ 的桥梁，做输入校验 |
| `kernel.cu` | 纯粹的 GPU 计算逻辑 |

---

## 2. CUDA Kernel 侧：用 torch/extension.h 写 kernel

### 2.1 头文件

```cpp
#include <torch/extension.h>   // 一个头文件包含了：
                                //   - torch/torch.h (Tensor API)
                                //   - pybind11 headers
                                //   - ATen 相关
#include <cuda.h>
#include <cuda_runtime.h>
```

不需要手动 include pybind11——`torch/extension.h` 已经包含了。

### 2.2 使用 torch::Tensor

在 host 函数中，直接使用 `torch::Tensor` 作为参数和返回值：

```cpp
torch::Tensor my_forward_cuda(torch::Tensor input) {
    // 输入校验
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "need float32");

    // 创建输出 tensor（自动在同一设备、同一 dtype）
    auto output = torch::empty_like(input);

    // 提取裸指针传给 kernel
    float* in_ptr  = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    // launch kernel ...

    return output;
}
```

**本项目示例**：`01_custom_gelu/csrc/custom_gelu_kernel.cu` 中的 `custom_gelu_forward_cuda`。

### 2.3 TORCH_CHECK vs assert

- `TORCH_CHECK(condition, message)` — 抛出 Python 可捕获的异常，**生产代码必须用这个**
- `assert` — 仅在 debug 构建有效，不能用于输入校验

### 2.4 返回多个 Tensor

用 `std::vector<torch::Tensor>` 返回，pybind11 会自动转为 Python tuple：

```cpp
std::vector<torch::Tensor> my_forward_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto O = torch::zeros_like(Q);
    auto L = torch::empty({B, N}, Q.options());   // .options() 复制 device + dtype
    // ... launch kernel ...
    return {O, L};
}
```

**本项目示例**：`02_flash_attention/csrc/flash_attn_kernel.cu` 中的 `flash_attn_forward_cuda`。

---

## 3. C++ 绑定层：pybind11 与 TORCH_EXTENSION_NAME

### 3.1 最小绑定文件

```cpp
// binding.cpp
#include <torch/extension.h>

// 声明 CUDA 函数（定义在 .cu 文件中）
torch::Tensor my_forward_cuda(torch::Tensor input);
torch::Tensor my_backward_cuda(torch::Tensor grad_output, torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &my_forward_cuda,  "My op forward  (CUDA)");
    m.def("backward", &my_backward_cuda, "My op backward (CUDA)");
}
```

### 3.2 为什么分两个文件？

| 文件 | 编译器 | 原因 |
|------|--------|------|
| `binding.cpp` | g++ / clang++ | pybind11 不需要 nvcc |
| `kernel.cu` | nvcc | CUDA kernel 必须用 nvcc |

分开可以加速编译——只有 `.cu` 文件需要走 nvcc。

### 3.3 TORCH_EXTENSION_NAME 是什么？

这是一个编译时宏，由 `setup.py` 或 JIT 编译器自动定义，值为 `CUDAExtension(name="xxx")` 中指定的名字。Python 端通过 `import xxx` 加载该模块。

---

## 4. 构建方式：JIT vs setup.py

把 CUDA 扩展编进 Python 可 import 的模块，常见两种：**JIT（`torch.utils.cpp_extension.load`）** 与 **`setup.py` + `pip install`**。二者 **运行时算子语义可以完全相同**，差别主要在 **何时编译、产物放哪、工程流程**。

### 4.1 核心区别（一览）

| 维度 | JIT（`load`） | `setup.py` + `pip install` |
|------|----------------|-----------------------------|
| **何时编译** | 第一次 **import** 用到 `load()` 的模块时（或命中缓存则跳过） | 执行 **`pip install -e .` / `pip install .`** 时 |
| **产物位置** | 多在用户目录缓存，如 **`~/.cache/torch_extensions/`**（路径常含哈希） | 装到当前 **Python 环境**（site-packages 或可编辑链接到项目 `build/`） |
| **首次 import / 启动** | 首编译 **慢**；之后通常较快 | 安装后 **import 快**（已是 `.so`） |
| **改 `.cu` / `.cpp` 后** | 常 **自动检测并重编**（或清缓存后重编） | 需 **重新** `pip install -e .`（或 `build_ext`） |
| **典型场景** | 本地实验、快速迭代 kernel | 发 **wheel**、生产部署、团队统一二进制 |
| **部署** | 需能访问 **源码** 或依赖每台机器上的 **JIT 缓存**（难保证一致） | 只发 **安装包** 即可（版本固定） |
| **常见依赖** | 建议 **`pip install ninja`** 加速编译 | 构建时同样需要 nvcc；**不要求**为运行时而装 ninja |
| **Python 里怎么接** | `custom_ops_cuda = load(name=..., sources=[...])` | **`import custom_ops_cuda`**（模块名与 `CUDAExtension(name=...)` 一致） |

**结论**：开发阶段多用 **JIT**；要打包、复现、上线多用 **`setup.py`**。

---

### 4.2 JIT 编译（开发阶段推荐）

```python
from torch.utils.cpp_extension import load

my_cuda = load(
    name="my_cuda",
    sources=["csrc/binding.cpp", "csrc/kernel.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,   # 第一次建议 True，看编译输出
)
```

- 第一次 import 时编译，之后缓存在 `~/.cache/torch_extensions/`
- 修改 `.cu` 文件后往往会触发重新编译（视缓存键与工具链而定）
- **建议安装 ninja**：`pip install ninja`

---

### 4.3 setup.py（发布阶段推荐）

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_ops",
    ext_modules=[
        CUDAExtension(
            name="my_cuda",
            sources=["csrc/binding.cpp", "csrc/kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

安装：**`pip install -e .`**（可编辑模式，改 C++/CUDA 后需再执行一次）或 **`pip install .`**。

---

### 4.4 本项目 `01_custom_gelu` 怎么用

`custom_gelu.py` 里用注释区分了两种接法（**二选一，不要同时开**）：

| 方式 | `custom_gelu.py` 写法 | 配置要求 |
|------|----------------------|--------------|
| **JIT（当前默认）** | 使用 **`load(...)`**，`# import custom_ops_cuda` 保持注释 | 在 `01_custom_gelu` 下直接 **`python test_custom_gelu.py`**，无需先 pip |
| **setup.py** | **注释掉** 整段 `load(...)`，改为 **`import custom_ops_cuda`** | 在 **`01_custom_gelu`** 目录执行 **`pip install -e .`**，再运行测试 |

**`test_custom_gelu.py` 本身不写构建逻辑**，只 `from custom_gelu import CustomGELU`，因此：

- **默认仓库状态**下测试走的是 **JIT**（与 `custom_gelu.py` 当前实现一致）。

**setup.py 方式命令示例**：

```bash
cd pytorch_custom_ops/01_custom_gelu
pip install -e .
python test_custom_gelu.py
```

---

### 4.5 两种方式对比（简表，与 4.1 呼应）

| | JIT (`load`) | `setup.py` |
|---|---|---|
| 首次延迟 | 高（当场编译） | 安装时编译；之后 import 低延迟 |
| 修改后重编译 | 多自动 | 需重新 `pip install -e .` |
| 适用场景 | 开发/调试 | 发布/部署/固定环境 |
| 依赖 | 建议 ninja | 构建时需 nvcc + setuptools |

---

## 5. Python 侧：torch.autograd.Function

这是将 CUDA kernel 接入 PyTorch 自动微分系统的**核心机制**。

### 5.1 基本结构

```python
class MyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # 1. 保存 backward 需要的张量
        ctx.save_for_backward(input)

        # 2. 调用 C++ 扩展
        output = my_cuda.forward(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 1. 取出保存的张量
        (input,) = ctx.saved_tensors

        # 2. 调用 C++ 反向 kernel
        grad_input = my_cuda.backward(grad_output.contiguous(), input)

        # 3. 返回与 forward 参数一一对应的梯度
        return grad_input
```

### 5.2 关键规则

| 规则 | 说明 |
|------|------|
| `forward` 参数的个数 = `backward` 返回值的个数 | 每个 forward 输入都对应一个梯度（不需要梯度的返回 `None`） |
| `ctx.save_for_backward()` 只能保存 Tensor | 非 Tensor 用 `ctx.my_attr = value` |
| `grad_output` 可能不连续 | 调用 `.contiguous()` 再传给 kernel |
| backward 中不要调用 `.backward()` | 会导致无限递归 |

### 5.3 Flash Attention 的 save_for_backward 策略

普通 Attention 的做法：保存 N×N 的注意力矩阵 P → O(N²) 内存

Flash Attention 的做法：
```python
def forward(ctx, Q, K, V):
    O, L = flash_attn_cuda.forward(Q, K, V)
    ctx.save_for_backward(Q, K, V, O, L)    # 保存 O(N) 的 logsumexp，不存 P
    return O

def backward(ctx, dO):
    Q, K, V, O, L = ctx.saved_tensors
    # backward kernel 从 Q, K, L 重新算出 P（分 tile），不需要完整 P 矩阵
    dQ, dK, dV = flash_attn_cuda.backward(dO, Q, K, V, O, L)
    return dQ, dK, dV
```

**省了什么？** `P` 的大小是 `[B, N, N]`，而 `L` 只有 `[B, N]` —— 对 N=4096，省了 ~64MB/head。

**本项目示例**：`02_flash_attention/flash_attention_op.py`。

---

## 6. 封装为 nn.Module

```python
class FlashAttention(torch.nn.Module):
    def forward(self, Q, K, V):
        return FlashAttentionFunction.apply(Q, K, V)
```

为什么用 `.apply()` 而不是直接调用 `FlashAttentionFunction.forward()`？
- `.apply()` 会正确地设置 autograd 计算图
- 直接调用 `.forward()` 不会注册反向传播

如果算子有可学习参数（如 bias），在 `__init__` 中用 `nn.Parameter` 定义。

---

## 7. PyTorch Dispatcher 与 ATen 算子注册

### 7.1 什么是 Dispatcher？

PyTorch 的所有算子（`torch.add`、`torch.mm` 等）都通过一个统一的调度系统（Dispatcher）分发。
当你调用 `torch.add(a, b)` 时，Dispatcher 根据输入的 **dispatch key**（CPU? CUDA? Autograd?
XLA?）选择对应的 kernel 实现。

```
torch.add(a, b)
    │
    ▼
Dispatcher 查找 dispatch key → CUDA
    │
    ▼
at::cuda::add_kernel(...)
```

### 7.2 三种注册 custom op 的方式

| 方式 | 适用场景 | 复杂度 |
|------|---------|--------|
| pybind11 (`PYBIND11_MODULE`) | 小型项目 | 低 |
| `torch.library` (Python API, PyTorch 2.4+) | 需要 torch.compile 兼容 | 中 |
| `TORCH_LIBRARY` (C++ API) | 大型项目、完整 dispatcher 集成 | 高 |

### 7.3 pybind11 方式（本项目使用的方式）

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &my_forward, "description");
}
```

- 优点：最简单，5 行搞定
- 缺点：不走 PyTorch Dispatcher，`torch.compile` 可能无法追踪

### 7.4 torch.library 方式（推荐的现代方式）

```python
import torch

# 第一步：定义算子签名
torch.library.define("mylib::flash_attn", "(Tensor Q, Tensor K, Tensor V) -> Tensor")

# 第二步：注册 CUDA 实现
@torch.library.impl("mylib::flash_attn", "cuda")
def flash_attn_cuda_impl(Q, K, V):
    return flash_attn_cuda.forward(Q, K, V)[0]

# 第三步（可选）：注册 FakeTensor 实现（给 torch.compile 用）
@torch.library.impl_abstract("mylib::flash_attn")
def flash_attn_fake(Q, K, V):
    return torch.empty_like(Q)    # 只返回正确的 shape/dtype/device

# 第四步：调用
result = torch.ops.mylib.flash_attn(Q, K, V)
```

这种方式注册的算子：
- 走 PyTorch Dispatcher（可以为 CPU / CUDA / Meta 注册不同实现）
- `torch.compile` 能正确追踪
- `torch.export` 能正确导出

### 7.5 TORCH_LIBRARY 方式（C++ 注册）

```cpp
#include <torch/library.h>

TORCH_LIBRARY(mylib, m) {
    m.def("flash_attn(Tensor Q, Tensor K, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("flash_attn", flash_attn_forward_cuda);
}
```

等价于 Python 的 `torch.library`，但在 C++ 层做。适合大型项目（如 FlashAttention 官方库）。

---

## 8. torch.compile / Inductor 兼容性

### 8.1 问题

使用 `pybind11` + `autograd.Function` 时，`torch.compile` 遇到该 op 会：
- 默认行为：graph break（跳出编译图，回退到 eager 模式）
- 后果：编译无法覆盖该 op，失去 fusion 等优化机会

### 8.2 解决方案：torch.library + custom_op

```python
# PyTorch 2.4+ 推荐方式
@torch.library.custom_op("mylib::flash_attn", mutates_args=())
def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return flash_attn_cuda.forward(Q, K, V)[0]

@flash_attn.register_fake
def flash_attn_fake(Q, K, V):
    return torch.empty_like(Q)

# 注册 autograd
def flash_attn_setup_context(ctx, inputs, output):
    Q, K, V = inputs
    ctx.save_for_backward(Q, K, V, output)

def flash_attn_backward(ctx, grad_output):
    Q, K, V, O = ctx.saved_tensors
    # ... compute grads
    return dQ, dK, dV

flash_attn.register_autograd(flash_attn_backward, setup_context=flash_attn_setup_context)
```

### 8.3 集成级别

```
Level 0: pybind11 + autograd.Function    → 能用，torch.compile 会 graph break
Level 1: + @torch.compiler.allow_in_graph → 告诉 compile 别 break，但不安全
Level 2: torch.library.custom_op          → 正确集成，compile 能完全追踪
```

Level 0 支持基本调用；Level 2 提供 `torch.compile` 所需的完整追踪能力。

---

## 9. 调试与性能分析

### 9.1 正确性验证

```python
# 方法 1：与 PyTorch 参考实现对比
O_custom = my_attn(Q, K, V)
O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
assert (O_custom - O_ref).abs().max() < 1e-3

# 方法 2：gradcheck（数值梯度 vs 解析梯度）
from torch.autograd import gradcheck
input = torch.randn(2, 8, 32, device='cuda', dtype=torch.float64, requires_grad=True)
assert gradcheck(MyFunction.apply, (input,), eps=1e-6, atol=1e-4)
```

### 9.2 性能分析

```python
# PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(100):
        attn(Q, K, V)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

命令行工具：
```bash
# Nsight Systems — 整体 timeline
nsys profile python my_test.py

# Nsight Compute — 单个 kernel 深入分析
ncu --set full python my_test.py
```

### 9.3 常见 Bug

| 症状 | 可能原因 |
|------|---------|
| 结果全 0 | `torch::zeros_like` 但 kernel 没写入；grid/block 配置错误 |
| 结果 NaN | 除零；`expf` 溢出；logsumexp 计算错误 |
| backward 梯度不对 | `save_for_backward` 存错了；`grad_output` 没有 `.contiguous()` |
| 段错误 | 越界访问；shared memory 分配不够 |
| 编译报错 | `.cu` 里不能用 C++17 特性；`torch/extension.h` 版本不匹配 |

---

## 10. 最佳实践 Checklist

- [ ] kernel 的 host 函数签名用 `torch::Tensor`，不用裸指针
- [ ] 所有输入用 `TORCH_CHECK` 校验（device / dtype / shape / contiguous）
- [ ] `forward` 用 `.contiguous()` 确保输入内存布局
- [ ] `backward` 中 `grad_output` 也要 `.contiguous()`
- [ ] 用 `.options()` 创建辅助 tensor，自动复制 device 和 dtype
- [ ] 用 `AT_DISPATCH_FLOATING_TYPES` 支持多种 dtype（或先只支持 float32）
- [ ] binding.cpp 和 kernel.cu 分开编译
- [ ] 写测试：forward 对比参考实现 + backward gradcheck
- [ ] 用 `torch.cuda.synchronize()` 确保 kernel 执行完再测时间

---

## 附录：本项目示例索引

| 概念 | 示例位置 |
|------|---------|
| 最简单的 elementwise kernel | `01_custom_gelu/csrc/custom_gelu_kernel.cu` |
| pybind11 绑定 | `01_custom_gelu/csrc/binding.cpp` |
| autograd.Function (简单) | `01_custom_gelu/custom_gelu.py` |
| JIT 编译 | `01_custom_gelu/custom_gelu.py` |
| setup.py 构建 | `01_custom_gelu/setup.py` |
| 复杂 kernel (tiling + online softmax) | `02_flash_attention/csrc/flash_attn_kernel.cu` |
| 保存 logsumexp 做 backward | `02_flash_attention/flash_attention_op.py` |
| tile-based backward (recomputation) | `02_flash_attention/csrc/flash_attn_kernel.cu` |
| 正确性 + 性能测试 | `02_flash_attention/test_flash_attention.py` |
