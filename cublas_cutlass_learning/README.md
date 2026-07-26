# cuBLAS 技术笔记

cuBLAS 是 NVIDIA 提供的 GPU BLAS 库，用于高性能向量和矩阵计算。传统接口默认按 **列主序（column-major）** 解释矩阵。

## 1. 基本使用流程

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// 分配显存、复制数据、调用 cuBLAS

cublasDestroy(handle);
```

编译时链接 cuBLAS：

```bash
nvcc example.cu -lcublas -o example
```

实际项目中应检查每个 CUDA、cuBLAS API 的返回值。

## 2. Handle 管理

### `cublasCreate`

```cpp
cublasStatus_t cublasCreate(cublasHandle_t *handle);
```

- `handle`：输出参数，返回 cuBLAS 上下文句柄；后续计算 API 都需要它。

### `cublasDestroy`

```cpp
cublasStatus_t cublasDestroy(cublasHandle_t handle);
```

- `handle`：待销毁的句柄。handle 创建有开销，通常应复用而不是每次计算都创建。

## 3. 基础概念

### API 名称中的类型前缀

- `S`：单精度 `float`，如 `cublasSgemm`。
- `D`：双精度 `double`，如 `cublasDgemm`。
- `C`：单精度复数 `cuComplex`。
- `Z`：双精度复数 `cuDoubleComplex`。
- `H`：半精度 `__half`，部分 API 使用。

### 转置参数 `cublasOperation_t`

- `CUBLAS_OP_N`：不转置。
- `CUBLAS_OP_T`：转置。
- `CUBLAS_OP_C`：共轭转置；对于实数等价于转置。

### cuBLAS 默认采用列主序

传统 cuBLAS 的 BLAS 接口默认按照 **列主序（column-major）** 解释矩阵内存，这与 Fortran 一致。cuBLAS 不会在显存中保存“行主序/列主序”的标签；所谓默认列主序，是指 `cublasSgemm` 等 API 根据列主序规则计算元素地址。

假设逻辑矩阵为：

```text
A = [1 2 3
     4 5 6]
```

它有 2 行 3 列。连续列主序存储依次保存每一列：

```text
内存：[1, 4, 2, 5, 3, 6]
```

列主序元素地址为：

```cpp
A[row + col * lda]
```

如果矩阵连续存储且没有 padding，`lda` 等于矩阵行数，因此本例中 `lda=2`。

普通 C/C++ 二维数组通常采用 **行主序（row-major）**，同一矩阵的内存为：

```text
内存：[1, 2, 3, 4, 5, 6]
```

其元素地址为：

```cpp
A[row * lda + col]
```

对于连续行主序矩阵，这里的行跨度 `lda` 通常等于列数。需要注意：传统 cuBLAS 参数中的 leading dimension 仍按其列主序矩阵视角解释，不能直接套用 C/C++ 行主序的地址公式。

#### 用传统 cuBLAS 计算行主序 GEMM

要计算行主序矩阵：

```text
C(M×N) = A(M×K) × B(K×N)
```

不一定要真的转置数据。因为同一块行主序内存在 cuBLAS 看来相当于一个列主序的转置矩阵，可以利用：

```text
(A × B)^T = B^T × A^T
```

交换 A、B 的传入顺序，并交换结果的 M、N：

```cpp
float alpha = 1.0f;
float beta = 0.0f;

cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N);
```

此时 cuBLAS 实际看到的是：

```text
C^T(N×M) = B^T(N×K) × A^T(K×M)
```

结果内存不需要再转置：cuBLAS 眼中的列主序 `C^T` 与程序需要的行主序 `C` 使用完全相同的线性内存排列。

#### cuBLASLt 显式指定行主序

cuBLASLt 的矩阵布局描述符可以明确指定行主序，无需通过交换 A、B 来适配。创建 layout 后，通过 `CUBLASLT_MATRIX_LAYOUT_ORDER` 设置 `CUBLASLT_ORDER_ROW`：

```cpp
cublasLtMatrixLayout_t aDesc;

// 行主序 A 的逻辑形状是 M×K，连续存储时行跨度为 K
cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_32F, M, K, K);

cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
cublasLtMatrixLayoutSetAttribute(
    aDesc,
    CUBLASLT_MATRIX_LAYOUT_ORDER,
    &order,
    sizeof(order));
```

`B`、`C`、`D` 的 layout 也要分别设置对应的 order、逻辑形状和 leading dimension。简要结论如下：

| API | 矩阵布局方式 |
| --- | --- |
| 传统 `cublasSgemm`、`cublasGemmEx` | 按列主序解释；行主序输入通常通过交换 A/B 与 M/N 适配 |
| `cublasLtMatmul` | 通过 matrix layout 描述符显式设置行主序或列主序 |

### Leading dimension

`lda`、`ldb`、`ldc` 表示矩阵相邻两列起始地址之间相差的元素数，即“物理行跨度”，并不是元素总数。

对于连续存储的列主序矩阵：

- `m × n` 矩阵的 leading dimension 通常为 `m`。
- GEMM 中 `A` 不转置时通常要求 `lda >= m`，转置时通常要求 `lda >= k`。
- leading dimension 也可用于带 padding 的矩阵或大矩阵中的子矩阵。

## 4. 矩阵乘法：GEMM

### `cublasSgemm` / `cublasDgemm`

计算公式：`C = alpha * op(A) * op(B) + beta * C`。

`op(A)` 的尺寸是 `m × k`，`op(B)` 是 `k × n`，结果 `C` 是 `m × n`。

```cpp
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);
```

- `handle`：cuBLAS 句柄。
- `transa`：是否对 `A` 转置或共轭转置。
- `transb`：是否对 `B` 转置或共轭转置。
- `m`：结果矩阵 `C` 的行数。
- `n`：结果矩阵 `C` 的列数。
- `k`：矩阵乘法的归约维度。
- `alpha`：矩阵乘积的缩放系数。
- `A`：设备端矩阵 `A` 的首地址。
- `lda`：`A` 相邻两列起始地址之间相差的元素个数。
- `B`：设备端矩阵 `B` 的首地址。
- `ldb`：`B` 相邻两列起始地址之间相差的元素个数。
- `beta`：原矩阵 `C` 的缩放系数；不保留旧值时通常设为 `0`。
- `C`：设备端输入/输出矩阵 `C`。
- `ldc`：`C` 相邻两列起始地址之间相差的元素个数，通常至少为 `m`。

#### `lda`、`ldb`、`ldc` 到底是什么

`ld` 是 **leading dimension** 的缩写，可以理解成矩阵在内存中的“主维度跨度”。传统 cuBLAS 按列主序解释矩阵，所以这里具体表示：

> 从当前列的第 0 个元素移动到下一列的第 0 个元素，需要跨过多少个元素。

对于列主序矩阵，元素地址计算公式为：

```cpp
matrix[row + col * ld]
```

因此：

```cpp
A(row, col) = A[row + col * lda];
B(row, col) = B[row + col * ldb];
C(row, col) = C[row + col * ldc];
```

leading dimension 不是：

- 矩阵元素总数。
- 矩阵列数。
- 字节数。
- 一定等于矩阵的逻辑行数。

它的单位是**元素个数**。对于没有 padding 的连续列主序矩阵，它通常恰好等于物理矩阵的行数，但两者并不始终相等。

例如一个连续存储的 `3×2` 列主序矩阵：

```text
A = [a00 a01
     a10 a11
     a20 a21]

内存顺序：a00, a10, a20, a01, a11, a21
lda = 3
```

第一列从 `A[0]` 开始，第二列从 `A[3]` 开始，因此两列首地址相差 3 个元素。

如果为了对齐，在每列后面加入两个 padding 元素：

```text
第 0 列：a00, a10, a20, padding, padding
第 1 列：a01, a11, a21, padding, padding

内存顺序：a00, a10, a20, x, x, a01, a11, a21, x, x
lda = 5
```

逻辑矩阵仍然只有 3 行，但下一列从 `A[5]` 开始，所以 `lda=5`。这说明 leading dimension 描述的是**物理内存跨度**，而不是单纯的逻辑形状。

#### GEMM 中应该怎样填写 `lda`、`ldb`、`ldc`

先看转置前、实际存放在内存中的物理矩阵形状：

| 参数 | `CUBLAS_OP_N` | `CUBLAS_OP_T` / `CUBLAS_OP_C` | 连续列主序时的 ld |
| --- | --- | --- | --- |
| `A` | 物理形状为 `m×k` | 物理形状为 `k×m` | 不转置时 `lda=m`；转置时 `lda=k` |
| `B` | 物理形状为 `k×n` | 物理形状为 `n×k` | 不转置时 `ldb=k`；转置时 `ldb=n` |
| `C` | 始终为 `m×n` | 不适用 | `ldc=m` |

更严格地说，通常应满足：

```text
transa == CUBLAS_OP_N  -> lda >= max(1, m)
transa != CUBLAS_OP_N  -> lda >= max(1, k)

transb == CUBLAS_OP_N  -> ldb >= max(1, k)
transb != CUBLAS_OP_N  -> ldb >= max(1, n)

ldc >= max(1, m)
```

这里说的是“至少”，因为实际矩阵可能包含对齐 padding，或 A/B/C 只是某个更大矩阵中的子矩阵。这时必须填写真实的物理列跨度，不能只填写子矩阵行数。

#### 为什么 cuBLAS 必须让调用者传入 ld

如果 API 只知道 `m/n/k`，它只能知道需要计算的逻辑矩阵大小，却不知道矩阵在显存中的真实排布。显式传入 `lda/ldb/ldc` 后，cuBLAS 才能支持：

- 连续存储的普通矩阵。
- 每列带 padding、为了内存对齐的矩阵。
- 大矩阵中的子矩阵视图，而不必额外复制数据。
- 使用相同底层缓冲区表示不同逻辑区域。

最常见的无 padding、均不转置情况可以直接记为：

```text
A: m×k -> lda = m
B: k×n -> ldb = k
C: m×n -> ldc = m
```

列主序连续矩阵示例：

```cpp
float alpha = 1.0f;
float beta = 0.0f;

cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M);
```

### 行主序矩阵的处理

若内存中是行主序 `A(M×K)`、`B(K×N)`，可以利用 `(AB)^T = B^T A^T`，交换输入顺序与 `m/n`：

```cpp
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N);
```

此时 cuBLAS 将行主序 `C(M×N)` 的内存看成列主序 `C^T(N×M)`。

### `cublasGemmEx`

支持分别指定输入、输出和计算精度，也可使用 Tensor Core：

```cpp
cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const void *beta,
    void *C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);
```

尺寸、矩阵及 leading dimension 参数与 `cublasSgemm` 相同；新增参数如下：

- `Atype`：`A` 的存储类型，如 `CUDA_R_16F`、`CUDA_R_32F`。
- `Btype`：`B` 的存储类型。
- `Ctype`：`C` 的存储类型。
- `computeType`：乘加使用的计算类型，如 `CUBLAS_COMPUTE_32F` 或 `CUBLAS_COMPUTE_32F_FAST_TF32`。
- `algo`：算法选择；通用默认值为 `CUBLAS_GEMM_DEFAULT`。

#### `computeType`：内部到底用什么精度计算

`Atype/Btype/Ctype` 描述矩阵在显存中的**存储类型**，而 `computeType` 描述 GEMM 内部乘法、累加以及允许使用的硬件计算路径。两者不是一回事。

例如：

```text
A/B 存储为 FP16
C   存储为 FP32
computeType = CUBLAS_COMPUTE_32F
```

其含义大致是：读取 FP16 的 A/B，将乘积以该模式支持的路径计算，并使用至少 FP32 范围和精度的内部累加，最后写入 FP32 的 C。这样可以减少 A/B 的显存占用和带宽，同时避免使用 FP16 长链累加带来的严重误差。

常用 `computeType`：

| `computeType` | 主要含义 | 常见用途 |
| --- | --- | --- |
| `CUBLAS_COMPUTE_16F` | 使用 FP16 计算模式，内部精度和范围较低 | 对精度要求较低的纯 FP16 计算 |
| `CUBLAS_COMPUTE_32F` | FP32 计算模式；允许库使用符合该模式的高性能实现 | FP16/BF16 输入、FP32 累加，或标准 FP32 GEMM |
| `CUBLAS_COMPUTE_32F_PEDANTIC` | 更严格遵循指定的 FP32 算术行为，限制某些为了性能采用的替代路径 | 更看重可预期数值行为而非峰值性能 |
| `CUBLAS_COMPUTE_32F_FAST_TF32` | 允许把 FP32 输入按 TF32 精度进入 Tensor Core，通常仍以 FP32 累加 | Ampere 及以后 GPU 上加速可容忍一定误差的 FP32 GEMM |
| `CUBLAS_COMPUTE_32F_FAST_16F` | 允许将输入转换到 FP16 等更快路径进行计算 | 能接受更明显精度损失、追求吞吐的场景 |
| `CUBLAS_COMPUTE_32F_FAST_16BF` | 允许使用 BF16 等更快路径；BF16 动态范围接近 FP32，但尾数更短 | BF16 训练或推理 |
| `CUBLAS_COMPUTE_64F` | FP64 计算模式 | 高精度科学计算 |
| `CUBLAS_COMPUTE_32I` | 整数乘法、INT32 累加模式 | 支持的数据类型组合下的 INT8 GEMM |

`PEDANTIC`、默认和 `FAST` 可以这样理解：

```text
PEDANTIC：更严格遵守指定算术模式，通常牺牲部分性能
DEFAULT ：在该计算类型保证范围/精度要求的前提下选择高性能路径
FAST    ：允许降低输入乘法精度，以换取更高 Tensor Core 吞吐
```

##### TF32 为什么仍然叫 `COMPUTE_32F`

TF32 使用 FP32 的指数范围，但乘法输入的有效尾数比完整 FP32 少。Tensor Core 完成低精度乘法并进行 FP32 累加，因此它通常比标准 FP32 路径快，但结果不保证与严格 FP32 GEMM 逐位一致。

```cpp
// 更注重普通 FP32 数值精度
computeType = CUBLAS_COMPUTE_32F;

// 明确允许 TF32 Tensor Core 路径换取速度
computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
```

对于误差敏感的科学计算，应使用 `CUBLAS_COMPUTE_32F`、`CUBLAS_COMPUTE_32F_PEDANTIC` 或 FP64，并通过实际误差测试验证；对于深度学习推理和训练，TF32、FP16 或 BF16 往往能提供更高吞吐。

##### `alpha`、`beta` 的类型也必须匹配

`alpha` 和 `beta` 虽然声明为 `const void*`，并不表示可以传入任意类型。cuBLAS 会根据 `computeType` 和受支持的数据类型组合解释这两个地址。

最常见的 FP32 计算模式使用：

```cpp
float alpha = 1.0f;
float beta = 0.0f;
```

如果错误地传入 `double*` 或其他不匹配的指针，编译器因为参数是 `void*` 可能不会报错，但 cuBLAS 会按错误的二进制格式读取标量，导致结果错误。选择具体类型组合时，应同时确认 A/B/C 类型、`computeType` 和标量类型是受支持的组合。

#### `algo`：选择哪一种内部 GEMM 实现

同一个数学公式可以有多种 GPU 实现。例如内部算法可以采用不同的：

- thread-block tile、warp tile 和每线程计算区域。
- shared memory 使用量和流水线 stage 数量。
- K 维切分与归约方式。
- SIMT Core 或 Tensor Core 指令路径。
- 对齐、向量化加载和矩阵尺寸约束。

`algo` 的作用就是告诉 cuBLAS：这次 `cublasGemmEx` 应优先采用哪个算法系列。

##### 最常用选择：`CUBLAS_GEMM_DEFAULT`

```cpp
cublasGemmEx(/* ... */,
             computeType,
             CUBLAS_GEMM_DEFAULT);
```

`CUBLAS_GEMM_DEFAULT` 表示不手工指定某个编号，让 cuBLAS 根据数据类型、尺寸、转置、对齐和 GPU 架构选择兼容实现。它通常是最稳妥的起点，也最容易跨不同 GPU 和 CUDA 版本运行。

`DEFAULT` 不等于“永远选择全局最快算法”，而是让库使用默认的内部选择逻辑。对固定 shape 的极致优化仍需基准测试。

##### `CUBLAS_GEMM_ALGO0` 到 `CUBLAS_GEMM_ALGO23`

这些枚举用于请求特定的传统算法变体：

```cpp
CUBLAS_GEMM_ALGO0
CUBLAS_GEMM_ALGO1
// ...
CUBLAS_GEMM_ALGO23
```

数字只表示 cuBLAS 定义的算法编号，不能简单理解为“编号越大越快”。某个编号可能：

- 只支持部分数据类型或转置组合。
- 对指针对齐、leading dimension 或矩阵形状有要求。
- 在某个 GPU 架构上较快，在另一个架构上较慢。
- 对当前配置不受支持并返回错误。
- 随 CUDA 版本变化而表现不同。

因此不应该在没有 benchmark 的情况下随意把默认值改成某个数字算法。

##### `*_TENSOR_OP` 算法枚举

`CUBLAS_GEMM_DEFAULT_TENSOR_OP` 和 `CUBLAS_GEMM_ALGO*_TENSOR_OP` 用于较早期的显式 Tensor Core 算法选择方式。现代代码更应通过输入类型和 `computeType` 表达数值需求，让默认算法选择兼容路径；需要系统搜索算法时，优先使用 cuBLASLt heuristic。

是否真正使用 Tensor Core 不只由 `algo` 决定，还取决于：

```text
GPU 架构
+ A/B/C 数据类型
+ computeType
+ M/N/K 和 leading dimension
+ 地址对齐与转置方式
+ 所选算法是否支持
```

##### 为什么更复杂的调优推荐 cuBLASLt

传统 `cublasGemmEx` 的 `algo` 只是一个枚举，不能完整表达 workspace 上限、tile、stage、split-K 和 epilogue 等现代 Matmul 配置。

cuBLASLt 的流程更适合调优：

```text
描述计算和布局
  -> 设置可用 workspace
  -> cublasLtMatmulAlgoGetHeuristic 返回兼容候选
  -> 对多个候选实际计时
  -> 缓存固定 shape 的最佳算法
```

##### 实际选择建议

| 需求 | `computeType` 建议 | `algo` 建议 |
| --- | --- | --- |
| 初次写通 FP32 GEMM | `CUBLAS_COMPUTE_32F` | `CUBLAS_GEMM_DEFAULT` |
| FP16 输入、FP32 累加 | `CUBLAS_COMPUTE_32F` | `CUBLAS_GEMM_DEFAULT` |
| FP32 深度学习任务允许 TF32 | `CUBLAS_COMPUTE_32F_FAST_TF32` | `CUBLAS_GEMM_DEFAULT` |
| 数值行为要求更严格 | 对应的 `*_PEDANTIC` | 从 `CUBLAS_GEMM_DEFAULT` 开始 |
| 固定 shape 追求极致性能 | 先确定可接受的计算精度 | 使用 cuBLASLt heuristic 并实测候选 |

固定 `algo=CUBLAS_GEMM_DEFAULT` 可单独比较存储类型和 `computeType` 对精度、性能的影响；cuBLASLt 则提供进一步的算法搜索能力。

## 5. 矩阵向量乘：GEMV

### `cublasSgemv` / `cublasDgemv`

计算公式：`y = alpha * op(A) * x + beta * y`。

```cpp
cublasStatus_t cublasSgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta,
    float *y, int incy);
```

- `handle`：cuBLAS 句柄。
- `trans`：矩阵 `A` 是否转置。
- `m`、`n`：原始矩阵 `A` 的行数、列数。
- `alpha`：矩阵向量乘结果的缩放系数。
- `A`：设备端列主序矩阵。
- `lda`：`A` 的 leading dimension，通常至少为 `m`。
- `x`：输入向量。
- `incx`：`x` 相邻逻辑元素的步长，连续向量通常为 `1`。
- `beta`：原向量 `y` 的缩放系数。
- `y`：输入/输出向量。
- `incy`：`y` 相邻逻辑元素的步长。

## 6. 常用向量 API（BLAS Level 1）

下列接口以单精度版本为例。`n` 是参与运算的元素数，`incx`/`incy` 是相邻逻辑元素的步长。

### `cublasSaxpy`

**目的：把一个经过缩放的向量加到另一个向量上，并将结果直接写回 `y`。**

计算 `y = alpha * x + y`：

```text
x     = [1, 2, 3]
y     = [4, 5, 6]
alpha = 2

y = 2 * x + y = [6, 9, 12]
```

名称中的 `S` 表示单精度 `float`，`AXPY` 来自 **A·X Plus Y**。这个操作看起来简单，但它是许多数值算法的基本步骤，例如：

- 梯度下降更新：`weights = weights - learning_rate * gradient`。此时令 `y=weights`、`x=gradient`、`alpha=-learning_rate`。
- 向量线性组合与残差更新：`residual += alpha * direction`。
- 迭代求解器中更新解向量或搜索方向。
- 在 GPU 上对大向量执行融合的“缩放加法”，避免先启动一个缩放 kernel，再启动一个加法 kernel。

```cpp
cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                           const float *alpha,
                           const float *x, int incx,
                           float *y, int incy);
```

- `handle`：cuBLAS 句柄。
- `n`：向量长度。
- `alpha`：`x` 的缩放系数。
- `x`、`y`：输入向量与输入/输出向量。
- `incx`、`incy`：两个向量的步长。

完整调用示例：

```cpp
int n = 3;
float alpha = 2.0f;
// d_x = [1, 2, 3]，d_y = [4, 5, 6]

cublasSaxpy(handle, n,
            &alpha,
            d_x, 1,
            d_y, 1);

// 执行完成后 d_y = [6, 9, 12]，d_x 不变
```

这里 `incx=1`、`incy=1` 表示连续访问。如果 `incx=2`，cuBLAS 会依次使用 `x[0]、x[2]、x[4]...`。`cublasSaxpy` 不分配结果内存，而且会修改 `y`，因此调用前 `y` 必须已经包含有效数据。

### `cublasSdot`

**目的：衡量两个向量的对应元素乘积之和。** 点积常用于计算向量相似度、投影、神经网络中的加权求和，以及判断两个向量是否接近正交。

计算点积 `result = x^T y`：

```text
x = [1, 2, 3], y = [4, 5, 6]
result = 1*4 + 2*5 + 3*6 = 32
```

```cpp
cublasStatus_t cublasSdot(cublasHandle_t handle, int n,
                          const float *x, int incx,
                          const float *y, int incy,
                          float *result);
```

- `x`、`y`：输入向量。
- `result`：点积结果，所在位置取决于 pointer mode。
- 其余参数含义同上。

### `cublasSnrm2`

**目的：计算向量的欧几里得长度。** 常用于向量归一化、梯度裁剪、误差度量和迭代算法的收敛判断。例如得到 `norm` 后，再使用 `cublasSscal` 乘以 `1/norm`，即可把向量归一化。

计算二范数 `sqrt(sum(x[i]^2))`：

```cpp
cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n,
                           const float *x, int incx,
                           float *result);
```

- `result`：范数结果，位置取决于 pointer mode。

### `cublasSasum`

**目的：计算向量所有元素绝对值之和，也就是 L1 范数。** 常用于稀疏正则化、误差统计，以及快速衡量向量整体幅度。注意它不是普通元素求和，负数会先取绝对值。

计算 `sum(abs(x[i]))`：

```cpp
cublasStatus_t cublasSasum(cublasHandle_t handle, int n,
                           const float *x, int incx,
                           float *result);
```

### `cublasSscal`

**目的：用一个标量原地缩放整个向量。** 常用于归一化、学习率缩放、单位换算或衰减。例如 `alpha=0.5` 会把 `x` 中所有参与运算的元素减半。

原地计算 `x = alpha * x`：

```cpp
cublasStatus_t cublasSscal(cublasHandle_t handle, int n,
                           const float *alpha,
                           float *x, int incx);
```

### `cublasIsamax` / `cublasIsamin`

**目的：寻找向量中绝对值最大或最小的元素位置。** `Isamax` 可用于查找最大误差、最大梯度或异常值；`Isamin` 可用于寻找幅度最接近零的元素。它们比较的是绝对值，而不是带符号的原值。

返回绝对值最大/最小元素的索引：

```cpp
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n,
                            const float *x, int incx,
                            int *result);
```

- `result`：索引结果。遵循 BLAS 传统，该索引通常 **从 1 开始**。

### `cublasScopy` / `cublasSswap`

**目的：在设备端复制或交换两个向量。** `Scopy` 用于保存向量副本或复制带步长的切片；`Sswap` 用于无需额外临时向量地交换两块向量数据，例如某些分解和排序步骤。

分别复制向量 `y = x` 和交换向量 `x`、`y`：

```cpp
cublasStatus_t cublasScopy(cublasHandle_t handle, int n,
                           const float *x, int incx,
                           float *y, int incy);

cublasStatus_t cublasSswap(cublasHandle_t handle, int n,
                           float *x, int incx,
                           float *y, int incy);
```

## 7. 批量矩阵乘法

批量 GEMM 用于一次提交多组相互独立、形状相同的矩阵乘法：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
i = 0, 1, ..., batchCount - 1
```

如果用 CPU 循环调用 `cublasSgemm`：

```cpp
for (int i = 0; i < batchCount; ++i) {
    cublasSgemm(/* 第 i 组矩阵 */);
}
```

每次调用都要经过 CPU、CUDA runtime/driver 和 kernel 调度。矩阵很小时，提交开销可能占据明显比例。Batched API 用一次调用描述整批工作，让 cuBLAS 更集中地调度这些小矩阵，提高 GPU 利用率。

需要先明确：这里的 batch 表示多次**独立 GEMM**，并不是把矩阵在数学上拼成一个大矩阵。所有 batch 共用同一组 `m/n/k`、转置方式、`alpha/beta` 和 leading dimension。

### `cublasSgemmBatched`

**目的：处理形状相同、但地址不规则或彼此不连续的一批矩阵。** 它通过三个“指针数组”分别找到每一组 A、B、C。

地址关系为：

```text
第 i 次 GEMM 使用：
A_i = Aarray[i]
B_i = Barray[i]
C_i = Carray[i]
```

```cpp
cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *const Aarray[], int lda,
    const float *const Barray[], int ldb,
    const float *beta,
    float *const Carray[], int ldc,
    int batchCount);
```

- `handle`：cuBLAS handle。
- `transa/transb`：对整批 A/B 统一采用的转置方式。
- `m/n/k`：每一次 GEMM 的统一逻辑尺寸；不能让不同 batch 使用不同尺寸。
- `alpha/beta`：整批共享的缩放系数。
- `Aarray`：包含 `batchCount` 个设备矩阵地址的指针数组；`Aarray[i]` 指向 `A_i`。
- `Barray`：同理，`Barray[i]` 指向 `B_i`。
- `Carray`：同理，`Carray[i]` 指向输入/输出矩阵 `C_i`。
- `lda/ldb/ldc`：整批统一使用的物理列跨度。
- `batchCount`：独立 GEMM 的数量。

#### 指针数组和矩阵数据是两层不同的内存

以 A 为例：

```text
设备端指针数组 d_Aarray
  ├── d_Aarray[0] -> 设备矩阵 A0
  ├── d_Aarray[1] -> 设备矩阵 A1
  ├── d_Aarray[2] -> 设备矩阵 A2
  └── ...
```

经典 `cublasSgemmBatched` 调用中，矩阵位于设备内存，传给 API 的 pointer array 也需要准备成设备可访问的指针数组。常见做法是先在主机端构造设备地址列表，再复制到设备：

```cpp
std::vector<float*> h_Aarray(batchCount);
std::vector<float*> h_Barray(batchCount);
std::vector<float*> h_Carray(batchCount);

for (int i = 0; i < batchCount; ++i) {
    h_Aarray[i] = d_A[i];  // d_A[i] 本身是某个设备矩阵地址
    h_Barray[i] = d_B[i];
    h_Carray[i] = d_C[i];
}

float **d_Aarray, **d_Barray, **d_Carray;
cudaMalloc(&d_Aarray, batchCount * sizeof(float*));
cudaMalloc(&d_Barray, batchCount * sizeof(float*));
cudaMalloc(&d_Carray, batchCount * sizeof(float*));

cudaMemcpy(d_Aarray, h_Aarray.data(),
           batchCount * sizeof(float*), cudaMemcpyHostToDevice);
cudaMemcpy(d_Barray, h_Barray.data(),
           batchCount * sizeof(float*), cudaMemcpyHostToDevice);
cudaMemcpy(d_Carray, h_Carray.data(),
           batchCount * sizeof(float*), cudaMemcpyHostToDevice);
```

然后调用：

```cpp
float alpha = 1.0f;
float beta = 0.0f;

cublasSgemmBatched(handle,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   M, N, K,
                   &alpha,
                   (const float**)d_Aarray, M,
                   (const float**)d_Barray, K,
                   &beta,
                   d_Carray, M,
                   batchCount);
```

这种方式的优势是地址灵活：`A0/A1/A2` 可以来自完全不同的 `cudaMalloc`，也可以指向不同大缓冲区中的不规则位置。代价是需要额外创建、复制和维护指针数组。

#### `cublasSgemmBatched` 适合什么情况

- 各矩阵地址不连续或间隔不固定。
- 已经有现成的设备端矩阵指针数组。
- 每个 batch 的矩阵来自不同内存池或不同对象。
- 矩阵形状相同，但无法用一个固定 stride 描述地址。

它不支持每组完全不同的 `m/n/k`。如果 batch 内形状也不同，需要考虑 grouped GEMM、cuBLASLt 的相应能力或按形状分组调用。

### `cublasSgemmStridedBatched`

正确名称是 `cublasSgemmStridedBatched`，其中是 **Strided**，不是 `cublasSgemmStrideBatched`。

**目的：处理形状相同、并且相邻 batch 地址间隔固定的一批矩阵。** 它不传指针数组，而是传 A/B/C 的三个基地址和固定 stride。

地址由 cuBLAS 自动计算：

```text
A_i = A + i * strideA
B_i = B + i * strideB
C_i = C + i * strideC
```

这里执行的是 C/C++ 指针意义上的元素偏移，所以 stride 单位是**元素个数**，不是字节数。

```cpp
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, long long int strideA,
    const float *B, int ldb, long long int strideB,
    const float *beta,
    float *C, int ldc, long long int strideC,
    int batchCount);
```

- `handle`：cuBLAS handle。
- `transa/transb`：整批统一的转置方式。
- `m/n/k`：每次 GEMM 的统一逻辑尺寸。
- `alpha/beta`：整批共享的缩放系数。
- `A/B/C`：第 0 个 batch 的矩阵基地址。
- `lda/ldb/ldc`：单个矩阵内部从一列移动到下一列的元素跨度。
- `strideA/strideB/strideC`：从当前 batch 矩阵首地址移动到下一 batch 对应矩阵首地址的元素跨度。
- `batchCount`：batch 数量。

#### `ld` 与 `stride` 不要混淆

二者描述不同层级的地址跨度：

```text
lda：同一个 A_i 内部，相邻两列之间的跨度
strideA：A_i 与 A_(i+1) 两个矩阵首地址之间的跨度
```

例如有 10 个连续存放的列主序 `A_i(M×K)`：

```text
单个 A 内部：lda = M
单个 A 元素数：M * K
相邻 A 之间：strideA = M * K
```

内存布局为：

```text
A0 的 M*K 个元素 | A1 的 M*K 个元素 | A2 的 M*K 个元素 | ...
^                  ^
A                  A + strideA
```

#### 无 padding 时怎样计算 stride

stride 应根据**转置前实际存放的物理矩阵**占用多少元素来计算：

| 矩阵 | 不转置时的物理形状 | 转置时的物理形状 | 连续无 padding 时的 stride |
| --- | --- | --- | --- |
| A | `m×k` | `k×m` | 都是 `m*k` |
| B | `k×n` | `n×k` | 都是 `k*n` |
| C | `m×n` | 不转置 | `m*n` |

虽然转置会交换物理行列数，但元素总数不变，所以无 padding 时 stride 仍是行数乘列数。

若每列存在 padding，则不能再直接使用逻辑元素总数。例如 `A` 不转置、物理列跨度为 `lda`、共有 K 列时，最简单且最常用的整槽分配方式是：

```text
strideA = lda * k
```

按照这种“每一列都占满一个 ld 槽位”的分配方式，可使用：

```text
transa == N -> strideA = lda * k
transa != N -> strideA = lda * m

transb == N -> strideB = ldb * n
transb != N -> strideB = ldb * k

strideC = ldc * n
```

这会在矩阵最后一列后也保留 padding，地址计算简单。如果只关心矩阵最后一个有效元素，列主序 `rows×cols` 矩阵的最小内存覆盖范围是：

```text
footprint = ld * (cols - 1) + rows
```

因此，不保留最后一列尾部 padding 时，stride 最小可以等于该 footprint；如果 batch 之间还有额外间隔，则 stride 可以更大。工程中通常使用 `ld*cols`，因为它与 pitched/整槽分配方式一致，也更不容易算错。

#### 完整调用示例

假设 `batchCount` 组列主序矩阵连续无 padding：

```cpp
long long strideA = static_cast<long long>(M) * K;
long long strideB = static_cast<long long>(K) * N;
long long strideC = static_cast<long long>(M) * N;

float alpha = 1.0f;
float beta = 0.0f;

cublasSgemmStridedBatched(handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          M, N, K,
                          &alpha,
                          d_A, M, strideA,
                          d_B, K, strideB,
                          &beta,
                          d_C, M, strideC,
                          batchCount);
```

需要为整个 batch 分配的元素数量为：

```cpp
size_t elementsA = static_cast<size_t>(batchCount) * strideA;
size_t elementsB = static_cast<size_t>(batchCount) * strideB;
size_t elementsC = static_cast<size_t>(batchCount) * strideC;
```

#### `stride=0`：跨 batch 广播只读矩阵

如果所有 batch 共用同一个只读 A 或 B，可以将对应 stride 设为 0：

```text
strideB = 0 -> B_i = B + i*0 = B
```

这表示每一次 GEMM 都读取同一个 B，常见于多组输入共享一份权重。A/B 是只读输入，因此这种重叠可以表达广播；输出 C 通常不能让不同 batch 写入重叠区域，否则会产生数据竞争或未定义结果。

#### `cublasSgemmStridedBatched` 适合什么情况

- batch 数据打包在一块连续大缓冲区中。
- 相邻矩阵之间有固定 padding 或固定间隔。
- Attention 的 batch/head 等维度能够展平成规则 batch。
- 希望避免构造和复制设备端指针数组。
- 相同权重需要在多个 batch 中广播。

### 两种 Batched API 如何选择

| 对比项 | `cublasSgemmBatched` | `cublasSgemmStridedBatched` |
| --- | --- | --- |
| 如何定位矩阵 | `Aarray[i]` 指针 | `A + i*strideA` 计算 |
| 地址是否必须规则 | 不需要 | 必须具有固定间隔 |
| 是否需要指针数组 | 需要 | 不需要 |
| 内存管理复杂度 | 较高 | 较低 |
| 适合的数据组织 | 分散、来自不同分配 | 连续或固定 padding 的大缓冲区 |
| 广播只读矩阵 | 可让多个数组项指向同一地址 | 将对应 stride 设为 0 |

选择原则很简单：如果固定 stride 能描述内存，优先使用 Strided Batched；只有地址不规则时才使用 pointer-array Batched。

### 常见错误

1. 把 `strideA/B/C` 当成字节数；它们的单位是元素。
2. 把 `lda` 和 `strideA` 当成同一个概念；前者跨列，后者跨 batch。
3. 为 A/B/C 只分配单个矩阵的空间，却让 `batchCount > 1`。
4. pointer-array 版本只把主机指针数组直接传入，而没有正确准备 API 所需的设备指针数组。
5. 忽略转置后物理矩阵的行数，导致 `lda/ldb` 填错。
6. 使用 padding 后仍把 stride 写成简单的 `m*k` 或 `k*n`，造成 batch 地址重叠。
7. 让不同 batch 的输出 C 重叠，引起并发写入冲突。
8. 认为 batched 一定比单个大 GEMM 快；它主要优化大量小型独立 GEMM，不能替代可以数学合并的大矩阵乘。

## 8. 数据复制辅助 API

实际工程也常直接使用 `cudaMemcpy` / `cudaMemcpyAsync`。

### `cublasSetVector` / `cublasGetVector`

```cpp
cublasStatus_t cublasSetVector(int n, int elemSize,
                               const void *x, int incx,
                               void *y, int incy);
```

- `n`：元素个数。
- `elemSize`：每个元素的字节数，如 `sizeof(float)`。
- `x`：源地址；`Set` 中通常在主机，`Get` 中通常在设备。
- `incx`：源向量步长。
- `y`：目标地址。
- `incy`：目标向量步长。

`cublasGetVector` 参数含义相同，但复制方向通常是设备到主机。

### `cublasSetMatrix` / `cublasGetMatrix`

```cpp
cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                               const void *A, int lda,
                               void *B, int ldb);
```

- `rows`、`cols`：复制区域的行数、列数。
- `elemSize`：每个元素的字节数。
- `A`、`B`：源矩阵和目标矩阵地址。
- `lda`、`ldb`：源矩阵和目标矩阵的 leading dimension。

`cublasGetMatrix` 参数含义相同，复制方向通常是设备到主机。

## 9. 常见错误与注意事项

1. **混淆行主序与列主序**：结果像被转置时，首先检查数据布局和矩阵传入顺序。
2. **把 `lda` 当成列数**：列主序连续矩阵的 leading dimension 通常是行数。
3. **维度与转置不匹配**：先写出 `op(A)`、`op(B)` 的尺寸，再填写 `m/n/k`。
4. **忽略 `beta` 对 `C` 的读取**：`beta != 0` 时，调用前必须保证 `C` 已初始化。
5. **标量地址放错位置**：默认 pointer mode 下，`alpha`、`beta` 和标量结果位于主机内存。
6. **过早读取或释放内存**：cuBLAS 通常异步执行，读取结果或释放缓冲区前应同步。
7. **频繁创建 handle**：应尽量复用；多线程共享时要留意 handle 状态和线程安全设计。
8. **性能测试包含初始化开销**：正式计时前先 warm-up，并使用 CUDA event 计时。
9. **小矩阵调用开销占比高**：大量小矩阵优先考虑 batched API。
10. **混合精度带来误差差异**：Tensor Core、TF32、FP16/BF16 的精度与 FP32 不同。

## 10. API 选择速查

| 需求 | 常用 API |
| --- | --- |
| 矩阵 × 矩阵 | `cublasSgemm` / `cublasDgemm` |
| 混合精度矩阵乘 | `cublasGemmEx` |
| 矩阵 × 向量 | `cublasSgemv` / `cublasDgemv` |
| `y = alpha*x + y` | `cublasSaxpy` / `cublasDaxpy` |
| 向量点积 | `cublasSdot` / `cublasDdot` |
| 向量二范数 | `cublasSnrm2` / `cublasDnrm2` |
| 向量缩放 | `cublasSscal` / `cublasDscal` |
| 地址不连续的批量矩阵乘 | `cublasSgemmBatched` |
| 固定间隔的批量矩阵乘 | `cublasSgemmStridedBatched` |

## 11. API 覆盖范围

- Level 1：`cublasSaxpy` 等向量操作。
- Level 2：`cublasSgemv` 等矩阵向量操作。
- Level 3：`cublasSgemm`、`cublasGemmEx` 与 batched GEMM。
- 执行控制：handle、stream、CUDA event 与异步计时。
- 数值格式：FP32、FP16、BF16、TF32 与 Tensor Core。

## 12. 九个核心 API 详解

本节集中说明九个核心 API。传统 cuBLAS 的核心对象是 `cublasHandle_t`，而 cuBLASLt 使用独立的 `cublasLtHandle_t` 和多个描述符；两套 handle 不能混用。

### 12.1 `cublasCreate`：创建传统 cuBLAS 上下文

**目的：为后续 cuBLAS 调用创建一个可复用的运行上下文。** cuBLAS 需要保存当前 GPU、stream、pointer mode 和数学模式等状态。可以把 handle 理解成“一组 cuBLAS 运行配置的载体”；它不是 GPU stream，也不保存矩阵数据。

```cpp
cublasStatus_t cublasCreate(cublasHandle_t *handle);
```

调用成功后，`*handle` 保存当前 cuBLAS 上下文。它记录当前 device、stream、pointer mode、math mode 等状态。

- `handle`：输出参数，必须传入有效的 `cublasHandle_t` 变量地址，不能传空指针。
- 返回值：`CUBLAS_STATUS_SUCCESS` 表示成功；也可能返回未初始化或内存分配失败等状态。

```cpp
cublasHandle_t handle = nullptr;
CHECK_CUBLAS(cublasCreate(&handle));
```

注意事项：

- 通常在程序或工作线程初始化阶段创建一次，然后反复使用。
- `cudaSetDevice()` 应放在 `cublasCreate()` 前面，使 handle 与预期 GPU 关联。
- handle 包含可变状态。多线程程序更容易维护的做法是每个工作线程使用自己的 handle。

### 12.2 `cublasDestroy`：释放传统 cuBLAS 上下文

**目的：释放 `cublasCreate` 为 cuBLAS 上下文申请的内部资源。** 它只管理 handle 本身，不会替用户释放矩阵、workspace 或 CUDA stream。

```cpp
cublasStatus_t cublasDestroy(cublasHandle_t handle);
```

- `handle`：由 `cublasCreate` 创建、尚未销毁的句柄。

```cpp
CHECK_CUBLAS(cublasDestroy(handle));
handle = nullptr;
```

销毁 handle 并不会替你释放 `cudaMalloc` 得到的矩阵内存，也不会销毁用户创建的 CUDA stream。资源通常按以下逆序释放：先确保计算结束，再释放矩阵和 workspace，最后销毁 handle 与 stream。

### 12.3 `cublasSgemm`：FP32 通用矩阵乘

**目的：在 GPU 上执行单精度矩阵乘加。** 它对应全连接层、线性变换和科学计算中的矩阵乘，是理解 `M/N/K`、转置和 leading dimension 的基础接口。适合固定 FP32、布局简单且不需要融合 bias/activation 的场景。

```cpp
C = alpha * op(A) * op(B) + beta * C
```

```cpp
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc);
```

参数应按照“逻辑形状”和“物理存储”分别理解：

- `transa`：决定逻辑矩阵 `op(A)`。`N` 时原始 `A` 是 `m×k`；`T/C` 时原始 `A` 是 `k×m`。
- `transb`：`N` 时原始 `B` 是 `k×n`；`T/C` 时原始 `B` 是 `n×k`。
- `m`：`op(A)` 和 `C` 的行数。
- `n`：`op(B)` 和 `C` 的列数。
- `k`：`op(A)` 的列数，也是 `op(B)` 的行数。
- `alpha`、`beta`：标量指针。默认 pointer mode 下指向 CPU 内存。
- `A/B/C`：设备内存地址。`C` 同时是输入和输出；`beta=0` 时通常不需要预先保存有效的 `C` 数据。
- `lda/ldb/ldc`：原始物理矩阵相邻两列之间的元素跨度。

连续列主序时的典型 leading dimension：

| 参数 | 不转置 | 转置/共轭转置 |
| --- | --- | --- |
| `lda` | `m` | `k` |
| `ldb` | `k` | `n` |
| `ldc` | `m` | `m` |

`alpha=1, beta=0` 表示普通矩阵乘；`beta=1` 表示把乘积累加到已有 `C`。维度较大时还需注意 `int` 上限，可查看对应的 64 位接口。

### 12.4 `cublasGemmEx`：混合精度 GEMM

**目的：解除“矩阵存储类型”和“内部计算类型”必须相同的限制。** 例如让 A/B 用 FP16 节省显存和带宽，同时用 FP32 累加控制误差。它解决了 `cublasSgemm` 类型固定的问题，也是使用 Tensor Core 加速推理和训练矩阵乘的传统核心接口。

它与 `cublasSgemm` 的数学公式相同，但把存储类型和累加计算类型拆开：

```cpp
cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const void *beta,
    void *C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);
```

- `Atype/Btype/Ctype`：矩阵实际存储类型，例如 `CUDA_R_16F`、`CUDA_R_16BF`、`CUDA_R_32F`。
- `computeType`：乘法与累加采用的计算模式。典型的 FP16 输入、FP32 累加使用 `CUBLAS_COMPUTE_32F`。
- `alpha/beta` 的 C++ 类型由 scale/compute 配置决定，不能因为矩阵是 FP16 就总是假设标量也必须是 `__half`。
- `algo`：算法枚举；通常从 `CUBLAS_GEMM_DEFAULT` 开始，复杂算法搜索更推荐使用 cuBLASLt。

FP16 输入、FP32 输出和累加的示意调用：

```cpp
float alpha = 1.0f, beta = 0.0f;
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             d_A, CUDA_R_16F, M,
             d_B, CUDA_R_16F, K,
             &beta,
             d_C, CUDA_R_32F, M,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT);
```

它是 AI 推理核心 API 的原因在于：权重和激活可以低精度存储以减少带宽，同时使用更高精度累加控制误差，并让支持的形状进入 Tensor Core 路径。

### 12.5 `cublasSgemmStridedBatched`：固定步长的批量 GEMM

**目的：用一次 API 调用提交大量形状相同、内存间隔固定的小矩阵乘。** 逐个调用 `cublasSgemm` 时，CPU 提交开销可能很明显；使用 `cublasSgemmBatched` 又要构造设备指针数组。Strided Batched 只需三个基地址和固定 stride，适合 batch、head 等规则布局。

对 `batchCount` 组矩阵执行：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
A_i = A + i * strideA
B_i = B + i * strideB
C_i = C + i * strideC
```

其中 stride 的单位是**元素**，不是字节。

```cpp
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, long long strideA,
    const float *B, int ldb, long long strideB,
    const float *beta,
    float *C, int ldc, long long strideC,
    int batchCount);
```

- `strideA/B/C`：相邻 batch 对应矩阵首元素之间的元素距离。
- `batchCount`：独立 GEMM 的数量。
- 其余参数对所有 batch 统一生效。

无 padding、列主序且都不转置时，通常使用：

```cpp
long long strideA = (long long)M * K;
long long strideB = (long long)K * N;
long long strideC = (long long)M * N;
```

一个操作数需要在所有 batch 中广播时，可在支持的使用方式下令相应 stride 为 `0`。Attention 中不同 head 或 batch 常被组织成规则间隔，因此适合该 API；不过现代模型也经常直接使用支持 batch layout 和 epilogue 的 cuBLASLt。

### 12.6 `cublasLtCreate`：创建轻量级 Matmul 上下文

**目的：进入 cuBLASLt 的描述符式 Matmul 工作流。** 传统 GEMM 难以表达复杂 layout、算法偏好和 bias/activation 融合；cuBLASLt handle 用来支撑这些灵活的 Matmul 能力和算法查询。

```cpp
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle);
```

- `lightHandle`：输出参数，返回 cuBLASLt handle。

```cpp
cublasLtHandle_t ltHandle = nullptr;
CHECK_CUBLAS(cublasLtCreate(&ltHandle));
// ...
CHECK_CUBLAS(cublasLtDestroy(ltHandle));
```

cuBLASLt 的“Lt”强调灵活的 Matmul：它把计算属性、矩阵布局、算法偏好分成描述符，并支持 bias、激活等 epilogue 融合。创建 handle 后仍需创建 operation descriptor 和 matrix layout，不能直接调用 `cublasLtMatmul`。

### 12.7 `cublasLtMatmulDescCreate`：描述计算属性

**目的：把“这次矩阵乘要怎样计算”封装成可配置对象。** 计算精度、标量类型、A/B 是否转置、是否融合 bias、使用哪种 epilogue 都属于计算属性。同一描述符可同时用于算法搜索和最终执行。

```cpp
cublasStatus_t cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t *matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType);
```

- `matmulDesc`：输出的计算描述符。
- `computeType`：乘加与累加采用的计算模式，例如 `CUBLAS_COMPUTE_32F`。
- `scaleType`：`alpha`、`beta` 的数据类型，例如 `CUDA_R_32F`。

```cpp
cublasLtMatmulDesc_t opDesc;
CHECK_CUBLAS(cublasLtMatmulDescCreate(
    &opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

cublasOperation_t transA = CUBLAS_OP_N;
CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
    opDesc, CUBLASLT_MATMUL_DESC_TRANSA,
    &transA, sizeof(transA)));
```

常用 attribute 包括 `TRANSA`、`TRANSB`、`EPILOGUE` 和 `BIAS_POINTER`。例如设置 bias epilogue 后，Matmul 可以在写出 D 前加上 bias，从而减少一次 kernel 启动和一次全局内存往返。使用结束后调用 `cublasLtMatmulDescDestroy(opDesc)`。

不要混淆两种描述符：

- `cublasLtMatmulDesc_t` 描述“怎么算”，例如转置、计算精度、epilogue。
- `cublasLtMatrixLayout_t` 描述“数据怎么放”，例如类型、行列数、leading dimension、batch stride。

### 12.8 `cublasLtMatmulAlgoGetHeuristic`：自动筛选算法

**目的：根据矩阵形状、布局、计算类型、epilogue 和 workspace 限制，筛选当前配置下可用且预计较快的内核算法。** 同一 Matmul 可以有不同 tile、stage、split-K 和 Tensor Core 实现，不存在对所有形状都最快的固定算法。这个 API 能避免硬编码算法编号，并过滤不支持当前配置的算法。

```cpp
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference,
    int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount);
```

- `lightHandle`：cuBLASLt handle。
- `operationDesc`：计算属性描述符。
- `Adesc/Bdesc/Cdesc/Ddesc`：四个矩阵的布局描述符；输出 D 可与输入 C 使用不同类型或布局。
- `preference`：算法偏好，最常用属性是允许的最大 workspace 字节数。
- `requestedAlgoCount`：结果数组容量，也是最多请求的算法数，必须大于 0。
- `heuristicResultsArray`：输出候选算法，通常按预计执行时间递增排序。
- `returnAlgoCount`：实际返回的候选数量。必须检查它是否大于 0。

```cpp
cublasLtMatmulPreference_t preference;
CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));

size_t workspaceSize = 4 * 1024 * 1024;
CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
    preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspaceSize, sizeof(workspaceSize)));

cublasLtMatmulHeuristicResult_t result{};
int returned = 0;
CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
    ltHandle, opDesc, aDesc, bDesc, cDesc, dDesc,
    preference, 1, &result, &returned));

if (returned == 0) {
    // 当前类型、布局、epilogue 和 workspace 限制下没有可用算法
}
```

heuristic 给出的是基于配置的候选，并不保证对实际数据和运行环境永远最快。性能敏感的固定 shape 可对多个返回候选进行 CUDA event 实测，并缓存最佳算法。

### 12.9 `cublasLtMatmul`：现代通用 Matmul 核心接口

**目的：按照计算描述符、矩阵布局和选定算法，真正把 Matmul 提交到 GPU。** `cublasLtMatmulDescCreate` 只描述运算，`cublasLtMatmulAlgoGetHeuristic` 只寻找候选，只有 `cublasLtMatmul` 才执行计算。它还可以融合 bias 和 activation，减少 kernel 启动及中间结果的显存读写。

数学语义为：

```text
D = alpha * op(A) * op(B) + beta * C
```

然后可按 `computeDesc` 执行 bias、ReLU、GELU 等受支持的 epilogue。

```cpp
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void *alpha,
    const void *A, cublasLtMatrixLayout_t Adesc,
    const void *B, cublasLtMatrixLayout_t Bdesc,
    const void *beta,
    const void *C, cublasLtMatrixLayout_t Cdesc,
    void *D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo,
    void *workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream);
```

- `computeDesc`：计算精度、转置和 epilogue 等计算属性。
- `alpha/beta`：缩放标量，类型必须匹配创建 operation descriptor 时的 `scaleType`。
- `A/B/C/D`：矩阵地址；传统 GEMM 原地更新 C，而 Lt 明确分开输入 C 和输出 D。
- `Adesc/Bdesc/Cdesc/Ddesc`：分别解释四块矩阵内存的类型、形状、leading dimension、order 和 batch 属性。
- `algo`：heuristic 返回的具体算法；应与查询时的全部描述符配置一致。
- `workspace`：算法可用的临时设备内存；不需要 workspace 时可为 `nullptr`。
- `workspaceSizeInBytes`：workspace 的字节数，必须不小于所选算法要求的大小。
- `stream`：此次运算直接指定的 CUDA stream。

调用前必须先创建矩阵布局，例如连续列主序 FP16 的 A：

```cpp
cublasLtMatrixLayout_t aDesc;
CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
    &aDesc, CUDA_R_16F, M, K, M));
```

一个完整的 cuBLASLt 生命周期是：

```text
cublasLtCreate
  -> cublasLtMatmulDescCreate
  -> cublasLtMatrixLayoutCreate（A/B/C/D 各一次）
  -> cublasLtMatmulPreferenceCreate
  -> cublasLtMatmulAlgoGetHeuristic
  -> cublasLtMatmul
  -> 销毁 preference、layout、matmul descriptor 和 handle
```

### 12.10 传统 GEMM 与 cuBLASLt Matmul 如何选择

| 场景 | 推荐接口 | 原因 |
| --- | --- | --- |
| 简单 FP32 矩阵乘 | `cublasSgemm` | 参数少，结果容易验证 |
| 简单混合精度矩阵乘 | `cublasGemmEx` | 能分别控制存储与计算类型 |
| 大量规则间隔的小矩阵 | `cublasSgemmStridedBatched` | 不需要设备指针数组 |
| 需要 bias/activation 融合 | `cublasLtMatmul` | 支持 epilogue |
| 需要行主序、复杂 layout 或算法搜索 | `cublasLtMatmul` | 描述符和 heuristic 更灵活 |

## 13. cuBLASLt 最小实现检查清单

1. `M/N/K` 描述的是 `op(A)`、`op(B)` 的逻辑形状。
2. layout 中的 `rows/cols/ld` 描述的是转置前的物理矩阵。
3. `computeType` 决定计算方式，matrix type 决定存储方式，`scaleType` 决定 `alpha/beta` 类型。
4. heuristic 查询和实际 Matmul 必须使用一致的 operation/layout/preference 配置。
5. 必须检查 `returnAlgoCount > 0` 和候选结果状态。
6. workspace 是设备内存，容量至少满足候选算法返回的需求。
7. bias 等 epilogue 指针必须在执行 Matmul 时仍有效。
8. 所有描述符和 workspace 都应在最后释放，异步运算结束前不能提前释放。

## 14. 九个核心 API 背后的原理

### 14.1 `cublasCreate`：状态对象与延迟初始化

cuBLAS 不能只靠一个普通函数完成所有工作，因为一次计算还依赖当前 GPU、CUDA context、stream、pointer mode、math mode 等运行状态。`cublasCreate` 创建的 handle，就是这些状态的容器。

其原理可以概括为：

```text
CPU 线程
  -> cuBLAS handle（保存配置和内部状态）
      -> 当前 CUDA context/device
      -> 当前 stream
      -> pointer mode / math mode
      -> cuBLAS 内部资源
```

handle 不保存矩阵，也不是 GPU 上执行计算的线程。它更像一个“命令生成器的配置对象”：后续 `cublasSgemm(handle, ...)` 根据 handle 状态选择实现，并把 GPU 工作提交出去。

第一次创建 handle 或第一次执行某类运算可能触发库、CUDA context 或内部模块的延迟初始化，因此首次调用通常比稳定阶段慢。基准测试需要先 warm-up。

### 14.2 `cublasDestroy`：对象生命周期与异步工作的边界

`cublasDestroy` 的原理是销毁 handle 管理的 CPU 侧和库内部资源。矩阵显存由用户通过 `cudaMalloc` 申请，不属于 handle，因此必须单独 `cudaFree`。

cuBLAS 计算通常异步提交：API 返回只说明任务已成功进入 CUDA 执行系统，不一定说明 GPU 已经算完。因此生命周期管理的关键不是简单地“调用 Destroy”，而是保证仍在执行的工作不再依赖将被释放的资源：

```text
提交 GEMM
  -> 等待相关 stream 完成
  -> 释放矩阵/workspace
  -> 销毁 handle
```

### 14.3 `cublasSgemm`：分块矩阵乘与数据复用

朴素矩阵乘的每个输出元素为：

```text
C[i,j] = sum(A[i,p] * B[p,j]), p = 0...K-1
```

若每次乘加都从显存重新读取 A 和 B，带宽开销很高。高性能 GEMM 的核心是**分块和数据复用**：

```text
全局内存中的 A/B
  -> 切成 tile
  -> 载入 shared memory / register
  -> 一个 tile 被多个线程反复使用
  -> 每个线程在寄存器中累加部分 C
  -> 写回全局内存
```

一个 GEMM kernel 通常把输出矩阵划分成多个 thread-block tile。每个 block 沿 K 维分段加载 A、B 子块，并在寄存器中累加。这样一次全局内存读取可以服务多次乘加，提高 arithmetic intensity。

`m/n/k` 描述逻辑运算，`lda/ldb/ldc` 描述物理地址跨度。cuBLAS 默认列主序，因此元素地址近似为：

```text
A(row, col) 地址 = A + row + col * lda
```

`transa/transb` 通常不需要真的生成一份转置矩阵，而是改变 kernel 对原内存的索引方式。cuBLAS 会根据数据类型、形状、转置、对齐和 GPU 架构选择内部 kernel。

### 14.4 `cublasGemmEx`：混合精度与 Tensor Core

`cublasGemmEx` 将三个概念拆开：

```text
存储精度：A/B/C 在显存里占多少位
乘法精度：输入以什么格式参与乘法
累加精度：部分和以什么精度保存在累加器中
```

例如 FP16 输入、FP32 累加：FP16 减少显存容量和内存流量，乘法使用低精度高吞吐硬件，累加器使用 FP32 减少大量加法造成的误差。

Tensor Core 并不是逐个标量执行乘加，而是以小矩阵片段为单位执行矩阵乘累加（MMA）：

```text
D_fragment = A_fragment * B_fragment + C_fragment
```

大型 GEMM 会被继续拆成许多 MMA fragment。`computeType` 决定可使用的计算路径和数值行为；矩阵尺寸、地址对齐、leading dimension、转置和数据类型也会影响 Tensor Core kernel 是否适用及效率。

混合精度的代价是舍入、溢出和误差累积行为发生变化。它提升吞吐，并不意味着数学结果与全 FP32 逐位相同。

### 14.5 `cublasSgemmStridedBatched`：摊薄启动开销

许多小矩阵逐个调用 GEMM，会产生大量 CPU 到 CUDA runtime/driver 的提交开销，GPU 也可能因每个 kernel 工作量太小而利用率不足。

Strided Batched 把规则描述压缩成：

```text
第 i 个 A = baseA + i * strideA
第 i 个 B = baseB + i * strideB
第 i 个 C = baseC + i * strideC
```

库只接收一次调用，就能在内部 kernel 或一组优化后的 kernel 中处理整个 batch。它省去了：

- CPU 循环调用大量 GEMM 的开销。
- `cublasSgemmBatched` 所需的设备端指针数组。
- 逐个小任务造成的部分调度浪费。

各 batch 数学上相互独立，因此 GPU 可以在 block 调度层面交错执行它们。`stride=0` 的广播场景表示每个 batch 重用同一操作数，有利于权重共享，但能否高效利用缓存仍取决于实现和形状。

### 14.6 `cublasLtCreate`：从固定 API 转向描述符系统

传统 GEMM 把布局和计算配置直接放在函数参数中。随着需求增加——行主序、不同输出类型、批布局、bias、activation、算法 workspace——继续扩充一个函数签名会越来越难维护。

cuBLASLt 改用描述符模型：

```text
Matmul 描述符：怎么算
MatrixLayout：数据怎么存
Preference：允许算法使用多少资源
Algorithm：具体采用哪个内核方案
```

`cublasLtCreate` 建立这一系统所需的库上下文。它将“问题定义”和“实现选择”分开，使库可以在 API 保持稳定的同时增加新的 attribute、epilogue 和算法。

### 14.7 `cublasLtMatmulDescCreate`：把计算语义编码为元数据

该 API 本身不计算，而是创建一份 operation metadata。`computeType` 和 `scaleType` 是基础属性，转置、epilogue、bias 指针等通过 attribute 继续写入。

这类似编译器的中间表示：

```text
用户意图
  -> FP16 输入、FP32 累加
  -> A 不转置、B 转置
  -> GEMM 后加 bias，再执行 GELU
  -> 形成 operation descriptor
```

后续 heuristic 根据这份描述过滤算法，执行阶段也根据同一描述解释参数。这样可保证算法选择针对的正是最终要执行的运算。

Epilogue 融合的原理是：GEMM 的累加结果还在寄存器或片上数据路径时，直接完成 bias/activation，再写回 D。与“GEMM 写显存 -> 新 kernel 读显存 -> bias/activation -> 再写显存”相比，可减少全局内存流量和 kernel 启动。

### 14.8 `cublasLtMatmulAlgoGetHeuristic`：约束过滤与经验选核

cuBLASLt 内部存在许多 Matmul 算法变体，例如不同的：

- thread-block tile 和 warp tile。
- pipeline stage 数量。
- Tensor Core 指令类型。
- split-K 策略。
- swizzle、对齐要求和 workspace 用量。

heuristic 会读取 operation descriptor、四个 layout 和 preference，先排除不兼容的候选，再依据 NVIDIA 内部的性能模型或经验数据对候选排序。

```text
全部算法
  -> 类型/架构是否支持
  -> 布局与对齐是否满足
  -> epilogue 是否支持
  -> workspace 是否超限
  -> 性能模型排序
  -> 返回若干候选
```

它叫 heuristic，是因为结果是“预计较好”，不是对当前机器进行完整 benchmark 后得到的绝对最优解。固定且高频的 shape 可以请求多个候选，实际测量后缓存最快算法；动态 shape 则常直接使用第一个有效候选。

### 14.9 `cublasLtMatmul`：描述符驱动的执行与融合写回

`cublasLtMatmul` 是最终执行入口。调用时，库把以下信息组合起来：

```text
handle：库上下文
computeDesc：数学与融合语义
A/B/C/D layout：物理内存解释方式
algo：tile、stage 等实现方案
workspace：算法临时存储
stream：提交队列
```

内部 kernel 仍遵循分块 GEMM 的基本原理，但它可以根据 layout 支持更灵活的地址计算，并根据 epilogue 在输出写回前完成融合操作。

Lt 将 C 与 D 分开，是因为数学输入和物理输出可能不同：

```text
D = alpha * op(A) * op(B) + beta * C
```

当 `C == D` 且布局兼容时可以原地更新；分开时则能支持 C 与 D 使用不同地址，部分配置还可使用不同布局或类型。workspace 为 split-K、额外归约或其他算法策略提供临时存储。允许更多 workspace 往往意味着候选算法更多，但不保证一定更快。

### 14.10 把九个 API 串起来理解

传统 cuBLAS 路径：

```text
cublasCreate
  -> cublasSgemm / cublasGemmEx / cublasSgemmStridedBatched
  -> stream 同步或 event 建立依赖
  -> cublasDestroy
```

cuBLASLt 路径：

```text
cublasLtCreate
  -> cublasLtMatmulDescCreate：定义怎么算
  -> MatrixLayout：定义数据怎么存
  -> cublasLtMatmulAlgoGetHeuristic：选择怎么高效实现
  -> cublasLtMatmul：提交到 stream 真正执行
  -> 销毁描述符和 handle
```

两条路径底层都围绕同一件事：把矩阵分块、尽可能复用片上数据、用大量并行线程或 Tensor Core 完成乘加。传统 API 强调简单直接；cuBLASLt 则把计算、布局和算法拆开，以换取更强的布局表达、自动选核与算子融合能力。
