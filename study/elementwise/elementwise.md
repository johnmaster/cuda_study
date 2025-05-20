# cuda的内建类型
| 类型 | 含义 | 容量 | 字节大小|
| ----|----|----|----|
|int4|包含4个`int`元素|`int x,y,z,w`|4x4=16Bytes|
|float4|包含4个`float`元素|`float x,y,z,w`|4x4=16Bytes|
|half2|包含2个`__half`元素|`__half x, y`|2x2=4Bytes|

使用内建类型的目的
* 一次访存多个数据
  * 使用`float4`表示一次性加载了4个`float`数据（128bit），而不是每次加载1个`float`(32 bits)。CUDA global memory是按照128 bits（16字节）对齐优先访问
  * 这样做可以
    * 减少访存指令数量，一条指令=读取4个元素
    * 充分利用global memory 带宽（每次读取都尽量填满128bits）
* 提高内存带宽利用率（load/store更高效）
  * 用标量方式（float*）发出4次32-bit load/store
  * 用float4，一次发出1次128bit load/store
  * CUDA的global memory访问性能依赖于
    * 内存coalescing（合并）
      * CUDA的合并访问与warp(32个线程)紧密相关
        * 每个warp的线程同时发出的访存请求会尝试coalesce
        * 如果
          * 地址是连续的
          * 地址是对齐的（128bit对齐最佳）
          * 数据类型一致
    * 对齐访问
      * 内建向量类型天然是对齐的，提升带宽利用率
* 使用SIMD（single instruction multiple data）指令加速运算
  * 利用GPU的指令集，在一个指令中同时处理多个数据元素
  * `half2`是支持SIMD加速指令的，`__device__ __half__ __hadd2(__half2 a, __half2 b);`这条指令会在一个CUDA core （ALU）周期内执行两个FP16的加法
* 减少指令数量和访存次数

# half 和 half2
|类型|位数|描述|
|---|----|---|
|half|16|单个半精度浮点数|
|half2|32|两个`__half`打包成一个结构，用于SIMD计算|

half2主要用于对两个半精度浮点数`__half`打包在一起，然后一次性运算。
## 常见的half2运算函数
* `__hadd2(a, b)`：两个half2相加
* `__hsub2(a, b)`：两个half2相减
* `__hmul2(a, b)`：两个half2相乘
* `__hdiv2(a, b)`：两个half2相除

# 头文件
* `#include <cuda_runtime.h>`
  * `cudaMalloc()`，`cudaMemcpy()`，`cudaFree()`，`cudaGetDevice()`，`cudaSetDevice()`
* `#include <cuda_fp16.h>`
  * 提供half精度浮点数（16-bit）的操作函数和类型
* `#include <torch/types.h>`
  * 提供pytorch的基本类型定义


# 编译选项
`__CUDA_NO_HALF_OPERATORS__`：默认情况下，CUDA不会启用`half`类型的加减乘除操作符，使用这个选项是为了取消默认的禁止定义，允许使用这些操作符；
`__CUDA_NO_HALF_CONVERSIONS__`：允许`half`和`float/int`类型之间的转换；
`__CUDA_NO_HALF2_OPERATORS__`：允许`half2`的SIMD运算符；
`--expt-relaxed-constexpr`：允许`constexpr`表达式中使用等多的CUDA特性，某些 constexpr（编译期常量表达式）在 CUDA 中被限制。这个选项放宽限制，让你在 constexpr 语境中使用例如 `__host__ __device__` 函数;
`--expt-extended-lambda`：允许cuda kernel中使用lambda表达式；

