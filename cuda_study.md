![images/gpu-cpu-system-diagram.png](images/gpu-cpu-system-diagram.png)


* GPU计算不是指单独的GPU计算，而是指CPU+GPU的异构计算，GPU必须在CPU的调度下才能完成特定任务。
* 起控制作用的CPU称为主机（host），起加速作用的GPU称为设备（device）。
* 主机对设备的调用是通过核函数（kernel function）来实现，
* 一个典型的，简单的CUDA程序的结构
  * `int main() { 主机代码; 核函数的调用; 主机代码}`
* 核函数必须被`__global__`修饰，核函数的返回类型必须是`void`，限定符`__global__`和`void`的次序可随意
* `cudaDeviceSynchronize`的作用是同步主机与设备

* CUDA对能够定义的网格大小和线程块大小做了限制，网格大小在x,y,z三个方向的最大允许值是2^31 - 1, 65535, 65535;线程块大小在x,y,z这三个方向的最大允许值分别是1024，1024和64。另外还要求线程块总的大小，即blockDim.x，blockDim.y,blockDim.z的乘积不能大于1024，也就是说，不管如何定义，一个线程块最多只能有1024个线程。
* 一个线程块中的线程可以细分为不同的线程束（thread warp），一个线程束是同一个线程块中相邻的warpSize个线程，warpSize的值是32，一个线程束就是连续的32个线程；

一个典型的CUDA程序的基本框架
```
头文件包含
常量定义（或者宏定义）
c++自定义函数和CUDA核函数的声明（原型）
int main(void)
{
        分配主机与设备内存
        初始化主机中的数据
        将某些数据从主机复制到设备
        调用核函数在设备中进行计算
        将某些数据从设备复制到主机
        释放主机与设备内存
}
c++自定义函数和CUDA核函数的定义（实现）
```

* 在CUDA中，设备内存的动态分配可由`cudaMalloc`函数实现
  * 函数原型如下`cudaError_t cudaMalloc(void **address, size_t size)`
* `cudaMalloc`函数分配的设备内存需要用`cudaFree()`函数释放
  * 函数原型如下`cudaError_t cudaFree(void *address)`
* `cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)`
  * 第一个参数dst是目标地址
  * 第二个参数src是源地址
  * 第三个参数count是复制数据的字节数
  * 第四个参数kind是枚举类型的变量，标志数据传递方向
    * cudaMemcpyHostToHost, 表示从主机复制到主机
    * cudaMemcpyHostToDevice,表示从主机复制到设备
    * cudaMemcpyDeviceToHost,表示从设备复制到主机
    * cudaMemcpyDeviceToDevice,表示从设备复制到设备
    * cudaMemcpyDefault,表示根据指针dst和src所指地址自动判断数据传输的方向

* 编写核函数时要注意的几点
  * 函数名无特殊要求，而且支持c++中的重载
  * 不支持可变数量的参数列表，参数的个数必须指定
  * 可以向核函数传递非指针变量，其内容对每个线程可见
  * 核函数不能成为一个类的成员，通常的做法是用一个包装函数调用核函数，而将包装函数定义为类的成员
  * 无论是从主机调用还是从设备调用，核函数都是在设备中执行，调用核函数必须指定执行配置，即三括号和它里面的参数
  * 除非使用统一内存编程机制，否则传递给核函数的数组（指针）必须指向设备内存

* 在CUDA程序中，由以下标识符确定一个函数在哪里被调用以及在哪里执行
  * 用`__global__`修饰的函数称为核函数，一般由主机调用，在设备中执行，如果使用动态并行，则也可以在核函数中调用自己或其他核函数
  * 用`__device__`修饰的函数称为设备函数，只能被核函数或其他设备函数调用，在设备中执行
  * 用`__host__`修饰的函数就是主机函数，在主机中被调用，在主机中执行。对于主机端的函数，该修饰符可省略。之所以提供这样的一个修饰符，是因为有时可以用`__host__`和`__device__`同时修饰一个函数，使得该函数既是一个c++的普通函数，又是一个设备函数，这样的操作告诉编译器（nvcc），为这个函数编译两份，一份是GPU版本，供`__global__`或`__device__`使用，另一份CPU版本，供普通c++代码调用；
  * 不能同时用`__device__`和`__global__`修饰一个函数，即不能将一个函数同时定义为设备函数和核函数
  * 不能同时用`__host__`和`__global__`修饰一个函数，即不能将一个函数同时定义为主机函数和核函数
  * 编译器决定把设备函数当作内联函数或非内联函数，但可以用修饰符`__noinline__`建议一个设备函数为非内联函数，也可以用修饰符`__forceinline__`建议一个设备函数为内联函数

## `__restrict__` 限定符（指针别名约定）

* **含义**：`__restrict__` 来自 C99 的 `restrict` 语义（CUDA / nvcc 中常用双下划线写法 `__restrict__`，也可用 `__restrict`）。加在某个**指针参数**上，表示：在**该指针的有效作用域内**，程序员保证**不会**再通过**其它指针**去读写同一块内存（即不存在 **aliasing / 别名**）。
* **编译器能做什么**：知道「只有这条指针会访问这段对象」之后，可以更激进地做 **load/store 复用、指令重排、向量化** 等优化；若实际上存在别名却写了 `__restrict__`，属于 **未定义行为**，可能算错结果。
* **典型写法**（核函数里很常见）：
  ```cuda
  __global__ void kernel(const float* __restrict__ src,
                         float* __restrict__ dst,
                         int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) dst[i] = src[i] * 2.f;
  }
  ```
* **与 `const` 的区别**：`const float*` 只表示**不能通过该指针修改**所指数据；**不**表示「没有别的可写指针指向同一块内存」。`__restrict__` 专门约束**别名关系**，二者可一起用：`const float* __restrict__ p`。
* **使用注意**：
  * 若 `src` 与 `dst` **可能重叠**（同一缓冲区、子区间重叠等），**不要**对它们同时标 `__restrict__`。
  * 主机端 C++ 代码里 MSVC 常用 `__restrict`，GCC/Clang 用 `__restrict__` 或 `restrict`（C 模式）；CUDA 源文件里 **`__restrict__` 最通用**。
  * 这是**优化提示 + 程序员契约**，不是「自动防止越界」或「自动防止数据竞争」；多线程同时写同一全局区仍需自己用原子或同步保证正确性。

* 有一种方法可以捕捉调用核函数可能发生的错误，即在调用核函数之后加上如下两个语句
  * `CHECK(cudaGetLastError())`
  * `CHECK(cudaDeviceSynchronize())`
  * 第一个语句是捕捉第二个语句之前的最后一个错误，第二个语句的作用是同步主机与设备，因为核函数的调用是异步的，即主机发出调用核函数的命令后会立即执行后面的语句，不会等待核函数执行完毕

* CUDA提供CUDA-MEMCHECK工具
  * cuda-memcheck --tool memcheck [option] app_name [options]
  * cuda-memcheck --tool racecheck [option] app_name [options]
  * cuda-memcheck --tool initcheck [option] app_name [options]
  * cuda-memcheck --tool synccheck [option] app_name [options]
* 提高CUDA程序获得高性能的必要不充分条件
  * 减少主机与设备之间的数据传输
  * 提高核函数的算术强度
  * 增大核函数的并行规模

![cuda_memory_architecture](images/cuda_memory_architecture.png)

| 内存类型 | 物理位置 | 访问权限 | 可见范围 | 生命周期 |
| ------ | ------ | ------ | ------ | ------ |
| 全局内存 | 在芯片外 |可读可写 |所有线程和主机端|由主机分配和释放|
|常量内存|在芯片外|仅可读|所有线程和主机端|由主机分配和释放|
|纹理和表面内存|在芯片外|一般仅可读|所有线程和主机端|由主机分配和释放|
|寄存器内存|在芯片内|可读可写|单个线程|所在线程|
|局部内存|在芯片外|可读可写|单个线程|所在线程|
|共享内存|在芯片内|可读可写|单个线程块|所在线程块|

* <mark>全局内存</mark>的含义是核函数中的所有线程都能访问其中的数据，全局内存的主要角色是为核函数提供数据，并在主机和设备以及设备和设备之间传递数据，使用`cudaMalloc`函数为全局内存分配设备内存；全局内存对整个网络的所有线程可见，也就是说，一个网格的所有线程都可以访问（读或写）传入核函数的设备指针所指向的全局内存中的全部数据;全局内存的生命周期不是由核函数决定的，而是由主机端决定的，从使用`cudaMalloc`分配开始，到`cudaFree`释放内存结束；
* cuda允许使用静态全局内存变量，其所占的内存数量在编译期间就确定，静态全局内存变量必须在所有主机与设备函数外部定义，是一种“全局的静态全局内存变量”
* 静态全局内存变量由以下方式在任何函数外部定义
  * `__device__ T x;`
  * `__device__ T y[N];`
  * 修饰符`__device__`说明该变量是设备中的变量，而不是主机的变量；在核函数中，可直接对静态全局内存变量进行访问，并不需要将它们以参数的形式传给核函数；不可以在主机函数中直接访问静态全局内存变量，可以使用`cudaMemcpyToSymbol`和`cudaMemcpyFromSymbol`在静态全局变量和主机内存之间传输数据。
* <mark>常量内存</mark>是有常量缓存的全局内存，共有64KB，可见范围和生命周期与全局内存一致，不同的是，常量内存仅可读，不可写，拥有缓存，常量内存的访问速度比全局内存快，<mark>得到高访问速度的前提是一个线程束中的线程（一个线程块中的相邻的32个线程）要读取相同的常量内存数据</mark>；
* 在核函数外使用`__constant__`定义变量来使用常量内存
* 纹理内存和表面内存类似常量内存，是一种具有缓存的全局内存，有相同的可见范围和生命周期，一般仅可读，
* 在核函数中定义的不加任何限定符的变量一般来说就存放在寄存器中，核函数中定义的不加任何限定符的数组有可能存放在局部内存中，寄存器可读可写，寄存器变量仅仅被一个线程可见，寄存器的生命周期与所属线程的生命周期一致，从定义它开始，到线程消失时结束；
* 共享内存对整个线程块可见，生命周期与线程块一致；
* <mark>SM线程的执行是以线程束为单位的，最好将线程块的大小取为线程束大小（32个线程）的整数倍</mark>

* 全局内存的合理使用（合并访问与内存事务）
  * **硬件粒度与「32」的关系**：在讨论全局内存合并时，现代 NVIDIA GPU 在 L2 / 全局路径上常以 **32 字节（sector，扇区）** 作为**参与合并与计费的常见最小块**。注意这与 **一个线程束有 32 个线程** 只是数字巧合，**没有因果关系**——不是「32 个线程 ⇒ 一次只传 32 字节」。
  * **「一次」读多少字节**：上述 32 字节指的是**单次事务/最小块**的常见大小，**不是**「整条 warp 指令总共只读 32 字节」。例如一个 warp 内 32 个线程各读一个连续的 `float`（4B），逻辑上需要 **128B** 连续数据，底层往往会对应 **多个** 32B 级别的传输（例如约 4 个 sector）。若数据已在 L1/L2 命中，则可能不再从 DRAM 取数。
  * **内存事务**：对全局内存的访问会经过缓存层次并最终体现为**内存事务**（可能命中缓存，也可能从更慢的存储层次取数）；从优化角度常关注「为满足这次 warp 访问，实际搬动了多少字节、有多少被浪费」。
  * **合并访问与非合并访问**：**合并访问（coalesced）**指：一个线程束对全局内存的一次访问请求（读或写），在硬件上引起的**数据传输总量接近该 warp 真正需要的字节数**（浪费少）。否则称为**非合并访问**。可定义 **合并度（degree of coalescing）** =（该 warp 逻辑上需要的字节数）÷（因该请求而实际参与传输/处理的总字节数）。当分母中没有「白搬」的字节、全部为该 warp 所需时，合并度为 **100%**，即理想合并访问。实践上应尽量让同一 warp 内线程访问**连续、对齐**的地址（如 `idx = blockIdx.x * blockDim.x + threadIdx.x` 访问 `data[idx]`），以提高合并度。

* 在核函数中可以直接使用在函数外部由`#define`或`const`定义的常量，包括整型常量和浮点型常量，但是不能在核函数中使用这种常量的引用或地址；

* 共享内存是一种可被程序员直接操控的缓存，主要作用有两个：一个是减少核函数对全局内存的访问次数，实现高效的线程块内部的通信；另一个是提高全局内存访问的合并度。
* 如果需要保证核函数中语句的执行顺序与出现顺序一致，必须使用某种同步机制，在cuda中，使用`__syncthreads__`，这个函数只能使用在核函数中，该函数可保证一个线程块中的所有线程，在执行该语句后面的语句之前都完全执行了该语句前面的语句，然而，该函数只针对同一个线程块中的线程，不同线程块中线程的执行次序是不确定的。
* 在核函数中，使用`__shared__`将一个变量定义为共享内存变量；在一个核函数中定义一个共享内存变量，相当于在每一个线程块中有了一个该变量的副本，每个副本都不一样；
* 使用动态的共享内存
  * 在调用核函数的执行配置中设置第三个参数
    * `<<<grid_size, block_size, sizeof(real) * block_size>>>`，前两个参数是网格大小和线程块大小，第三个参数是核函数中每个线程块需要定义的动态共享内存的字节数；
    * 动态共享内存的声明方式：`extern __shared__`，不可以将动态共享内存数组声明为指针-->>`extern __shared__ real *s_y`这是错误的

![shared_memory_bank](images/shared_memory_bank.png)

* 避免共享内存的bank冲突
  * 为了获得高的内存带宽，共享内存在物理上被分为32个（刚好等于线程束的数量）同样宽度的，能同时访问的内存bank。可以将32个bank从0到31编号，在每一个bank中，可以对其中的内存地址从0开始编号，将所有bank中编号为0的内存称为第一层内存，将所有bank中编号为1的内存称为第二层内存，每个bank的宽度为4字节。
  * 对于bank宽度为4字节的架构，共享内存数组是按照如下方式线性的映射到内存bank的，共享内存数组中连续的128字节的内容分摊到32个bank的某一层中，每个bank负责4字节的内容。例如，对于一个长度为128的单精度浮点数变量的共享内存数组来说，第0-31个数组元素依次对应到32个bank的第一层，第32-63个数组元素依次对应到32个bank的第二层，第64-95个数组元素依次对应到32个bank的第三层，第96-127个数组元素依次对应到32个bank的第四层。
  * 只要同一个线程束中的多个线程不同时访问同一个bank中不同层的数据，该线程束对共享内存的访问就只需要一次内存事务，当同一个线程束的多个线程试图访问同一个bank的不同层的数据时，就会发生bank冲突。在一个线程束中对同一个bank的n层数据同时访问将导致n次内存事务，这种情况称为n路bank冲突。
  * 原子函数`atomicAdd(address, val)`第一个参数是待累加变量的地址address，第二个参数是累加的值val，该函数的作用是将地址address中的旧值old读出，计算old+val后，将计算得到的值存入地址address中。

* 从硬件上来看，一个GPU被分为若干个流多处理器（SM）。核函数中定义的线程块在执行时将被分配到还没有完全占满的SM中，一个线程块不会被分配到不同的SM中，而总是在一个SM中，但一个SM可以有一个或多个线程块。
* 一个SM以32个线程为单位产生，管理，调度，执行线程，一个SM可以处理一个或多个线程块，一个线程块又分为若干个线程束。
* 当一个线程束中的线程顺序地执行判断语句中的不同分支时，这种情况称为分支发散。<mark>分支发散是针对同一个线程束内部的线程的</mark>，如果不同的线程束执行条件语句的不同分支，则不属于分支发散。
* 当使用的线程都在一个线程束内时，可以将线程同步函数`__syncthreads__`替换为`__syncwarp`，这个函数的名字是束内同步函数
  * 函数原型`void __syncwarp(unsigned mask = 0xffffffff);`函数有一个可选的参数，这个参数代表掩码的无符号整型数，默认值的全部32个二进制位都为1,代表线程束中的所有线程都参与同步；
* 协作组可以看作是线程块和线程束同步机制的推广，包括线程内部的同步与协作，线程块之间的（网格级别）同步与协作及设备之间的同步与协作
  * 使用协作组的功能时，需要包含`#include <cooperative_groups.h>`
  * 所有与协作组相关的数据类型和函数都定义在`cooperative_groups`命名空间下
* 线程块级别的协作组
  * 最基本的类型是线程组`thread_group`
    * `void sync()`：同步组内所有线程
    * `unsigned size()`：返回组内总的线程数目
    * `unsigned thread_rank()`：返回当前调用该函数的线程在组内的编号
    * `bool is_valid`：返回逻辑值，如果定义的组违反了任何CUDA的限制，返回false,否则返回true
  * 线程组类型有一个称为线程块`thread_block`的导出类型
    * `dim group_index()`：函数返回当前调用该函数的线程的线程块指标，等同于blockIdx
    * `dim thread_index()`：函数返回当前调用该函数的线程的线程指标，等同于threadIdx
  * 可以使用`thread_block g = this_thread_block()`定义初始化一个`thread_block`对象，其中`this_thread_block()`相当于一个线程块类型的常量，`g.sync()`等同于`__syncthreads()`,`g.group_index()`等同于`blockIdx`，`g.thread_index()`等同于`threadIdx`
  * 可以使用`tiled_partition`将一个线程块划分为若干片，每一个片构成一个新的线程组，目前仅仅可以将片的大小设置为2的正整数次方且不大于32
* 一个CUDA流指的是由主机发出的在一个设备中执行的CUDA操作，一个CUDA流中各个操作的次序都是由主机控制的，按照主机发布的次序来决定，然而，来自于两个不同的CUDA流中的操作不一定按照某个次序执行，有可能并发或交错的执行。
* 任何CUDA操作都存在于某个cuda流中，要么是默认流，要么是明确指定的非空流。
* 非默认的CUDA流是在主机端产生和销毁的，一个CUDA流由类型`cudaStream_t`表示
  * `cudaError_t cudaStreamCreate(cudaStream_t*)`
  * `cudaError_t cudaStreamDestroy(cudaStream_t);`

## Shuffle操作（线程束内通信）

* Shuffle操作允许线程束内的线程直接交换寄存器中的数据，无需通过共享内存，比使用共享内存更加高效
* CUDA提供四种shuffle函数（CUDA 9.0+都需要带`_sync`后缀）：

### 第二个参数 `var` 是什么？（四个 API 通用）

四个函数的形参顺序都是 `(mask, var, …)`，**第二个参数 `var` 表示：当前线程把自己寄存器里的这个值，交给 shuffle 硬件参与本次 warp 内交换**。

* **从本线程角度**：`var` 是**本线程提供出去**的标量（`int` / `float` / `unsigned` 等支持的类型），一般就是你想让「别的 lane 可能读到」或「参与广播/归约」的那个变量。
* **从整条指令角度**：warp 里 32 个线程**各自**传一个 `var`，硬件根据 `srcLane` / `delta` / `laneMask` 等规则，**从某个（或规则对应的）lane 的 `var` 取值**，作为**本线程的返回值**。
* **和返回值的关系**：函数返回值 = **按规则选中的那条 lane 在调用时传入的 `var`**（若规则指向不存在的 lane，则按文档返回本线程自己的 `var` 等定义行为）。
* **简单记**：`var` = **「我这一线程交出去的寄存器值」**；返回 = **「规则指定的那条线程当时交出去的值」**。例如 `__shfl_sync(mask, myVal, 0)`：大家都拿到 **lane 0 传入的 `myVal`**。

### 1. `__shfl_sync(mask, var, srcLane, width)`
* 从指定的lane（线程束内线程编号0-31）直接读取数据
* 示例：`int val = __shfl_sync(0xffffffff, myVal, 0);` // 所有线程都从lane 0读取数据

### 2. `__shfl_up_sync(mask, var, delta, width)`
* 每个线程从比自己编号小delta的线程读取数据
* 如果源lane不存在（编号<0），则返回调用线程自己的值
* 示例：`int val = __shfl_up_sync(0xffffffff, myVal, 2);` // 从前面第2个线程读取

### 3. `__shfl_down_sync(mask, var, delta, width)`
* 每个线程从比自己编号大delta的线程读取数据
* 如果源lane不存在（编号≥width），则返回调用线程自己的值
* 常用于归约操作
* 示例：`int val = __shfl_down_sync(0xffffffff, myVal, 2);` // 从后面第2个线程读取

### 4. `__shfl_xor_sync(mask, var, laneMask, width)`
* 每个线程从自己的lane编号与laneMask进行异或运算得到的lane读取数据
* 常用于蝶形交换（butterfly exchange）
* 示例：`int val = __shfl_xor_sync(0xffffffff, myVal, 1);` // lane 0和1交换，2和3交换...

### Shuffle参数说明
| 参数 | 说明 |
|------|------|
| mask | 32位掩码，指定参与操作的线程，`0xffffffff`表示所有32个线程都参与 |
| **var（第2个参数）** | **当前线程提供的寄存器值**；全 warp 的 `var` 组成「可被读取的源」，返回值来自规则选中的 lane 的 `var` |
| srcLane/delta/laneMask | 确定**从哪条 lane 的 var 取值**（或相对偏移、异或目标 lane） |
| width | 可选参数，逻辑线程束宽度（必须是2的幂且≤32），默认32 |

### 使用Shuffle实现高效归约求和
```cpp
__device__ int warpReduce(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0 包含最终结果
}
```

### Shuffle操作的优势
* 低延迟：直接在寄存器级别交换数据
* 无需共享内存：节省共享内存资源
* 无需额外同步：`_sync`版本内置同步
* 更高带宽：比共享内存访问更快

* 4种shuffle操作

## WMMA 常用 API（Warp Matrix Multiply-Accumulate）

WMMA 是 CUDA 提供的 **Warp 级矩阵乘加** 接口，由 **Tensor Core** 执行。典型配置为 **`m16n16k16`**（`half`×`half` → `float` 累加）。**一个 warp（32 线程）** 协同调用下列 API；不能只用部分 lane。

* **头文件与命名空间**：`#include <mma.h>`，使用 `using namespace nvcuda;` 后前缀写 `wmma::`，或写全名 `nvcuda::wmma::`。

### `wmma::fragment`

* **作用**：在寄存器里表示参与 WMMA 的一小块矩阵（**逻辑视图**；数据实际分布在 warp 各 lane 的寄存器中）。
* **模板参数**（常见）：
  * **角色**：`wmma::matrix_a`、`wmma::matrix_b`、`wmma::accumulator`
  * **尺寸**：`M, N, K`（如 `16, 16, 16`），须与架构支持的 WMMA 配置一致
  * **A/B 的元素类型**：如 `half`、`__nv_bfloat16` 等（视 GPU 而定）
  * **A 的布局**：`wmma::row_major` 或 `wmma::col_major`（**必须与内存中数据布局一致**）
  * **B 的布局**：同样有 row/col 要求，且要与 `mma_sync` 规定的 A×B 语义匹配
* **accumulator** 常为 **`float`**（半精度乘、单精度累加）。

```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
```

### `wmma::fill_fragment`

* **作用**：把累加器 fragment **初始化为常数**（通常为 `0`，做 GEMM 累加前清零）。
* **调用**：`wmma::fill_fragment(acc_frag, 0.0f);`

### `wmma::load_matrix_sync`

* **作用**：从 **全局内存或共享内存** 按指定布局把 **`M×K`（A）或 `K×N`（B）** 的子块加载到 fragment；**整 warp 同步**执行。
* **典型形式**：`load_matrix_sync(frag, ptr, ldm)`
  * **`ptr`**：子块**左上角**元素在内存中的地址（`const half*` 等）
  * **`ldm`（leading dimension）**：行主序下 **「一整行」在内存中占多少个元素**（即下一行相对上一行起始的步长）。例如：大矩阵 A 为 `M×K` 行主序存则 **`ldm = K`**；共享数组 `As[BLOCK_M][BLOCK_K]` 则 **`ldm = BLOCK_K`**。填错会导致读串行、结果错误。

### `wmma::mma_sync`

* **作用**：执行 **`D = A×B + D`**（矩阵乘加），**D** 为 accumulator fragment；**整 warp** 同步。
* **典型形式**：`mma_sync(acc_frag, a_frag, b_frag, acc_frag);`
* **语义**：在 K 维上需在外层循环中多次调用（每次处理 `K` 步中的 16），把多段结果累加到同一 `acc_frag`。

### `wmma::store_matrix_sync`

* **作用**：把 accumulator fragment 写回内存（全局或共享）；**整 warp** 同步。
* **典型形式**：`store_matrix_sync(ptr, acc_frag, ldm, wmma::mem_row_major);`
  * **`ptr`**：输出子块左上角在 C 中的地址
  * **`ldm`**：C 行主序时一般为 **矩阵列数 `N`**（与 `load` 同理：行的 leading dimension）
  * **最后一参**：输出内存布局，如 `wmma::mem_row_major`

### 使用注意小结

| 项目 | 说明 |
|------|------|
| 线程要求 | 每次 `load` / `mma` / `store` 需 **同一 warp 内 32 线程同时参与** |
| `ldm` | **行主序「行跨度」**；与物理存储一致，不是子块宽度那么简单 |
| 布局 | `fragment` 声明的布局须与 `ptr` 所指数据一致 |
| 算力与架构 | 需支持 Tensor Core 的 SM；具体 `M,N,K` 与类型以 **官方文档 / `cuda_fp16.h` + 架构** 为准 |
| 边界 | 若 `M,N` 非 16 整数倍，需 **padding** 或 **额外边界处理**，避免越界或脏写 |

### 与 Tensor Core 的关系（对应前文 SM 图）

* **Tensor Core**：专做小块矩阵乘加的硬件单元；WMMA API 是其主要编程入口之一（另有 CUDA C++ `mma.sync` PTX 等更底层用法）。
* 高吞吐 GEMM 往往：**block 协作搬运到 shared → 各 warp 反复 `load`→`mma`→最后 `store`**（见 `wmma_gemm` 等实现）。

## `cp.async` 异步拷贝（Global → Shared）

与 WMMA / Tensor Core GEMM 常一起出现：**在数据还在从显存往共享内存搬的时候，让 SM 继续算上一批 tile**，属于 **指令级异步内存子系统**，不是主机端的 `cudaMemcpy`。

### 是什么技术

* **名称**：NVIDIA PTX 里的 **`cp.async`**（**asynchronous copy**）指令族。
* **典型方向**：**Global Memory → Shared Memory**（kernel 内细粒度搬运）。
* **架构**：**Ampere（SM 8.0）及更新架构**上广泛使用（具体能力与变体以 **CUDA / PTX 文档** 为准）。
* **和什么并列**：与较高层的 **`cuda::memcpy_async`**（C++ API）等属于同一类「异步数据搬运」思路；也可手写 **`asm volatile` 内联 PTX** 直接发射 `cp.async`。

### 典型指令形式（内联汇编）

```cuda
asm volatile(
    "cp.async.cg.shared.global [%0], [%1], 16;\n"
    :: "r"(smem_addr), "l"(gmem_ptr));
```

| 片段 | 含义 |
|------|------|
| `cp.async` | **异步**发起拷贝，可在约束下与后续指令 **重叠执行**（由硬件完成搬运） |
| `.shared.global` | **目的地址**在 **shared**，**源地址**在 **global** |
| `.cg` | **缓存提示**（cache global）：倾向把数据当「全局可复用」类访问处理（具体以架构文档为准，非改变语义必选） |
| `16` | 本次拷贝 **16 字节**（常配合 `half`/`float4` 等对齐向量） |
| `smem_addr` / `gmem_ptr` | 共享内存与全局内存地址，须满足 **对齐与访问规则**（见官方约束） |

### 内联汇编里 `"r"` 与 `"l"` 是什么？（GCC / clang **扩展汇编**约束）

模板里 **`%0`、`%1`** 依次对应 **冒号后第三段（输入操作数）** 的第 1、2 个操作数。前面的 **`"r"`、`"l"`** 是 **约束字母**，告诉编译器：**这个 C 表达式允许放在什么位置、用什么方式传给汇编**。

| 约束 | 常见含义（NVCC + PTX 场景） |
|------|---------------------------|
| **`"r"`** | **寄存器（register）**：操作数放在 **通用整型寄存器**里。这里 **`smem_addr`** 已是 **`uint32_t`**（经 `__cvta_generic_to_shared`），PTX 里 shared 目的地址常以 **32 位寄存器操作数** 形式出现，故用 **`"r"`**。 |
| **`"l"`** | **依赖目标/工具链**：在 CUDA 设备代码里常用于 **全局指针 / 立即数类** 操作数，使编译器生成 **PTX 期望的地址形式**（如 **64 位 global 地址**）。**`gmem_ptr`** 指向 **global**，用 **`"l"`** 与 NVIDIA 示例、CUTLASS 等写法一致。 |

**注意**：约束字母的 **精确集合** 以 **所用编译器文档** 为准；若换约束导致 PTX 非法，需按 **PTX ISA** 与 **内联汇编文档** 调整。**不要**把 `"r"`/`"l"` 理解成「内存类型」，它们是 **「编译器怎么编操作数」** 的提示。

**汇编模板语法提醒**：`asm volatile("..." : : "r"(a), "l"(b));` 中 **无输出、无破坏描述** 时，第三、四个 `:` 之间为 **输入列表**。

### 同步：何时算「搬完了」

* 异步拷贝发出后，若马上要 **读 shared** 做 WMMA，需用 **`cp.async.wait_group`**（或 `wait_all` 等）保证 **对应批次** 已完成，避免读未就绪数据。
* 典型 **软件流水线**：多段 **`__shared__` 缓冲（double/triple buffer）** + 循环内 **先发 `cp.async` 预取下一段** → **等 `wait_group`** → **对已到齐的 stage 做 `load_matrix_sync` / `mma_sync`**。

### 「流水线」在直觉上是什么？（与 V6 心智模型）

不必先啃完硬件细节，可先记 **生活类比**：

* **没有流水**：搬货（Global→Shared）时 **站着等搬完**，再开始算（WMMA）→ **搬运工和计算工总有一个在闲着**。  
* **有流水**：**一边搬下一箱货**，**一边算上一箱已经到的货** → 同一时间段里 **搬运与计算重叠**，总时间变短。

**V6（`wmma_gemm_v6_async_pipeline`）** 里：

* **`NUM_STAGES` 套 `As/Bs`** = **多个「货位」**：有的货位 **刚到货**（`cp.async`），有的 **正在算**（WMMA）。  
* **Prologue**：先 **把前几段 K 的货异步摆进货位**，主循环 **第一轮就有得算**。  
* **`stage = k_block % NUM_STAGES`**：**环形用货位**；**末尾 prefetch** 把 **「再隔 `NUM_STAGES` 段 K」** 提前摆回 **刚腾空的货位**。  
* **`wait_group`**：**在「可以读这个货位」之前**，按硬件规则 **等异步拷贝收到足够程度**；**收尾** 常用 **`wait_group 0`** **排空**，避免最后几段还在路上就读 shared。

### `cp.async.commit_group` 与 `wait_group` 的配合

PTX 里 **`wait_group n`** 统计的是 **「尚未完成的异步拷贝组（group）」** 的数量上界；**组**要靠 **`cp.async.commit_group`** 划界。

* **`cp.async.commit_group`**（内联写法示例）：
  ```cuda
  asm volatile("cp.async.commit_group;\n");
  ```
  * **作用**：把 **此指令之前** 已发出的 **`cp.async`** 划为 **同一组** 并 **提交（commit）** 给硬件的组计数逻辑；之后再发的 `cp.async` 属于 **下一组**，直到下一次 `commit_group`。
  * **直觉**：类似「这一批异步单发完了，打包成一个 logical batch」；**`wait_group`** 等的就是 **这种 batch 完成到还剩几个**。

* **与 `wait_group` 的关系**（记流程即可）：
  1. 连续多条 **`cp.async...`**（搬一段 tile / 一段 K）  
  2. **`cp.async.commit_group`** → 记为 **1 个 group**  
  3. 可再重复 1～2 做下一组（多 stage 流水）  
  4. **`cp.async.wait_group k`** → 阻塞直到 **未完成 group 数 ≤ k**（例如允许 pipeline 里仍剩 `k` 组在飞，更早的组须已收尾到一定程度）

* **`wait_group` 里 `k` 取多大？能改成 `1` 吗？**  
  * **语义**：**`k` 越小越严**——等到 **仍在飞的 commit 组** 越少才放行。  
  * **与 `NUM_STAGES` 配套**：多缓冲深度为 **`NUM_STAGES`** 时，稳态常用 **`wait_group(NUM_STAGES - 1)`**（如 3 路缓冲用 **`2`**），与 **「同时最多还有几组异步在飞」** 的设计一致。  
  * **改成 `wait_group 1`**：比 **`NUM_STAGES-1`**（当其为 2 时）**更严**，多数情况下 **仍可能正确**，但 **更早阻塞**、**重叠变少**、**吞吐可能下降**；是否等价要以 **实际发了多少 `commit_group`、组计数行为** 为准，**不能**保证任意改参都最优。  
  * **随意把 `k` 改大**（比设计允许的未完成组数更松）：可能在 **数据未完全到 shared** 时继续执行 → **错误结果**。  
  * **收尾**：**`wait_group 0`** 表示 **尽量排空**，与 **`k_block + (NUM_STAGES-1) >= num_k_blocks`** 分支配套，**不要用大 `k` 替代**。

* **若只用 `wait_group` 从不 `commit_group`**：部分工具链 / 用法下 **组边界依赖默认规则**；显式 **`commit_group`** 是 **推荐** 写法，流水与文档示例（CUTLASS 等）常成对出现。

* **`asm volatile`**：防止优化掉这条「只影响异步子系统状态」的指令。

### 与同步全局加载的对比

| 方式 | 特点 |
|------|------|
| 普通 `=` 或同步 `ld.global` | 简单；访存与后续指令顺序由内存依赖串起来，难重叠 |
| `cp.async` + `wait_*` | 显式 **流水**：**隐藏 Global→Shared 延迟**，提高 **Tensor Core 利用率** |
| `wmma::load_matrix_sync`（从 global） | 仍是 **warp 同步** 路径；与 `cp.async` 可先 **异步灌 shared**，再从 **shared** `load_matrix_sync` 到 fragment |

### 使用注意

* **架构**：旧 GPU 可能 **不支持** 或变体不同，需查 **compute capability**。
* **对齐与大小**：`cp.async` 常要求 **16B 的倍数**、地址对齐；非法组合会 **trap** 或未定义行为。
* **`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`**：调节的是 **launch 第三参动态 shared** 的**上限**；**静态** `__shared__` 数组大小由声明决定。若仅用静态 shared 且未超默认 per-block 上限，该调用可能 **无必要**；传 `0` 表示不额外请求动态 shared 字节数。
* **调试**：异步与多 stage 交错时，**wait 漏写或 stage 索引错**易导致 **偶发错误结果**，需仔细对照流水线状态机。

### 新架构扩展（Hopper 及以后，概要）

Ampere 上的 **`cp.async` + `wait_group`** 仍是基础模型。从 **Hopper（SM 9.x，如 H100）** 起，异步路径更丰富，常见知识点如下（细节以 **CUDA / PTX Release Notes** 为准）：

* **`mbarrier`（memory barrier in shared memory）**：在共享内存里放 **屏障对象**，用于 **异步拷贝完成** 与 **消费者 warp** 之间的同步；PTX 中有与 **`cp.async`** 联用的变体（如带 **mbarrier** 的异步拷贝），语义比仅 `wait_group` 更灵活（多生产者/多阶段、与 WGMMA 等配合）。
* **`cp.async.bulk` / Bulk Copy**：**更大粒度** 的异步搬运（Bulk），适合 **整块 tile** 搬运；与早期 **16B 步进** 的 `cp.async` 并存，按场景选用。
* **TMA（Tensor Memory Accelerator）**：Hopper 引入的 **硬件单元**，支持从 **全局内存** 按 **张量描述符（tensor map）** 向 **共享内存** 发起 **异步、多维步长** 的加载/存储；CUDA 中可通过 **CUDA C++ / PTX / CUTLASS** 等使用。可理解为 **比手写循环 `cp.async` 更「整块、可描述」** 的搬运路径，常与 **Tensor Core 第四代（WGMMA）** 流水线搭配。
* **后续架构（如 Blackwell）**：在 **TMA、异步内存、新 Tensor Core 指令** 上继续演进；学习时仍以 **当前 toolkit 的 Programming Guide + PTX ISA** 为准，**不可**假定 Hopper 代码零修改即得最优点。

**关系一句话**：**Ampere `cp.async` 是「细粒度异步 Global→Shared」的入门；Hopper+ 在 `mbarrier`、Bulk、`TMA` 上把「谁完成、谁等待、一次搬多大」做得更硬件化、更适合大 tile GEMM。**

### 一句话

**`cp.async` = Ampere 起常用的 PTX 级 Global→Shared 异步拷贝；用 `commit_group` 划批、`wait_group` 控制未完成批数，再配合多缓冲，实现 GEMM 等算子中访存与 Tensor Core 计算重叠。**

# 按速度划分
* Register: 线程独享
* shared memory: block内线程共享
* L1 cache: 每个SM共享，自动缓存
* L2 cache: 所有SM共享
* global memory: 所有线程共享

![H100-Streaming-Multiprocessor-SM-625x869.png](images/H100-Streaming-Multiprocessor-SM-625x869.png)

* LD/ST unit: 加载/存储单元
* SFU: 执行复杂数学函数
* Tensor Core: 执行矩阵乘法的硬件单元
* warp scheduler
  * 监控warp状态，哪些就绪，哪些等待
  * 选择并发射指令，挑选就绪的warp执行
  * 隐藏内存延迟，当一个warp等待时，切换到另一个

# shared memory 分配机制
* 每个SM有一个固定大小的shared memory池；当多个block thread被调度到同一个SM时，shared memory池会被分区，每个block获得自己独立的一块shared memory。



| 内存类型 | 位置 | 作用域 | 生命周期 | 速度 | 容量 | 声明方式 | 典型用途 |
|---------|------|--------|---------|------|------|----------|---------|
| **Register** | SM片内 | 线程私有 | 线程 | 最快(~1 cycle) | 64KB/SM | 自动变量 | 循环变量、临时计算 |
| **Shared Memory** | SM片内 | Block内共享 | Block | 很快(~28 cycles) | 48-192KB/SM | `__shared__` | Tile数据、线程协作 |
| **Local Memory** | 片外(Global) | 线程私有 | 线程 | 慢(~400 cycles) | 无限制 | 大数组/spill | ⚠️尽量避免 |
| **Constant Memory** | 片外(Global) | Grid只读 | 应用程序 | 快(缓存命中) | 64KB | `__constant__` | 卷积核、常量参数 |
| **Global Memory** | 片外(HBM) | 所有线程读写 | cudaFree前 | 最慢(~800 cycles) | GB级 | `cudaMalloc` | 主要数据存储 |

## 面试常考点

### Q1: Register vs Local Memory
**关键区别：**
- Register在SM片内，极快；Local Memory虽然叫"local"但实际在片外Global Memory中，很慢
- 使用太多寄存器或定义大数组会导致**寄存器溢出(register spilling)**，被迫使用Local Memory
- 可通过`nvcc --ptxas-options=-v`查看寄存器使用情况

```cuda
// ❌ 会用Local Memory（太大）
__global__ void bad() {
    float arr[1000];  // 寄存器放不下 -> Local Memory
}

// ✅ 使用Shared Memory
__global__ void good() {
    __shared__ float arr[1000];  // 片内，快
}
```

### Q2: Shared Memory的优势和注意事项
**优势：**
- 比Global Memory快10-20倍
- Block内线程可共享数据，减少重复读取Global Memory
- 可编程控制的缓存

**注意事项：**
1. **必须同步**：使用`__syncthreads()`避免race condition
2. **Bank Conflict**：32个bank，相邻4字节映射到不同bank
3. **容量限制**：影响occupancy（每个SM能同时运行的block数）

```cuda
__shared__ float data[32];

// ❌ 32-way bank conflict（同一warp的32个线程访问同一个bank的不同地址）
data[threadIdx.x * 32] = 0;

// ✅ 无bank conflict（相邻线程访问相邻地址 -> 不同bank）
data[threadIdx.x] = 0;
```

### Q3: Constant Memory的使用场景
**最佳场景：warp内所有线程读取相同地址（广播机制）**
```cuda
__constant__ float kernel_3x3[9];  // 卷积核

__global__ void conv() {
    // ✅ 所有线程读相同值 -> 1次内存访问，广播给32个线程
    float w0 = kernel_3x3[0];
    
    // ❌ 每个线程读不同值 -> 串行化，32次内存访问
    float val = kernel_3x3[threadIdx.x];  // 慢！
}
```

**限制：**
- 总容量仅64KB
- 只读（kernel中不能修改）
- 使用`cudaMemcpyToSymbol()`从host传输数据

### Q4: Global Memory访问优化
**关键：Coalesced Access（合并访问）**

**与上文「32」的关系（面试常澄清）：**
- **Warp 有 32 个线程**：执行调度单位，与「一次传多少字节」无必然对应。
- **约 32 字节的 sector**：全局/L2 路径上常见的**最小合并块**，一次 warp 若各读 4B 且地址连续，逻辑上约 **128B**，往往对应**多个** 32B 级别的传输，而不是「整条指令只搬 32B」。
- **优化目标**：提高**合并度**——让「实际搬动的字节」尽量等于「warp 真正需要的字节」，减少白读的 sector。
- **缓存**：若数据已在 L1/L2 命中，则可能**不再从 DRAM 取数**；合并访问仍重要，因为未命中时 DRAM 带宽与 sector 浪费会直接体现为性能。

```cuda
// ✅ 合并访问：同一 warp 内线程访问连续、对齐的地址
__global__ void coalesced(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];  // lane0→data[0], lane1→data[1], ...
    // 32 个 float ≈ 128B 连续；硬件通常用少量事务（多按 32B sector 计）即可满足，合并度高
}

// ❌ 非合并访问：跨步/分散，易触发大量 sector，合并度低
__global__ void strided(float* data) {
    int idx = threadIdx.x * 32;
    float val = data[idx];  // lane0→data[0], lane1→data[32], ...
    // 各线程地址相距远，可能为少量数据各自拉整 sector，浪费大、事务数多
}
```

**口诀：** 让 `threadIdx.x`（或等价线性下标）对应**连续内存**；避免 `threadIdx.x * stride`（大 stride）或错位访问主序与线程序不一致的二维数组。

### Q5: L1/L2 Cache与Shared Memory的关系
**关键点：**
- **L1 Cache和Shared Memory共享同一块物理SRAM**（现代架构如Ampere）
  - Volta: 128KB统一L1/Shared，可配置比例
  - Ampere A100: 192KB统一L1/Shared
- **L2 Cache是独立的**：所有SM共享，缓存Global Memory数据
  - A100: 40MB L2
  - 程序员无法直接控制，硬件自动管理

```
内存层次（从快到慢）：
Register (最快) 
    ↓ 
Shared Memory / L1 Cache (物理上同一块SRAM)
    ↓
L2 Cache (所有SM共享，缓存Global Memory)
    ↓
Global Memory (HBM/GDDR，实际数据存储位置)
```

### Q6: 如何选择合适的内存类型？
1. **单个变量** → Register（编译器自动）
2. **需要Block内线程共享** → Shared Memory
3. **小量只读数据，所有线程读相同值** → Constant Memory
4. **大量数据** → Global Memory（注意合并访问）
5. **避免使用** → Local Memory（会自动产生，尽量避免）
