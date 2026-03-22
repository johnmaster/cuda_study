# GPU 设备信息查询

在 CUDA 开发中，了解当前 GPU 的硬件参数（SM 数量、线程限制、共享内存大小、是否支持 Tensor Core 等）是性能调优的基础。查询方法分为**命令行工具**和**编程 API** 两大类。

## 一、命令行工具：nvidia-smi

`nvidia-smi`（NVIDIA System Management Interface）是驱动自带的命令行工具，无需编程即可使用。

### 1. 直接运行（最常用）

```bash
nvidia-smi
```

显示仪表盘式输出：GPU 型号、温度、功耗、显存使用率、正在运行的 GPU 进程。**想看"显存还剩多少"时就敲这个。**

### 2. 结构化查询：`--query-gpu`

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,compute_cap --format=csv
```

- `--query-gpu=字段1,字段2,...`：指定要查的字段
- `--format=csv`：以 CSV 格式输出，方便脚本解析

常用字段：

| 字段 | 含义 |
|------|------|
| `name` | GPU 型号名称 |
| `driver_version` | 驱动版本 |
| `memory.total` | 总显存 |
| `memory.used` | 已用显存 |
| `memory.free` | 剩余显存 |
| `compute_cap` | 计算能力 (Compute Capability) |
| `temperature.gpu` | GPU 温度 |
| `power.draw` | 当前功耗 |
| `clocks.sm` | SM 时钟频率 |
| `utilization.gpu` | GPU 利用率 |

查看所有支持的字段：`nvidia-smi --help-query-gpu`

### 3. 全量详细信息

```bash
nvidia-smi -q
```

输出非常长，包含 PCI 总线、功耗限制、ECC 状态等几乎所有驱动层面的信息。

### 4. nvidia-smi 的局限

`nvidia-smi` **无法查询** SM 数量、每 SM 线程数、共享内存大小、寄存器数量等硬件微架构参数——这些必须通过 CUDA Runtime API 编程获取。

## 二、CUDA 编译器版本：nvcc

```bash
nvcc --version
```

仅查看 **CUDA Toolkit 版本**（如 12.4），不查硬件参数。但需要确认 Toolkit 版本是否支持目标特性（如 WMMA 需要 CUDA 9.0+）。

## 三、编程查询：cudaGetDeviceProperties（最全面）

这是获取 GPU 硬件参数的**核心 API**，SM 数、线程数、共享内存等信息只有通过编程才能拿到。

### 核心用法（两步）

```cpp
cudaDeviceProp prop;                         // 第1步：声明结构体
cudaGetDeviceProperties(&prop, device_id);   // 第2步：传入设备ID，填充结构体
// 然后通过 prop.xxx 访问各字段
```

### cudaDeviceProp 关键字段

#### 身份信息

| 字段 | 含义 | RTX 3080 示例值 |
|------|------|----------------|
| `prop.name` | GPU 名称 | "NVIDIA GeForce RTX 3080" |
| `prop.major` | Compute Capability 主版本号 | 8 |
| `prop.minor` | Compute Capability 次版本号 | 6 |

`prop.major` 和 `prop.minor` 合起来就是计算能力，如 8.6。

#### SM 与线程（最重要）

| 字段 | 含义 | RTX 3080 值 |
|------|------|------------|
| `prop.multiProcessorCount` | SM（流多处理器）数量 | 68 |
| `prop.maxThreadsPerMultiProcessor` | 每个 SM 最大驻留线程数 | 1536 |
| `prop.warpSize` | Warp 大小（永远32） | 32 |
| `prop.maxThreadsPerBlock` | 单个 Block 最大线程数 | 1024 |
| `prop.maxBlocksPerMultiProcessor` | 每个 SM 最大驻留 Block 数 | 16 |

**这些数字之间的关系**：
- 每 SM 最多 1536 线程 ÷ warp 大小 32 = 每 SM 最多 **48 个 warp**
- 全卡理论最大线程 = SM 数 × 每 SM 最大线程 = 68 × 1536 = **104,448**
- 每个 Block 最多 1024 线程，但每 SM 最多 16 个 Block，且总线程不超过 1536

#### 内存层次

| 字段 | 含义 | RTX 3080 值 |
|------|------|------------|
| `prop.totalGlobalMem` | 全局显存（字节） | ~10 GB |
| `prop.sharedMemPerMultiprocessor` | 每 SM 共享内存总量 | 100 KB |
| `prop.sharedMemPerBlock` | 每 Block 可用共享内存（默认） | 48 KB |
| `prop.regsPerMultiprocessor` | 每 SM 寄存器总数 | 65536 |
| `prop.regsPerBlock` | 每 Block 可用寄存器数 | 65536 |
| `prop.l2CacheSize` | L2 缓存大小 | 5 MB |

### 判断 WMMA / Tensor Core 支持

WMMA（Warp Matrix Multiply-Accumulate）是 Tensor Core 的编程接口：

```cpp
bool wmma_support = (prop.major > 7) || (prop.major == 7 && prop.minor >= 0);
// Compute Capability >= 7.0 即支持 WMMA
```

不同代的 Tensor Core 支持的精度：

| Compute Capability | 架构 | 支持精度 |
|---|---|---|
| 7.0 (V100) | Volta | FP16 |
| 7.5 (RTX 2080) | Turing | FP16 |
| 8.0 (A100) | Ampere | FP16, BF16, TF32, INT8 |
| 8.6 (RTX 3080) | Ampere | FP16, BF16, TF32 |
| 9.0 (H100) | Hopper | FP16, BF16, TF32, FP8 |

## 四、单属性查询：cudaDeviceGetAttribute

如果只需查**某一个属性**，不需要填充整个结构体：

```cpp
int value;
cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0);
// 参数：输出指针, 属性枚举值, 设备ID
```

常用枚举值：

| 枚举值 | 含义 |
|--------|------|
| `cudaDevAttrMultiProcessorCount` | SM 数量 |
| `cudaDevAttrMaxThreadsPerMultiProcessor` | 每 SM 最大线程 |
| `cudaDevAttrMaxThreadsPerBlock` | 每 Block 最大线程 |
| `cudaDevAttrMaxSharedMemoryPerMultiprocessor` | 每 SM 共享内存 |
| `cudaDevAttrMaxRegistersPerMultiprocessor` | 每 SM 寄存器 |

## 五、完整查询工具代码

项目根目录下的 `device_query.cu` 是一个完整的查询工具：

```bash
nvcc -o device_query device_query.cu
./device_query
```

程序逻辑：
1. `cudaGetDeviceCount(&count)` → 获取 GPU 数量
2. 循环每块 GPU，调用 `cudaGetDeviceProperties(&prop, id)` 填充属性
3. 打印 SM、线程、内存、WMMA 等信息
4. 根据 `prop.major` 判断 Tensor Core 支持和精度

## 六、查询方法速查表

| 你想查的 | 用什么 | 命令/代码 |
|----------|--------|-----------|
| 显存用量、温度、功耗 | nvidia-smi | `nvidia-smi` |
| GPU 型号、驱动版本 | nvidia-smi | `nvidia-smi --query-gpu=name,driver_version --format=csv` |
| CUDA Toolkit 版本 | nvcc | `nvcc --version` |
| SM 数量 | 编程 | `prop.multiProcessorCount` |
| 每 SM 最大线程数 | 编程 | `prop.maxThreadsPerMultiProcessor` |
| 共享内存大小 | 编程 | `prop.sharedMemPerMultiprocessor` |
| 是否支持 WMMA | 编程 | `prop.major >= 7` |
| 全部硬件参数 | 编程 | 编译运行 `device_query.cu` |

---

# 从硬件参数到 WMMA 程序设计：最大化 GPU 性能

本节以 **RTX 3080 (Compute 8.6)** 为例，讲解如何根据硬件参数推导代码中的每一个设计决策。

## 硬件参数速查（RTX 3080）

```
SM 数量:                       68
每 SM 最大线程数:               1536
每 SM 最大 warp 数:             48
每 SM 最大 block 数:            16
每 block 最大线程数:            1024
Warp 大小:                     32
每 SM 共享内存:                 100 KB
每 block 共享内存 (默认):       48 KB
每 SM 寄存器:                   65536
WMMA 基础 tile 大小:            16×16×16 (FP16)
```

## 核心思路：五个参数的决策链

写一个 WMMA GEMM，要确定五个核心参数，它们环环相扣：

```
Step 1: WMMA tile 大小 → 由硬件决定，不可选
Step 2: 每个 warp 处理多少个 tile → 影响寄存器用量
Step 3: 每个 block 有多少个 warp → 影响共享内存复用
Step 4: 每个 block 的共享内存分块大小 (BLOCK_K) → 影响共享内存总量
Step 5: grid 大小 → 需要铺满所有 68 个 SM
```

## Step 1: WMMA Tile 大小（硬件固定）

```cpp
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
```

这是 **硬件决定的**，不可更改。在 Compute 8.6 (Ampere) 上，FP16 WMMA 支持的 tile 尺寸为：
- 16×16×16（最常用，性能最好）
- 32×8×16 和 8×32×16（特殊场景）

一次 `wmma::mma_sync` 调用 = 一个 warp (32线程) 协作完成 16×16×16 的矩阵乘加。

**关键理解**：WMMA 是 warp 级操作，一个 warp 的 32 个线程共同持有一个 fragment，不是每个线程独立算。

## Step 2: 每个 Warp 处理多少个 Tile（寄存器约束）

这是第一个设计决策：`WARP_ROW_TILES` 和 `WARP_COL_TILES`。

### 寄存器代价分析

每个 `wmma::fragment<accumulator, 16, 16, 16, float>` 在每个线程中占用的寄存器：
- 16×16 = 256 个 float 结果，分摊到 32 个线程 → 每线程 8 个 float → **8 个寄存器**

如果每个 warp 处理 `WARP_ROW_TILES × WARP_COL_TILES` 个 tile：

| WARP_ROW_TILES × WARP_COL_TILES | 累加器数量 | 每线程寄存器（仅累加器） |
|---|---|---|
| 1×1 | 1 | 8 |
| 2×2 | 4 | 32 |
| 4×2 | 8 | 64 |
| 4×4 | 16 | 128 |

加上 a_frag、b_frag 和其他变量，每线程实际使用的寄存器约为累加器的 1.5~2 倍。

### 寄存器预算计算

RTX 3080：每 SM 有 65536 个寄存器，每 SM 最多 1536 个线程。

```
每线程可用寄存器 = 65536 / 1536 = 42.67 → 实际最多 42 个
```

但如果每线程用更多寄存器，就无法达到满 occupancy：

| 每线程寄存器数 | 每 SM 最多线程 | 占用率 (Occupancy) |
|---|---|---|
| 32 | 1536 (=65536/42.67, 受限于1536) | 100% |
| 64 | 1024 (=65536/64) | 66.7% |
| 128 | 512 (=65536/128) | 33.3% |

**设计决策**：`WARP_ROW_TILES=2, WARP_COL_TILES=2`（如你的 V2-V4）是一个好的平衡点。累加器用 32 寄存器，加上其他开销约 50-64，occupancy 约 66-100%。

**更激进**：`4×2=8` 个 tile 每 warp，寄存器多但计算密度更高——适合 compute-bound 场景（大矩阵）。但 occupancy 会降到 ~33%，需要靠高计算密度弥补。

## Step 3: 每个 Block 有多少个 Warp（共享内存约束）

这决定了 `BLOCK_ROW_WARPS` 和 `BLOCK_COL_WARPS`。

### 设计原则

一个 block 的多个 warp 共享同一份 shared memory 数据——**warp 越多，数据复用率越高**。

```
block 的 warp 数 = BLOCK_ROW_WARPS × BLOCK_COL_WARPS
block 的线程数 = warp 数 × 32
```

当前代码配置：
```cpp
BLOCK_ROW_WARPS = 2, BLOCK_COL_WARPS = 2 → 4 warps → 128 线程/block
```

block 处理的输出区域：
```
BLOCK_M = BLOCK_ROW_WARPS × WARP_ROW_TILES × WMMA_M = 2 × 2 × 16 = 64
BLOCK_N = BLOCK_COL_WARPS × WARP_COL_TILES × WMMA_N = 2 × 2 × 16 = 64
```

即每个 block 计算 C 矩阵的一个 **64×64** 子块。

### 共享内存用量计算

一次 K 方向的分块加载：
```
As[BLOCK_M][BLOCK_K] = 64 × 32 × 2 bytes = 4096 bytes = 4 KB
Bs[BLOCK_K][BLOCK_N] = 32 × 64 × 2 bytes = 4096 bytes = 4 KB
总共 = 8 KB/block
```

双缓冲 (V3)：8 KB × 2 = **16 KB/block**

带 padding 的 V4：
```
As[64][32+8] × 2 = 64 × 40 × 2 = 5120 bytes
Bs[32][64+8] × 2 = 32 × 72 × 2 = 4608 bytes
总共 ≈ 9.5 KB/block
```

### 每 SM 能放多少个 Block？

```
每 SM 共享内存 = 100 KB
每 block 共享内存 = 8 KB（V2）/ 16 KB（V3）/ 9.5 KB（V4）

V2: floor(100/8)  = 12 blocks/SM（受限于 16 max → 取 12）
V3: floor(100/16) = 6 blocks/SM
V4: floor(100/9.5) = 10 blocks/SM
```

### Occupancy 综合计算（多重约束取最小值）

以 V2 为例，4 warps = 128 线程/block，8 KB shared memory/block：

```
线程约束: floor(1536 / 128) = 12 blocks/SM
共享内存约束: floor(100KB / 8KB) = 12 blocks/SM
block 数约束: 16 blocks/SM (硬件上限)
寄存器约束: 取决于每线程寄存器数

→ 取最小值 = 12 blocks/SM
→ 活跃线程 = 12 × 128 = 1536
→ Occupancy = 1536 / 1536 = 100% （理想情况，实际受寄存器影响）
```

**关键洞察**：提高 occupancy 不是唯一目标。有时降低 occupancy 但增加每 warp 的计算量（更多 tiles），总吞吐量反而更高，因为 Tensor Core 的运算非常快，瓶颈往往在内存。

## Step 4: BLOCK_K 的选择（计算/访存比）

BLOCK_K 决定每次从全局内存加载多少数据到共享内存。

### 计算/访存比分析

一次共享内存分块的计算量和访存量：

```
加载数据量 = BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N (A片 + B片)
         = 64 × 32 + 32 × 64 = 4096 elements (half, 2 bytes each)
         = 8 KB

计算量(FLOPs) = BLOCK_M × BLOCK_N × BLOCK_K × 2 (乘加各算1次)
             = 64 × 64 × 32 × 2 = 524,288 FLOPs
```

计算/访存比 = 524288 / (8192 bytes) = **64 FLOPs/byte**

RTX 3080 的理论比值：
```
Tensor Core FP16 算力 ≈ 238 TFLOPS
显存带宽 ≈ 760 GB/s
计算/带宽比 = 238e12 / 760e9 ≈ 313 FLOPs/byte
```

**我们的 64 远小于 313**，说明当前是 **memory-bound（受限于访存）**。

提升方法：
- 增大 BLOCK_M / BLOCK_N → 增加数据复用（但共享内存增加）
- 增大 WARP_ROW_TILES / WARP_COL_TILES → 每次加载算更多（但寄存器增加）

如果用 128×128 的 block（8 warps, 4×2 tiles per warp）：
```
加载 = 128×32 + 32×128 = 16 KB
计算 = 128×128×32×2 = 2,097,152 FLOPs
比值 = 2097152 / 16384 = 128 FLOPs/byte （好了一倍）
```

## Step 5: Grid 大小（铺满所有 SM）

```cpp
dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
```

grid 的 block 总数应 **远大于 SM 数 × 每 SM blocks**：

```
目标: grid_blocks >> 68 × blocks_per_SM

例: M=N=4096, BLOCK_M=BLOCK_N=64
    grid = (64, 64) = 4096 blocks
    68 SM × 12 blocks/SM = 816 → 4096 >> 816 ✓

例: M=N=1024, BLOCK_M=BLOCK_N=64
    grid = (16, 16) = 256 blocks
    256 < 816 → SM 利用不充分！有些 SM 空闲
```

**规律**：矩阵小的时候，block 不够多，SM 利用率低。这就是为什么小矩阵（1024×1024）性能相对差。

## 完整参数决策总结

以你的 V2 配置为例，标注每个参数来自哪个硬件约束：

```
WMMA_M/N/K = 16      ← 硬件固定 (Compute 8.6 FP16)
WARP_ROW_TILES = 2   ← 平衡寄存器 (4 acc × 8 reg = 32 reg)
WARP_COL_TILES = 2
BLOCK_ROW_WARPS = 2  ← 平衡共享内存复用和 occupancy
BLOCK_COL_WARPS = 2
BLOCK_K = 32         ← 2×WMMA_K, 平衡计算访存比和共享内存

→ BLOCK_M = 2×2×16 = 64
→ BLOCK_N = 2×2×16 = 64
→ 线程数 = 4 warps × 32 = 128
→ 共享内存 = 8 KB/block
→ 每 SM 最多 12 blocks
```

## 七个版本的优化逻辑（对应 wmma_gemm 的 V1-V7）

| 版本 | 核心优化 | 解决什么瓶颈 |
|------|---------|-------------|
| V1 Naive | 直接从 Global Memory 加载 fragment | 基线，每次 mma 都访问全局内存 |
| V2 Shared | 先加载到 Shared Memory，多个 warp 复用 | 减少全局内存访问次数 |
| V3 Double Buffer | 计算当前块的同时，加载下一块 | 隐藏全局内存延迟（计算和访存重叠） |
| V4 Vectorized | float4 向量化加载 + padding 避免 bank conflict | 提高内存带宽利用率 |
| V5 Large Tile | 128×64 大 tile, 8 warps, 向量化 | 提高计算/访存比（数据复用率） |
| V6 cp.async | cp.async 硬件异步拷贝 + 3 阶段流水线 | 绕过寄存器，更深地隐藏延迟 |
| V7 Combined | 大 tile + cp.async + 向量化 + padding | 组合所有优化技术 |

## 进一步优化：V5 — 增大 Block Tile 到 128×64

### 核心思路

V4 的 block tile 为 64×64，计算/访存比为：
```
(2 × 64 × 64 × 32) / (64 × 32 + 32 × 64) × 2 = 262144 / 8192 = 32 FLOPs/byte
```

增大到 128×64 后：
```
(2 × 128 × 64 × 32) / (128 × 32 + 32 × 64) × 2 = 524288 / 12288 = 42.7 FLOPs/byte
```

更大的 tile 意味着**每次从全局内存加载的数据被更多 MMA 指令复用**，提升计算密度。

### 配置参数

```
BLOCK_ROW_WARPS = 4, BLOCK_COL_WARPS = 2   → 8 warps = 256 线程
WARP_ROW_TILES  = 2, WARP_COL_TILES  = 2   → 每 warp 处理 2×2 = 4 个 16×16 tile
BLOCK_M = 4×2×16 = 128,  BLOCK_N = 2×2×16 = 64
```

### 资源分析（RTX 3080, sm_86）

```
编译器报告:
  Used 56 registers, 14848 bytes smem (14.5 KB)

Occupancy 计算:
  寄存器: 56 reg × 256 threads = 14336 reg/block, 65536/14336 = 4 blocks/SM
  共享内存: 14848 B/block, 102400/14848 = 6 blocks/SM
  Warp: 8 warps/block, 48/8 = 6 blocks/SM
  Block 硬件限制: 16 blocks/SM
  瓶颈 = 寄存器 → 4 blocks/SM → 32 warps → 32/48 = 66.7% occupancy
```

虽然 occupancy 比 V4 (75%) 低，但更大的 tile 提升了数据复用率，实测 **性能提升约 10%**。

### 关键代码差异（vs V4）

相比 V4，只改了 launcher 的模板参数：
```cpp
// V4: 4 warps, 64×64 tile
constexpr int BLOCK_ROW_WARPS = 2, BLOCK_COL_WARPS = 2;  // 128 threads

// V5: 8 warps, 128×64 tile
constexpr int BLOCK_ROW_WARPS = 4, BLOCK_COL_WARPS = 2;  // 256 threads
```

kernel 本身完全复用 V4 的模板化实现（向量化加载 + padding），只是 tile 变大了。这就是**模板化设计的好处**：不改 kernel 代码，只调参数就能探索不同配置。

---

## 进一步优化：V6 — cp.async 异步拷贝 + 多阶段流水线

### 核心思路

V3 的双缓冲用**软件**方式重叠计算和访存：一边计算 buffer[0]，一边用线程把数据从 Global 搬到 buffer[1]。但这种搬运**经过寄存器**，占用线程资源。

Ampere (Compute 8.0+) 引入了 `cp.async` 硬件指令，可以**绕过寄存器**直接从 Global Memory 拷贝到 Shared Memory：

```cpp
// cp.async: Global → Shared，不经过寄存器，由硬件 DMA 执行
uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
    :: "r"(smem_addr), "l"(gmem_ptr));
```

配合 **3 阶段流水线**（而非双缓冲的 2 阶段），可以更深地隐藏内存延迟。

### cp.async 流水线模型

```
时间 →
Stage 0: [Load k=0 ] [Compute k=0] [Load k=3 ] [Compute k=3] ...
Stage 1: [Load k=32] [Compute k=32] [Load k=4 ] ...
Stage 2: [Load k=64] [Compute k=64] ...

三个 stage 轮转，保证每时每刻都有 1 个在计算、1 个在加载、1 个已就绪
```

### 关键 API：commit_group 和 wait_group

```cpp
// 发射异步拷贝
cp_async_16B(smem_dst, gmem_src);
cp_async_16B(smem_dst2, gmem_src2);
...
// 把以上所有 cp.async 打包成一个 "group"
asm volatile("cp.async.commit_group;\n");

// 等到未完成的 group 数 ≤ N
asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
// N = NUM_STAGES - 1 表示：最老的那个 group 完成（正好是当前要读的 stage）
```

### 重要 Bug 修复：流水线排空阶段

当 K 维度循环接近尾部，不再发射新的 prefetch，outstanding group 数减少。此时 `wait_group(NUM_STAGES-1)` **不再保证当前 stage 就绪**：

```
假设 NUM_STAGES=3, num_k_blocks=32:

k_block=29: 3 groups outstanding, wait_group(2) → 最老的完成 ✓ 无新 prefetch
k_block=30: 2 groups outstanding, wait_group(2) → 2 ≤ 2, 不等待! ✗ 数据可能未就绪!
k_block=31: 1 group  outstanding, wait_group(2) → 1 ≤ 2, 不等待! ✗
```

**修复方案**：排空阶段用 `wait_group(0)` 等待所有 group 完成：

```cpp
if (k_block + (NUM_STAGES - 1) >= num_k_blocks) {
    asm volatile("cp.async.wait_group 0;\n");       // 排空：等全部完成
} else {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 1));  // 稳态：流水线
}
__syncthreads();
```

### 资源分析

```
编译器报告:
  Used 55 registers, 29184 bytes smem (28.5 KB)

共享内存构成:
  3 stages × (64×40 + 32×72) × 2 bytes = 3 × (2560 + 2304) × 2 = 29184 B

Occupancy:
  寄存器: 55 × 128 = 7040 reg/block, 65536/7040 = 9 blocks/SM
  共享内存: 29184 B/block, 102400/29184 = 3 blocks/SM  ← 瓶颈！
  瓶颈 = 共享内存 → 3 blocks/SM → 12 warps → 25% occupancy
```

V6 在 64×64 tile 上因为 3 阶段共享内存太大，occupancy 仅 25%，性能反而不如 V4/V5。**教训：cp.async 的收益必须超过 occupancy 下降的损失**。

---

## 进一步优化：V7 — 终极组合（大 Tile + cp.async + 向量化 + Padding）

### 核心思路

V7 将 V5（大 tile 128×64）和 V6（cp.async 3 阶段流水线）合并，再加上 V4 的 padding 避免 bank conflict。目标：**同时获得大 tile 的高复用率和 cp.async 的硬件加速**。

### 配置参数

```
BLOCK_ROW_WARPS = 4, BLOCK_COL_WARPS = 2   → 8 warps = 256 线程
WARP_ROW_TILES  = 2, WARP_COL_TILES  = 2
BLOCK_M = 128, BLOCK_N = 64, BLOCK_K = 32
NUM_STAGES = 3
```

### 资源分析

```
编译器报告:
  Used 55 registers, 44544 bytes smem (43.5 KB)

共享内存构成:
  3 stages × (128×40 + 32×72) × 2 bytes = 3 × (5120 + 2304) × 2 = 44544 B

Occupancy:
  寄存器: 55 × 256 = 14080 reg/block, 65536/14080 = 4 blocks/SM
  共享内存: 44544 B/block, 102400/44544 = 2 blocks/SM  ← 瓶颈
  瓶颈 = 共享内存 → 2 blocks/SM → 16 warps → 33.3% occupancy
```

### 性能总结（所有版本对比）

实测结果（RTX 3080, 4096×4096×4096 矩阵）：

| 版本 | 核心优化 | Time (ms) | TFLOPS | vs cuBLAS | 编译器信息 |
|------|---------|-----------|--------|-----------|-----------|
| cuBLAS | NVIDIA 官方实现 | 2.24 | 61.5 | 1.00x | — |
| V1 Naive | 直接从 Global 加载 | 9.12 | 15.1 | 0.25x | 40 reg, 0 KB smem |
| V2 Shared | 共享内存缓存 | 10.79 | 12.7 | 0.21x | 54 reg, 8 KB smem |
| V3 Double Buffer | 双缓冲重叠计算/访存 | 18.30 | 7.5 | 0.12x | 55 reg, 16 KB smem |
| **V4 Vectorized** | float4 向量化 + padding | **2.89** | **47.5** | **0.77x** | 56 reg, 9.5 KB smem |
| **V5 Large Tile** | 128×64 大 tile + 向量化 | **2.69** | **51.0** | **0.83x** | 56 reg, 14.5 KB smem |
| V6 cp.async | 3 阶段异步流水线 | 3.49 | 39.4 | 0.64x | 55 reg, 28.5 KB smem |
| **V7 Combined** | 大 tile + cp.async + padding | **2.84** | **48.4** | **0.79x** | 55 reg, 43.5 KB smem |

不同矩阵尺寸下的最佳版本：

| 矩阵尺寸 | 最佳版本 | TFLOPS | vs cuBLAS |
|----------|---------|--------|-----------|
| 1024×1024 | V7 Combined | 37.6 | 0.88x |
| 2048×2048 | V7 Combined | 48.1 | 0.94x |
| 4096×4096 | V5 Large Tile | 51.0 | 0.83x |
| 8192×8192 | V5 Large Tile | 51.1 | 0.82x |

### 优化经验总结

1. **向量化加载是最大的性能跳跃**：V3→V4 从 7.5 TFLOPS 到 47.5 TFLOPS（6.3 倍），说明内存带宽利用率是第一瓶颈
2. **增大 tile 有稳定收益**：V4→V5 约 10% 提升，因为提高了计算/访存比
3. **cp.async 需要大 tile 配合**：V6 单独用 cp.async（64×64 tile）反而变慢，因为 3 阶段的共享内存翻了 3 倍，occupancy 暴跌
4. **V7 组合在小矩阵上最优**：1024/2048 尺寸下 V7 达到 cuBLAS 的 88-94%，因为小矩阵的 grid 较小，流水线的延迟隐藏更有价值
5. **V5 在大矩阵上最优**：4096/8192 尺寸下 V5 最佳，因为大矩阵有足够的 block 填满 SM，occupancy 比 cp.async 的延迟隐藏更重要
6. **共享内存是 occupancy 的隐形杀手**：V3（双缓冲 16 KB）和 V6（三阶段 28.5 KB）的 occupancy 分别只有 42% 和 25%，严重限制了性能

---

# 验证：理论推导 vs 实测数据

理论分析必须用工具验证。有三层验证手段，从编译期到运行时。

## 验证方法一：编译期 — `nvcc --ptxas-options=-v`

编译时加 `--ptxas-options=-v`，编译器会报告每个 kernel 的**寄存器数**和**共享内存用量**：

```bash
nvcc -arch=sm_86 -O3 --use_fast_math --ptxas-options=-v -std=c++17 \
     -o wmma_gemm main.cu -lcublas
```

### 实测编译器输出（RTX 3080, sm_86）

| Kernel | 每线程寄存器 | 共享内存 | spill |
|--------|-------------|---------|-------|
| V1 Naive | 40 | 0 KB | 0 |
| V2 Shared | 54 | 8 KB (8192 B) | 0 |
| V3 Double Buffer | 55 | 16 KB (16384 B) | 0 |
| V4 Vectorized | 56 | 9.5 KB (9728 B) | 0 |

**对比文档中的理论推导**：

- 共享内存：V2 = 8 KB, V3 = 16 KB, V4 ≈ 9.5 KB → **完全吻合**
- 寄存器：理论估计累加器 32 个 + 其他开销 ≈ 50-64 → 实测 54-56 → **吻合**
- spill = 0：没有寄存器溢出到 Local Memory，这很好

### 关键指标解读

```
ptxas info    : Used 54 registers, 8192 bytes smem, 388 bytes cmem[0]
                 ↑                   ↑                  ↑
            每线程寄存器数      共享内存/block      常量内存(kernel参数)
```

**spill stores / spill loads = 0** 表示没有寄存器溢出。如果这个数字不为 0，说明寄存器用超了，编译器被迫把数据存到 Local Memory（很慢）。

## 验证方法二：运行时 — Nsight Compute (`ncu`)

`ncu` 是 NVIDIA 的 kernel 级 profiler，可以测量**实际运行时**的 occupancy、寄存器使用、共享内存使用等。

### 基本命令

```bash
ncu --metrics <指标列表> --kernel-name regex:"kernel名" ./可执行文件
```

### 验证 occupancy 的常用指标

```bash
ncu --metrics \
  launch__registers_per_thread,\
  launch__shared_mem_per_block_driver,\
  launch__block_size,\
  launch__grid_size,\
  launch__occupancy_limit_registers,\
  launch__occupancy_limit_shared_mem,\
  launch__occupancy_limit_blocks,\
  launch__occupancy_limit_warps,\
  sm__maximum_warps_per_active_cycle_pct,\
  sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name regex:"wmma_gemm" ./wmma_gemm
```

### 实测 ncu 输出（8192×8192 矩阵）

| 指标 | V1 Naive | V2 Shared | V3 Double Buffer | V4 Vectorized |
|------|---------|----------|-----------------|--------------|
| 每线程寄存器 | 40 | 54 | 55 | 56 |
| block 大小 | 128 | 128 | 128 | 128 |
| grid 大小 | 1024 | 256 | 256 | 256 |
| 寄存器限制 (blocks/SM) | 12 | 9 | 9 | 9 |
| 共享内存限制 (blocks/SM) | 16 | 11 | 5 | 9 |
| warp 限制 (blocks/SM) | 12 | 12 | 12 | 12 |
| block 数硬件限制 | 16 | 16 | 16 | 16 |
| **理论最大 occupancy** | **100%** | **75%** | **41.67%** | **75%** |
| **实际 warp 活跃率** | **84.2%** | **31.3%** | **31.3%** | **31.0%** |

### ncu 指标含义详解

| ncu 指标 | 含义 |
|---------|------|
| `launch__registers_per_thread` | 每线程实际使用的寄存器数 |
| `launch__occupancy_limit_registers` | 受寄存器限制，每 SM 最多放多少 block |
| `launch__occupancy_limit_shared_mem` | 受共享内存限制，每 SM 最多放多少 block |
| `launch__occupancy_limit_warps` | 受 warp 数限制，每 SM 最多放多少 block |
| `launch__occupancy_limit_blocks` | 硬件 block 数上限（16） |
| `sm__maximum_warps_per_active_cycle_pct` | 理论最大 occupancy（取上面四个限制的最小值） |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **实际 occupancy**（运行时真正活跃的 warp 比例） |

### 对比理论推导 vs 实测

**V2 为例：**

文档中的理论推导（假设 100% occupancy）：
```
寄存器约束: floor(65536 / 54 / 32) × 32 / 1536 ... 
```
更精确地算：54 reg/thread × 128 threads/block = 6912 reg/block。
65536 / 6912 = 9.48 → floor = **9 blocks/SM** → 9 × 4 warps = 36 warps → 36/48 = **75%**

ncu 实测：`sm__maximum_warps_per_active_cycle_pct = 75%` → **完全吻合！**

文档中预估"occupancy 约 66-100%"，实际是 75%，在预估范围内。

**为什么文档预估有偏差？**
文档用的是粗略估计（"50-64 寄存器"），而编译器实际分配了 54 个。精确计算需要编译后才知道确切寄存器数。这就是为什么**必须用工具验证**。

**V3 为什么 occupancy 最低？**
因为双缓冲让共享内存翻倍（16 KB），`occupancy_limit_shared_mem = 5`，成为瓶颈 → 只有 5 × 4 = 20 warps → 20/48 = 41.67%。这直接解释了 V3 反而比 V2 慢的现象。

## 验证方法三：Nsight Systems (`nsys`) — 系统级时间线

```bash
nsys profile -o report ./wmma_gemm
nsys stats report.nsys-rep
```

`nsys` 看的是全局时间线：kernel 执行时间、内存拷贝、CPU/GPU 重叠等。适合发现"大块时间花在哪里"。而 `ncu` 看的是单个 kernel 内部细节。

## 总结：验证工具速查

| 你想验证的 | 用什么 | 命令 |
|-----------|--------|------|
| 编译期寄存器/共享内存用量 | nvcc | `nvcc --ptxas-options=-v ...` |
| 是否有寄存器溢出 | nvcc | 看 `spill stores / spill loads` |
| 运行时实际 occupancy | ncu | `ncu --metrics sm__warps_active...` |
| occupancy 的瓶颈是什么 | ncu | `launch__occupancy_limit_*` |
| kernel 执行时间 | ncu / nsys | `ncu --metrics gpu__time_duration.sum` |
| 全局时间线、CPU-GPU 重叠 | nsys | `nsys profile -o report ./app` |

**验证流程**：
1. 先编译加 `--ptxas-options=-v`，确认寄存器和共享内存在预期范围
2. 用 `ncu` 跑一遍，看实际 occupancy 和各项限制
3. 对比理论推导，如果有偏差，找出瓶颈并调整参数

---

# Nsight Compute (`ncu`) 完整使用教程

`ncu` 是 NVIDIA 的 **kernel 级别** profiler，分析单个 GPU kernel 的内部细节：寄存器、共享内存、occupancy、内存带宽、指令吞吐等。

## 安装位置

```bash
which ncu
# 通常在 /usr/local/cuda-XX.X/bin/ncu
ncu --version
```

## 基本用法

### 1. 最简单：默认分析

```bash
ncu ./my_cuda_app
```

对程序中**每个 kernel 的每次调用**都做默认分析。输出非常多，一般不这么用。

### 2. 指定 kernel 名称

```bash
ncu --kernel-name regex:"wmma_gemm_v2" ./wmma_gemm
```

- `--kernel-name`：只分析名称匹配的 kernel
- `regex:`：使用正则表达式匹配。比如 `regex:"v2|v4"` 同时匹配 V2 和 V4

### 3. 控制分析哪几次调用

同一个 kernel 可能被调用很多次（比如 warmup + benchmark），用 `--launch-skip` 和 `--launch-count` 控制：

```bash
ncu --kernel-name regex:"v4" --launch-skip 20 --launch-count 1 ./wmma_gemm
```

- `--launch-skip 20`：跳过前 20 次调用（跳过 warmup）
- `--launch-count 1`：只分析 1 次调用

### 4. 使用预定义分析集 (`--set`)

ncu 有四个预定义的分析集，从简到详：

| 分析集 | 包含内容 | 指标数 | 速度 |
|--------|---------|--------|------|
| `basic` | 启动参数、Occupancy、SpeedOfLight、负载分布 | ~144 | 快 |
| `detailed` | basic + 计算分析 + 内存分析 + Roofline | ~459 | 中 |
| `full` | 所有分析（含指令统计、调度器统计等） | ~613 | 慢 |
| `roofline` | SpeedOfLight + Roofline 图 | ~241 | 中 |

```bash
# 推荐日常使用 basic
ncu --set basic --kernel-name regex:"v4" --launch-skip 20 --launch-count 1 ./wmma_gemm

# 需要深入分析时用 detailed
ncu --set detailed --kernel-name regex:"v4" --launch-skip 20 --launch-count 1 ./wmma_gemm
```

### 5. 指定具体指标 (`--metrics`)

如果只想看特定几个指标：

```bash
ncu --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name regex:"v2" --launch-skip 20 --launch-count 1 ./wmma_gemm
```

## 常用场景和指标

### 场景1：查看 Occupancy 及其瓶颈

```bash
ncu --metrics \
  launch__registers_per_thread,\
  launch__shared_mem_per_block_driver,\
  launch__block_size,\
  launch__grid_size,\
  launch__occupancy_limit_registers,\
  launch__occupancy_limit_shared_mem,\
  launch__occupancy_limit_blocks,\
  launch__occupancy_limit_warps,\
  sm__maximum_warps_per_active_cycle_pct,\
  sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name regex:"v2" --launch-skip 20 --launch-count 1 ./wmma_gemm
```

输出示例：
```
launch__registers_per_thread          register/thread    54
launch__block_size                                      128
launch__occupancy_limit_registers             block        9   ← 寄存器限制：9 blocks/SM
launch__occupancy_limit_shared_mem            block       11   ← 共享内存限制：11 blocks/SM
launch__occupancy_limit_warps                 block       12   ← warp 限制：12 blocks/SM
launch__occupancy_limit_blocks                block       16   ← 硬件上限：16 blocks/SM
sm__maximum_warps_per_active_cycle_pct           %       75   ← 理论最大 occupancy（取最小值）
sm__warps_active.avg.pct_of_peak_sustained_active  %    31.3  ← 实际运行时 occupancy
```

**怎么读**：四个 `occupancy_limit_*` 中的最小值 = 瓶颈。这里是 `registers = 9`，所以理论最大 = 9 blocks × 4 warps / 48 max warps = 75%。

### 场景2：用 `--set basic` 获取综合报告

```bash
ncu --set basic --kernel-name regex:"v4" --launch-skip 23 --launch-count 1 ./wmma_gemm
```

输出包含人类可读的诊断建议，例如：

```
OPT   The difference between calculated theoretical (75.0%) and measured
      achieved occupancy (31.8%) can be the result of warp scheduling
      overheads or workload imbalances during the kernel execution.

OPT   This kernel's theoretical occupancy (75.0%) is limited by the number
      of required registers. This kernel's theoretical occupancy (75.0%) is
      limited by the required amount of shared memory.
```

ncu 会直接告诉你**瓶颈是什么**以及**建议怎么优化**。

### 场景3：查看内存带宽利用率

```bash
ncu --metrics \
  dram__bytes.sum,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__throughput.avg.pct_of_peak_sustained_active,\
  lts__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:"v4" --launch-skip 23 --launch-count 1 ./wmma_gemm
```

| 指标 | 含义 |
|------|------|
| `dram__bytes.sum` | 全局内存总读写字节数 |
| `dram__throughput...pct` | 全局内存带宽利用率（% of peak） |
| `l1tex__throughput...pct` | L1 缓存/共享内存带宽利用率 |
| `lts__throughput...pct` | L2 缓存带宽利用率 |

### 场景4：查看计算吞吐率

```bash
ncu --metrics \
  sm__inst_executed.sum,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  smsp__inst_executed.sum \
  --kernel-name regex:"v4" --launch-skip 23 --launch-count 1 ./wmma_gemm
```

| 指标 | 含义 |
|------|------|
| `sm__inst_executed.sum` | 总执行指令数 |
| `sm__pipe_tensor_cycles_active...pct` | **Tensor Core 利用率**（最关键！） |

### 场景5：导出报告文件

```bash
# 导出为 ncu 专用格式（可用 Nsight Compute GUI 打开）
ncu --set detailed -o my_report --kernel-name regex:"v4" \
    --launch-skip 23 --launch-count 1 ./wmma_gemm
# 生成 my_report.ncu-rep，可以传回本地用 GUI 分析
```

## ncu 指标名称规则

ncu 的指标名遵循 `单元__子系统_操作.聚合方式` 的命名规则：

```
sm__warps_active.avg.pct_of_peak_sustained_active
│    │            │    │
│    │            │    └── 聚合：平均值占峰值持续活跃的百分比
│    │            └─────── 聚合：平均值
│    └──────────────────── 子系统：活跃 warp
└───────────────────────── 单元：SM
```

常见前缀：
- `sm__` — SM 级别
- `dram__` — 全局内存 (DRAM)
- `l1tex__` — L1/纹理缓存
- `lts__` — L2 缓存
- `launch__` — kernel 启动参数

常见后缀：
- `.sum` — 总和
- `.avg` — 平均
- `.pct_of_peak_sustained_active` — 占峰值的百分比（活跃周期内）
- `.pct_of_peak_sustained_elapsed` — 占峰值的百分比（总时间内）

## ncu 完整常用指标速查

| 指标 | 含义 | 用途 |
|------|------|------|
| `launch__registers_per_thread` | 每线程寄存器数 | 验证寄存器预算 |
| `launch__shared_mem_per_block_driver` | 每 block 共享内存 | 验证共享内存计算 |
| `launch__occupancy_limit_*` | occupancy 各项限制 | 找瓶颈 |
| `sm__maximum_warps_per_active_cycle_pct` | 理论最大 occupancy | 对比理论值 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际 occupancy | 实际运行效率 |
| `sm__pipe_tensor_cycles_active...pct` | Tensor Core 利用率 | WMMA 效率 |
| `dram__throughput...pct` | 显存带宽利用率 | 是否 memory-bound |
| `sm__inst_executed.sum` | 总指令数 | 计算量分析 |
| `gpu__time_duration.sum` | kernel 执行时间(ns) | 耗时 |

---

# Nsight Systems (`nsys`) 完整使用教程

`nsys` 是 NVIDIA 的 **系统级** profiler，看的是全局时间线：所有 kernel 的执行时间、内存拷贝耗时、CPU 和 GPU 的重叠情况。和 `ncu` 的区别：

| 工具 | 粒度 | 看什么 | 类比 |
|------|------|--------|------|
| `ncu` | 单个 kernel 内部 | 寄存器、occupancy、带宽 | 显微镜 |
| `nsys` | 整个程序的时间线 | 哪个 kernel 最慢、内存拷贝占多久 | 望远镜 |

**使用顺序**：先用 `nsys` 找出"大块时间花在哪"，再用 `ncu` 深入分析最慢的 kernel。

## 基本用法

### 1. 采集 + 命令行统计（最常用）

```bash
nsys profile --stats=true -o report ./wmma_gemm
```

- `profile`：采集模式
- `--stats=true`：采集完后自动输出统计表
- `-o report`：保存为 `report.nsys-rep`（可选）

### 2. 只采集，不输出统计

```bash
nsys profile -o report ./wmma_gemm
```

之后手动查看统计：

```bash
nsys stats report.nsys-rep
```

### 3. 覆盖已有文件

```bash
nsys profile -f true -o report ./wmma_gemm
```

`-f true`（force）：如果 `report.nsys-rep` 已存在，覆盖它。

## 输出解读

`nsys --stats=true` 会输出多个统计表，最重要的是以下三个：

### 表1：CUDA API 调用统计 (`cuda_api_sum`)

```
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)        Name
 --------  ---------------  ---------  -------------  -------------------------
     92.6   15,570,579,997         56  278,046,071.4  cudaDeviceSynchronize
      6.2    1,042,443,374         28   37,230,120.5  cudaMemcpy
      0.8      130,733,144         24    5,447,214.3  cudaMalloc
```

**怎么读**：
- 92.6% 的时间花在 `cudaDeviceSynchronize`（等 GPU 完成）→ 说明 GPU 是瓶颈，正常
- 6.2% 花在 `cudaMemcpy`（数据传输）→ 如果这个比例很大，说明传输是瓶颈
- 如果 `cudaMalloc` 占比高，考虑预分配内存

### 表2：GPU Kernel 执行时间统计 (`cuda_gpu_kern_sum`)（最重要）

```
 Time (%)  Total Time (ns)  Instances    Avg (ns)       Name
 --------  ---------------  ---------  ------------  ----------------------------------
     51.1    7,958,602,585       100   79,586,025.9  wmma_gemm_v1_naive
     25.8    4,021,380,541       100   40,213,805.4  wmma_gemm_v3_double_buffer
     15.5    2,417,106,707       100   24,171,067.1  wmma_gemm_v2_shared
      4.3      672,582,395       100    6,725,824.0  wmma_gemm_v4_vectorized
      2.8      436,972,345        25   17,478,893.8  cutlass_80_tensorop_s16816gemm... (cuBLAS)
```

**怎么读**：
- V1 最慢（51.1%），V4 最快（4.3%），cuBLAS 只占 2.8%
- **这就告诉你该优化哪个 kernel** → 如果要继续优化，先用 `ncu` 分析 V4
- Instances = 调用次数，Avg = 每次平均耗时

### 表3：内存拷贝统计 (`cuda_gpu_mem_time_sum`)

```
 Time (%)  Total Time (ns)  Count    Avg (ns)         Operation
 --------  ---------------  -----  ------------  ----------------------------
     91.9      937,290,805     20  46,864,540.3  [CUDA memcpy Device-to-Host]
      7.9       80,967,771      8  10,120,971.4  [CUDA memcpy Host-to-Device]
      0.2        1,960,580     16     122,536.3  [CUDA memset]
```

**怎么读**：
- D→H（设备到主机）拷贝 91.9%，因为要把结果拷回来验证
- 如果 H→D 或 D→H 占比极大，说明数据传输是瓶颈，考虑减少传输或用 pinned memory

## 高级用法

### 只追踪 CUDA 活动（过滤 OS 调用）

```bash
nsys profile --trace=cuda,cudnn,cublas -o report ./wmma_gemm
```

`--trace` 选项控制追踪哪些活动：
- `cuda`：CUDA Runtime 和 Driver API
- `cublas`：cuBLAS 调用
- `cudnn`：cuDNN 调用
- `nvtx`：用户自定义标记
- `osrt`：OS 运行时（默认包含，可排除）

### 导出为 GUI 可用的报告

```bash
nsys profile -o report ./wmma_gemm
# 生成 report.nsys-rep
# 可以传回本地，用 Nsight Systems GUI 打开查看时间线图
```

在 GUI 中可以看到：
- CPU 和 GPU 活动的时间线重叠图
- kernel 并发情况
- 内存拷贝与 kernel 的重叠

### 限制采集时间

```bash
nsys profile --duration 10 -o report ./wmma_gemm
```

只采集前 10 秒。

## nsys vs ncu 使用流程

```
第 1 步：nsys profile --stats=true ./app
         → 看哪个 kernel 最慢、内存拷贝占多少
         → 确定优化目标

第 2 步：ncu --set basic --kernel-name regex:"目标kernel" ./app
         → 看 occupancy、带宽利用率、Tensor Core 利用率
         → 找到瓶颈（寄存器？共享内存？带宽？）

第 3 步：修改代码，调整参数

第 4 步：重复 1-3，直到满意
```

## 完整工具速查

| 你想做什么 | 用什么 | 命令 |
|-----------|--------|------|
| 看所有 kernel 谁最慢 | nsys | `nsys profile --stats=true ./app` |
| 看内存拷贝占多少时间 | nsys | 同上，看 `cuda_gpu_mem_time_sum` 表 |
| 看单个 kernel 的 occupancy | ncu | `ncu --set basic --kernel-name regex:"xxx" ./app` |
| 看 occupancy 瓶颈是什么 | ncu | `ncu --metrics launch__occupancy_limit_* ...` |
| 看 Tensor Core 利用率 | ncu | `ncu --metrics sm__pipe_tensor_cycles_active...` |
| 看显存带宽利用率 | ncu | `ncu --metrics dram__throughput...` |
| 编译时看寄存器/共享内存 | nvcc | `nvcc --ptxas-options=-v ...` |
| 看寄存器溢出 | nvcc | 看输出中的 `spill stores / spill loads` |
| GUI 时间线分析 | nsys GUI | 把 `.nsys-rep` 文件传回本地打开 |
| GUI kernel 分析 | ncu GUI | 把 `.ncu-rep` 文件传回本地打开 |
