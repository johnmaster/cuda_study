# wmma_gemm Nsight 分析报告

在 **NVIDIA GeForce RTX 4090 (Compute 8.9)** 上使用 Nsight Systems 与 Nsight Compute 对 `wmma_gemm` 程序进行分析。

---

## 1. Nsight Systems 分析 (nsys)

### 1.1 CUDA API 时间分布

| API | 时间占比 | 总时间 | 调用次数 | 说明 |
|-----|----------|--------|----------|------|
| **cudaDeviceSynchronize** | **86.5%** | ~5.82 s | 56 | 同步等待 GPU，说明大部分时间在跑 kernel |
| **cudaMemcpy** | 11.1% | ~744 ms | 28 | 主机与设备间拷贝（验证结果等） |
| cudaMalloc | 1.5% | ~99 ms | 24 | 每轮 benchmark 的分配 |
| 其他 | <1% | - | - | cuLibraryLoadData、cuKernelGetFunction 等 |

**结论**：程序以 GPU 计算为主，同步和结果回读占约 11%，符合 benchmark 形态。

### 1.2 GPU Kernel 时间分布（所有规模汇总）

| Kernel | 时间占比 | 总时间 | 调用次数 | 平均 (ns) |
|--------|----------|--------|----------|------------|
| **wmma_gemm_v1_naive** | **44.2%** | ~2.57 s | 100 | 25.7 ms |
| wmma_gemm_v3_double_buffer | 29.3% | ~1.71 s | 100 | 17.1 ms |
| wmma_gemm_v2_shared | 18.4% | ~1.07 s | 100 | 10.7 ms |
| **wmma_gemm_v4_vectorized** | **4.6%** | ~270 ms | 100 | 2.7 ms |
| cuBLAS (ampere_s16816gemm 等) | 3.0% | ~172 ms | 25 | 6.9 ms |

**结论**：
- 自研 WMMA 中 **V1 (Naive) 耗时最多**，V4 (Vectorized) 最快、总时间约为 V1 的 1/10。
- cuBLAS Tensor Core 仅 25 次调用（4 个规模 × 约 1 次计时），单次约 6.9 ms，仍是单次最快的实现。

### 1.3 GPU 内存操作

| 操作 | 时间占比 | 数据量 (MB) |
|------|----------|-------------|
| Device → Host | 94.7% | ~1782 MB |
| Host → Device | 5.1% | ~356 MB |
| memset | 0.2% | - |

大量 D2H 来自每轮验证时的 `cudaMemcpy(d_C → h_C)`，与 benchmark 设计一致。

### 1.4 生成文件

- `wmma_gemm_nsys.nsys-rep`：可在 **Nsight Systems GUI** (`nsys-ui`) 中打开，查看时间线、API、kernel 分布。
- `wmma_gemm_nsys.sqlite`：同一数据的 SQLite 形式。

---

## 2. Nsight Compute 分析 (ncu)

对 **wmma_gemm_v4_vectorized** 的**第一次启动**（矩阵 1024×1024×1024）做了 Roofline 集合分析。

### 2.1 Launch 配置

- **Grid**: (16, 16, 1) = 256 blocks  
- **Block**: (128, 1, 1)  
- **矩阵**: M=N=K=1024

### 2.2 Speed of Light / 吞吐 (RTX 4090)

| 指标 | 数值 | 说明 |
|------|------|------|
| Duration | 57.98 µs | 单次 kernel 执行时间 |
| Elapsed Cycles | 129,375 |  |
| **Memory Throughput** | **21.06%** | 显存带宽利用率 |
| DRAM Throughput | 7.38% |  |
| L1/TEX Throughput | 18.21% |  |
| L2 Throughput | 21.06% |  |
| **Compute (SM) Throughput** | **12.67%** | 计算单元利用率 |
| SM Active Cycles | 125,543 |  |

### 2.3 Nsight Compute 规则 (Rules)

- **SOLBottleneck (OPT)**：  
  - 说明：*This kernel grid is too small to fill the available resources on this device, resulting in only **0.2 full waves** across all SMs.*  
  - 含义：当前 grid 只有 256 个 block，而 GPU 有 128 个 SM，每波约 128 个 block，约 2 波才能占满；实际只有约 0.2 波，**SM 占用率低**，属于 **launch 规模过小** 的瓶颈。

- **SOLFPRoofline (INF)**：  
  - 与 Roofline 相关，当前 kernel 相对 fp32/fp64 峰值的占比为 0%（因是 FP16 Tensor Core 等，此处仅作参考）。

### 2.4 小结

- V4 在 1024 规模下：**计算与显存利用率都不高**（SM ~12.7%，Memory ~21%），主要受 **grid 太小、并行度不足** 影响。
- 增大问题规模（如 4096、8192）时，block 数增多，SM 利用率和带宽利用率会提升，与你在 benchmark 中看到的 V4 在大规模下更接近 cuBLAS 的现象一致。

### 2.5 生成文件

- `wmma_gemm_ncu.ncu-rep`：可在 **Nsight Compute GUI** (`ncu-ui`) 中打开，查看详细指标与 Roofline。
- 命令行查看摘要：  
  `ncu -i wmma_gemm_ncu.ncu-rep --print-summary per-kernel`  
- 导出 CSV：  
  `ncu -i wmma_gemm_ncu.ncu-rep --csv --page details`

---

## 3. 如何复现与扩展

```bash
# Nsight Systems：采集时间线与 API/Kernel 统计
nsys profile --stats=true --trace=cuda,nvtx --output=wmma_gemm_nsys ./wmma_gemm

# Nsight Compute：只分析 V4 kernel，第一次 launch，Roofline 指标
ncu --set roofline -k "regex:wmma_gemm_v4" --launch-count 1 -o wmma_gemm_ncu ./wmma_gemm

# 查看 ncu 报告
ncu -i wmma_gemm_ncu.ncu-rep --print-summary per-kernel
ncu -i wmma_gemm_ncu.ncu-rep --csv --page details
```

若要分析**大规模**（如 8192）下的 V4 或其它 kernel，可调节 `--launch-skip` 与 `--launch-count`，或对单一规模写一个小程序只跑该规模再 profiling。

---

## 4. 总结

| 工具 | 作用 | 主要结论 |
|------|------|----------|
| **Nsight Systems** | 时间线、API/Kernel 时间占比、内存拷贝 | 耗时主要在 V1，V4 与 cuBLAS 占比小；同步与 D2H 占约 11%。 |
| **Nsight Compute** | Kernel 级吞吐、Roofline、规则建议 | V4 在 1024 规模下 SM 与显存利用率低，主要因 **grid 过小（0.2 waves）**；建议增大问题规模或增大 grid 以提升占用率。 |

报告生成时间：与本次分析运行时间一致（见 nsys/ncu 输出时间戳）。
