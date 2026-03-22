"""
不同精度的 GEMM 性能 & 精度 Benchmark

对比 FP32 / TF32 / FP16 / BF16 在矩阵乘法上的:
  - 计算速度 (TFLOPS)
  - 数值精度 (vs FP64 参考)
  - 显存占用

这直接反映了 Tensor Core 在不同精度下的吞吐量差异。

运行:
    python precision_benchmark.py
"""

import torch
import time


def bench_matmul(M, N, K, dtype, warmup=50, iters=200):
    """Benchmark matrix multiply at given dtype, return time in ms."""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.mm(A, B)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000

    tflops = 2 * M * N * K / (ms / 1000) / 1e12
    return ms, tflops


def measure_error(M, N, K, dtype):
    """Measure numerical error vs FP64 reference."""
    A_f64 = torch.randn(M, K, device="cuda", dtype=torch.float64)
    B_f64 = torch.randn(K, N, device="cuda", dtype=torch.float64)
    C_ref = torch.mm(A_f64, B_f64)

    A = A_f64.to(dtype)
    B = B_f64.to(dtype)
    C = torch.mm(A, B).double()

    abs_err = (C - C_ref).abs().max().item()
    rel_err = ((C - C_ref).abs() / (C_ref.abs() + 1e-12)).max().item()
    return abs_err, rel_err


def main():
    print("=" * 75)
    print("  Precision Benchmark: FP32 / TF32 / FP16 / BF16 GEMM")
    print("=" * 75)

    cc = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    print(f"  GPU: {gpu_name}  Compute Capability: {cc[0]}.{cc[1]}")

    dtypes = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
    ]
    if cc[0] >= 8:
        dtypes.append(("BF16", torch.bfloat16))

    sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

    # ── Performance ──

    print(f"\n  Performance (higher TFLOPS = better)")
    print(f"  {'Size':>20s}", end="")
    for name, _ in dtypes:
        print(f"  {name:>12s}", end="")
    print()
    print(f"  {'─'*20}", end="")
    for _ in dtypes:
        print(f"  {'─'*12}", end="")
    print()

    for M, N, K in sizes:
        label = f"{M}x{N}x{K}"
        print(f"  {label:>20s}", end="")
        for name, dtype in dtypes:
            ms, tflops = bench_matmul(M, N, K, dtype)
            print(f"  {tflops:>8.1f} TF/s", end="")
        print()

    # ── TF32 mode (Ampere+) ──

    if cc[0] >= 8:
        print(f"\n  TF32 Effect on FP32 GEMM (Ampere+ only)")
        print(
            f"  {'Size':>20s}  {'TF32=True':>12s}  {'TF32=False':>12s}  {'Speedup':>8s}"
        )
        print(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*8}")

        for M, N, K in sizes:
            label = f"{M}x{N}x{K}"

            torch.backends.cuda.matmul.allow_tf32 = True
            _, tflops_tf32 = bench_matmul(M, N, K, torch.float32)

            torch.backends.cuda.matmul.allow_tf32 = False
            _, tflops_fp32 = bench_matmul(M, N, K, torch.float32)

            speedup = tflops_tf32 / tflops_fp32
            print(
                f"  {label:>20s}  {tflops_tf32:>8.1f} TF/s  "
                f"{tflops_fp32:>8.1f} TF/s  {speedup:>6.2f}x"
            )

        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Numerical Error ──

    print(f"\n  Numerical Error (vs FP64 reference, M=N=K=1024)")
    print(f"  {'Dtype':<8s}  {'Max Abs Err':>12s}  {'Max Rel Err':>12s}  {'Bits':>20s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*20}")

    dtype_info = {
        "FP32": (torch.float32, "1+8+23 = 32 bits"),
        "FP16": (torch.float16, "1+5+10 = 16 bits"),
    }
    if cc[0] >= 8:
        dtype_info["BF16"] = (torch.bfloat16, "1+8+7  = 16 bits")

    for name, (dtype, bits) in dtype_info.items():
        abs_err, rel_err = measure_error(1024, 1024, 1024, dtype)
        print(f"  {name:<8s}  {abs_err:>12.2e}  {rel_err:>12.2e}  {bits:>20s}")

    # ── Memory ──

    print(f"\n  Memory per 4096x4096 matrix")
    print(f"  {'Dtype':<8s}  {'Bytes/elem':>10s}  {'Matrix size':>12s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}")
    for name, dtype in dtypes:
        elem_size = torch.tensor(0, dtype=dtype).element_size()
        mat_mb = 4096 * 4096 * elem_size / 1024**2
        print(f"  {name:<8s}  {elem_size:>10d}  {mat_mb:>9.1f} MB")

    print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  Tensor Core throughput by precision (A100):                   │
  │                                                                │
  │  FP64:   19.5 TFLOPS                                          │
  │  FP32:   19.5 TFLOPS (without TF32)                           │
  │  TF32:  156   TFLOPS (FP32 input, TF32 compute) ← 8x FP32!   │
  │  FP16:  312   TFLOPS                             ← 16x FP32!  │
  │  BF16:  312   TFLOPS                                           │
  │  INT8:  624   TOPS                                             │
  │                                                                │
  │  这就是为什么混合精度训练能快 2-3 倍:                             │
  │  大部分 GEMM 用 FP16/BF16 → Tensor Core 吞吐翻倍               │
  └────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
