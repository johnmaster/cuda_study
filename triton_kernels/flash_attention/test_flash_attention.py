"""
Triton kernels — correctness + performance test suite.

Run:
    cd triton_kernels
    python test_all.py
"""

import torch
import time


# ── Flash Attention Tests ────────────────────────────────────────────────────

def reference_attention(Q, K, V):
    d = Q.size(-1)
    S = Q @ K.transpose(-1, -2) / (d ** 0.5)
    P = torch.softmax(S, dim=-1)
    return P @ V


def test_flash_attn_forward():
    from flash_attention import FlashAttentionTriton
    attn = FlashAttentionTriton()

    print("Flash Attention — Forward Correctness")
    for B, N, D in [(1, 32, 32), (1, 64, 64), (2, 64, 32), (1, 128, 64), (2, 128, 128)]:
        torch.manual_seed(42)
        Q = torch.randn(B, N, D, device="cuda")
        K = torch.randn(B, N, D, device="cuda")
        V = torch.randn(B, N, D, device="cuda")

        O = attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)
        diff = (O - O_ref).abs().max().item()
        ok = diff < 1e-3
        print(f"  B={B} N={N:>4} D={D:>3}  max_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")
        assert ok


def test_flash_attn_backward():
    from flash_attention import FlashAttentionTriton
    attn = FlashAttentionTriton()

    print("\nFlash Attention — Backward Correctness")
    for B, N, D in [(1, 32, 32), (1, 64, 64), (2, 64, 32), (1, 128, 64)]:
        torch.manual_seed(42)
        Q = torch.randn(B, N, D, device="cuda", requires_grad=True)
        K = torch.randn(B, N, D, device="cuda", requires_grad=True)
        V = torch.randn(B, N, D, device="cuda", requires_grad=True)
        Q_ref = Q.clone().detach().requires_grad_(True)
        K_ref = K.clone().detach().requires_grad_(True)
        V_ref = V.clone().detach().requires_grad_(True)

        attn(Q, K, V).sum().backward()
        reference_attention(Q_ref, K_ref, V_ref).sum().backward()

        diffs = {
            "dQ": (Q.grad - Q_ref.grad).abs().max().item(),
            "dK": (K.grad - K_ref.grad).abs().max().item(),
            "dV": (V.grad - V_ref.grad).abs().max().item(),
        }
        ok = all(v < 1e-2 for v in diffs.values())
        parts = "  ".join(f"{k}={v:.2e}" for k, v in diffs.items())
        print(f"  B={B} N={N:>4} D={D:>3}  {parts}  {'PASS' if ok else 'FAIL'}")
        assert ok


# ── GEMM Tests ───────────────────────────────────────────────────────────────

def test_gemm_correctness():
    from gemm import triton_gemm_v1, triton_gemm_v2, triton_gemm_v3

    print("\nGEMM — Correctness")
    for M, N, K in [(128, 128, 64), (256, 512, 128), (512, 512, 256), (1024, 1024, 512)]:
        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        ref = A @ B

        for name, fn in [("V1", triton_gemm_v1), ("V2", triton_gemm_v2), ("V3", triton_gemm_v3)]:
            C = fn(A, B)
            diff = (C - ref).abs().max().item()
            ok = diff < 1e-1
            print(f"  {name} M={M:>4} N={N:>4} K={K:>3}  max_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")
            assert ok, f"{name} failed at M={M} N={N} K={K} diff={diff}"


# ── Performance ──────────────────────────────────────────────────────────────

def bench(fn, *args, warmup=20, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def benchmark_gemm():
    from gemm import triton_gemm_v1, triton_gemm_v2, triton_gemm_v3

    print("\nGEMM — Performance (ms)")
    print(f"  {'Size':>20s}  {'V1':>8s}  {'V2':>8s}  {'V3':>8s}  {'cuBLAS':>8s}")
    for M, N, K in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")

        t1 = bench(triton_gemm_v1, A, B)
        t2 = bench(triton_gemm_v2, A, B)
        t3 = bench(triton_gemm_v3, A, B)
        t_ref = bench(torch.mm, A, B)

        print(f"  {M}x{N}x{K:>4}  {t1:8.3f}  {t2:8.3f}  {t3:8.3f}  {t_ref:8.3f}")


def benchmark_flash_attn():
    from flash_attention import FlashAttentionTriton
    attn = FlashAttentionTriton()

    print("\nFlash Attention — Performance (ms)")
    print(f"  {'Config':>20s}  {'Triton':>8s}  {'PyTorch':>8s}  {'Speedup':>8s}")
    for B, N, D in [(1, 128, 64), (1, 256, 64), (1, 512, 64), (4, 256, 64)]:
        Q = torch.randn(B, N, D, device="cuda")
        K = torch.randn(B, N, D, device="cuda")
        V = torch.randn(B, N, D, device="cuda")

        t_tri = bench(attn, Q, K, V)
        t_ref = bench(reference_attention, Q, K, V)
        speedup = t_ref / t_tri

        label = f"B={B} N={N} D={D}"
        print(f"  {label:>20s}  {t_tri:8.3f}  {t_ref:8.3f}  {speedup:7.2f}x")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Triton Kernels — Test Suite")
    print("=" * 65)

    test_flash_attn_forward()
    test_flash_attn_backward()
    test_gemm_correctness()

    print("\n" + "=" * 65)
    print("  Performance Benchmarks")
    print("=" * 65)

    benchmark_gemm()
    benchmark_flash_attn()

    print("\nAll tests passed.")
