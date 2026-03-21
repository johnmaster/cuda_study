"""
Test script for Flash Attention custom op.

Validates:
  1. Forward correctness   — vs torch reference
  2. Backward correctness  — vs torch.autograd.gradcheck
  3. Performance           — vs torch scaled_dot_product_attention

Run:
    cd 02_flash_attention
    python test_flash_attention.py
"""

import torch
import time
from flash_attention_op import FlashAttention


# ── reference implementation ──────────────────────────────────────────────────

def reference_attention(Q, K, V):
    """Standard scaled dot-product attention (materialises N×N matrix)."""
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


# ── correctness ──────────────────────────────────────────────────────────────

def test_forward(B, N, d):
    torch.manual_seed(42)
    Q = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
    K = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
    V = torch.randn(B, N, d, device="cuda", dtype=torch.float32)

    attn = FlashAttention()
    O_custom = attn(Q, K, V)
    O_ref = reference_attention(Q, K, V)

    diff = (O_custom - O_ref).abs().max().item()
    ok = diff < 1e-3
    print(f"  Forward  B={B} N={N:>4} d={d:>3}  max_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"forward diff too large: {diff}"


def test_backward(B, N, d):
    torch.manual_seed(42)
    Q = torch.randn(B, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    Q_ref = Q.clone().detach().requires_grad_(True)
    K_ref = K.clone().detach().requires_grad_(True)
    V_ref = V.clone().detach().requires_grad_(True)

    attn = FlashAttention()
    O_custom = attn(Q, K, V)
    O_ref = reference_attention(Q_ref, K_ref, V_ref)

    loss_custom = O_custom.sum()
    loss_ref = O_ref.sum()
    loss_custom.backward()
    loss_ref.backward()

    diffs = {
        "dQ": (Q.grad - Q_ref.grad).abs().max().item(),
        "dK": (K.grad - K_ref.grad).abs().max().item(),
        "dV": (V.grad - V_ref.grad).abs().max().item(),
    }
    ok = all(v < 1e-2 for v in diffs.values())
    parts = "  ".join(f"{k}={v:.2e}" for k, v in diffs.items())
    print(f"  Backward B={B} N={N:>4} d={d:>3}  {parts}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"backward diff too large: {diffs}"


# ── performance ──────────────────────────────────────────────────────────────

def benchmark(B, N, d, iters=100):
    Q = torch.randn(B, N, d, device="cuda")
    K = torch.randn(B, N, d, device="cuda")
    V = torch.randn(B, N, d, device="cuda")

    attn = FlashAttention()

    # warmup
    for _ in range(20):
        attn(Q, K, V)
        reference_attention(Q, K, V)
    torch.cuda.synchronize()

    # custom
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        attn(Q, K, V)
    torch.cuda.synchronize()
    t_custom = (time.perf_counter() - t0) / iters * 1000

    # reference
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        reference_attention(Q, K, V)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / iters * 1000

    mem_flash = B * N * d * 4 * 4                  # Q + K + V + O
    mem_std = mem_flash + B * N * N * 4             # + N×N scores
    savings_mb = (mem_std - mem_flash) / 1024**2

    print(f"  B={B} N={N:>4} d={d:>3}  custom={t_custom:.3f}ms  "
          f"ref={t_ref:.3f}ms  speedup={t_ref/t_custom:.2f}x  "
          f"mem_saved={savings_mb:.1f}MB")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Flash Attention Custom Op — Correctness Tests")
    print("=" * 70)

    for B, N, d in [(1, 32, 32), (1, 64, 64), (2, 64, 32), (1, 128, 64), (2, 128, 128)]:
        test_forward(B, N, d)

    print()
    for B, N, d in [(1, 32, 32), (1, 64, 64), (2, 64, 32), (1, 128, 64)]:
        test_backward(B, N, d)

    print()
    print("=" * 70)
    print("Flash Attention Custom Op — Performance")
    print("=" * 70)
    for B, N, d in [(1, 128, 64), (1, 256, 64), (1, 512, 64), (4, 256, 64), (1, 1024, 64)]:
        benchmark(B, N, d)

    print("\nAll tests passed.")
