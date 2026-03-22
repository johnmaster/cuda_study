"""
Test script: validates correctness and compares performance against torch.nn.GELU.

Run:
    cd pytorch_custom_ops
    python test_custom_gelu.py
"""

import torch
import time
from custom_gelu import CustomGELU

def test_correctness():
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # --- Forward ---
    custom = CustomGELU()
    ref = torch.nn.GELU(approximate="tanh")

    y_custom = custom(x)
    y_ref = ref(x_ref)

    fwd_diff = (y_custom - y_ref).abs().max().item()
    print(f"  Forward  max abs diff: {fwd_diff:.2e}")
    assert fwd_diff < 1e-5, f"Forward mismatch! diff={fwd_diff}"

    # --- Backward ---
    y_custom.sum().backward()
    y_ref.sum().backward()

    bwd_diff = (x.grad - x_ref.grad).abs().max().item()
    print(f"  Backward max abs diff: {bwd_diff:.2e}")
    assert bwd_diff < 1e-5, f"Backward mismatch! diff={bwd_diff}"

    print("  PASSED\n")


def test_performance():
    print("=" * 60)
    print("Performance Test")
    print("=" * 60)

    x = torch.randn(4096, 4096, device="cuda")

    custom = CustomGELU()
    ref = torch.nn.GELU(approximate="tanh")

    # Warmup
    for _ in range(20):
        custom(x)
        ref(x)
    torch.cuda.synchronize()

    # Benchmark custom
    iters = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        custom(x)
    torch.cuda.synchronize()
    t_custom = (time.perf_counter() - t0) / iters * 1000

    # Benchmark PyTorch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ref(x)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / iters * 1000

    print(f"  Custom GELU : {t_custom:.3f} ms")
    print(f"  PyTorch GELU: {t_ref:.3f} ms")
    print(f"  Speedup     : {t_ref / t_custom:.2f}x\n")


if __name__ == "__main__":
    test_correctness()
    test_performance()
    print("All tests done.")
