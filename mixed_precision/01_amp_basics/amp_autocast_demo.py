"""
autocast 内部行为可视化

展示 torch.amp.autocast 到底把哪些算子转成了 FP16，哪些保留 FP32。

运行:
    python amp_autocast_demo.py
"""

import torch
import torch.nn as nn


def show_dtypes(name, *tensors):
    dtypes = ", ".join(str(t.dtype) for t in tensors)
    print(f"  {name:<40s} → {dtypes}")


def main():
    device = torch.device("cuda")
    x_fp32 = torch.randn(4, 128, device=device, dtype=torch.float32)
    w_fp32 = torch.randn(128, 256, device=device, dtype=torch.float32)

    print("=" * 70)
    print("  autocast 自动 dtype 选择策略")
    print("=" * 70)
    print()

    # ── 不用 autocast ──
    print("Without autocast (all FP32):")
    y = x_fp32 @ w_fp32
    show_dtypes("matmul(FP32, FP32)", y)
    print()

    # ── 用 autocast ──
    print("With autocast(float16):")
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

        # --- Cast to FP16 (compute-intensive, 受益于 Tensor Core) ---
        y_mm = x_fp32 @ w_fp32
        show_dtypes("matmul(FP32, FP32)", y_mm)

        linear = nn.Linear(128, 256, device=device)
        y_linear = linear(x_fp32)
        show_dtypes("nn.Linear(FP32 input)", y_linear)

        conv = nn.Conv2d(3, 64, 3, padding=1, device=device)
        img = torch.randn(1, 3, 32, 32, device=device)
        y_conv = conv(img)
        show_dtypes("nn.Conv2d(FP32 input)", y_conv)

        print()

        # --- Keep FP32 (numerically sensitive) ---
        y_softmax = torch.softmax(y_mm, dim=-1)
        show_dtypes("softmax(FP16 input)", y_softmax)

        ln = nn.LayerNorm(256, device=device)
        y_ln = ln(y_mm.float())
        show_dtypes("LayerNorm(FP32 input)", y_ln)

        y_loss = nn.functional.cross_entropy(
            torch.randn(4, 10, device=device),
            torch.randint(0, 10, (4,), device=device),
        )
        show_dtypes("cross_entropy", y_loss)

        print()

        # --- Follow input dtype ---
        y_relu = torch.relu(y_mm)
        show_dtypes("relu(FP16 input)", y_relu)

        y_relu32 = torch.relu(x_fp32)
        show_dtypes("relu(FP32 input)", y_relu32)

    print()
    print("=" * 70)
    print("  autocast 分类总结")
    print("=" * 70)
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Cast to FP16 (compute-bound, 受益于 Tensor Core):             │
  │    • matmul, linear, conv1d/2d/3d, bmm                         │
  │    • GRU, LSTM, RNN                                             │
  │                                                                 │
  │  Keep FP32 (numerically sensitive):                             │
  │    • softmax, log_softmax                                       │
  │    • layer_norm, group_norm, batch_norm                         │
  │    • cross_entropy, nll_loss, binary_cross_entropy              │
  │    • pow, exp, log, sum (reductions)                            │
  │                                                                 │
  │  Follow input dtype (不强制转换):                                │
  │    • relu, gelu, sigmoid, tanh                                  │
  │    • dropout, max_pool, avg_pool                                │
  │    • cat, stack, index_select                                   │
  └─────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
