"""
手动实现混合精度训练 (不用 torch.amp)

目的: 理解 AMP 在底层到底做了什么。拆解成 4 个核心机制:
  1. FP16 前向/反向 + FP32 master weights
  2. Loss Scaling (防止 FP16 梯度下溢)
  3. Dynamic Loss Scaling (自动调整 scale factor)
  4. Op-level casting (哪些算子用 FP16, 哪些保持 FP32)

运行:
    python manual_amp.py
"""

import torch
import torch.nn as nn


# ============================================================================
# 核心概念 1: FP32 Master Weights
# ============================================================================

class MixedPrecisionOptimizer:
    """
    手动维护 FP32 master weights。

    为什么需要?
      - 模型参数存两份: FP16 (用于前向/反向) + FP32 (用于优化器更新)
      - 如果直接用 FP16 参数做 optimizer.step():
        - 学习率 1e-4 * 梯度 1e-3 = 1e-7
        - FP16 最小正数 ≈ 6e-8, 精度只有 ~1e-3 的相对误差
        - 很多参数更新会被舍入为 0 ("梯度消失")
      - FP32 master weights 保证 optimizer.step() 的精度

    流程:
      1. FP32 master → cast to FP16 model
      2. FP16 forward + backward → FP16 gradients
      3. FP16 gradients → cast to FP32
      4. FP32 optimizer.step() on master weights
      5. 回到 1
    """

    def __init__(self, model, lr=1e-3):
        self.fp16_params = list(model.parameters())
        self.fp32_master = [
            p.float().detach().clone().requires_grad_(True)
            for p in self.fp16_params
        ]
        self.optimizer = torch.optim.Adam(self.fp32_master, lr=lr)

    def step(self):
        for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_master):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()

        self.optimizer.step()

        for fp16_p, fp32_p in zip(self.fp16_params, self.fp32_master):
            fp16_p.data.copy_(fp32_p.data.half())

    def zero_grad(self):
        for p in self.fp16_params:
            if p.grad is not None:
                p.grad = None
        self.optimizer.zero_grad()


# ============================================================================
# 核心概念 2: Dynamic Loss Scaling
# ============================================================================

class LossScaler:
    """
    手动实现 Dynamic Loss Scaling。

    为什么需要?
      FP16 的表示范围: [6e-8, 65504]
      梯度在反向传播中经常很小 (1e-7 ~ 1e-5), 容易下溢为 0。

    解决方案:
      1. 前向后: loss = loss * scale  (放大)
      2. backward 后: grad = grad / scale  (缩回)
      3. 动态调整 scale:
         - 连续 N 步没有 overflow → scale *= 2 (尝试更大的 scale)
         - 出现 overflow (inf/nan) → scale /= 2, 跳过这步更新

    这就是 torch.amp.GradScaler 做的事情。
    """

    def __init__(self, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_step = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, params):
        inv_scale = 1.0 / self.scale
        found_inf = False
        for p in params:
            if p.grad is not None:
                p.grad.data *= inv_scale
                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    found_inf = True
        return found_inf

    def update(self, found_inf):
        if found_inf:
            self.scale *= self.backoff_factor
            self.growth_step = 0
        else:
            self.growth_step += 1
            if self.growth_step >= self.growth_interval:
                self.scale *= self.growth_factor
                self.growth_step = 0


# ============================================================================
# 演示: 手动混合精度训练
# ============================================================================

def demo_manual_amp():
    device = torch.device("cuda")

    model = nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ).half().to(device)

    mixed_optim = MixedPrecisionOptimizer(model, lr=1e-3)
    loss_scaler = LossScaler(init_scale=2**16)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(64, 784, device=device)
    y = torch.randint(0, 10, (64,), device=device)

    print("Manual Mixed Precision Training (no torch.amp)")
    print("=" * 60)

    for step in range(20):
        mixed_optim.zero_grad()

        # FP16 前向
        x_fp16 = x.half()
        out = model(x_fp16)

        # Loss 必须在 FP32 下计算
        loss = criterion(out.float(), y)

        # Loss Scaling + 反向
        scaled_loss = loss_scaler.scale_loss(loss)
        scaled_loss.backward()

        # Unscale 梯度 & 检查 overflow
        found_inf = loss_scaler.unscale_grads(list(model.parameters()))

        if not found_inf:
            mixed_optim.step()
        else:
            print(
                f"  Step {step:>3d}: overflow detected, skipping update "
                f"(scale: {loss_scaler.scale:.0f})"
            )

        loss_scaler.update(found_inf)

        if step % 5 == 0:
            print(
                f"  Step {step:>3d}: loss={loss.item():.4f}  "
                f"scale={loss_scaler.scale:.0f}"
            )

    print("\nDone.")


# ============================================================================
# 对比: FP16 梯度下溢演示
# ============================================================================

def demo_gradient_underflow():
    """展示为什么 FP16 需要 loss scaling。"""
    print("=" * 60)
    print("  FP16 梯度下溢演示")
    print("=" * 60)

    grad_fp32 = torch.tensor(1e-6, dtype=torch.float32)
    grad_fp16 = grad_fp32.half()

    print(f"\n  FP32 gradient: {grad_fp32.item():.2e}")
    print(f"  FP16 gradient: {grad_fp16.item():.2e}")
    print(f"  Lost? {'YES — underflow!' if grad_fp16.item() == 0 else 'No'}")

    scale = 1024.0
    grad_scaled = (grad_fp32 * scale).half()
    grad_recovered = grad_scaled.float() / scale

    print(f"\n  With loss scaling (scale={scale:.0f}):")
    print(f"  Scaled FP16 gradient: {grad_scaled.item():.2e}")
    print(f"  Recovered FP32:       {grad_recovered.item():.2e}")
    print(f"  Lost? {'YES' if grad_recovered.item() == 0 else 'No — recovered!'}")

    print(f"\n  FP16 range: [{torch.finfo(torch.float16).tiny:.2e}, "
          f"{torch.finfo(torch.float16).max:.2e}]")
    print(f"  FP16 eps (precision):    {torch.finfo(torch.float16).eps:.2e}")

    print(f"\n  BF16 range: [{torch.finfo(torch.bfloat16).tiny:.2e}, "
          f"{torch.finfo(torch.bfloat16).max:.2e}]")
    print(f"  BF16 eps (precision):    {torch.finfo(torch.bfloat16).eps:.2e}")

    print("""
  ┌──────────────────────────────────────────────────────────┐
  │  FP32: 1 sign + 8 exp + 23 mantissa  → 高精度, 大范围   │
  │  FP16: 1 sign + 5 exp + 10 mantissa  → 低精度, 小范围   │
  │  BF16: 1 sign + 8 exp + 7  mantissa  → 低精度, 大范围!  │
  │  TF32: 1 sign + 8 exp + 10 mantissa  → 中精度, 大范围   │
  └──────────────────────────────────────────────────────────┘

  BF16 的关键优势: 和 FP32 一样的指数位 (8 bit)
    → 和 FP32 一样的表示范围
    → 不需要 loss scaling!
    → 这就是为什么 BF16 训练不需要 GradScaler
    """)


if __name__ == "__main__":
    demo_gradient_underflow()
    demo_manual_amp()
