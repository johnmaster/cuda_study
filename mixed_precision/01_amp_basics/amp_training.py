"""
PyTorch AMP (Automatic Mixed Precision) 训练完整示例

混合精度训练的核心思想:
  - 前向/反向: 用 FP16/BF16 计算 (快 2-3x, 省一半显存)
  - 参数更新: 用 FP32 (保证数值精度)
  - 某些算子 (如 softmax, layernorm, loss): 自动保持 FP32 (数值敏感)

本脚本训练一个小 ResNet 在 CIFAR-10 上，对比:
  1. 纯 FP32 训练
  2. AMP FP16 训练
  3. AMP BF16 训练

运行:
    python amp_training.py
"""

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ── 模型 ──────────────────────────────────────────────────────────────────

class SmallResNet(nn.Module):
    """用于精度与性能对比的简化 ResNet。"""

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return torch.relu(out + x)

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.layer1 = nn.Sequential(self.Block(64), self.Block(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            self.Block(128), self.Block(128),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            self.Block(256), self.Block(256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


# ── 数据 ──────────────────────────────────────────────────────────────────

def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )


# ── 训练函数 ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, amp_dtype=None):
    """
    amp_dtype:
      None          → 纯 FP32
      torch.float16 → AMP FP16 (需要 GradScaler)
      torch.bfloat16→ AMP BF16 (不需要 GradScaler)
    """
    model.train()
    use_amp = amp_dtype is not None
    # GradScaler 只在 FP16 时需要 (BF16 的动态范围够大, 不需要 loss scaling)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── 前向 ──
        # autocast 区域内: Conv, Linear, MatMul 自动用 FP16/BF16
        #                   Softmax, LayerNorm, Loss 自动保持 FP32
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # ── 反向 ──
        optimizer.zero_grad()
        if amp_dtype == torch.float16:
            # FP16: 需要 loss scaling 防止梯度下溢
            scaler.scale(loss).backward()   # loss * scale → backward → grad * scale
            scaler.step(optimizer)           # unscale grad → clip → optimizer.step
            scaler.update()                  # 根据是否 overflow 调整 scale factor
        else:
            # FP32 或 BF16: 直接 backward
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


# ── 主函数 ────────────────────────────────────────────────────────────────

def run_experiment(name, amp_dtype, epochs=3):
    device = torch.device("cuda")
    model = SmallResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=256)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {num_params:,}  Batch: 256")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()

    total_time = 0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        loss, acc = train_one_epoch(
            model, loader, optimizer, criterion, device, amp_dtype
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        print(f"  Epoch {epoch+1}/{epochs}  loss={loss:.4f}  acc={acc:.1f}%  time={elapsed:.1f}s")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = total_time / epochs
    throughput = len(loader.dataset) / avg_time

    print(f"\n  Peak GPU memory: {peak_mem:.0f} MB")
    print(f"  Avg epoch time:  {avg_time:.1f}s")
    print(f"  Throughput:      {throughput:.0f} img/s")

    return {"name": name, "mem_mb": peak_mem, "time_s": avg_time, "throughput": throughput}


if __name__ == "__main__":
    print("Mixed Precision Training Comparison")
    print("=" * 60)

    results = []
    results.append(run_experiment("FP32 (baseline)", amp_dtype=None))
    results.append(run_experiment("AMP FP16 (with GradScaler)", amp_dtype=torch.float16))

    # BF16 需要 Ampere+ (compute capability >= 8.0)
    if torch.cuda.get_device_capability()[0] >= 8:
        results.append(
            run_experiment("AMP BF16 (no GradScaler needed)", amp_dtype=torch.bfloat16)
        )
    else:
        print("\n  [Skipping BF16: requires Ampere+ GPU (sm80+)]")

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<35s} {'Memory':>8s} {'Time':>8s} {'Speedup':>8s}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8}")
    baseline_time = results[0]["time_s"]
    for r in results:
        speedup = baseline_time / r["time_s"]
        print(
            f"  {r['name']:<35s} {r['mem_mb']:>6.0f}MB "
            f"{r['time_s']:>6.1f}s {speedup:>6.2f}x"
        )
