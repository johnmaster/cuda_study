"""DDP 分布式数据并行实战

双卡训练一个小模型，演示:
1. DDP 基本训练流程
2. 梯度同步验证
3. Gradient Accumulation (no_sync)
4. DDP vs 单卡 速度对比

运行方式: torchrun --nproc_per_node=2 ddp_training.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


class SimpleNet(nn.Module):
    def __init__(self, dim=512, hidden=1024, n_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_data(n_samples=4096, dim=512):
    """合成分类数据"""
    torch.manual_seed(42)
    X = torch.randn(n_samples, dim)
    y = (X[:, :10].sum(dim=1) > 0).long()
    return TensorDataset(X, y)


# ==================== 实验 1: DDP 基本训练 ====================

def train_ddp(rank, world_size):
    """标准 DDP 训练流程"""
    # 所有卡用相同的初始权重
    torch.manual_seed(0)
    model = SimpleNet().cuda()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = make_data()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    ddp_model.train()
    total_loss = 0.0
    n_batches = 0

    for epoch in range(3):
        sampler.set_epoch(epoch)
        for X, y in loader:
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            out = ddp_model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches

    # 验证所有卡的参数一致
    param_sum = sum(p.data.sum().item() for p in model.parameters())
    all_sums = [torch.tensor(0.0).cuda() for _ in range(world_size)]
    dist.all_gather(all_sums, torch.tensor(param_sum).cuda())
    param_diff = abs(all_sums[0].item() - all_sums[-1].item())

    return avg_loss, param_diff


# ==================== 实验 2: 梯度同步验证 ====================

def verify_gradient_sync(rank, world_size):
    """验证 DDP 梯度是否真的被 AllReduce 了"""
    torch.manual_seed(0)
    model = SimpleNet(dim=64, hidden=128, n_layers=1).cuda()
    ddp_model = DDP(model, device_ids=[rank])

    # 每卡用不同的数据
    torch.manual_seed(rank)
    X = torch.randn(32, 64).cuda()
    y = torch.randint(0, 10, (32,)).cuda()

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    loss = criterion(ddp_model(X), y)
    loss.backward()

    # 抓取梯度
    grad_first_layer = list(model.parameters())[0].grad.clone()

    # 所有卡的梯度应该相同（DDP 已经 AllReduce 过了）
    all_grads = [torch.zeros_like(grad_first_layer) for _ in range(world_size)]
    dist.all_gather(all_grads, grad_first_layer)

    grad_diff = (all_grads[0] - all_grads[-1]).abs().max().item()
    return grad_diff


# ==================== 实验 3: Gradient Accumulation ====================

def train_with_gradient_accumulation(rank, world_size, accum_steps=4):
    """梯度累积：用 no_sync() 减少通信次数"""
    torch.manual_seed(0)
    model = SimpleNet(dim=64, hidden=128, n_layers=1).cuda()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = make_data(n_samples=1024, dim=64)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    ddp_model.train()
    n_comm = 0

    optimizer.zero_grad()
    for i, (X, y) in enumerate(loader):
        X, y = X.cuda(), y.cuda()
        is_accumulating = (i + 1) % accum_steps != 0

        if is_accumulating:
            with ddp_model.no_sync():
                loss = criterion(ddp_model(X), y) / accum_steps
                loss.backward()
        else:
            loss = criterion(ddp_model(X), y) / accum_steps
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n_comm += 1

    return n_comm, len(loader)


# ==================== 实验 4: 速度对比 ====================

def benchmark_throughput(rank, world_size):
    """DDP 吞吐量测试"""
    torch.manual_seed(0)
    model = SimpleNet(dim=256, hidden=512, n_layers=4).cuda()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    batch_size = 128
    X = torch.randn(batch_size, 256).cuda()
    y = torch.randint(0, 10, (batch_size,)).cuda()

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(ddp_model(X), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    n_steps = 50
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = criterion(ddp_model(X), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    samples_per_sec = n_steps * batch_size * world_size / elapsed
    return samples_per_sec, elapsed


def main():
    rank, world_size = setup()

    if rank == 0:
        print("=" * 60)
        print(f" DDP 分布式数据并行 — {world_size} GPUs")
        print("=" * 60)

    # 实验 1
    dist.barrier()
    if rank == 0:
        print("\n[1] DDP 基本训练")
    avg_loss, param_diff = train_ddp(rank, world_size)
    dist.barrier()
    if rank == 0:
        print(f"  平均 loss: {avg_loss:.4f}")
        print(f"  各卡参数差异: {param_diff:.2e} "
              f"{'(一致 ✓)' if param_diff < 1e-3 else '(不一致 ✗)'}")

    # 实验 2
    dist.barrier()
    if rank == 0:
        print("\n[2] 梯度同步验证")
    grad_diff = verify_gradient_sync(rank, world_size)
    dist.barrier()
    if rank == 0:
        print(f"  各卡梯度最大差异: {grad_diff:.2e} "
              f"{'(同步正确 ✓)' if grad_diff < 1e-6 else '(同步异常 ✗)'}")

    # 实验 3
    dist.barrier()
    if rank == 0:
        print("\n[3] Gradient Accumulation (accum_steps=4)")
    n_comm, n_total = train_with_gradient_accumulation(rank, world_size, accum_steps=4)
    dist.barrier()
    if rank == 0:
        print(f"  总 batch 数: {n_total}, AllReduce 次数: {n_comm}")
        print(f"  通信减少: {n_total}/{n_comm} = {n_total/max(n_comm,1):.0f}x "
              f"(只在累积完成时同步)")

    # 实验 4
    dist.barrier()
    if rank == 0:
        print("\n[4] DDP 吞吐量")
    throughput, elapsed = benchmark_throughput(rank, world_size)
    dist.barrier()
    if rank == 0:
        print(f"  {world_size} GPU 吞吐: {throughput:.0f} samples/s")
        print(f"  (理想情况下应接近单卡的 {world_size}x)")

    # 模型参数量
    dist.barrier()
    if rank == 0:
        model = SimpleNet(dim=256, hidden=512, n_layers=4)
        n_params = sum(p.numel() for p in model.parameters())
        mem_per_card = n_params * 4 / 1024**2  # FP32
        print(f"\n[info] 模型参数: {n_params:,} ({mem_per_card:.1f} MB)")
        print(f"[info] DDP 中每卡都存完整副本 → 总显存 = {mem_per_card:.1f} MB × {world_size}")

    dist.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(" 全部完成!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
