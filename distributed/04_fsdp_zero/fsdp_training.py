"""FSDP (ZeRO-3) 实战

对比 DDP vs FSDP 的显存占用差异。
运行方式: torchrun --nproc_per_node=2 fsdp_training.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SmallTransformer(nn.Module):
    """可调大小的 Transformer 模型"""
    def __init__(self, vocab_size=1000, hidden_dim=512, ffn_dim=2048,
                 n_heads=8, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, ffn_dim, n_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


def get_gpu_memory_mb():
    return torch.cuda.memory_allocated() / 1024**2


def make_data(n_samples=2048, seq_len=64, vocab_size=1000):
    torch.manual_seed(42)
    X = torch.randint(0, vocab_size, (n_samples, seq_len))
    y = torch.randint(0, vocab_size, (n_samples, seq_len))
    return TensorDataset(X, y)


def train_loop(model, loader, optimizer, criterion, device, n_epochs=2):
    """训练循环，返回峰值显存"""
    torch.cuda.reset_peak_memory_stats()

    model.train()
    total_loss = 0.0
    n_batches = 0

    for epoch in range(n_epochs):
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, peak_mem


def main():
    rank, world_size = setup()

    if rank == 0:
        print("=" * 60)
        print(f" FSDP vs DDP 显存对比 — {world_size} GPUs")
        print("=" * 60)

    hidden_dim = 512
    ffn_dim = 2048
    n_layers = 8
    batch_size = 32

    dataset = make_data()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    criterion = nn.CrossEntropyLoss()

    # ==================== DDP 训练 ====================
    dist.barrier()
    if rank == 0:
        print(f"\n[1] DDP 训练")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)

    ddp_model = SmallTransformer(hidden_dim=hidden_dim, ffn_dim=ffn_dim,
                                  n_layers=n_layers).cuda()
    n_params = sum(p.numel() for p in ddp_model.parameters())

    mem_after_model = get_gpu_memory_mb()
    ddp_model = DDP(ddp_model, device_ids=[rank])
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-4)

    ddp_loss, ddp_peak = train_loop(ddp_model, loader, optimizer, criterion, "cuda")

    dist.barrier()
    if rank == 0:
        print(f"  模型参数: {n_params:,} ({n_params * 4 / 1024**2:.1f} MB FP32)")
        print(f"  模型加载后显存: {mem_after_model:.1f} MB")
        print(f"  训练峰值显存:   {ddp_peak:.1f} MB")
        print(f"  训练 loss:      {ddp_loss:.4f}")

    del ddp_model, optimizer
    torch.cuda.empty_cache()

    # ==================== FSDP 训练 (FULL_SHARD = ZeRO-3) ====================
    dist.barrier()
    if rank == 0:
        print(f"\n[2] FSDP (FULL_SHARD = ZeRO-3) 训练")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)

    fsdp_model = SmallTransformer(hidden_dim=hidden_dim, ffn_dim=ffn_dim,
                                   n_layers=n_layers).cuda()
    fsdp_model = FSDP(fsdp_model, sharding_strategy=ShardingStrategy.FULL_SHARD)

    mem_after_fsdp = get_gpu_memory_mb()
    optimizer = optim.Adam(fsdp_model.parameters(), lr=1e-4)

    fsdp_loss, fsdp_peak = train_loop(fsdp_model, loader, optimizer, criterion, "cuda")

    dist.barrier()
    if rank == 0:
        print(f"  模型加载后显存: {mem_after_fsdp:.1f} MB (参数已分片)")
        print(f"  训练峰值显存:   {fsdp_peak:.1f} MB")
        print(f"  训练 loss:      {fsdp_loss:.4f}")

    del fsdp_model, optimizer
    torch.cuda.empty_cache()

    # ==================== FSDP SHARD_GRAD_OP (= ZeRO-2) ====================
    dist.barrier()
    if rank == 0:
        print(f"\n[3] FSDP (SHARD_GRAD_OP = ZeRO-2) 训练")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)

    z2_model = SmallTransformer(hidden_dim=hidden_dim, ffn_dim=ffn_dim,
                                 n_layers=n_layers).cuda()
    z2_model = FSDP(z2_model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)

    mem_after_z2 = get_gpu_memory_mb()
    optimizer = optim.Adam(z2_model.parameters(), lr=1e-4)

    z2_loss, z2_peak = train_loop(z2_model, loader, optimizer, criterion, "cuda")

    dist.barrier()
    if rank == 0:
        print(f"  模型加载后显存: {mem_after_z2:.1f} MB")
        print(f"  训练峰值显存:   {z2_peak:.1f} MB")
        print(f"  训练 loss:      {z2_loss:.4f}")

    # ==================== 对比总结 ====================
    dist.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f" 显存对比总结 (rank 0)")
        print(f"{'=' * 60}")
        print(f"  {'方法':<25} {'峰值显存':>10} {'相对DDP':>10}")
        print(f"  {'-'*45}")
        print(f"  {'DDP (全量副本)':<25} {ddp_peak:>8.1f} MB {'1.00x':>10}")
        print(f"  {'FSDP ZeRO-2':<25} {z2_peak:>8.1f} MB "
              f"{z2_peak/ddp_peak:>9.2f}x")
        print(f"  {'FSDP ZeRO-3 (FULL)':<25} {fsdp_peak:>8.1f} MB "
              f"{fsdp_peak/ddp_peak:>9.2f}x")
        print(f"\n  模型越大，FSDP 的显存优势越明显。")
        print(f"  当前小模型 ({n_params:,} 参数) 下差异不大，")
        print(f"  但对于 7B+ 模型，ZeRO-3 可以省 ~4x 显存。")

    dist.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(" 全部完成!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
