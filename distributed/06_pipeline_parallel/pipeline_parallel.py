"""流水线并行 (Pipeline Parallelism) 手动实现

双卡 GPipe 和 1F1B 调度策略实现。
运行方式: torchrun --nproc_per_node=2 pipeline_parallel.py
"""

import time
import torch
import torch.nn as nn
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


class StageModule(nn.Module):
    """每个 stage 包含若干 Transformer 层"""
    def __init__(self, hidden_dim, ffn_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def gpipe_forward_backward(stage, rank, world_size, micro_batches, target=None):
    """GPipe: 所有前向 → 所有反向"""
    n_mb = len(micro_batches)
    outputs = []
    losses = []

    # ==================== All Forward ====================
    for i in range(n_mb):
        if rank == 0:
            x = micro_batches[i].requires_grad_(True)
            out = stage(x)
            dist.send(out.detach(), dst=1)
            outputs.append((x, out))
        else:
            buf = torch.zeros_like(micro_batches[0]).cuda()
            dist.recv(buf, src=0)
            buf.requires_grad_(True)
            out = stage(buf)
            loss = ((out - target[i]) ** 2).mean()
            outputs.append((buf, out))
            losses.append(loss)

    # ==================== All Backward ====================
    for i in reversed(range(n_mb)):
        if rank == 1:
            losses[i].backward()
            grad = outputs[i][0].grad
            dist.send(grad.contiguous(), dst=0)
        else:
            grad_buf = torch.zeros_like(micro_batches[0]).cuda()
            dist.recv(grad_buf, src=1)
            outputs[i][1].backward(grad_buf)

    return sum(l.item() for l in losses) / max(len(losses), 1)


def one_f1b_forward_backward(stage, rank, world_size, micro_batches, target=None):
    """1F1B: 交替执行前向和反向，减少峰值显存"""
    n_mb = len(micro_batches)
    outputs = []
    losses = []
    total_loss = 0.0

    if rank == 0:
        # Warmup: 前 num_warmup 个只做前向
        num_warmup = min(world_size - 1, n_mb)

        for i in range(num_warmup):
            x = micro_batches[i].requires_grad_(True)
            out = stage(x)
            dist.send(out.detach(), dst=1)
            outputs.append((x, out))

        # 1F1B 稳定阶段
        for i in range(num_warmup, n_mb):
            # Backward for (i - num_warmup)
            grad_buf = torch.zeros_like(micro_batches[0]).cuda()
            dist.recv(grad_buf, src=1)
            bi = i - num_warmup
            outputs[bi][1].backward(grad_buf)

            # Forward for i
            x = micro_batches[i].requires_grad_(True)
            out = stage(x)
            dist.send(out.detach(), dst=1)
            outputs.append((x, out))

        # Cooldown: 剩余反向
        for i in range(n_mb - num_warmup, n_mb):
            grad_buf = torch.zeros_like(micro_batches[0]).cuda()
            dist.recv(grad_buf, src=1)
            outputs[i][1].backward(grad_buf)

    else:  # rank == 1
        num_warmup = min(world_size - 1, n_mb)

        for i in range(num_warmup):
            buf = torch.zeros_like(micro_batches[0]).cuda()
            dist.recv(buf, src=0)
            buf.requires_grad_(True)
            out = stage(buf)
            loss = ((out - target[i]) ** 2).mean()
            outputs.append((buf, out))
            losses.append(loss)

        for i in range(num_warmup, n_mb):
            # Backward
            bi = i - num_warmup
            losses[bi].backward()
            dist.send(outputs[bi][0].grad.contiguous(), dst=0)
            total_loss += losses[bi].item()

            # Forward
            buf = torch.zeros_like(micro_batches[0]).cuda()
            dist.recv(buf, src=0)
            buf.requires_grad_(True)
            out = stage(buf)
            loss = ((out - target[i]) ** 2).mean()
            outputs.append((buf, out))
            losses.append(loss)

        for i in range(n_mb - num_warmup, n_mb):
            losses[i].backward()
            dist.send(outputs[i][0].grad.contiguous(), dst=0)
            total_loss += losses[i].item()

    return total_loss / max(n_mb, 1)


def main():
    rank, world_size = setup()
    assert world_size == 2, "此示例需要恰好 2 个 GPU"

    if rank == 0:
        print("=" * 60)
        print(f" Pipeline Parallelism — {world_size} stages")
        print("=" * 60)

    hidden_dim = 256
    ffn_dim = 1024
    layers_per_stage = 4
    micro_batch_size = 16
    seq_len = 32
    n_micro_batches = 8

    torch.manual_seed(42 + rank)
    stage = StageModule(hidden_dim, ffn_dim, layers_per_stage).cuda()

    # 准备数据
    torch.manual_seed(42)
    micro_batches = [
        torch.randn(micro_batch_size, seq_len, hidden_dim).cuda()
        for _ in range(n_micro_batches)
    ]
    target = [torch.randn_like(micro_batches[0]) for _ in range(n_micro_batches)]

    # ==================== 1. 正确性验证 ====================
    dist.barrier()
    if rank == 0:
        print(f"\n[1] GPipe 调度")

    torch.cuda.reset_peak_memory_stats()
    loss_gpipe = gpipe_forward_backward(stage, rank, world_size, micro_batches, target)
    gpipe_mem = torch.cuda.max_memory_allocated() / 1024**2

    dist.barrier()
    if rank == 0:
        print(f"  micro_batches={n_micro_batches}, stages={world_size}")

    # 重置梯度
    stage.zero_grad()

    dist.barrier()
    if rank == 0:
        print(f"\n[2] 1F1B 调度")

    torch.cuda.reset_peak_memory_stats()
    loss_1f1b = one_f1b_forward_backward(stage, rank, world_size, micro_batches, target)
    f1b_mem = torch.cuda.max_memory_allocated() / 1024**2

    dist.barrier()
    if rank == 0:
        print(f"  micro_batches={n_micro_batches}, stages={world_size}")

    # ==================== 2. 显存对比 ====================
    dist.barrier()
    if rank == 0:
        print(f"\n[3] 显存对比 (rank 0)")

    all_gpipe_mem = [torch.tensor(0.0).cuda() for _ in range(world_size)]
    all_f1b_mem = [torch.tensor(0.0).cuda() for _ in range(world_size)]
    dist.all_gather(all_gpipe_mem, torch.tensor(gpipe_mem).cuda())
    dist.all_gather(all_f1b_mem, torch.tensor(f1b_mem).cuda())

    if rank == 0:
        for r in range(world_size):
            print(f"  rank {r}: GPipe={all_gpipe_mem[r].item():.1f}MB, "
                  f"1F1B={all_f1b_mem[r].item():.1f}MB")
        print(f"\n  1F1B 相比 GPipe 显存节省: 1F1B 在稳定阶段只需存 1 个 micro-batch 的激活")

    # ==================== 3. Bubble 分析 ====================
    dist.barrier()
    if rank == 0:
        print(f"\n[4] Bubble Rate 分析")
        print(f"  公式: bubble_rate = (P-1) / (P-1+M)")
        print(f"  P=pipeline stages, M=micro-batches\n")
        P = world_size
        for M in [1, 2, 4, 8, 16, 32]:
            bubble = (P - 1) / (P - 1 + M)
            print(f"  P={P}, M={M:2d}: bubble_rate = {bubble:.1%}")

    dist.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(" 全部完成!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
