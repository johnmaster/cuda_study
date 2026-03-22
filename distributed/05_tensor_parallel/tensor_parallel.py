"""张量并行 (Tensor Parallelism) 手动实现

手动实现 Megatron-style Column/Row Parallel Linear，
在双卡上演示 Transformer MLP 的张量并行。

运行方式: torchrun --nproc_per_node=2 tensor_parallel.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


# ==================== 核心：Column / Row Parallel Linear ====================

class ColumnParallelLinear(nn.Module):
    """列并行线性层：权重按输出维度切分

    W_full: [in_features, out_features]
    本卡持有: W_local: [in_features, out_features // world_size]

    前向: Y_local = X @ W_local  (每卡得到输出的一部分)
    """
    def __init__(self, in_features, out_features, rank, world_size, gather_output=True):
        super().__init__()
        assert out_features % world_size == 0
        self.out_per_rank = out_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(in_features, self.out_per_rank))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, in_features] — 每卡都有完整输入
        y_local = F.linear(x, self.weight.T, self.bias)  # [batch, out/N]

        if self.gather_output:
            # AllGather 拼接完整输出
            y_list = [torch.zeros_like(y_local) for _ in range(self.world_size)]
            dist.all_gather(y_list, y_local)
            return torch.cat(y_list, dim=-1)  # [batch, out]
        return y_local  # 不 gather，留给下一层 RowParallel 用


class RowParallelLinear(nn.Module):
    """行并行线性层：权重按输入维度切分

    W_full: [in_features, out_features]
    本卡持有: W_local: [in_features // world_size, out_features]

    前向: Y_local = X_local @ W_local  (部分结果)
          Y = AllReduce(Y_local)        (求和得到完整结果)
    """
    def __init__(self, in_features, out_features, rank, world_size, input_is_parallel=False):
        super().__init__()
        assert in_features % world_size == 0
        self.in_per_rank = in_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.input_is_parallel = input_is_parallel

        self.weight = nn.Parameter(torch.empty(self.in_per_rank, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.input_is_parallel:
            x_local = x  # 输入已经按卡切分好了
        else:
            # 从完整输入中取本卡负责的部分
            x_local = x[..., self.rank * self.in_per_rank:(self.rank + 1) * self.in_per_rank]

        y_local = F.linear(x_local, self.weight.T)  # [batch, out]

        # AllReduce 求和
        dist.all_reduce(y_local, op=dist.ReduceOp.SUM)
        y_local = y_local + self.bias
        return y_local


# ==================== Megatron-style MLP ====================

class TensorParallelMLP(nn.Module):
    """张量并行 MLP: Column Parallel → GELU → Row Parallel

    关键优化：Column Parallel 不 gather，输出直接作为 Row Parallel 的输入
    → 两层只需要 1 次 AllReduce（而不是 AllGather + AllReduce）
    """
    def __init__(self, hidden_dim, ffn_dim, rank, world_size):
        super().__init__()
        self.w1 = ColumnParallelLinear(hidden_dim, ffn_dim, rank, world_size,
                                        gather_output=False)
        self.w2 = RowParallelLinear(ffn_dim, hidden_dim, rank, world_size,
                                     input_is_parallel=True)

    def forward(self, x):
        h = F.gelu(self.w1(x))  # Column Parallel, 不 gather
        return self.w2(h)       # Row Parallel, AllReduce


class NormalMLP(nn.Module):
    """普通 MLP（对照组）"""
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


# ==================== 实验 ====================

def load_from_full_weights(tp_mlp, full_mlp, rank, world_size):
    """从完整权重中加载对应分片到 TP 模型"""
    with torch.no_grad():
        ffn_per_rank = full_mlp.w1.weight.shape[0] // world_size
        hidden_per_rank = full_mlp.w2.weight.shape[1] // world_size

        # w1 Column Parallel: 按输出维度切
        tp_mlp.w1.weight.copy_(
            full_mlp.w1.weight[rank * ffn_per_rank:(rank + 1) * ffn_per_rank].T
        )
        tp_mlp.w1.bias.copy_(
            full_mlp.w1.bias[rank * ffn_per_rank:(rank + 1) * ffn_per_rank]
        )

        # w2 Row Parallel: 按输入维度切
        tp_mlp.w2.weight.copy_(
            full_mlp.w2.weight[:, rank * hidden_per_rank:(rank + 1) * hidden_per_rank].T
        )
        # bias 只在一张卡上加，或者每卡加 bias/N
        tp_mlp.w2.bias.copy_(full_mlp.w2.bias / world_size)


def main():
    rank, world_size = setup()

    if rank == 0:
        print("=" * 60)
        print(f" Tensor Parallelism — {world_size} GPUs")
        print("=" * 60)

    hidden_dim = 512
    ffn_dim = 2048
    batch_size = 32

    # ==================== 实验 1: 正确性验证 ====================
    dist.barrier()
    if rank == 0:
        print("\n[1] 正确性验证")

    torch.manual_seed(42)
    full_mlp = NormalMLP(hidden_dim, ffn_dim).cuda()
    X = torch.randn(batch_size, hidden_dim).cuda()

    with torch.no_grad():
        Y_ref = full_mlp(X)

    tp_mlp = TensorParallelMLP(hidden_dim, ffn_dim, rank, world_size).cuda()
    load_from_full_weights(tp_mlp, full_mlp, rank, world_size)

    with torch.no_grad():
        Y_tp = tp_mlp(X)

    max_err = (Y_ref - Y_tp).abs().max().item()
    cos_sim = F.cosine_similarity(Y_ref.flatten().unsqueeze(0),
                                   Y_tp.flatten().unsqueeze(0)).item()

    if rank == 0:
        print(f"  Max error: {max_err:.2e}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  {'正确 ✓' if max_err < 1e-4 else '异常 ✗'}")

    # ==================== 实验 2: 显存对比 ====================
    dist.barrier()
    if rank == 0:
        print("\n[2] 显存分析")

    full_params = sum(p.numel() for p in full_mlp.parameters())
    tp_params = sum(p.numel() for p in tp_mlp.parameters())

    if rank == 0:
        print(f"  完整模型参数量: {full_params:,} ({full_params * 4 / 1024**2:.1f} MB)")
        print(f"  TP 每卡参数量:  {tp_params:,} ({tp_params * 4 / 1024**2:.1f} MB)")
        print(f"  切分比: {tp_params / full_params:.2f}x (理想值: {1/world_size:.2f})")

    # ==================== 实验 3: 通信量分析 ====================
    dist.barrier()
    if rank == 0:
        print("\n[3] 通信量分析 (Megatron MLP)")
        comm_bytes = batch_size * hidden_dim * 4  # float32
        print(f"  一次前向的 AllReduce 通信量: {comm_bytes / 1024:.1f} KB")
        print(f"  (仅需 1 次 AllReduce，Column+Row 巧妙组合省去了 AllGather)")

        print(f"\n  对比：如果两层都用独立的 TP：")
        naive_comm = batch_size * ffn_dim * 4 + batch_size * hidden_dim * 4
        print(f"  需要 AllGather({batch_size * ffn_dim * 4 / 1024:.1f}KB) "
              f"+ AllReduce({batch_size * hidden_dim * 4 / 1024:.1f}KB) "
              f"= {naive_comm / 1024:.1f} KB")
        print(f"  Megatron 优化节省: {(naive_comm - comm_bytes) / naive_comm * 100:.0f}%")

    # ==================== 实验 4: 速度对比 ====================
    dist.barrier()
    if rank == 0:
        print("\n[4] 速度对比")

    large_hidden = 2048
    large_ffn = 8192
    large_batch = 64

    torch.manual_seed(42)
    large_full = NormalMLP(large_hidden, large_ffn).cuda()
    large_tp = TensorParallelMLP(large_hidden, large_ffn, rank, world_size).cuda()
    X_large = torch.randn(large_batch, large_hidden).cuda()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            if rank == 0:
                large_full(X_large)
            large_tp(X_large)
    torch.cuda.synchronize()

    n_iters = 100

    # TP
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            large_tp(X_large)
    torch.cuda.synchronize()
    tp_time = (time.perf_counter() - start) / n_iters

    # Full (只在 rank 0 跑)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            large_full(X_large)
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - start) / n_iters

    dist.barrier()
    if rank == 0:
        print(f"  模型: hidden={large_hidden}, ffn={large_ffn}, batch={large_batch}")
        print(f"  单卡完整模型: {full_time * 1000:.3f} ms/iter")
        print(f"  TP {world_size}卡:      {tp_time * 1000:.3f} ms/iter")
        print(f"  (TP 有通信开销，在 PCIe 互联下加速比有限；NVLink 下会好得多)")

    dist.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(" 全部完成!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
