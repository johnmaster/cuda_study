"""Ring AllReduce 真实多卡 NCCL 实现

使用 torch.distributed P2P Send/Recv 手动实现 Ring AllReduce，
然后与 NCCL 内置的 AllReduce 对比验证正确性和性能。

运行方式: torchrun --nproc_per_node=2 ring_allreduce_nccl.py
"""

import time
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def ring_send_recv(send_data, recv_buf, dst, src):
    """使用 batch_isend_irecv 同时发送和接收，避免死锁"""
    ops = [
        dist.P2POp(dist.isend, send_data, dst),
        dist.P2POp(dist.irecv, recv_buf, src),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


def ring_allreduce(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    手动实现 Ring AllReduce

    Phase 1 (ReduceScatter): 沿环传递并累加，每卡最终持有一个 chunk 的完整归约
    Phase 2 (AllGather):     沿环传递完整 chunk，每卡收集所有归约结果
    """
    N = world_size
    assert tensor.numel() % N == 0, f"数据量 {tensor.numel()} 必须被 {N} 整除"

    chunk_size = tensor.numel() // N
    left = (rank - 1) % N
    right = (rank + 1) % N

    result = tensor.clone()
    recv_buf = torch.zeros(chunk_size, device=tensor.device, dtype=tensor.dtype)

    # ==================== Phase 1: ReduceScatter ====================
    for step in range(N - 1):
        send_chunk_id = (rank - step) % N
        recv_chunk_id = (rank - step - 1) % N

        send_start = send_chunk_id * chunk_size
        recv_start = recv_chunk_id * chunk_size

        send_data = result[send_start:send_start + chunk_size].contiguous()

        ring_send_recv(send_data, recv_buf, dst=right, src=left)

        result[recv_start:recv_start + chunk_size] += recv_buf

    # ==================== Phase 2: AllGather ====================
    for step in range(N - 1):
        send_chunk_id = (rank - step + 1) % N
        recv_chunk_id = (rank - step) % N

        send_start = send_chunk_id * chunk_size
        recv_start = recv_chunk_id * chunk_size

        send_data = result[send_start:send_start + chunk_size].contiguous()

        ring_send_recv(send_data, recv_buf, dst=right, src=left)

        result[recv_start:recv_start + chunk_size] = recv_buf

    return result


def benchmark(fn, *args, warmup=5, repeat=20, name=""):
    """计时工具"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat

    return elapsed


def main():
    rank, world_size = setup()

    if rank == 0:
        print(f"{'=' * 60}")
        print(f" Ring AllReduce — {world_size} GPUs (NCCL backend)")
        print(f"{'=' * 60}")

    # ==================== 1. 正确性验证 ====================
    dist.barrier()
    if rank == 0:
        print(f"\n[1] 正确性验证 (小数据)")

    torch.manual_seed(42 + rank)
    small = torch.randn(world_size * 4, device="cuda")

    # 保存各卡原始数据用于验证
    all_tensors = [torch.zeros_like(small) for _ in range(world_size)]
    dist.all_gather(all_tensors, small)

    # 手动 Ring AllReduce
    my_result = ring_allreduce(small.clone(), rank, world_size)

    # 计算期望结果
    expected = torch.stack(all_tensors).sum(dim=0)

    # NCCL 内置 AllReduce
    nccl_result = small.clone()
    dist.all_reduce(nccl_result, op=dist.ReduceOp.SUM)

    err_vs_expected = (my_result - expected).abs().max().item()
    err_vs_nccl = (my_result - nccl_result).abs().max().item()

    print(f"  rank {rank}: Ring vs Expected max_err = {err_vs_expected:.2e}")
    print(f"  rank {rank}: Ring vs NCCL     max_err = {err_vs_nccl:.2e}")

    dist.barrier()
    if rank == 0:
        if err_vs_expected < 1e-5:
            print(f"  正确性验证通过 ✓")
        else:
            print(f"  正确性验证失败 ✗")

    # ==================== 2. 性能对比 ====================
    for size_mb in [1, 10, 100]:
        n_elements = size_mb * 1024 * 1024 // 4  # float32
        # 确保能被 world_size 整除
        n_elements = (n_elements // world_size) * world_size

        data = torch.randn(n_elements, device="cuda")

        dist.barrier()
        if rank == 0:
            print(f"\n[2] 性能对比 — 数据量: {size_mb} MB ({n_elements} floats)")

        # 手动 Ring AllReduce
        def run_ring():
            ring_allreduce(data.clone(), rank, world_size)

        # NCCL 内置 AllReduce
        def run_nccl():
            t = data.clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        ring_time = benchmark(run_ring, warmup=3, repeat=10)
        nccl_time = benchmark(run_nccl, warmup=3, repeat=10)

        dist.barrier()
        if rank == 0:
            bw_ring = 2 * size_mb * (world_size - 1) / world_size / ring_time / 1024
            bw_nccl = 2 * size_mb * (world_size - 1) / world_size / nccl_time / 1024
            print(f"  手动 Ring:  {ring_time * 1000:8.3f} ms  "
                  f"(有效带宽: {bw_ring:.2f} GB/s)")
            print(f"  NCCL 内置:  {nccl_time * 1000:8.3f} ms  "
                  f"(有效带宽: {bw_nccl:.2f} GB/s)")
            print(f"  NCCL 加速比: {ring_time / nccl_time:.2f}x")

    # ==================== 3. 模拟 DDP 梯度同步 ====================
    dist.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f" [3] 模拟 DDP 梯度同步")
        print(f"{'=' * 60}")

    # 模拟每卡独立计算的梯度
    torch.manual_seed(rank)
    fake_grads = torch.randn(1024 * 1024, device="cuda")  # ~4MB 梯度

    if rank == 0:
        print(f"  每卡独立梯度的均值:")
    dist.barrier()
    print(f"    rank {rank}: grad_mean = {fake_grads.mean():.6f}")
    dist.barrier()

    # AllReduce 求和 → 除以 world_size = 平均梯度
    synced_grads = ring_allreduce(fake_grads.clone(), rank, world_size)
    synced_grads /= world_size

    dist.barrier()
    if rank == 0:
        print(f"  AllReduce 后平均梯度的均值:")
    dist.barrier()
    print(f"    rank {rank}: synced_grad_mean = {synced_grads.mean():.6f}")

    # 验证所有卡的同步梯度一致
    dist.barrier()
    all_synced = [torch.zeros_like(synced_grads) for _ in range(world_size)]
    dist.all_gather(all_synced, synced_grads)
    max_diff = (all_synced[0] - all_synced[-1]).abs().max().item()
    if rank == 0:
        print(f"  各卡同步梯度最大差异: {max_diff:.2e} "
              f"{'(一致 ✓)' if max_diff < 1e-5 else '(不一致 ✗)'}")

    dist.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f" 全部完成!")
        print(f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
