"""NCCL 通信原语实战 Demo

在双卡上演示所有核心通信操作的输入输出。
运行方式: torchrun --nproc_per_node=2 nccl_ops_demo.py
"""

import os
import torch
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def demo_broadcast(rank, world_size):
    """Broadcast: rank 0 的数据广播到所有卡"""
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    else:
        tensor = torch.zeros(4, device="cuda")

    print(f"  [Broadcast] Before: rank {rank} = {tensor.tolist()}")
    dist.broadcast(tensor, src=0)
    print(f"  [Broadcast] After:  rank {rank} = {tensor.tolist()}")


def demo_reduce(rank, world_size):
    """Reduce: 所有卡的数据求和到 rank 0"""
    tensor = torch.tensor([rank + 1.0] * 4, device="cuda")

    print(f"  [Reduce] Before: rank {rank} = {tensor.tolist()}")
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f"  [Reduce] After:  rank {rank} = {tensor.tolist()}")


def demo_allreduce(rank, world_size):
    """AllReduce: 所有卡求和，每卡都有完整结果"""
    tensor = torch.tensor([rank + 1.0] * 4, device="cuda")

    print(f"  [AllReduce] Before: rank {rank} = {tensor.tolist()}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"  [AllReduce] After:  rank {rank} = {tensor.tolist()}")


def demo_allgather(rank, world_size):
    """AllGather: 每卡贡献一块，所有卡得到完整拼接"""
    tensor = torch.tensor([rank * 10 + 1.0, rank * 10 + 2.0], device="cuda")
    gathered = [torch.zeros(2, device="cuda") for _ in range(world_size)]

    print(f"  [AllGather] Before: rank {rank} = {tensor.tolist()}")
    dist.all_gather(gathered, tensor)
    result = torch.cat(gathered)
    print(f"  [AllGather] After:  rank {rank} = {result.tolist()}")


def demo_reduce_scatter(rank, world_size):
    """ReduceScatter: 先 Reduce 再 Scatter，每卡得到部分聚合结果"""
    # 每卡准备 world_size 个 chunk
    input_tensor = torch.tensor(
        [(rank + 1.0) * (i + 1) for i in range(world_size * 2)],
        device="cuda"
    )
    output_tensor = torch.zeros(2, device="cuda")

    input_list = list(input_tensor.chunk(world_size))
    print(f"  [ReduceScatter] Before: rank {rank} input = {input_tensor.tolist()}")
    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
    print(f"  [ReduceScatter] After:  rank {rank} output = {output_tensor.tolist()}")


def demo_p2p(rank, world_size):
    """P2P Send/Recv: 点对点通信"""
    if rank == 0:
        tensor = torch.tensor([99.0, 88.0, 77.0, 66.0], device="cuda")
        print(f"  [P2P] rank 0 sending: {tensor.tolist()}")
        dist.send(tensor, dst=1)
        print(f"  [P2P] rank 0 send complete")
    else:
        tensor = torch.zeros(4, device="cuda")
        print(f"  [P2P] rank 1 before recv: {tensor.tolist()}")
        dist.recv(tensor, src=0)
        print(f"  [P2P] rank 1 received: {tensor.tolist()}")


def main():
    rank, world_size = setup()

    demos = [
        ("1. Broadcast", demo_broadcast),
        ("2. Reduce", demo_reduce),
        ("3. AllReduce", demo_allreduce),
        ("4. AllGather", demo_allgather),
        ("5. ReduceScatter", demo_reduce_scatter),
        ("6. P2P Send/Recv", demo_p2p),
    ]

    for name, fn in demos:
        dist.barrier()
        if rank == 0:
            print(f"\n{'='*50}")
            print(f" {name}")
            print(f"{'='*50}")
        dist.barrier()

        fn(rank, world_size)
        dist.barrier()

    if rank == 0:
        print(f"\n{'='*50}")
        print(" All demos completed!")
        print(f"{'='*50}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
