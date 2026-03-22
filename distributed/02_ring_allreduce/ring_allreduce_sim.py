"""Ring AllReduce 纯 Python 模拟

单进程运行，不需要 GPU。用 list 模拟多张 GPU，打印每一步的状态变化。
运行方式: python ring_allreduce_sim.py
"""

import copy


def ring_allreduce(gpus: list[list[float]], verbose: bool = True) -> list[list[float]]:
    """
    Ring AllReduce 算法模拟

    Args:
        gpus: gpus[i] 是第 i 张 GPU 上的数据（长度必须被 N 整除）
    Returns:
        每张 GPU 上的数据都变成所有 GPU 数据的逐元素之和
    """
    N = len(gpus)
    M = len(gpus[0])
    chunk_size = M // N
    assert M % N == 0, f"数据长度 {M} 必须被 GPU 数 {N} 整除"

    gpus = copy.deepcopy(gpus)

    def get_chunk(gpu_data, chunk_id):
        start = chunk_id * chunk_size
        return gpu_data[start:start + chunk_size]

    def set_chunk(gpu_data, chunk_id, values):
        start = chunk_id * chunk_size
        gpu_data[start:start + chunk_size] = values

    def add_chunks(a, b):
        return [x + y for x, y in zip(a, b)]

    if verbose:
        print(f"=== Ring AllReduce: {N} GPUs, {M} elements, chunk_size={chunk_size} ===\n")
        print("初始状态:")
        for i, g in enumerate(gpus):
            chunks = [get_chunk(g, c) for c in range(N)]
            print(f"  GPU {i}: {chunks}")
        print()

    # ==================== Phase 1: ReduceScatter ====================
    if verbose:
        print("=" * 60)
        print(" Phase 1: ReduceScatter")
        print("=" * 60)

    for step in range(N - 1):
        if verbose:
            print(f"\n--- Step {step + 1}/{N - 1} ---")

        # 每张卡同时发送和接收
        # GPU i 发送 chunk[(i - step) % N] 给 GPU (i+1) % N
        # GPU i 从 GPU (i-1) % N 接收 chunk[(i - step - 1) % N]
        send_bufs = {}
        for i in range(N):
            send_chunk_id = (i - step) % N
            send_bufs[i] = get_chunk(gpus[i], send_chunk_id)
            if verbose:
                dst = (i + 1) % N
                print(f"  GPU {i} → GPU {dst}: chunk[{send_chunk_id}] = {send_bufs[i]}")

        for i in range(N):
            src = (i - 1) % N
            recv_chunk_id = (i - step - 1) % N
            received = send_bufs[src]
            current = get_chunk(gpus[i], recv_chunk_id)
            new_val = add_chunks(current, received)
            set_chunk(gpus[i], recv_chunk_id, new_val)

        if verbose:
            print(f"\n  结果:")
            for i, g in enumerate(gpus):
                chunks = [get_chunk(g, c) for c in range(N)]
                print(f"    GPU {i}: {chunks}")

    if verbose:
        print(f"\n  ReduceScatter 完成! 每卡持有一个 chunk 的完整求和:")
        for i in range(N):
            owner_chunk = (i + 1) % N
            val = get_chunk(gpus[i], owner_chunk)
            print(f"    GPU {i} → chunk[{owner_chunk}] = {val} ✓")

    # ==================== Phase 2: AllGather ====================
    if verbose:
        print(f"\n{'=' * 60}")
        print(" Phase 2: AllGather")
        print("=" * 60)

    for step in range(N - 1):
        if verbose:
            print(f"\n--- Step {step + 1}/{N - 1} ---")

        send_bufs = {}
        for i in range(N):
            # 发送自己已有完整结果的 chunk
            send_chunk_id = (i - step + 1) % N
            send_bufs[i] = get_chunk(gpus[i], send_chunk_id)
            if verbose:
                dst = (i + 1) % N
                print(f"  GPU {i} → GPU {dst}: chunk[{send_chunk_id}] = {send_bufs[i]}")

        for i in range(N):
            src = (i - 1) % N
            recv_chunk_id = (i - step) % N
            received = send_bufs[src]
            set_chunk(gpus[i], recv_chunk_id, received)  # 直接覆盖，不累加

        if verbose:
            print(f"\n  结果:")
            for i, g in enumerate(gpus):
                chunks = [get_chunk(g, c) for c in range(N)]
                print(f"    GPU {i}: {chunks}")

    return gpus


def main():
    print("=" * 60)
    print(" 示例 1: 4 GPUs, 简单整数")
    print("=" * 60)
    print()

    gpus_4 = [
        [1, 2, 3, 4],      # GPU 0
        [10, 20, 30, 40],   # GPU 1
        [100, 200, 300, 400],  # GPU 2
        [1000, 2000, 3000, 4000],  # GPU 3
    ]

    result = ring_allreduce(gpus_4, verbose=True)

    expected = [1 + 10 + 100 + 1000,
                2 + 20 + 200 + 2000,
                3 + 30 + 300 + 3000,
                4 + 40 + 400 + 4000]

    print(f"\n\n期望结果: {expected}")
    print(f"实际结果:")
    all_correct = True
    for i, g in enumerate(result):
        ok = g == expected
        all_correct &= ok
        print(f"  GPU {i}: {g} {'✓' if ok else '✗'}")
    print(f"\n验证: {'通过 ✓' if all_correct else '失败 ✗'}")

    # 示例 2: 2 GPUs（对应你的双卡机器）
    print(f"\n\n{'=' * 60}")
    print(" 示例 2: 2 GPUs（模拟你的 2×2080Ti）")
    print("=" * 60)
    print()

    gpus_2 = [
        [1.0, 2.0, 3.0, 4.0],    # GPU 0
        [0.5, 1.5, 2.5, 3.5],    # GPU 1
    ]

    result = ring_allreduce(gpus_2, verbose=True)

    expected = [1.5, 3.5, 5.5, 7.5]
    print(f"\n期望: {expected}")
    print(f"GPU 0: {result[0]} {'✓' if result[0] == expected else '✗'}")
    print(f"GPU 1: {result[1]} {'✓' if result[1] == expected else '✗'}")

    # 通信量分析
    print(f"\n\n{'=' * 60}")
    print(" 通信量分析")
    print("=" * 60)
    for N in [2, 4, 8, 16, 32, 64]:
        M = 1_000_000_000  # 1B 参数 × 4 bytes = 4GB
        bytes_per_gpu = 2 * M * 4 * (N - 1) / N  # 4 bytes per float
        print(f"  N={N:3d} GPUs: 每卡通信 {bytes_per_gpu / 1e9:.2f} GB "
              f"(理论总时间与N几乎无关)")


if __name__ == "__main__":
    main()
