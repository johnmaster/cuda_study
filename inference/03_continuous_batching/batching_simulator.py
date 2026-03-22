"""Continuous Batching 模拟器

模拟 Static Batching vs Continuous Batching 的吞吐量差异。
运行方式: python batching_simulator.py
"""

import random
import statistics


class Request:
    def __init__(self, req_id, arrival_time, prompt_len, gen_len):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.generated = 0
        self.start_time = None
        self.end_time = None

    @property
    def is_done(self):
        return self.generated >= self.gen_len

    @property
    def ttft(self):
        """Time To First Token"""
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None

    @property
    def total_latency(self):
        if self.end_time is not None:
            return self.end_time - self.arrival_time
        return None

    def __repr__(self):
        return f"Req({self.req_id}, gen={self.gen_len})"


def generate_requests(n_requests=50, seed=42):
    """生成模拟请求"""
    random.seed(seed)
    requests = []
    time = 0
    for i in range(n_requests):
        prompt_len = random.randint(10, 200)
        gen_len = random.randint(10, 500)
        requests.append(Request(i, time, prompt_len, gen_len))
        time += random.randint(0, 3)  # 请求间隔
    return requests


def simulate_static_batching(requests, max_batch_size=8):
    """Static Batching: 一批请求全部完成才处理下一批"""
    requests = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in requests]
    pending = list(requests)
    completed = []
    current_time = 0
    total_iters = 0
    gpu_busy_tokens = 0  # 实际处理的 token 数

    while pending:
        # 取一批
        batch = []
        while pending and len(batch) < max_batch_size:
            if pending[0].arrival_time <= current_time:
                batch.append(pending.pop(0))
            else:
                break

        if not batch:
            current_time = pending[0].arrival_time
            continue

        for r in batch:
            r.start_time = current_time

        # 处理直到所有请求完成
        max_gen = max(r.gen_len for r in batch)
        for step in range(max_gen):
            active_count = 0
            for r in batch:
                if not r.is_done:
                    r.generated += 1
                    active_count += 1
            total_iters += 1
            gpu_busy_tokens += active_count
            current_time += 1

        for r in batch:
            r.end_time = current_time
            completed.append(r)

    gpu_utilization = gpu_busy_tokens / (total_iters * max_batch_size) if total_iters > 0 else 0
    return completed, total_iters, gpu_utilization


def simulate_continuous_batching(requests, max_batch_size=8):
    """Continuous Batching: 每步检查完成/新到达，动态调整 batch"""
    requests = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in requests]
    pending = list(requests)
    running = []
    completed = []
    current_time = 0
    total_iters = 0
    gpu_busy_tokens = 0

    while pending or running:
        # 添加新到达的请求（填满 batch）
        while pending and len(running) < max_batch_size:
            if pending[0].arrival_time <= current_time:
                req = pending.pop(0)
                req.start_time = current_time
                running.append(req)
            else:
                break

        if not running:
            if pending:
                current_time = pending[0].arrival_time
            continue

        # 所有 running 的请求生成一个 token
        for r in running:
            r.generated += 1

        total_iters += 1
        gpu_busy_tokens += len(running)
        current_time += 1

        # 移除完成的请求
        newly_done = [r for r in running if r.is_done]
        for r in newly_done:
            r.end_time = current_time
            completed.append(r)
        running = [r for r in running if not r.is_done]

    gpu_utilization = gpu_busy_tokens / (total_iters * max_batch_size) if total_iters > 0 else 0
    return completed, total_iters, gpu_utilization


def print_stats(name, completed, total_iters, gpu_util, total_tokens):
    ttfts = [r.ttft for r in completed if r.ttft is not None]
    latencies = [r.total_latency for r in completed if r.total_latency is not None]
    end_time = max(r.end_time for r in completed)

    throughput = total_tokens / end_time if end_time > 0 else 0

    print(f"\n  --- {name} ---")
    print(f"  总迭代次数:     {total_iters}")
    print(f"  总耗时:         {end_time} time units")
    print(f"  吞吐量:         {throughput:.1f} tokens/unit")
    print(f"  GPU 利用率:     {gpu_util:.1%}")
    print(f"  平均 TTFT:      {statistics.mean(ttfts):.1f} units")
    print(f"  中位 TTFT:      {statistics.median(ttfts):.1f} units")
    print(f"  P99 TTFT:       {sorted(ttfts)[int(len(ttfts)*0.99)]:.1f} units")
    print(f"  平均延迟:       {statistics.mean(latencies):.1f} units")

    return throughput, gpu_util


def main():
    print("=" * 60)
    print(" Continuous Batching 模拟器")
    print("=" * 60)

    requests = generate_requests(n_requests=50)
    total_tokens = sum(r.gen_len for r in requests)

    print(f"\n[配置] {len(requests)} 个请求, 共 {total_tokens} tokens")
    print(f"  生成长度范围: 10-500 tokens")

    for batch_size in [4, 8, 16]:
        print(f"\n{'=' * 60}")
        print(f" Batch Size = {batch_size}")
        print(f"{'=' * 60}")

        s_comp, s_iters, s_util = simulate_static_batching(requests, batch_size)
        s_thru, _ = print_stats("Static Batching", s_comp, s_iters, s_util, total_tokens)

        c_comp, c_iters, c_util = simulate_continuous_batching(requests, batch_size)
        c_thru, _ = print_stats("Continuous Batching", c_comp, c_iters, c_util, total_tokens)

        print(f"\n  >> Continuous Batching 吞吐提升: {c_thru / s_thru:.2f}x")
        print(f"  >> GPU 利用率提升: {c_util / s_util:.2f}x")

    # 不同请求长度分布的影响
    print(f"\n\n{'=' * 60}")
    print(" 请求长度方差对吞吐的影响")
    print("=" * 60)

    print(f"\n  长度越不均匀 → Static 浪费越多 → Continuous 优势越大\n")
    print(f"  {'长度分布':<20} {'Static':>12} {'Continuous':>12} {'提升':>8}")
    print(f"  {'-' * 55}")

    for desc, low, high in [
        ("均匀 (90-110)", 90, 110),
        ("中等 (50-200)", 50, 200),
        ("很不均匀 (10-500)", 10, 500),
        ("极端 (1-1000)", 1, 1000),
    ]:
        random.seed(42)
        reqs = [Request(i, i // 5, 50, random.randint(low, high)) for i in range(50)]
        total = sum(r.gen_len for r in reqs)

        s_comp, s_iters, _ = simulate_static_batching(reqs, 8)
        s_end = max(r.end_time for r in s_comp)
        c_comp, c_iters, _ = simulate_continuous_batching(reqs, 8)
        c_end = max(r.end_time for r in c_comp)

        s_thru = total / s_end if s_end > 0 else 0
        c_thru = total / c_end if c_end > 0 else 0
        ratio = c_thru / s_thru if s_thru > 0 else float('inf')

        print(f"  {desc:<20} {s_thru:>10.1f} t/u {c_thru:>10.1f} t/u {ratio:>7.2f}x")

    print(f"\n{'=' * 60}")
    print(" 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
