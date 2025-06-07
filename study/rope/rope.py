import time
from typing import Optional, Tuple
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

lib = load(
    name="rope",
    sources=["rope.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, out)
    else:
        for i in range(warmup):
            _ = perf_func(a)
    torch.cuda.synchronize()
    start = time.time()

    if out is not None:
        for i in range(iters):
            perf_func(a, out)
    else:
        for i in range(iters):
            out = perf_func(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time

def naive_rope(
    x: torch.Tensor,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x.shape[-1]最后一个维度
    # x.shape[-2]倒数第二个维度
    dim = x.shape[-1]
    seq_len = x.shape[-2]
    
    