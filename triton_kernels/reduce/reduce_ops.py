"""
Reduce 归约 — Triton

1) sum_all:  将整个 1D（或展平）张量归约为标量
   - 每个 program 对一段 BLOCK 做 tl.sum，再用 tl.atomic_add 累加到单个输出
   - 注意: 输出需事先清零；atomic 在极大规模时可能成为瓶颈（教学用足够）

2) row_sum:  对 2D 张量 [M, N] 沿最后一维求和，得到 [M]
   - 每个 program 负责一行，内层沿 N 分块累加
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_all_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    """每个 block 计算局部和，原子加到 *out_ptr（标量）。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    block_sum = tl.sum(x)
    tl.atomic_add(out_ptr, block_sum)


def sum_all(x: torch.Tensor, block_size: int = 4096) -> torch.Tensor:
    """
    返回 shape [] 的标量张量（与 x 同 device），值为 x 所有元素之和。
    内部用 float32 累加（与 PyTorch sum 默认行为接近）；输入可为 fp16/bf16/fp32。
    """
    assert x.is_cuda
    xc = x.contiguous().view(-1).float()
    n = xc.numel()
    out = torch.zeros((), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(n, block_size),)
    _sum_all_kernel[grid](xc, out, n, BLOCK=block_size)
    return out


@triton.jit
def _row_sum_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    acc = tl.zeros((), dtype=tl.float32)
    col = tl.arange(0, BLOCK_N)
    for col_start in range(0, N, BLOCK_N):
        cols = col_start + col
        mask = cols < N
        idx = row * N + cols
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals)
    tl.store(out_ptr + row, acc)


def row_sum(x: torch.Tensor, block_n: int = 1024) -> torch.Tensor:
    """
    x: [M, N]，返回 [M]，在最后一维上求和。输出 float32。
    """
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    xc = x.contiguous().float()
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = (M,)
    _row_sum_kernel[grid](xc, out, M, N, BLOCK_N=block_n)
    return out
