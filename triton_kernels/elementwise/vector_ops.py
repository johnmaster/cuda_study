"""
Elementwise 向量运算 — Triton 实现

展示最基础的 1D 并行：每个 program 负责一段连续下标，load → 计算 → store。
涉及 program_id、arange、mask 边界和合并访问（连续 BLOCK 对齐）。

包含:
  - vector_add: z = x + y
  - vector_mul: z = x * y（可选对比）
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x + y
    tl.store(z_ptr + offs, z, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    z = x + y，要求 x/y 同 shape、同 device、同 dtype（与 kernel 一致）。

    Args:
        x, y: 任意 shape，内部视作 1D contiguous 元素序列。
        block_size: 每个 program 处理的元素个数。
    """
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    assert x.dtype == y.dtype
    n = x.numel()
    z = torch.empty_like(x)
    xc = x.contiguous().view(-1)
    yc = y.contiguous().view(-1)
    zc = z.view(-1)
    grid = (triton.cdiv(n, block_size),)
    _add_kernel[grid](xc, yc, zc, n, BLOCK=block_size)
    return z


@triton.jit
def _mul_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x * y, mask=mask)


def vector_mul(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """z = x * y，接口与 vector_add 相同。"""
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    assert x.dtype == y.dtype
    n = x.numel()
    z = torch.empty_like(x)
    xc, yc = x.contiguous().view(-1), y.contiguous().view(-1)
    zc = z.view(-1)
    grid = (triton.cdiv(n, block_size),)
    _mul_kernel[grid](xc, yc, zc, n, BLOCK=block_size)
    return z
