"""
Python wrapper that integrates the CUDA kernel into PyTorch's autograd.

Usage:
    from custom_gelu import CustomGELU

    gelu = CustomGELU()
    y = gelu(x)          # forward — calls our CUDA kernel
    y.sum().backward()    # backward — calls our CUDA backward kernel
"""

import torch
from torch.autograd import Function

# Option A: import the pre-built extension (after `pip install -e .`)
# import custom_ops_cuda

# Option B: JIT compile at first import (no setup.py needed)
import os
from torch.utils.cpp_extension import load
_dir = os.path.dirname(os.path.abspath(__file__))
custom_ops_cuda = load(
    name="custom_ops_cuda",
    sources=[
        os.path.join(_dir, "csrc/binding.cpp"),
        os.path.join(_dir, "csrc/custom_gelu_kernel.cu"),
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class CustomGELUFunction(Function):
    """Connects our CUDA kernels to PyTorch autograd."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return custom_ops_cuda.forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        grad_input = custom_ops_cuda.backward(grad_output.contiguous(), input)
        return grad_input


class CustomGELU(torch.nn.Module):
    """Drop-in replacement for torch.nn.GELU using our custom CUDA kernel."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomGELUFunction.apply(input)
