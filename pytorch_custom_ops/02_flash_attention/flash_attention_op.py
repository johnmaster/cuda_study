"""
Flash Attention as a PyTorch custom op with full autograd support.

Usage:
    from flash_attention_op import FlashAttention

    attn = FlashAttention()
    O = attn(Q, K, V)        # Q, K, V: [B, N, d], float32, CUDA
    O.sum().backward()        # computes dQ, dK, dV via our CUDA backward kernels
"""

import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
flash_attn_cuda = load(
    name="flash_attn_cuda",
    sources=[
        os.path.join(_dir, "csrc/binding.cpp"),
        os.path.join(_dir, "csrc/flash_attn_kernel.cu"),
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class FlashAttentionFunction(Function):
    """
    autograd.Function that wires our CUDA kernels into PyTorch's computation graph.

    Forward saves Q, K, V, O, and logsumexp L — but NOT the N×N attention matrix.
    Backward recomputes the attention weights on-the-fly in tiles (the core Flash
    Attention insight: trade compute for memory).
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        O, L = flash_attn_cuda.forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ, dK, dV = flash_attn_cuda.backward(dO, Q, K, V, O, L)
        return dQ, dK, dV


class FlashAttention(torch.nn.Module):
    """
    Drop-in module for scaled dot-product attention using Flash Attention.

    Input:  Q, K, V  — each [B, N, d], float32, CUDA
    Output: O        — [B, N, d]

    Equivalent to: softmax(Q @ K^T / sqrt(d)) @ V
    But uses O(N) memory instead of O(N^2).
    """

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return FlashAttentionFunction.apply(Q, K, V)
