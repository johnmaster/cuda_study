"""
Flash Attention in Triton — Forward + Backward

对比 CUDA 版本 (pytorch_custom_ops/02_flash_attention/):
  - CUDA 版: ~380 行 .cu + 27 行 binding.cpp + 66 行 Python wrapper
  - Triton 版: 全在这一个 Python 文件里

Triton 的核心优势:
  1. 用 Python 写 GPU kernel，不需要 nvcc / pybind11 / setup.py
  2. 编译器自动处理 shared memory 分配、内存合并、寄存器分配
  3. 你只需要思考 "分块(tiling) + 计算逻辑"，不需要手动管理 threadIdx / blockIdx

核心概念:
  - tl.program_id(axis)  ≈  CUDA 的 blockIdx.x/y/z
  - BLOCK_SIZE 常量      ≈  CUDA 的 blockDim（但 Triton 是按 block 操作，不按 thread）
  - tl.load / tl.store   ≈  CUDA 的全局内存读写，但自动处理合并访问
  - tl.dot               ≈  CUDA 的矩阵乘（可以自动利用 Tensor Core）
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    每个 program instance 处理一行 query (batch b, row i)。

    算法 (Online Softmax):
      m = -inf, l = 0, o = 0
      for each K/V tile j:
          S_ij = Q_i @ K_j^T / sqrt(d)
          m_new = max(m, max(S_ij))
          P_ij = exp(S_ij - m_new)
          alpha = exp(m - m_new)
          o = o * alpha + P_ij @ V_j
          l = l * alpha + sum(P_ij)
          m = m_new
      O_i = o / l
      L_i = m + log(l)
    """
    batch = tl.program_id(1)
    row   = tl.program_id(0)

    # Q[batch, row, :] 的起始指针
    q_offset = batch * stride_qb + row * stride_qn
    # 加载整行 Q_i (长度 D)
    d_range = tl.arange(0, D)
    q = tl.load(Q_ptr + q_offset + d_range * stride_qd)  # [D]

    scale: tl.constexpr = 1.0 / tl.sqrt(float(D))

    # online softmax 累加器
    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.full([], 0.0, dtype=tl.float32)
    o_i = tl.zeros([D], dtype=tl.float32)

    # 遍历 K/V 的每个 tile
    for j_start in range(0, N, BLOCK_N):
        j_range = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask = j_range < N

        # 加载 K tile: [BLOCK_N, D]
        k_offset = batch * stride_kb + j_range[:, None] * stride_kn + d_range[None, :] * stride_kd
        k = tl.load(K_ptr + k_offset, mask=mask[:, None], other=0.0)

        # S_ij = Q_i @ K_j^T → [BLOCK_N]
        # q: [D], k: [BLOCK_N, D] → 手动点积
        s = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_N]
        s = tl.where(mask, s, float("-inf"))

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)  # [BLOCK_N]

        # 加载 V tile: [BLOCK_N, D]
        v_offset = batch * stride_vb + j_range[:, None] * stride_vn + d_range[None, :] * stride_vd
        v = tl.load(V_ptr + v_offset, mask=mask[:, None], other=0.0)

        # o = o * alpha + P @ V
        o_i = o_i * alpha + tl.sum(p[:, None] * v, axis=0)  # [D]
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # 归一化
    o_i = o_i / l_i

    # 写 O
    o_offset = batch * stride_ob + row * stride_on
    tl.store(O_ptr + o_offset + d_range * stride_od, o_i)

    # 写 logsumexp L (backward 需要)
    tl.store(L_ptr + batch * N + row, m_i + tl.log(l_i))


# ============================================================================
# Backward Kernel — dQ
# ============================================================================

@triton.jit
def _flash_attn_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N: tl.constexpr, Di: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    每个 program 处理一行 dQ[batch, row, :].

    dQ_i = sum_j  dS_ij * K_j * scale
    其中 dS_ij = P_ij * (dO_i @ V_j - D_i)
         P_ij  = exp(Q_i @ K_j * scale - L_i)
         D_i   = sum_k dO_{i,k} * O_{i,k}   (预计算)
    """
    batch = tl.program_id(1)
    row   = tl.program_id(0)

    d_range = tl.arange(0, Di)
    scale: tl.constexpr = 1.0 / tl.sqrt(float(Di))

    q_offset = batch * stride_qb + row * stride_qn
    q = tl.load(Q_ptr + q_offset + d_range * stride_qd)
    do = tl.load(dO_ptr + batch * stride_ob + row * stride_on + d_range * stride_od)
    lse_i = tl.load(L_ptr + batch * N + row)
    d_i = tl.load(D_ptr + batch * N + row)

    dq_acc = tl.zeros([Di], dtype=tl.float32)

    for j_start in range(0, N, BLOCK_N):
        j_range = j_start + tl.arange(0, BLOCK_N)
        mask = j_range < N

        k_offset = batch * stride_kb + j_range[:, None] * stride_kn + d_range[None, :] * stride_kd
        k = tl.load(K_ptr + k_offset, mask=mask[:, None], other=0.0)

        v_offset = batch * stride_vb + j_range[:, None] * stride_vn + d_range[None, :] * stride_vd
        v = tl.load(V_ptr + v_offset, mask=mask[:, None], other=0.0)

        # 重算 P_ij
        s = tl.sum(q[None, :] * k, axis=1) * scale
        s = tl.where(mask, s, float("-inf"))
        p = tl.exp(s - lse_i)  # [BLOCK_N]

        # dP_ij = dO_i @ V_j
        dp = tl.sum(do[None, :] * v, axis=1)  # [BLOCK_N]

        # dS_ij = P_ij * (dP_ij - D_i)
        ds = p * (dp - d_i)  # [BLOCK_N]

        # dQ_i += dS_ij * K_j * scale
        dq_acc += tl.sum(ds[:, None] * k, axis=0) * scale

    dq_offset = batch * stride_qb + row * stride_qn
    tl.store(dQ_ptr + dq_offset + d_range * stride_qd, dq_acc)


# ============================================================================
# Backward Kernel — dK, dV
# ============================================================================

@triton.jit
def _flash_attn_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    N: tl.constexpr, Di: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    每个 program 处理一行 dK[batch, col, :] 和 dV[batch, col, :].

    dK_j = sum_i  dS_ij * Q_i * scale
    dV_j = sum_i  P_ij  * dO_i
    """
    batch = tl.program_id(1)
    col   = tl.program_id(0)

    d_range = tl.arange(0, Di)
    scale: tl.constexpr = 1.0 / tl.sqrt(float(Di))

    k_offset = batch * stride_kb + col * stride_kn
    k = tl.load(K_ptr + k_offset + d_range * stride_kd)
    v = tl.load(V_ptr + batch * stride_vb + col * stride_vn + d_range * stride_vd)

    dk_acc = tl.zeros([Di], dtype=tl.float32)
    dv_acc = tl.zeros([Di], dtype=tl.float32)

    for i_start in range(0, N, BLOCK_N):
        i_range = i_start + tl.arange(0, BLOCK_N)
        mask = i_range < N

        q_offset = batch * stride_qb + i_range[:, None] * stride_qn + d_range[None, :] * stride_qd
        q_tile = tl.load(Q_ptr + q_offset, mask=mask[:, None], other=0.0)

        do_offset = batch * stride_ob + i_range[:, None] * stride_on + d_range[None, :] * stride_od
        do_tile = tl.load(dO_ptr + do_offset, mask=mask[:, None], other=0.0)

        lse_tile = tl.load(L_ptr + batch * N + i_range, mask=mask, other=0.0)
        d_tile = tl.load(D_ptr + batch * N + i_range, mask=mask, other=0.0)

        # 重算 S_{i, col} = Q_i @ K_col * scale
        s = tl.sum(q_tile * k[None, :], axis=1) * scale  # [BLOCK_N]
        s = tl.where(mask, s, float("-inf"))
        p = tl.exp(s - lse_tile)  # [BLOCK_N]

        dp = tl.sum(do_tile * v[None, :], axis=1)  # [BLOCK_N]
        ds = p * (dp - d_tile)  # [BLOCK_N]

        dk_acc += tl.sum(ds[:, None] * q_tile, axis=0) * scale
        dv_acc += tl.sum(p[:, None] * do_tile, axis=0)

    dk_offset = batch * stride_kb + col * stride_kn
    dv_offset = batch * stride_vb + col * stride_vn
    tl.store(dK_ptr + dk_offset + d_range * stride_kd, dk_acc)
    tl.store(dV_ptr + dv_offset + d_range * stride_vd, dv_acc)


# ============================================================================
# Python wrapper — autograd.Function
# ============================================================================

def _get_block_n(N):
    """选择 tile 大小 (必须是 2 的幂)。"""
    if N <= 32:
        return 32
    elif N <= 64:
        return 64
    else:
        return 128


class FlashAttentionTritonFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V):
        B, N, D = Q.shape
        assert D <= 256, "head dim must be <= 256"

        O = torch.empty_like(Q)
        L = torch.empty(B, N, device=Q.device, dtype=torch.float32)

        BLOCK_N = _get_block_n(N)
        grid = (N, B)

        _flash_attn_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            N=N, D=D, BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.BLOCK_N = BLOCK_N
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, N, D = Q.shape
        BLOCK_N = ctx.BLOCK_N

        # 预计算 D_i = dot(dO_i, O_i)
        D_vec = (dO * O).sum(dim=-1)  # [B, N]

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        grid = (N, B)

        _flash_attn_bwd_dq_kernel[grid](
            Q, K, V, dO, L, D_vec, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            N=N, Di=D, BLOCK_N=BLOCK_N,
        )

        _flash_attn_bwd_dkv_kernel[grid](
            Q, K, V, dO, L, D_vec, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            N=N, Di=D, BLOCK_N=BLOCK_N,
        )

        return dQ, dK, dV


class FlashAttentionTriton(torch.nn.Module):
    """Drop-in scaled dot-product attention using Triton Flash Attention."""

    def forward(self, Q, K, V):
        return FlashAttentionTritonFunction.apply(
            Q.contiguous(), K.contiguous(), V.contiguous()
        )


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, D = 2, 64, 64
    Q = torch.randn(B, N, D, device="cuda", requires_grad=True)
    K = torch.randn(B, N, D, device="cuda", requires_grad=True)
    V = torch.randn(B, N, D, device="cuda", requires_grad=True)

    attn = FlashAttentionTriton()
    O = attn(Q, K, V)

    # reference
    ref = torch.softmax(Q @ K.transpose(-1, -2) / D**0.5, dim=-1) @ V
    print(f"Forward max diff: {(O - ref).abs().max().item():.2e}")

    O.sum().backward()
    print(f"dQ norm: {Q.grad.norm().item():.4f}")
    print("Flash Attention Triton — OK")
