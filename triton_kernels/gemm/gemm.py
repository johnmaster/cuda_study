"""
GEMM (General Matrix Multiply) in Triton — 从 naive 到优化

本文件包含 3 个递进版本的 GEMM kernel，展示 Triton 的核心优化技巧:

  V1: Naive GEMM         — 最简单的分块矩阵乘
  V2: + Loop Tiling      — 沿 K 维分块，减少内存流量
  V3: + Auto-tuning      — Triton 的 autotune 自动搜索最优参数

对比 CUDA GEMM:
  - CUDA: 手动管理 shared memory tile、bank conflict、双缓冲、寄存器 tiling
  - Triton: 你只写 tile 级别逻辑，编译器自动做 shared memory 分配 + 合并访问 + 向量化

C = A @ B
A: [M, K]
B: [K, N]
C: [M, N]
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# V1: Naive Tiled GEMM
# ============================================================================

@triton.jit
def gemm_v1_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    V1: 最基础的 Triton GEMM.

    每个 program instance 计算 C 的一个 [BLOCK_M, BLOCK_N] 输出块。
    沿 K 维循环，每次加载 A 的 [BLOCK_M, BLOCK_K] 和 B 的 [BLOCK_K, BLOCK_N]，
    用 tl.dot 做块矩阵乘，累加到结果。

    Grid: (ceil(M/BLOCK_M) * ceil(N/BLOCK_N), )
    """
    pid = tl.program_id(0)

    # 把 1D pid 映射到 2D 的 (block_row, block_col)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    block_row = pid // num_n_blocks
    block_col = pid % num_n_blocks

    # 这个 block 负责的 M 维和 N 维范围
    rm = block_row * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = block_col * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # 累加器 (float32，即使 A/B 是 float16)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 K 维分块循环
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # 加载 A tile [BLOCK_M, BLOCK_K]
        a_offsets = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        # 加载 B tile [BLOCK_K, BLOCK_N]
        b_offsets = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)

        # 块矩阵乘  [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    # 写回 C
    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


# ============================================================================
# V2: + Swizzle (L2 cache 友好的 block 顺序)
# ============================================================================

@triton.jit
def gemm_v2_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    V2: 加入 Swizzle/Grouped ordering.

    V1 的问题: program 按行序遍历 C 的 blocks，相邻 program 访问 B 的不同列，
    L2 cache 命中率低。

    优化: 将 programs 分成 GROUP_SIZE 大小的组，同组内的 programs 覆盖 C 的
    一个 "super-tile"，共享 A 的行和 B 的列，提高 L2 复用。

    原理图:
      V1 遍历顺序:             V2 (grouped) 遍历顺序:
      [ 0  1  2  3]           [ 0  1  2  3]
      [ 4  5  6  7]           [ 4  5  6  7]    ← group 0 (8 blocks)
      [ 8  9 10 11]           [ 8  9 10 11]
      [12 13 14 15]           [12 13 14 15]    ← group 1

    同组的 block 共享 A 和 B 的部分行/列 → L2 cache hit ↑
    """
    pid = tl.program_id(0)

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    # grouped ordering: 把连续的 pid 映射到一个 group 内
    num_blocks_in_group = GROUP_SIZE * num_n_blocks
    group_id = pid // num_blocks_in_group
    first_block_m = group_id * GROUP_SIZE
    group_size_m = min(num_m_blocks - first_block_m, GROUP_SIZE)

    block_row = first_block_m + ((pid % num_blocks_in_group) % group_size_m)
    block_col = (pid % num_blocks_in_group) // group_size_m

    rm = block_row * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = block_col * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        a_offsets = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        b_offsets = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


# ============================================================================
# V3: + Autotune (自动搜索最优 block size)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_SIZE": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_SIZE": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_SIZE": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],  # 当 M/N/K 变化时重新搜索
)
@triton.jit
def gemm_v3_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    V3: 和 V2 的 kernel 代码完全一样，但加了 @triton.autotune。

    Triton 会在第一次运行时，对 configs 列表中的每种配置都跑一遍 benchmark，
    选择最快的那个。

    autotune 搜索的维度:
      - BLOCK_M, BLOCK_N, BLOCK_K: tile 大小
      - GROUP_SIZE: swizzle 分组大小
      - num_warps: 每个 block 用几个 warp
      - num_stages: software pipelining 的阶段数
    """
    pid = tl.program_id(0)

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    num_blocks_in_group = GROUP_SIZE * num_n_blocks
    group_id = pid // num_blocks_in_group
    first_block_m = group_id * GROUP_SIZE
    group_size_m = min(num_m_blocks - first_block_m, GROUP_SIZE)

    block_row = first_block_m + ((pid % num_blocks_in_group) % group_size_m)
    block_col = (pid % num_blocks_in_group) // group_size_m

    rm = block_row * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = block_col * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        a_offsets = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        b_offsets = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


# ============================================================================
# Python Wrappers
# ============================================================================

def triton_gemm_v1(A, B):
    """Naive tiled GEMM."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    gemm_v1_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


def triton_gemm_v2(A, B):
    """Tiled GEMM + grouped ordering (L2 cache friendly)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_SIZE = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    gemm_v2_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE=GROUP_SIZE,
    )
    return C


def triton_gemm_v3(A, B):
    """Autotuned GEMM — Triton automatically picks the best config."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )

    gemm_v3_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    M, N, K = 512, 512, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    ref = A @ B

    for name, fn in [("V1 naive", triton_gemm_v1),
                      ("V2 swizzle", triton_gemm_v2),
                      ("V3 autotune", triton_gemm_v3)]:
        C = fn(A, B)
        diff = (C - ref).abs().max().item()
        # tl.dot uses TF32 on Ampere+ GPUs → slightly lower precision than full FP32
        print(f"  {name:15s}  max_diff={diff:.2e}  {'PASS' if diff < 0.1 else 'FAIL'}")

    print("GEMM Triton — OK")
