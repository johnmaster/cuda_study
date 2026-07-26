"""GPTQ 简化实现
核心算法：逐列量化 + Hessian 误差补偿
"""

import torch
import torch.nn as nn


def symmetric_quantize_tensor(w: torch.Tensor, bits: int = 4):
    """对称量化"""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone(), scale
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale, scale


def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int = 4,
                  block_size: int = 128, percdamp: float = 0.01):
    """
    GPTQ 核心算法
    
    Args:
        W: 权重矩阵 [out_features, in_features]
        H: Hessian 矩阵 [in_features, in_features] = X^T @ X
        bits: 量化位宽
        block_size: 分块大小
        percdamp: Hessian 对角线的阻尼系数（防止数值不稳定）
    
    Returns:
        Q: 量化后的权重, scales: 每组的 scale
    """
    out_features, in_features = W.shape
    qmax = 2 ** (bits - 1) - 1

    W = W.clone().float()
    Q = torch.zeros_like(W)

    # Hessian 正则化
    damp = percdamp * H.diag().mean()
    H_diag = H.diagonal()
    H_diag += damp

    # Cholesky 分解（用于高效求逆）
    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
    except Exception:
        # fallback: 直接求逆
        H_inv = torch.inverse(H + damp * torch.eye(in_features))

    scales = []

    for i1 in range(0, in_features, block_size):
        i2 = min(i1 + block_size, in_features)
        count = i2 - i1

        W_block = W[:, i1:i2].clone()
        Q_block = torch.zeros_like(W_block)
        H_inv_block = H_inv[i1:i2, i1:i2]

        for j in range(count):
            col = i1 + j
            w = W_block[:, j]
            h_inv_jj = H_inv_block[j, j]

            # 量化当前列
            scale = w.abs().max() / qmax
            if scale == 0:
                scale = torch.tensor(1.0)
            q = torch.clamp(torch.round(w / scale), -qmax, qmax)
            q_val = q * scale
            Q_block[:, j] = q_val

            # 计算误差并补偿块内后续列
            err = (w - q_val) / h_inv_jj
            W_block[:, j:] -= err.unsqueeze(1) @ H_inv_block[j, j:].unsqueeze(0)

            scales.append(scale)

        Q[:, i1:i2] = Q_block

        # 块间更新：补偿后续所有未处理的列
        if i2 < in_features:
            err_block = W[:, i1:i2] - Q[:, i1:i2]
            W[:, i2:] -= err_block @ H_inv[i1:i2, i2:]

    return Q, scales


def naive_quantize(W: torch.Tensor, bits: int = 4):
    """朴素 Round-to-Nearest 量化（对照组）"""
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().max() / qmax
    q = torch.clamp(torch.round(W / scale), -qmax, qmax)
    return q * scale, scale


if __name__ == "__main__":
    torch.manual_seed(42)

    in_features = 512
    out_features = 256
    n_calibration = 128

    W = torch.randn(out_features, in_features) * 0.02
    X_cal = torch.randn(n_calibration, in_features)

    # 计算 Hessian
    H = (X_cal.T @ X_cal) / n_calibration

    # FP32 参考输出
    Y_ref = X_cal @ W.T

    print("===== GPTQ vs Naive Quantization =====\n")

    for bits in [8, 4, 3]:
        # 朴素量化
        Q_naive, _ = naive_quantize(W, bits)
        Y_naive = X_cal @ Q_naive.T
        mse_naive = ((Y_ref - Y_naive) ** 2).mean().item()

        # GPTQ
        Q_gptq, _ = gptq_quantize(W, H, bits)
        Y_gptq = X_cal @ Q_gptq.T
        mse_gptq = ((Y_ref - Y_gptq) ** 2).mean().item()

        ratio = mse_naive / mse_gptq if mse_gptq > 0 else float('inf')

        print(f"--- INT{bits} ---")
        print(f"  Naive RTN:  MSE = {mse_naive:.8f}")
        print(f"  GPTQ:       MSE = {mse_gptq:.8f}")
        print(f"  GPTQ 精度提升: {ratio:.1f}x")
        print()

    # 权重误差分布对比
    print("--- 权重误差分布 (INT4) ---")
    Q_naive, _ = naive_quantize(W, 4)
    Q_gptq, _ = gptq_quantize(W, H, 4)
    err_naive = (W - Q_naive).abs()
    err_gptq = (W - Q_gptq).abs()
    print(f"  Naive: mean={err_naive.mean():.6f}, max={err_naive.max():.6f}")
    print(f"  GPTQ:  mean={err_gptq.mean():.6f}, max={err_gptq.max():.6f}")
