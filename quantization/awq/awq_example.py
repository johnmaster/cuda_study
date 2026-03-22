"""AWQ (Activation-aware Weight Quantization) 简化实现
演示核心算法：激活值感知的 Per-Channel 缩放 + 量化
"""

import torch


def quantize_tensor(w: torch.Tensor, bits: int = 4, group_size: int = 128):
    """Per-Group 对称量化"""
    qmax = 2 ** (bits - 1) - 1
    orig_shape = w.shape

    if w.shape[-1] % group_size != 0:
        # padding
        pad = group_size - w.shape[-1] % group_size
        w = torch.nn.functional.pad(w, (0, pad))

    w_grouped = w.reshape(-1, group_size)
    scales = w_grouped.abs().amax(dim=-1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)
    q = torch.clamp(torch.round(w_grouped / scales), -qmax, qmax)
    w_hat = (q * scales).reshape(w.shape)

    return w_hat[..., :orig_shape[-1]], scales


def compute_output_error(X, W_orig, W_quant):
    """计算量化后的输出误差"""
    Y_orig = X @ W_orig.T
    Y_quant = X @ W_quant.T
    return ((Y_orig - Y_quant) ** 2).mean().item()


def awq_scale_search(W: torch.Tensor, X: torch.Tensor,
                     bits: int = 4, group_size: int = 128,
                     n_grid: int = 20):
    """
    AWQ 核心：搜索最优 per-channel 缩放因子

    Args:
        W: 权重 [out_features, in_features]
        X: 校准数据激活值 [n_samples, in_features]
        bits: 量化位宽
        group_size: Per-Group 大小
        n_grid: α 搜索的网格密度
    Returns:
        best_scales: 最优缩放因子 [in_features]
    """
    # 计算每个输入通道的激活值重要性
    activation_magnitude = X.abs().mean(dim=0)  # [in_features]

    best_error = float('inf')
    best_scales = torch.ones(W.shape[1])
    best_alpha = 0

    for i in range(n_grid):
        alpha = i / n_grid  # α ∈ [0, 1)

        # 计算缩放因子: s_j = a_j^α
        scales = activation_magnitude.pow(alpha)
        scales = scales.clamp(min=1e-4)

        # 缩放权重: W' = W * s (对输入维度缩放)
        W_scaled = W * scales.unsqueeze(0)

        # 量化缩放后的权重
        W_scaled_q, _ = quantize_tensor(W_scaled, bits, group_size)

        # 反缩放: W_final = W_q / s
        W_final = W_scaled_q / scales.unsqueeze(0)

        # 计算输出误差
        error = compute_output_error(X, W, W_final)

        if error < best_error:
            best_error = error
            best_scales = scales.clone()
            best_alpha = alpha

    return best_scales, best_alpha, best_error


def awq_quantize(W: torch.Tensor, X: torch.Tensor,
                 bits: int = 4, group_size: int = 128):
    """AWQ 量化完整流程"""
    # 搜索最优缩放
    scales, alpha, _ = awq_scale_search(W, X, bits, group_size)

    # 应用缩放 + 量化
    W_scaled = W * scales.unsqueeze(0)
    W_scaled_q, q_scales = quantize_tensor(W_scaled, bits, group_size)
    W_awq = W_scaled_q / scales.unsqueeze(0)

    return W_awq, scales, alpha


if __name__ == "__main__":
    torch.manual_seed(42)

    in_features = 512
    out_features = 256
    n_samples = 128

    W = torch.randn(out_features, in_features) * 0.02
    X = torch.randn(n_samples, in_features)

    # 人为注入异常值通道（模拟 LLM 中的 outlier）
    outlier_channels = [10, 50, 100, 200, 300]
    for ch in outlier_channels:
        X[:, ch] *= 20.0  # 激活值放大 20 倍

    Y_ref = X @ W.T

    print("===== AWQ vs Naive vs GPTQ-style Quantization =====\n")

    # 查看激活值分布
    act_mag = X.abs().mean(dim=0)
    top_channels = act_mag.topk(5)
    print(f"激活值最大的 5 个通道:")
    for idx, val in zip(top_channels.indices.tolist(), top_channels.values.tolist()):
        print(f"  通道 {idx}: avg|activation| = {val:.4f}")
    print()

    for bits in [4, 3]:
        print(f"--- INT{bits} 量化 ---")

        # 朴素量化（无缩放）
        W_naive, _ = quantize_tensor(W, bits)
        mse_naive = compute_output_error(X, W, W_naive)

        # AWQ
        W_awq, scales, alpha = awq_quantize(W, X, bits)
        mse_awq = compute_output_error(X, W, W_awq)

        ratio = mse_naive / mse_awq if mse_awq > 0 else float('inf')

        print(f"  Naive:  MSE = {mse_naive:.8f}")
        print(f"  AWQ:    MSE = {mse_awq:.8f}  (α={alpha:.2f})")
        print(f"  提升: {ratio:.1f}x")
        print()

    # 展示缩放因子在异常值通道上的效果
    print("--- 缩放因子分析 ---")
    _, scales_4bit, alpha = awq_scale_search(W, X, bits=4)
    print(f"最优 α = {alpha:.2f}")
    print(f"缩放因子统计: mean={scales_4bit.mean():.4f}, "
          f"max={scales_4bit.max():.4f}")
    print(f"\n异常值通道的缩放因子 (相比普通通道更大):")
    normal_scale = scales_4bit[[0, 1, 2, 3, 4]].mean().item()
    for ch in outlier_channels:
        print(f"  通道 {ch}: scale={scales_4bit[ch]:.4f} "
              f"(是普通通道的 {scales_4bit[ch]/normal_scale:.1f}x)")
