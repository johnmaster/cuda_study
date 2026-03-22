"""Per-Group 量化 PyTorch 实现"""

import torch


def per_group_quantize(tensor: torch.Tensor, group_size: int = 128, bits: int = 8):
    """Per-Group 对称量化（沿最后一维分组）"""
    qmax = 2 ** (bits - 1) - 1
    orig_shape = tensor.shape

    # reshape 为 [..., num_groups, group_size]
    assert tensor.shape[-1] % group_size == 0, \
        f"最后一维 {tensor.shape[-1]} 必须能被 group_size {group_size} 整除"
    t = tensor.reshape(*tensor.shape[:-1], -1, group_size)

    scales = t.abs().amax(dim=-1) / qmax  # [..., num_groups]
    q = torch.clamp(torch.round(t / scales.unsqueeze(-1)), -qmax, qmax).to(torch.int8)
    q = q.reshape(orig_shape)
    return q, scales


def per_group_dequantize(q: torch.Tensor, scales: torch.Tensor, group_size: int = 128):
    """Per-Group 反量化"""
    orig_shape = q.shape
    t = q.reshape(*q.shape[:-1], -1, group_size).float()
    result = t * scales.unsqueeze(-1)
    return result.reshape(orig_shape)


if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟 LLM 线性层权重 [out=1024, in=1024]
    # 人为制造不同区域数值范围差异
    weight = torch.randn(1024, 1024)
    for i in range(0, 1024, 128):
        weight[:, i:i+128] *= (i // 128 + 1) * 0.2

    print("===== Per-Group 量化 (不同 group_size 对比) =====\n")

    # Per-Tensor 作为基线
    qmax = 127
    scale_pt = weight.abs().max() / qmax
    q_pt = torch.clamp(torch.round(weight / scale_pt), -qmax, qmax).to(torch.int8)
    w_pt = q_pt.float() * scale_pt
    mse_pt = ((weight - w_pt) ** 2).mean().item()
    print(f"{'方法':<25} {'MSE':>12}  {'scale参数量':>10}")
    print(f"{'Per-Tensor':<25} {mse_pt:12.8f}  {'1':>10}")

    for gs in [32, 64, 128, 256]:
        q, scales = per_group_quantize(weight, group_size=gs)
        w_hat = per_group_dequantize(q, scales, group_size=gs)
        mse = ((weight - w_hat) ** 2).mean().item()
        n_scales = scales.numel()
        print(f"{'Per-Group (g=' + str(gs) + ')':<25} {mse:12.8f}  {n_scales:>10}")

    # INT4 Per-Group
    print(f"\n--- INT4 量化 ---")
    for gs in [32, 64, 128]:
        q, scales = per_group_quantize(weight, group_size=gs, bits=4)
        w_hat = per_group_dequantize(q, scales, group_size=gs)
        mse = ((weight - w_hat) ** 2).mean().item()
        n_scales = scales.numel()
        print(f"{'INT4 Per-Group (g=' + str(gs) + ')':<25} {mse:12.8f}  {n_scales:>10}")
