"""Per-Channel 量化 PyTorch 实现"""

import torch


def per_tensor_quantize(tensor: torch.Tensor, bits: int = 8):
    """Per-Tensor 对称量化（对照组）"""
    qmax = 2 ** (bits - 1) - 1
    scale = tensor.abs().max() / qmax
    q = torch.clamp(torch.round(tensor / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def per_tensor_dequantize(q: torch.Tensor, scale: float):
    return q.float() * scale


def per_channel_quantize(tensor: torch.Tensor, bits: int = 8):
    """Per-Channel 对称量化（沿 dim=0，即输出通道）"""
    qmax = 2 ** (bits - 1) - 1
    # 每个通道独立计算 scale
    scales = tensor.abs().amax(dim=list(range(1, tensor.dim()))) / qmax
    shape = [-1] + [1] * (tensor.dim() - 1)
    q = torch.clamp(torch.round(tensor / scales.view(shape)), -qmax, qmax).to(torch.int8)
    return q, scales


def per_channel_dequantize(q: torch.Tensor, scales: torch.Tensor):
    shape = [-1] + [1] * (q.dim() - 1)
    return q.float() * scales.view(shape)


if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟卷积权重 [out_channels=64, in_channels=32, kH=3, kW=3]
    # 人为让不同通道有不同的数值范围
    weight = torch.randn(64, 32, 3, 3)
    for i in range(64):
        weight[i] *= (i + 1) * 0.1

    print("===== Per-Tensor vs Per-Channel 量化 =====\n")

    # 查看通道数值范围差异
    ch_ranges = weight.abs().amax(dim=(1, 2, 3))
    print(f"通道 absmax 范围: [{ch_ranges.min():.4f}, {ch_ranges.max():.4f}]")
    print(f"最大/最小通道比值: {ch_ranges.max() / ch_ranges.min():.1f}x\n")

    # Per-Tensor
    q_pt, s_pt = per_tensor_quantize(weight)
    w_pt = per_tensor_dequantize(q_pt, s_pt)
    mse_pt = ((weight - w_pt) ** 2).mean().item()

    # Per-Channel
    q_pc, s_pc = per_channel_quantize(weight)
    w_pc = per_channel_dequantize(q_pc, s_pc)
    mse_pc = ((weight - w_pc) ** 2).mean().item()

    print(f"[Per-Tensor]  MSE = {mse_pt:.8f}")
    print(f"[Per-Channel] MSE = {mse_pc:.8f}")
    print(f"\nPer-Channel 精度提升: {mse_pt / mse_pc:.1f}x")

    # 查看每个通道的误差分布
    ch_mse_pt = ((weight - w_pt) ** 2).mean(dim=(1, 2, 3))
    ch_mse_pc = ((weight - w_pc) ** 2).mean(dim=(1, 2, 3))

    print(f"\n--- 各通道 MSE 对比（前 5 个通道） ---")
    print(f"{'通道':>4}  {'Per-Tensor MSE':>15}  {'Per-Channel MSE':>16}  {'提升':>6}")
    for i in range(5):
        ratio = ch_mse_pt[i] / ch_mse_pc[i] if ch_mse_pc[i] > 0 else float('inf')
        print(f"  {i:2d}  {ch_mse_pt[i]:15.8f}  {ch_mse_pc[i]:16.8f}  {ratio:5.1f}x")
