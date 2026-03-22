"""非对称量化 PyTorch 实现"""

import torch


def asymmetric_quantize(tensor: torch.Tensor, bits: int = 8):
    """FP32 → UINT8 非对称量化"""
    qmin, qmax = 0, 2 ** bits - 1
    x_min, x_max = tensor.min(), tensor.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = torch.round(-x_min / scale).clamp(qmin, qmax).int()
    q_tensor = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax).to(torch.uint8)
    return q_tensor, scale, zero_point


def asymmetric_dequantize(q_tensor: torch.Tensor, scale: float, zero_point: int):
    """UINT8 → FP32 非对称反量化"""
    return (q_tensor.float() - zero_point) * scale


if __name__ == "__main__":
    torch.manual_seed(42)
    # 模拟 ReLU 后激活值
    x = torch.relu(torch.randn(1024)) * 3.0

    q, scale, zp = asymmetric_quantize(x)
    x_hat = asymmetric_dequantize(q, scale, zp)

    mse = ((x - x_hat) ** 2).mean().item()
    max_err = (x - x_hat).abs().max().item()

    print("===== Asymmetric UINT8 Quantization =====")
    print(f"Data range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Scale: {scale:.6f}, Zero Point: {zp}")
    print(f"MSE: {mse:.8f}, Max Error: {max_err:.6f}")

    # 对比：同样的数据用对称量化
    from symmetric_quant import symmetric_quantize, symmetric_dequantize  # noqa: E402
    q_sym, scale_sym = symmetric_quantize(x)
    x_sym = symmetric_dequantize(q_sym, scale_sym)
    mse_sym = ((x - x_sym) ** 2).mean().item()
    print(f"\n[对比] 对称量化 MSE: {mse_sym:.8f}")
    print(f"[对比] 非对称量化 MSE: {mse:.8f}")
    print(f"非对称量化精度提升: {mse_sym / mse:.1f}x")
