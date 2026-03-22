"""对称量化 PyTorch 实现"""

import torch


def symmetric_quantize(tensor: torch.Tensor, bits: int = 8):
    """FP32 → INT8 对称量化"""
    qmax = 2 ** (bits - 1) - 1  # 127 for INT8
    scale = tensor.abs().max() / qmax
    q_tensor = torch.clamp(torch.round(tensor / scale), -qmax, qmax).to(torch.int8)
    return q_tensor, scale


def symmetric_dequantize(q_tensor: torch.Tensor, scale: float):
    """INT8 → FP32 对称反量化"""
    return q_tensor.float() * scale


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(1024) * 0.5

    q, scale = symmetric_quantize(x)
    x_hat = symmetric_dequantize(q, scale)

    mse = ((x - x_hat) ** 2).mean().item()
    max_err = (x - x_hat).abs().max().item()

    print("===== Symmetric INT8 Quantization =====")
    print(f"Scale: {scale:.6f}")
    print(f"MSE: {mse:.8f}")
    print(f"Max Error: {max_err:.6f}")
    print(f"Memory: FP32={x.nelement()*4} bytes, INT8={q.nelement()*1} bytes")
    print(f"\nSamples:")
    for i in range(5):
        print(f"  [{i}] original={x[i]:.6f}  quantized={q[i]:4d}  "
              f"recovered={x_hat[i]:.6f}  error={abs(x[i]-x_hat[i]):.6f}")
