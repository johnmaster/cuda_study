"""训练后量化 (PTQ) PyTorch 实现
演示动态量化和静态量化（含校准）的完整流程
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """简单的两层全连接网络"""
    def __init__(self, in_features=256, hidden=128, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ======================== 手动实现 PTQ ========================

def compute_scale_minmax(tensor: torch.Tensor):
    """MinMax 校准"""
    return (tensor.max() - tensor.min()) / 255.0


def compute_scale_percentile(tensor: torch.Tensor, percentile=99.99):
    """百分位校准（抗异常值）"""
    k = int(tensor.numel() * (1 - percentile / 100))
    k = max(k, 1)
    upper = tensor.kthvalue(tensor.numel() - k).values
    lower = tensor.kthvalue(k).values
    return (upper - lower) / 255.0


def compute_scale_mse(tensor: torch.Tensor, num_candidates=100):
    """MSE 最优校准：搜索最小化量化误差的 scale"""
    best_scale = None
    best_mse = float('inf')

    abs_max = tensor.abs().max()
    for i in range(1, num_candidates + 1):
        candidate_max = abs_max * i / num_candidates
        scale = candidate_max / 127.0
        q = torch.clamp(torch.round(tensor / scale), -127, 127)
        recovered = q * scale
        mse = ((tensor - recovered) ** 2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
    return best_scale


class ManualPTQModel:
    """手动实现的 PTQ 流程"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.weight_scales = {}
        self.activation_scales = {}

    def quantize_weights(self):
        """离线量化权重"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                scale = param.data.abs().max() / 127.0
                self.weight_scales[name] = scale
                print(f"  {name}: scale={scale:.6f}, "
                      f"range=[{param.data.min():.4f}, {param.data.max():.4f}]")

    def calibrate(self, calibration_data, method='minmax'):
        """用校准数据统计激活值范围"""
        activation_stats = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(output.detach())
            return hook_fn

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                self.model(batch)

        for h in hooks:
            h.remove()

        # 根据校准方法计算 scale
        compute_fn = {
            'minmax': compute_scale_minmax,
            'percentile': compute_scale_percentile,
            'mse': compute_scale_mse,
        }[method]

        for name, tensors in activation_stats.items():
            all_activations = torch.cat(tensors)
            scale = compute_fn(all_activations)
            self.activation_scales[name] = scale
            print(f"  {name}: scale={scale:.6f}")

    def quantized_inference(self, x):
        """模拟量化推理"""
        with torch.no_grad():
            # fc1: 量化权重 × 量化输入
            w1 = self.model.fc1.weight.data
            w1_scale = self.weight_scales['fc1.weight']
            w1_q = torch.clamp(torch.round(w1 / w1_scale), -127, 127)

            x_scale = x.abs().max() / 127.0
            x_q = torch.clamp(torch.round(x / x_scale), -127, 127)

            # INT8 矩阵乘 → INT32 累加 → 反量化
            out = (x_q @ w1_q.T).float() * (x_scale * w1_scale)
            out += self.model.fc1.bias.data

            out = torch.relu(out)

            # fc2
            w2 = self.model.fc2.weight.data
            w2_scale = self.weight_scales['fc2.weight']
            w2_q = torch.clamp(torch.round(w2 / w2_scale), -127, 127)

            out_scale = out.abs().max() / 127.0
            out_q = torch.clamp(torch.round(out / out_scale), -127, 127)

            result = (out_q @ w2_q.T).float() * (out_scale * w2_scale)
            result += self.model.fc2.bias.data

            return result


if __name__ == "__main__":
    torch.manual_seed(42)

    model = SimpleModel()
    x = torch.randn(32, 256)

    # FP32 推理
    model.eval()
    with torch.no_grad():
        y_fp32 = model(x)

    print("===== Post-Training Quantization =====\n")

    # 手动 PTQ
    ptq = ManualPTQModel(model)

    print("[1] 量化权重:")
    ptq.quantize_weights()

    print("\n[2] 校准激活值 (MinMax):")
    calibration_data = [torch.randn(32, 256) for _ in range(10)]
    ptq.calibrate(calibration_data, method='minmax')

    # 量化推理
    y_quant = ptq.quantized_inference(x)

    mse = ((y_fp32 - y_quant) ** 2).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        y_fp32.flatten().unsqueeze(0),
        y_quant.flatten().unsqueeze(0)
    ).item()

    print(f"\n[3] 推理精度对比:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Cosine Similarity: {cos_sim:.6f}")
    print(f"  FP32 output range: [{y_fp32.min():.4f}, {y_fp32.max():.4f}]")
    print(f"  INT8 output range: [{y_quant.min():.4f}, {y_quant.max():.4f}]")

    # 对比不同校准方法
    print(f"\n[4] 校准方法对比:")
    for method in ['minmax', 'percentile', 'mse']:
        ptq2 = ManualPTQModel(model)
        ptq2.quantize_weights()
        ptq2.calibrate(calibration_data, method=method)
        y_q = ptq2.quantized_inference(x)
        mse = ((y_fp32 - y_q) ** 2).mean().item()
        print(f"  {method:>12}: MSE = {mse:.8f}")
