"""量化感知训练 (QAT) PyTorch 实现
手动实现 STE (Straight-Through Estimator) 和伪量化
"""

import torch
import torch.nn as nn
import torch.optim as optim


class FakeQuantize(torch.autograd.Function):
    """伪量化操作：前向做量化+反量化，反向用 STE 直通梯度"""

    @staticmethod
    def forward(ctx, x, scale, bits=8):
        qmax = 2 ** (bits - 1) - 1
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        x_hat = q * scale
        # 保存 mask 用于 STE（只传递在量化范围内的梯度）
        ctx.save_for_backward(x, torch.tensor([qmax * scale]))
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        x, max_val = ctx.saved_tensors
        # STE：梯度直通，但超出量化范围的部分梯度置零
        mask = (x.abs() <= max_val).float()
        return grad_output * mask, None, None


def fake_quantize(x, scale, bits=8):
    return FakeQuantize.apply(x, scale, bits)


class QATLinear(nn.Module):
    """带量化感知的线性层"""

    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, x):
        # 伪量化权重
        w_scale = self.linear.weight.data.abs().max() / self.qmax
        w_fq = fake_quantize(self.linear.weight, w_scale, self.bits)

        # 伪量化输入激活值
        x_scale = x.detach().abs().max() / self.qmax
        if x_scale == 0:
            x_scale = torch.tensor(1.0)
        x_fq = fake_quantize(x, x_scale, self.bits)

        return nn.functional.linear(x_fq, w_fq, self.linear.bias)


class FP32Model(nn.Module):
    """原始 FP32 模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class QATModel(nn.Module):
    """QAT 模型：用 QATLinear 替换普通 Linear"""
    def __init__(self, bits=8):
        super().__init__()
        self.fc1 = QATLinear(64, 128, bits)
        self.relu = nn.ReLU()
        self.fc2 = QATLinear(128, 10, bits)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def generate_data(n_samples=1000):
    """生成简单的分类数据"""
    torch.manual_seed(42)
    x = torch.randn(n_samples, 64)
    # 简单规则：前 32 维的和 > 0 为正类
    y = (x[:, :32].sum(dim=1) > 0).long()
    return x, y


def train_model(model, x_train, y_train, epochs=20, lr=0.01):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_train).argmax(dim=1)
        acc = (pred == y_train).float().mean().item()
    return acc


def ptq_inference(model, x, bits=8):
    """PTQ 推理（直接量化权重和激活）"""
    qmax = 2 ** (bits - 1) - 1

    with torch.no_grad():
        w1 = model.fc1.weight.data
        s1 = w1.abs().max() / qmax
        w1_q = torch.clamp(torch.round(w1 / s1), -qmax, qmax)

        sx = x.abs().max() / qmax
        x_q = torch.clamp(torch.round(x / sx), -qmax, qmax)

        h = (x_q @ w1_q.T).float() * (sx * s1) + model.fc1.bias.data
        h = torch.relu(h)

        w2 = model.fc2.weight.data
        s2 = w2.abs().max() / qmax
        w2_q = torch.clamp(torch.round(w2 / s2), -qmax, qmax)

        sh = h.abs().max() / qmax
        h_q = torch.clamp(torch.round(h / sh), -qmax, qmax)

        out = (h_q @ w2_q.T).float() * (sh * s2) + model.fc2.bias.data
    return out


if __name__ == "__main__":
    x_train, y_train = generate_data(1000)
    x_test, y_test = generate_data(200)

    print("===== QAT vs PTQ 对比实验 =====\n")

    # 1. 训练 FP32 模型
    fp32_model = FP32Model()
    acc_fp32 = train_model(fp32_model, x_train, y_train, epochs=50)
    print(f"[FP32 模型] 训练集准确率: {acc_fp32:.4f}")

    fp32_model.eval()
    with torch.no_grad():
        acc_test = (fp32_model(x_test).argmax(1) == y_test).float().mean().item()
    print(f"[FP32 模型] 测试集准确率: {acc_test:.4f}")

    # 2. PTQ 直接量化
    ptq_out = ptq_inference(fp32_model, x_test)
    acc_ptq = (ptq_out.argmax(1) == y_test).float().mean().item()
    print(f"\n[PTQ INT8]  测试集准确率: {acc_ptq:.4f}")

    # 3. QAT 训练
    for bits in [8, 4]:
        qat_model = QATModel(bits=bits)
        # 从 FP32 模型复制权重
        qat_model.fc1.linear.load_state_dict(fp32_model.fc1.state_dict())
        qat_model.fc2.linear.load_state_dict(fp32_model.fc2.state_dict())

        acc_qat = train_model(qat_model, x_train, y_train, epochs=30, lr=0.001)

        qat_model.eval()
        with torch.no_grad():
            acc_test_qat = (qat_model(x_test).argmax(1) == y_test).float().mean().item()
        print(f"[QAT INT{bits}] 测试集准确率: {acc_test_qat:.4f}")
