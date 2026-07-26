# 混合精度训练完全指南 (Mixed Precision Training)

## 目录

1. [为什么需要混合精度](#1-为什么需要混合精度)
2. [浮点数格式详解](#2-浮点数格式详解)
3. [混合精度训练原理](#3-混合精度训练原理)
4. [PyTorch AMP 使用](#4-pytorch-amp-使用)
5. [Loss Scaling 机制](#5-loss-scaling-机制)
6. [Tensor Core 与性能](#6-tensor-core-与性能)
7. [常见问题与调优](#7-常见问题与调优)
8. [核心问题](#8-核心问题)

---

## 1. 为什么需要混合精度

### 问题: 训练大模型的瓶颈

| 资源 | FP32 训练 | 混合精度训练 |
|------|-----------|-------------|
| 显存 | 参数 + 梯度 + 优化器状态 ≈ 16x 参数量 | 约 **60-70%** |
| 计算 | CUDA Core only | **Tensor Core** (2-8x 吞吐) |
| 带宽 | 4 bytes/elem | 2 bytes/elem (**2x** 有效带宽) |

具体例子: 1B 参数模型
- FP32: 参数 4GB + 梯度 4GB + Adam 状态 8GB = **16 GB**
- 混合精度: FP16参数 2GB + FP16梯度 2GB + FP32 master 4GB + Adam 8GB = **16 GB** (但前向/反向的中间激活值是 FP16, 大幅节省)

真正的显存节省来自 **activation memory**:
- 前向传播的中间结果 (对大 batch size 是显存主要开销)
- FP16 直接减半

---

## 2. 浮点数格式详解

### 2.1 格式对比

```
FP64: [1 sign][11 exponent][52 mantissa]  64 bits
FP32: [1 sign][ 8 exponent][23 mantissa]  32 bits
TF32: [1 sign][ 8 exponent][10 mantissa]  19 bits  ← Ampere+
BF16: [1 sign][ 8 exponent][ 7 mantissa]  16 bits  ← Ampere+
FP16: [1 sign][ 5 exponent][10 mantissa]  16 bits
FP8:  [1 sign][ 4 exponent][ 3 mantissa]   8 bits  ← Hopper
```

### 2.2 关键参数

| 格式 | 最小正数 | 最大值 | 精度 (eps) | 用途 |
|------|---------|--------|-----------|------|
| FP32 | 1.2e-38 | 3.4e+38 | 1.2e-7 | 参数存储, 优化器 |
| FP16 | 6.1e-5 | 65504 | 9.8e-4 | 前向/反向, 需 loss scaling |
| BF16 | 1.2e-38 | 3.4e+38 | 7.8e-3 | 前向/反向, 不需 loss scaling |
| TF32 | 1.2e-38 | 3.4e+38 | 9.8e-4 | Tensor Core 内部计算格式 |

### 2.3 关键洞察

**BF16 vs FP16:**
```
FP16: 小指数 (5 bit) → 范围小 (max 65504)   → 容易 overflow/underflow → 需要 loss scaling
BF16: 大指数 (8 bit) → 范围大 (同 FP32)     → 不容易溢出            → 不需要 loss scaling

FP16: 大尾数 (10 bit) → 精度高
BF16: 小尾数 (7 bit)  → 精度低
```

**TF32:** 不是一种存储格式, 而是 Tensor Core 的内部计算模式。输入仍是 FP32, 但 Tensor Core 截断到 10-bit mantissa 计算, 再用 FP32 累加。效果: FP32 输入/输出, 但速度接近 FP16。

---

## 3. 混合精度训练原理

### 3.1 核心流程

```
   FP32 Master Weights
          │
          ├─── (1) Cast to FP16/BF16 ───→ FP16 Model Copy
          │                                     │
          │                                (2) Forward (FP16)
          │                                     │
          │                                (3) Loss (FP32!)
          │                                     │
          │                           (4) Loss × Scale Factor
          │                                     │
          │                                (5) Backward (FP16)
          │                                     │
          │                           (6) Grad / Scale Factor
          │                                     │
          ├─── (7) Cast grad to FP32 ←──── FP16 Gradients
          │
     (8) FP32 Optimizer Step (Adam, SGD...)
          │
          ├─── (9) 回到 (1)
```

### 3.2 三个核心技术

**a) FP32 Master Weights**
```python
# 为什么不能直接用 FP16 做 optimizer step?
lr = 1e-4
grad = 1e-3
update = lr * grad  # = 1e-7

# FP16 最小正数 ≈ 6e-5
# 1e-7 在 FP16 中被舍入为 0 → 参数永远不更新!

# 解决: 保留 FP32 副本做更新
fp32_param -= lr * fp32_grad       # FP32: 能精确表示 1e-7
fp16_param = fp32_param.half()     # 复制回 FP16 用于下一次前向
```

**b) Loss Scaling**
```python
# FP16 梯度容易下溢 (值太小变成 0)
# 解决: 放大 loss → 放大梯度 → 防止下溢 → 更新前缩回

scaled_loss = loss * 1024        # 放大
scaled_loss.backward()           # 梯度也被放大
grad = param.grad / 1024         # 缩回
```

**c) Op-level Casting**
```
需要 FP16 (compute-bound, 受益于 Tensor Core):
  ✓ matmul, linear, conv, bmm, GRU/LSTM

保持 FP32 (numerically sensitive):
  ✓ softmax, log_softmax, layer_norm, batch_norm
  ✓ cross_entropy, loss functions
  ✓ exp, log, pow, sum (reductions)

跟随输入 (不强制转换):
  ✓ relu, gelu, sigmoid, dropout
  ✓ max_pool, avg_pool, cat, stack
```

---

## 4. PyTorch AMP 使用

### 4.1 基本用法

```python
# 关键: 只需要改 3 行代码
scaler = torch.amp.GradScaler()                         # ← 新增

for images, labels in loader:
    with torch.amp.autocast(device_type="cuda"):        # ← 新增
        outputs = model(images)
        loss = criterion(outputs, labels)

    optimizer.zero_grad()
    scaler.scale(loss).backward()                        # ← 改动
    scaler.step(optimizer)                               # ← 改动
    scaler.update()                                      # ← 新增
```

### 4.2 BF16 更简单 (Ampere+)

```python
# BF16 不需要 GradScaler (动态范围够大)
for images, labels in loader:
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(images)
        loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 4.3 autocast 原理

```python
# autocast 是一个 context manager, 在进入时注册 dispatch hooks
# 当一个 op 被调用时:
#   1. 查 autocast policy 表 (cast-to-fp16 / keep-fp32 / follow-input)
#   2. 如果需要 cast: 将输入 tensor cast 到目标 dtype
#   3. 执行 op
#   4. 输出保持 op 计算后的 dtype

# autocast 只影响 CUDA ops, CPU 不受影响
# autocast 不影响已有 tensor 的 dtype, 只在 op dispatch 时 cast
```

### 4.4 GradScaler 工作流

```python
# GradScaler 内部流程:
#
# 1. scaler.scale(loss)
#    → return loss * self._scale
#
# 2. scaled_loss.backward()
#    → 所有梯度都被放大了 self._scale 倍
#
# 3. scaler.step(optimizer)
#    → 先 unscale: grad /= self._scale
#    → 检查 inf/nan
#    → 如果没有 overflow: optimizer.step()
#    → 如果有 overflow: 跳过这一步
#
# 4. scaler.update()
#    → 如果 step 成功: _growth_tracker += 1
#    → 如果连续成功 2000 步: _scale *= 2 (尝试更大的 scale)
#    → 如果 step 失败 (overflow): _scale /= 2
```

---

## 5. Loss Scaling 机制

### 5.1 为什么 FP16 需要 Loss Scaling

```
FP16 表示范围: [6.1e-5, 65504]
FP32 表示范围: [1.2e-38, 3.4e+38]

典型梯度范围: 1e-7 ~ 1e-3
→ 很多梯度 < 6.1e-5 (FP16 最小正数)
→ 在 FP16 中被舍入为 0 (underflow)
→ 参数不更新, 训练停滞
```

### 5.2 Static vs Dynamic Loss Scaling

**Static:**
```python
scale = 1024  # 固定值
scaled_loss = loss * scale
scaled_loss.backward()
for p in model.parameters():
    p.grad /= scale
optimizer.step()
```

**Dynamic (PyTorch GradScaler 默认):**
```
初始 scale = 2^16 = 65536
连续 2000 步没 overflow → scale *= 2
出现 overflow (inf/nan) → scale /= 2, 跳过更新
```

### 5.3 BF16 不需要 Loss Scaling

```
BF16 指数位 = 8 bit = FP32 的指数位
→ 表示范围和 FP32 完全相同
→ 梯度不会 underflow
→ 不需要 loss scaling
→ 训练更稳定, 代码更简单

这就是为什么 BF16 在大模型训练中更受欢迎:
  - GPT-3, LLaMA, Gemini 都用 BF16
  - 减少了 loss scaling 带来的不确定性
```

---

## 6. Tensor Core 与性能

### 6.1 为什么混合精度能加速

混合精度的加速不仅仅是 "用更少的 bit 表示":

```
加速来源 1: Tensor Core 吞吐量
  FP32 (CUDA Core): 19.5 TFLOPS (A100)
  FP16 (Tensor Core): 312 TFLOPS (A100) ← 16x!

  大部分训练时间花在 GEMM (matmul):
  - 前向: Y = XW  (matmul)
  - 反向: dW = X^T dY, dX = dY W^T  (matmul)

  autocast 把 GEMM 转成 FP16 → Tensor Core → 16x 峰值吞吐

加速来源 2: 内存带宽
  FP16 = 2 bytes vs FP32 = 4 bytes
  → 2x 有效内存带宽
  → 对 bandwidth-bound ops (如 activation, normalization) 也有帮助

加速来源 3: 显存节省 → 更大 batch size
  → 更好的 GPU 利用率
  → 更稳定的训练 (更大 batch = 更准确的梯度估计)
```

### 6.2 TF32 模式

```python
# PyTorch 在 Ampere+ GPU 上默认启用 TF32 for matmul
# 含义: FP32 输入, 但 Tensor Core 截断到 TF32 (10-bit mantissa) 计算

torch.backends.cuda.matmul.allow_tf32 = True   # 默认: True
torch.backends.cudnn.allow_tf32 = True          # 默认: True

# 效果:
#   - 用户代码不需要改 (仍然是 FP32 tensor)
#   - 但 GEMM 速度接近 FP16 (用 Tensor Core)
#   - 精度介于 FP16 和 FP32 之间

# 如果需要严格 FP32 精度 (如验证):
torch.backends.cuda.matmul.allow_tf32 = False
```

### 6.3 Tensor Core 要求

```
Tensor Core 需要对齐:
  - 矩阵维度最好是 8 的倍数 (FP16) 或 16 的倍数 (INT8)
  - 内存地址对齐

PyTorch cuBLAS 自动处理:
  - 如果维度不对齐, cuBLAS 会自动 padding
  - 但 padding 会浪费计算 → 设计模型时尽量让维度是 8 的倍数

Hidden size 推荐:
  ✓ 768, 1024, 2048, 4096 (8 的倍数)
  ✗ 765, 1000, 2000 (不是 8 的倍数)
```

---

## 7. 常见问题与调优

### 7.1 Loss Spike / NaN

```
症状: 训练中 loss 突然变成 inf 或 nan
原因: FP16 overflow (值 > 65504)

解决方案:
  1. 使用 BF16 代替 FP16 (如果 GPU 支持)
  2. 减小学习率
  3. 增加 gradient clipping:
     scaler.unscale_(optimizer)
     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     scaler.step(optimizer)
  4. 检查 GradScaler 的 scale 是否持续下降 (说明频繁 overflow)
```

### 7.2 训练不收敛

```
症状: 使用 AMP 后模型不收敛或精度下降
可能原因:
  1. 某些自定义 op 不在 autocast 的 policy 表中
     → 手动 cast: op(x.float()) 或 op(x.half())
  2. accumulation 精度不足
     → 确保 reduction ops (sum, mean) 用 FP32
  3. 梯度累积时忘了处理 scaling
```

### 7.3 梯度累积 + AMP

```python
# 正确写法:
for micro_step in range(accum_steps):
    with torch.amp.autocast(device_type="cuda"):
        loss = model(batch) / accum_steps
    scaler.scale(loss).backward()

# 累积完后再 step
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()

# scaler.scale(loss).backward() 可以调多次
# 梯度在 .grad 上累加
# 最后一次性 unscale + step
```

### 7.4 自定义算子 + AMP

```python
class MyCustomOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return custom_cuda_forward(input)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return custom_cuda_backward(input, grad_output)

# @custom_fwd 确保 forward 在 autocast 区域内正确处理 dtype
# @custom_bwd 确保 backward 的 dtype 一致
```

---

## 8. 核心问题

### Q1: 混合精度训练为什么能加速?

**答:**
三个层面的加速:
1. **Tensor Core 吞吐**: FP16 matmul 在 Tensor Core 上的吞吐是 FP32 CUDA Core 的 8-16 倍 (取决于 GPU 代数)
2. **内存带宽**: FP16 tensor 只需一半的数据传输, 对 bandwidth-bound ops 有 2x 加速
3. **显存节省**: activation memory 减半, 可以用更大 batch size, 提高 GPU 利用率

### Q2: FP16 和 BF16 的区别? 为什么 LLM 训练更倾向 BF16?

**答:**
- FP16: 5-bit exponent, 10-bit mantissa → 范围小 (max 65504), 精度高
- BF16: 8-bit exponent, 7-bit mantissa → 范围大 (同 FP32), 精度低

LLM 训练倾向 BF16 因为:
1. 不需要 loss scaling (范围够大, 梯度不会 underflow)
2. 训练更稳定 (不会因 loss scaling 导致的 skip step 浪费计算)
3. 代码更简单 (不需要 GradScaler)
4. 虽然精度低一点, 但对大模型的最终效果影响极小

### Q3: Loss Scaling 是怎么工作的?

**答:**
1. 前向后把 loss 乘一个大的 scale factor (如 2^16)
2. backward 时梯度也被同比例放大 → 防止小梯度在 FP16 中下溢为 0
3. optimizer step 前把梯度除以 scale factor 缩回原始值
4. Dynamic scaling: 连续 N 步没 overflow → scale 翻倍; 出现 overflow → scale 减半, 跳过这步更新

### Q4: 为什么需要 FP32 Master Weights?

**答:**
学习率 × 梯度 的结果通常很小 (如 1e-4 × 1e-3 = 1e-7), 这个值在 FP16 中会被舍入为 0 (FP16 最小正数约 6e-5)。如果直接用 FP16 做 optimizer step, 很多参数更新会消失。

所以保留一份 FP32 的参数副本:
- FP16 模型: 用于前向/反向 (快)
- FP32 master: 用于 optimizer step (精确)
- 每步结束后: FP32 → cast to FP16 → 下一次前向

### Q5: TF32 是什么? 和 FP32 有什么区别?

**答:**
TF32 不是一种存储格式, 而是 Ampere+ GPU 上 Tensor Core 的一种计算模式:
- 输入/输出仍是 FP32 (32-bit 存储)
- Tensor Core 内部截断到 19-bit (8-bit exp + 10-bit mantissa) 计算
- 累加仍用 FP32

效果: 用户代码完全不变, 但 matmul 速度接近 FP16, 精度介于 FP16 和 FP32 之间。
PyTorch 在 Ampere+ 上默认启用 TF32。

### Q6: autocast 区域内哪些 op 用 FP16, 哪些保持 FP32?

**答:**
- **FP16**: compute-bound 的 GEMM 类 op (matmul, linear, conv, bmm) → 受益于 Tensor Core
- **FP32**: 数值敏感的 op (softmax, layernorm, batchnorm, loss functions, exp/log) → 防止精度问题
- **跟随输入**: element-wise op (relu, gelu, dropout, pooling) → 不强制转换

### Q7: 混合精度和模型并行/数据并行怎么配合?

**答:**
- **DDP + AMP**: 每个进程独立做 autocast + GradScaler, 梯度同步 (allreduce) 在 FP16 完成
- **FSDP + AMP**: 参数分片可以在 FP16 存储, 减半通信量
- **Tensor Parallelism**: 跨 GPU 的 matmul 在 FP16 通信, 减少 NVLink/IB 带宽压力
- 关键: 通信用 FP16 可以减半通信量, 这对多卡训练非常重要

### Q8: 为什么有些模型用混合精度训练后精度下降?

**答:**
常见原因:
1. **自定义 op 没有正确处理 dtype**: 需要用 `@torch.amp.custom_fwd` 装饰器
2. **累加精度不够**: 如手写 attention 中 score 累加应该用 FP32
3. **数值敏感的操作没有保持 FP32**: 如自定义 loss function
4. **梯度累积时没正确处理 scaling**: 每个 micro-step 都 scale, 最后统一 unscale + step

解决: 用 `torch.autograd.detect_anomaly()` 定位 NaN, 对可疑 op 手动 `.float()`
