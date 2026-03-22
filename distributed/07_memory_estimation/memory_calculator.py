"""LLM 训练/推理显存估算器

面试常考：给定模型配置，估算训练和推理需要多少显存。
运行方式: python memory_calculator.py
"""


def estimate_training_memory(
    n_params_B: float,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    n_layers: int,
    n_heads: int,
    dtype_bytes: int = 2,
    optimizer: str = "adam",
    tp_degree: int = 1,
    zero_stage: int = 0,
    n_gpus: int = 1,
    gradient_checkpointing: bool = False,
):
    """
    估算训练显存

    Returns: dict of memory components in GB
    """
    Phi = n_params_B * 1e9  # 参数量

    # 1. 模型状态
    param_bytes = Phi * dtype_bytes                      # FP16 参数
    grad_bytes = Phi * dtype_bytes                       # FP16 梯度

    if optimizer == "adam":
        opt_bytes = Phi * (4 + 4 + 4)  # master weight + m + v (FP32)
    elif optimizer == "sgd":
        opt_bytes = Phi * 4  # 只有 momentum
    else:
        opt_bytes = 0

    # ZeRO 分片
    if zero_stage >= 1:
        opt_bytes /= n_gpus
    if zero_stage >= 2:
        grad_bytes /= n_gpus
    if zero_stage >= 3:
        param_bytes /= n_gpus

    model_state = (param_bytes + grad_bytes + opt_bytes) / 1024**3

    # 2. 激活值 (Megatron 论文公式的简化版)
    # 每层: ~34 * s * b * h bytes (FP16) 不含 checkpointing
    act_per_layer = 34 * seq_len * batch_size * hidden_dim * dtype_bytes
    act_per_layer /= tp_degree  # TP 切分激活

    if gradient_checkpointing:
        n_stored_layers = int(n_layers ** 0.5)  # sqrt(N) 层存激活
    else:
        n_stored_layers = n_layers

    activation = (act_per_layer * n_stored_layers) / 1024**3

    # 3. 临时缓冲 (约 5% overhead)
    buffer = (param_bytes * 0.05) / 1024**3

    total = model_state + activation + buffer

    return {
        "model_state_gb": model_state,
        "param_gb": param_bytes / 1024**3,
        "grad_gb": grad_bytes / 1024**3,
        "optimizer_gb": opt_bytes / 1024**3,
        "activation_gb": activation,
        "buffer_gb": buffer,
        "total_gb": total,
    }


def estimate_inference_memory(
    n_params_B: float,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    n_layers: int,
    n_heads: int,
    dtype_bytes: int = 2,
    kv_cache_dtype_bytes: int = 2,
):
    """估算推理显存"""
    Phi = n_params_B * 1e9
    head_dim = hidden_dim // n_heads

    model_mem = Phi * dtype_bytes / 1024**3

    # KV Cache: 2 (K+V) × n_layers × batch × seq × n_heads × head_dim × dtype
    kv_cache = 2 * n_layers * batch_size * seq_len * n_heads * head_dim * kv_cache_dtype_bytes
    kv_cache_gb = kv_cache / 1024**3

    total = model_mem + kv_cache_gb

    return {
        "model_gb": model_mem,
        "kv_cache_gb": kv_cache_gb,
        "total_gb": total,
    }


def main():
    print("=" * 70)
    print(" LLM 显存估算器")
    print("=" * 70)

    # ==================== 训练显存 ====================
    print("\n[1] 训练显存估算 (Mixed Precision + Adam)\n")

    models = [
        ("LLaMA-7B",   7,  4096, 11008, 32, 32),
        ("LLaMA-13B", 13,  5120, 13824, 40, 40),
        ("LLaMA-70B", 70,  8192, 28672, 80, 64),
    ]

    print(f"  {'模型':<12} {'参数':>5} {'模型状态':>10} {'激活值':>10} "
          f"{'总计':>10} {'所需GPU':>12}")
    print(f"  {'-' * 65}")

    for name, n_B, hidden, ffn, n_layers, n_heads in models:
        mem = estimate_training_memory(
            n_params_B=n_B, batch_size=1, seq_len=2048,
            hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
        )
        n_a100 = max(1, int(mem['total_gb'] / 80) + 1)
        print(f"  {name:<12} {n_B:>4}B {mem['model_state_gb']:>8.1f}GB "
              f"{mem['activation_gb']:>8.1f}GB {mem['total_gb']:>8.1f}GB "
              f"{n_a100:>3}× A100-80G")

    # ==================== ZeRO 各阶段对比 ====================
    print(f"\n[2] ZeRO 各阶段显存对比 (LLaMA-7B, 4 GPUs, batch=1, seq=2048)\n")

    print(f"  {'策略':<20} {'参数':>8} {'梯度':>8} {'优化器':>8} "
          f"{'激活':>8} {'总计':>8}")
    print(f"  {'-' * 60}")

    for stage_name, stage in [("DDP (无ZeRO)", 0), ("ZeRO-1", 1),
                                ("ZeRO-2", 2), ("ZeRO-3 (FSDP)", 3)]:
        mem = estimate_training_memory(
            n_params_B=7, batch_size=1, seq_len=2048,
            hidden_dim=4096, n_layers=32, n_heads=32,
            zero_stage=stage, n_gpus=4,
        )
        print(f"  {stage_name:<20} {mem['param_gb']:>6.1f}GB "
              f"{mem['grad_gb']:>6.1f}GB {mem['optimizer_gb']:>6.1f}GB "
              f"{mem['activation_gb']:>6.1f}GB {mem['total_gb']:>6.1f}GB")

    # ==================== Gradient Checkpointing 效果 ====================
    print(f"\n[3] Gradient Checkpointing 效果 (LLaMA-7B, batch=4, seq=2048)\n")

    for gc in [False, True]:
        mem = estimate_training_memory(
            n_params_B=7, batch_size=4, seq_len=2048,
            hidden_dim=4096, n_layers=32, n_heads=32,
            gradient_checkpointing=gc,
        )
        tag = "有" if gc else "无"
        print(f"  {tag} Checkpointing: 激活={mem['activation_gb']:.1f}GB, "
              f"总计={mem['total_gb']:.1f}GB")

    ratio_no = estimate_training_memory(7, 4, 2048, 4096, 32, 32,
                                         gradient_checkpointing=False)['activation_gb']
    ratio_gc = estimate_training_memory(7, 4, 2048, 4096, 32, 32,
                                         gradient_checkpointing=True)['activation_gb']
    print(f"  激活值节省: {(1 - ratio_gc / ratio_no) * 100:.0f}%")

    # ==================== 推理显存 ====================
    print(f"\n[4] 推理显存估算\n")

    print(f"  {'模型':<12} {'精度':>6} {'模型':>8} {'KV Cache':>10} "
          f"{'总计':>8} {'适配GPU':>15}")
    print(f"  {'-' * 65}")

    for name, n_B, hidden, n_layers, n_heads in [
        ("LLaMA-7B",   7,  4096, 32, 32),
        ("LLaMA-13B", 13,  5120, 40, 40),
        ("LLaMA-70B", 70,  8192, 80, 64),
    ]:
        for dtype_name, d_bytes, kv_bytes in [("FP16", 2, 2), ("INT4", 0.5, 2)]:
            mem = estimate_inference_memory(
                n_params_B=n_B, batch_size=1, seq_len=4096,
                hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
                dtype_bytes=d_bytes, kv_cache_dtype_bytes=kv_bytes,
            )
            if mem['total_gb'] <= 11:
                gpu = "1× 2080Ti"
            elif mem['total_gb'] <= 24:
                gpu = "1× 4090/A5000"
            elif mem['total_gb'] <= 80:
                gpu = "1× A100-80G"
            else:
                n_gpu = int(mem['total_gb'] / 80) + 1
                gpu = f"{n_gpu}× A100-80G"
            print(f"  {name:<12} {dtype_name:>6} {mem['model_gb']:>6.1f}GB "
                  f"{mem['kv_cache_gb']:>8.1f}GB {mem['total_gb']:>6.1f}GB "
                  f"{gpu:>15}")

    # ==================== 你的机器能跑什么 ====================
    print(f"\n[5] 你的 2×2080Ti (各 11GB) 能做什么?\n")

    scenarios = [
        ("推理 7B INT4", 3.5, "单卡"),
        ("推理 7B FP16", 14.0, "双卡 TP"),
        ("推理 13B INT4", 6.5, "单卡"),
        ("推理 13B FP16", 26.0, "放不下"),
        ("训练 <500M FP16+Adam", 8.0, "单卡 DDP"),
        ("微调 7B LoRA INT4", 5.0, "单卡"),
    ]

    for desc, mem_gb, strategy in scenarios:
        fits = "✓" if mem_gb <= 22 else "✗"  # 双卡共 22GB
        print(f"  {fits} {desc:<25} ~{mem_gb:.1f}GB → {strategy}")

    print(f"\n{'=' * 70}")
    print(" 完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
