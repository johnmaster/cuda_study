#!/usr/bin/env python3
"""
LLM 推理引擎 CLI 入口。

用法:
    # 基础推理
    python serve.py --prompt "The future of AI is"

    # 指定模型（gpt2 / gpt2-medium / gpt2-large）
    python serve.py --model gpt2-medium --prompt "Once upon a time"

    # 控制生成参数
    python serve.py --prompt "Hello world" --max-tokens 200 --temperature 0.8 --top-k 40

    # 批量推理（演示 Continuous Batching 吞吐）
    python serve.py --benchmark --model gpt2 --num-requests 8

    # 精度对比（手写引擎 vs HuggingFace 官方 pipeline）
    python serve.py --compare --prompt "The quick brown fox"

    # 验证 CUDA Fused Kernel（先编译：python setup.py build_ext --inplace）
    python serve.py --test-kernel
"""

import argparse
import time
import sys
import os

import torch


def run_inference(args):
    """单个/多个 prompt 的推理。"""
    from engine import LLMEngine

    print(f"\n{'='*60}")
    print(f"LLM Inference Engine — {args.model}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    engine = LLMEngine.from_pretrained(
        model_name=args.model,
        device=args.device,
        max_batch_size=args.batch_size,
        use_fused_kernel=args.use_fused_kernel,
    )

    prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]

    print(f"Prompts ({len(prompts)}):")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:80]}...")
    print()

    results = engine.generate(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        show_progress=True,
    )

    print(f"\n{'─'*60}")
    print("Results:")
    print(f"{'─'*60}")
    for r in results:
        print(f"\n[Request {r['request_id']}]")
        print(f"  Prompt: {r['prompt'][:60]}...")
        print(f"  Output ({r['output_len']} tokens): {r['text'][len(r['prompt']):len(r['prompt'])+200]}")
        if r['ttft_ms']:
            print(f"  TTFT: {r['ttft_ms']:.1f} ms | "
                  f"Speed: {r['throughput_tps']:.1f} tokens/s")


def run_benchmark(args):
    """Continuous Batching 吞吐基准测试。"""
    from engine import LLMEngine

    print(f"\n{'='*60}")
    print(f"Continuous Batching Benchmark — {args.model}")
    print(f"{'='*60}")

    engine = LLMEngine.from_pretrained(
        model_name=args.model,
        device=args.device,
        max_batch_size=args.batch_size,
    )

    results = engine.benchmark(
        num_requests=args.num_requests,
        prompt_len=50,
        output_len=100,
        batch_size=args.batch_size,
    )

    print(f"\n{'─'*60}")
    print("Summary:")
    cb = results["continuous_batching"]
    print(f"  Total time:    {cb['time_s']:.2f} s")
    print(f"  Total tokens:  {cb['total_tokens']}")
    print(f"  Throughput:    {cb['throughput_tps']:.1f} tokens/s")
    print(f"  Avg TTFT:      {cb['avg_ttft_ms']:.1f} ms")


def run_compare(args):
    """对比手写引擎和 HuggingFace pipeline 的输出质量。"""
    from transformers import pipeline as hf_pipeline
    from engine import LLMEngine

    prompt = args.prompt[0] if isinstance(args.prompt, list) else args.prompt

    print(f"\n{'='*60}")
    print(f"Output Comparison — {args.model}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    # HuggingFace 官方 pipeline
    print("[HuggingFace pipeline]")
    t0 = time.time()
    hf_gen = hf_pipeline("text-generation", model=args.model, device=0)
    hf_result = hf_gen(prompt, max_new_tokens=args.max_tokens, do_sample=False)
    hf_time = time.time() - t0
    hf_text = hf_result[0]["generated_text"]
    print(f"  Output: {hf_text[:300]}")
    print(f"  Time: {hf_time:.2f}s\n")

    # 手写引擎（temperature=0 ≈ 贪心）
    print("[Custom LLM Engine]")
    engine = LLMEngine.from_pretrained(args.model, args.device)
    t0 = time.time()
    results = engine.generate(
        [prompt], max_new_tokens=args.max_tokens,
        temperature=0.01, top_k=1, show_progress=False
    )
    my_time = time.time() - t0
    my_text = results[0]["text"]
    print(f"  Output: {my_text[:300]}")
    print(f"  Time: {my_time:.2f}s\n")

    # 简单的 token-level 对比
    hf_tokens = set(hf_text.split())
    my_tokens = set(my_text.split())
    overlap = len(hf_tokens & my_tokens) / max(len(hf_tokens | my_tokens), 1)
    print(f"Token overlap: {overlap*100:.1f}%  (贪心模式下应接近 100%)")


def test_cuda_kernel():
    """验证 CUDA fused attention kernel 的正确性和性能。"""
    print(f"\n{'='*60}")
    print("CUDA Fused Decode Attention Kernel Test")
    print(f"{'='*60}\n")

    # 尝试 JIT 编译
    print("Compiling CUDA extension (JIT)...")
    try:
        from torch.utils.cpp_extension import load
        decode_attn = load(
            name="decode_attention_cuda",
            sources=["csrc/decode_attention.cu"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_75,code=sm_75"],
            verbose=True,
        )
        print("Compilation successful!\n")
    except Exception as e:
        print(f"Compilation failed: {e}")
        print("Make sure CUDA toolkit is installed and nvcc is in PATH")
        return

    # 精度测试
    batch, n_heads, seq_len, head_dim = 2, 12, 256, 64
    q = torch.randn(batch, n_heads, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)

    # 自定义 kernel
    out_custom = decode_attn.decode_attention(q, k, v)

    # PyTorch 参考实现
    out_ref = decode_attn.decode_attention_ref(q.unsqueeze(2), k, v)

    max_diff = (out_custom - out_ref).abs().max().item()
    print(f"Correctness test (fp32):")
    print(f"  Max abs diff = {max_diff:.2e}  {'✓ PASS' if max_diff < 1e-4 else '✗ FAIL'}")

    # FP16 测试
    q16 = q.half()
    k16 = k.half()
    v16 = v.half()
    out_custom_16 = decode_attn.decode_attention(q16, k16, v16)
    max_diff_16 = (out_custom_16.float() - out_ref).abs().max().item()
    print(f"Correctness test (fp16):")
    print(f"  Max abs diff = {max_diff_16:.2e}  {'✓ PASS' if max_diff_16 < 5e-3 else '✗ FAIL'}")

    # 性能测试
    print(f"\nPerformance test (batch={batch}, n_heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}):")

    # 预热
    for _ in range(10):
        decode_attn.decode_attention(q, k, v)
    torch.cuda.synchronize()

    # 自定义 kernel 计时
    iters = 100
    t0 = time.time()
    for _ in range(iters):
        decode_attn.decode_attention(q, k, v)
    torch.cuda.synchronize()
    custom_ms = (time.time() - t0) / iters * 1000

    # PyTorch 原生 attention 计时
    scale = (head_dim ** -0.5)
    t0 = time.time()
    for _ in range(iters):
        scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)) * scale
        weights = torch.softmax(scores, -1)
        _ = torch.matmul(weights, v).squeeze(2)
    torch.cuda.synchronize()
    pt_ms = (time.time() - t0) / iters * 1000

    print(f"  Custom kernel:  {custom_ms:.3f} ms")
    print(f"  PyTorch native: {pt_ms:.3f} ms")
    print(f"  Speedup:        {pt_ms/custom_ms:.2f}x")

    # 测试更长序列
    print(f"\nScaling test (varying seq_len):")
    for slen in [64, 128, 256, 512, 1024]:
        k_ = torch.randn(batch, n_heads, slen, head_dim, device="cuda")
        v_ = torch.randn(batch, n_heads, slen, head_dim, device="cuda")
        for _ in range(5):
            decode_attn.decode_attention(q, k_, v_)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            decode_attn.decode_attention(q, k_, v_)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 50 * 1000
        print(f"  seq_len={slen:4d}: {ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Engine CLI")
    parser.add_argument("--model", type=str, default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large"],
                        help="GPT-2 model variant")
    parser.add_argument("--prompt", type=str, nargs="+",
                        default=["The future of artificial intelligence is"],
                        help="Input prompt(s)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Max concurrent requests")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run throughput benchmark")
    parser.add_argument("--num-requests", type=int, default=8,
                        help="Number of requests for benchmark")
    parser.add_argument("--compare", action="store_true",
                        help="Compare output with HuggingFace pipeline")
    parser.add_argument("--test-kernel", action="store_true",
                        help="Test the CUDA fused attention kernel")
    parser.add_argument("--use-fused-kernel", action="store_true",
                        help="Use compiled CUDA fused attention kernel")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.test_kernel:
        test_cuda_kernel()
    elif args.benchmark:
        run_benchmark(args)
    elif args.compare:
        run_compare(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
