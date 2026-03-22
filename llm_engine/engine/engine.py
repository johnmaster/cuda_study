"""
LLM Engine 主类 —— 串联 Model、KV Cache、Scheduler 三个核心组件。

核心流程：
  generate_batch() 方法：
    1. 将所有请求加入 Scheduler
    2. 调度循环：
       a. scheduler.schedule() 返回本步 running requests
       b. 对 PREFILLING 的 requests：单独做 prefill，写入 KV Cache
       c. 对 RUNNING 的 requests：并行 decode 一步
       d. 处理新生成的 token（采样、EOS 检测、写回 KV Cache）
       e. 重复直到所有 requests 完成

Prefill 和 Decode 的差异：
  Prefill: 输入整个 prompt，seq_len 大，计算密集型（compute-bound）
  Decode:  每步只输入 1 个 token，seq_len=1，内存密集型（memory-bound）
           这就是为什么 Decode 阶段用 fused attention kernel 能显著提升性能
"""

import time
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn.functional as F

from .model import GPT2Model, GPT2Config
from .kv_cache import KVCacheManager
from .scheduler import Scheduler, Request, RequestStatus


class LLMEngine:
    """
    LLM 推理引擎。

    Example:
        engine = LLMEngine.from_pretrained("gpt2")
        results = engine.generate(
            prompts=["The future of AI is", "Once upon a time"],
            max_new_tokens=50,
        )
        for r in results:
            print(r["text"])
    """

    def __init__(
        self,
        model: GPT2Model,
        tokenizer,
        max_batch_size: int = 8,
        kv_block_size: int = 16,
        kv_num_blocks: int = 128,
        device: str = "cuda",
        use_fused_kernel: bool = False,   # 是否使用自定义 CUDA fused attention
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.device = device
        self.use_fused_kernel = use_fused_kernel

        # KV Cache 管理器
        self.kv_manager = KVCacheManager(
            n_layers=self.config.n_layer,
            n_heads=self.config.n_head,
            head_dim=self.config.head_dim,
            block_size=kv_block_size,
            num_blocks=kv_num_blocks,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
        )

        # 调度器
        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_tokens_per_step=2048,
            max_seq_len=self.config.n_positions,
        )

        # 如果开启 fused kernel，尝试导入
        if use_fused_kernel:
            try:
                import decode_attention_cuda
                self._fused_decode_attn = decode_attention_cuda.decode_attention
                print("[LLMEngine] Using custom fused decode attention kernel")
            except ImportError:
                print("[LLMEngine] Warning: fused kernel not found, falling back to PyTorch")
                self.use_fused_kernel = False

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        device: str = "cuda",
        **kwargs,
    ) -> "LLMEngine":
        """方便的工厂方法：从 HuggingFace 加载模型并创建引擎。"""
        from transformers import GPT2Tokenizer
        print(f"[LLMEngine] Loading tokenizer: {model_name}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = GPT2Model.from_pretrained(model_name, device=device)
        return cls(model, tokenizer, device=device, **kwargs)

    @torch.no_grad()
    def _prefill(self, req: Request) -> None:
        """
        Prefill 阶段：处理完整 prompt，初始化 KV Cache。

        输入: prompt token ids [1, prompt_len]
        输出: 第一个新 token，以及所有层的 KV Cache

        计算密集型（每次 forward 处理大量 token），GPU 高效利用。
        """
        input_ids = torch.tensor(
            [req.prompt_token_ids], dtype=torch.long, device=self.device
        )

        # 分配 KV Cache 空间
        self.kv_manager.allocate_sequence(req.request_id)

        # 前向传播，不使用外部 KV Cache（模型内部生成完整 KV）
        logits, past_kv = self.model(input_ids, past_key_values=None, use_cache=True)

        # 将 KV Cache 写入 Block Manager
        # past_kv[layer]: K/V [1, n_head, seq_len, head_dim]
        for pos in range(req.prompt_len):
            # 每个 token 写一次（模拟 block 分配逻辑）
            self.kv_manager.append_token(req.request_id)
            for layer_idx in range(self.config.n_layer):
                k = past_kv[layer_idx][0][0, :, pos, :]  # [n_head, head_dim]
                v = past_kv[layer_idx][1][0, :, pos, :]
                self.kv_manager.write_kv(layer_idx, req.request_id, pos, k, v)

        # 采样第一个新 token
        first_token_logits = logits[0, -1, :]   # [vocab_size]
        next_token = self._sample(first_token_logits, req.temperature, req.top_k)
        req.add_token(next_token)

        # 写入第一个新 token 的 KV（需要再做一次单 token forward）
        new_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
        _, new_kv = self.model(new_input, past_key_values=past_kv, use_cache=True)
        token_pos = req.prompt_len
        self.kv_manager.append_token(req.request_id)
        for layer_idx in range(self.config.n_layer):
            k = new_kv[layer_idx][0][0, :, -1, :]
            v = new_kv[layer_idx][1][0, :, -1, :]
            self.kv_manager.write_kv(layer_idx, req.request_id, token_pos, k, v)

        self.scheduler.mark_prefilled(req)

    @torch.no_grad()
    def _decode_step(self, running_requests: List[Request]) -> None:
        """
        Decode 步骤：所有 running requests 各生成 1 个新 token。

        关键优化：batch 所有 requests 一起处理，避免每个 request 单独 forward。
        内存密集型（每次只处理 1 个 token，计算量少但内存访问多）。

        注意：这里为了清晰，逐个处理每个 request（可进一步优化为真正的 batch decode）。
        真实生产引擎会 pad/pack 所有 requests 成一个 batch tensor。
        """
        for req in running_requests:
            if req.status != RequestStatus.RUNNING:
                continue

            # 从 KV Cache 获取完整历史 KV
            past_kv = []
            for layer_idx in range(self.config.n_layer):
                k, v = self.kv_manager.get_kv_tensor(layer_idx, req.request_id)
                past_kv.append((k, v))

            # 当前要输入的是最新生成的 token
            last_token = req.output_token_ids[-1]
            input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)

            # 前向传播（只处理 1 个 token，速度极快）
            logits, new_kv = self.model(input_ids, past_key_values=past_kv, use_cache=True)

            # 采样下一个 token
            next_token_logits = logits[0, -1, :]
            next_token = self._sample(next_token_logits, req.temperature, req.top_k)
            req.add_token(next_token)

            # 将新 token 的 KV 写回 Block Manager
            token_pos = req.total_len - 1  # 新 token 在序列中的位置
            if self.kv_manager.can_append(req.request_id):
                self.kv_manager.append_token(req.request_id)
                for layer_idx in range(self.config.n_layer):
                    k = new_kv[layer_idx][0][0, :, -1, :]
                    v = new_kv[layer_idx][1][0, :, -1, :]
                    self.kv_manager.write_kv(layer_idx, req.request_id, token_pos, k, v)

            # 检查停止条件
            if req.should_stop():
                req.status = RequestStatus.FINISHED
                self.kv_manager.free_sequence(req.request_id)

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int) -> int:
        """Top-k 采样（temperature=1.0, top_k=1 等效于贪心）。"""
        logits = logits / max(temperature, 1e-8)
        if top_k > 1:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = topk_vals[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        主推理入口：接受多个 prompt，使用 Continuous Batching 高效生成。

        返回每个请求的 dict: {text, prompt, output_tokens, stats}
        """
        # 对所有 prompts 进行 tokenize 并提交给调度器
        requests = []
        for prompt in prompts:
            token_ids = self.tokenizer.encode(prompt)
            req = self.scheduler.add_request(
                prompt=prompt,
                prompt_token_ids=token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            requests.append(req)

        start_time = time.time()
        step = 0

        # 主调度循环
        while self.scheduler.has_requests:
            step += 1
            scheduled = self.scheduler.schedule()

            # Prefill 阶段：对新加入的 requests 做 prefill
            for req in scheduled:
                if req.status == RequestStatus.PREFILLING:
                    self._prefill(req)

            # Decode 阶段：所有 RUNNING requests 各生成 1 token
            running = [r for r in scheduled if r.status == RequestStatus.RUNNING]
            if running:
                self._decode_step(running)

            if show_progress and step % 20 == 0:
                stats = self.kv_manager.get_memory_stats()
                print(f"  Step {step:4d} | "
                      f"running={self.scheduler.num_running} "
                      f"waiting={self.scheduler.num_waiting} "
                      f"finished={len(self.scheduler.finished)} | "
                      f"KV Cache: {stats['utilization']:.1f}%")

        total_time = time.time() - start_time
        total_tokens = sum(r.output_len for r in requests)
        if show_progress:
            print(f"\n[LLMEngine] Done: {len(requests)} requests, "
                  f"{total_tokens} tokens in {total_time:.2f}s "
                  f"({total_tokens/total_time:.1f} tokens/s)")

        # 整理输出
        results = []
        for req in requests:
            output_ids = req.prompt_token_ids + req.output_token_ids
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append({
                "request_id": req.request_id,
                "prompt": req.prompt,
                "text": output_text,
                "output_token_ids": req.output_token_ids,
                "prompt_len": req.prompt_len,
                "output_len": req.output_len,
                "ttft_ms": req.ttft * 1000 if req.ttft else None,
                "generation_time_s": req.generation_time,
                "throughput_tps": req.throughput,
            })
        return results

    def benchmark(
        self,
        num_requests: int = 8,
        prompt_len: int = 50,
        output_len: int = 100,
        batch_size: int = 4,
    ) -> Dict:
        """
        基准测试：对比 Static Batching 和 Continuous Batching 的吞吐量差异。
        """
        from transformers import GPT2Tokenizer
        sample_prompt = "The quick brown fox jumps over the lazy dog " * 5
        sample_prompt = sample_prompt[:prompt_len * 4]  # 粗略控制长度

        print(f"\n{'='*60}")
        print(f"Benchmark: {num_requests} requests, "
              f"~{prompt_len} prompt tokens, ~{output_len} output tokens")
        print(f"{'='*60}")

        # --- Continuous Batching ---
        print(f"\n[Continuous Batching] max_batch_size={batch_size}")
        self.scheduler = Scheduler(max_batch_size=batch_size)

        prompts = [sample_prompt] * num_requests
        t0 = time.time()
        results = self.generate(prompts, max_new_tokens=output_len, show_progress=True)
        cb_time = time.time() - t0
        cb_tokens = sum(r["output_len"] for r in results)

        stats = self.scheduler.get_stats()
        print(f"  Throughput: {cb_tokens/cb_time:.1f} tokens/s")
        print(f"  Avg TTFT: {stats['avg_ttft_ms']:.1f} ms")

        return {
            "continuous_batching": {
                "time_s": cb_time,
                "total_tokens": cb_tokens,
                "throughput_tps": cb_tokens / cb_time,
                "avg_ttft_ms": stats["avg_ttft_ms"],
            }
        }
