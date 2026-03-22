"""
Continuous Batching 调度器。

核心理念（对比传统 Static Batching）：
  Static Batching:
    batch = [req_A(200 tokens), req_B(50 tokens)]
    B 在第 50 步就完成了，但必须等到 A 完成（200步）才能处理下一批
    → GPU 后 150 步有效 batch_size = 1，利用率低

  Continuous Batching (iteration-level scheduling):
    每一个 decode step 之后：
    1. 移除已完成的 requests
    2. 检查 waiting 队列，将能放入的 requests 加入 running
    3. 下一步所有 running requests 一起 decode
    → 始终保持高 batch size，GPU 利用率最大化

这正是 vLLM、TensorRT-LLM、SGLang 等生产级引擎的核心调度策略。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import torch


class RequestStatus(Enum):
    WAITING = "waiting"      # 等待被调度（还没开始）
    PREFILLING = "prefilling"  # 正在做 prefill（第一次处理整个 prompt）
    RUNNING = "running"      # 正在 decode（逐 token 生成）
    FINISHED = "finished"    # 生成完成（遇到 EOS 或达到 max_new_tokens）
    ABORTED = "aborted"      # 被中止（OOM 等原因）


@dataclass
class Request:
    """表示一个推理请求的完整生命周期信息。"""
    request_id: int
    prompt: str
    prompt_token_ids: List[int]
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int = 50
    eos_token_id: int = 50256   # GPT-2 的 <|endoftext|>

    # 运行时状态
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    prefill_start: Optional[float] = None
    first_token_time: Optional[float] = None   # Time to First Token (TTFT)
    finish_time: Optional[float] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    @property
    def ttft(self) -> Optional[float]:
        """Time to First Token，衡量响应延迟的关键指标。"""
        if self.first_token_time and self.prefill_start:
            return self.first_token_time - self.prefill_start
        return None

    @property
    def generation_time(self) -> Optional[float]:
        if self.finish_time and self.prefill_start:
            return self.finish_time - self.prefill_start
        return None

    @property
    def throughput(self) -> Optional[float]:
        """该请求的 token 生成速度 (tokens/s)"""
        gt = self.generation_time
        if gt and gt > 0:
            return self.output_len / gt
        return None

    def add_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if len(self.output_token_ids) == 1:
            self.first_token_time = time.time()

    def should_stop(self) -> bool:
        """检查是否满足停止条件。"""
        if self.output_len >= self.max_new_tokens:
            return True
        if self.output_token_ids and self.output_token_ids[-1] == self.eos_token_id:
            return True
        return False


class Scheduler:
    """
    Continuous Batching 调度器。

    设计参数：
      max_batch_size: 同时 running 的最大 sequence 数
      max_tokens_per_step: 单步最大总 token 数（控制显存和计算量上限）
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens_per_step: int = 2048,
        max_seq_len: int = 1024,
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step
        self.max_seq_len = max_seq_len

        self.waiting: List[Request] = []    # 待处理队列（FIFO）
        self.running: List[Request] = []    # 正在生成的 requests
        self.finished: List[Request] = []   # 已完成（用于统计）

        self._next_request_id = 0
        self._total_requests = 0
        self._step_count = 0

    def add_request(
        self,
        prompt: str,
        prompt_token_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Request:
        """提交新的推理请求。"""
        req = Request(
            request_id=self._next_request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        self._next_request_id += 1
        self._total_requests += 1
        self.waiting.append(req)
        return req

    def schedule(self) -> List[Request]:
        """
        Continuous Batching 调度的核心逻辑。

        每个 decode step 调用一次，返回本步需要处理的 requests。
        步骤：
          1. 从 running 中移除已完成的 requests
          2. 尝试从 waiting 队列填充空位（需通过 KV Cache 空间检查）
          3. 返回当前 running 列表
        """
        self._step_count += 1

        # 1. 清理已完成的 requests
        still_running = []
        for req in self.running:
            if req.is_finished:
                req.finish_time = time.time()
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

        # 2. 从 waiting 填充到 max_batch_size
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting[0]
            if req.prompt_len > self.max_seq_len:
                # prompt 太长，直接丢弃
                req.status = RequestStatus.ABORTED
                self.waiting.pop(0)
                self.finished.append(req)
                continue

            # 检查总 token 数是否超限（近似检查）
            current_tokens = sum(r.total_len for r in self.running)
            if current_tokens + req.prompt_len > self.max_tokens_per_step and self.running:
                break  # 本步先不加了，下步再试

            req.status = RequestStatus.PREFILLING
            req.prefill_start = time.time()
            self.waiting.pop(0)
            self.running.append(req)

        return list(self.running)

    def mark_prefilled(self, req: Request) -> None:
        """Prefill 完成后标记为 RUNNING 状态。"""
        req.status = RequestStatus.RUNNING

    def mark_finished(self, req: Request) -> None:
        """手动标记某个 request 完成。"""
        req.status = RequestStatus.FINISHED

    @property
    def has_requests(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def get_stats(self) -> Dict:
        """返回调度器统计信息。"""
        finished = [r for r in self.finished if not r.status == RequestStatus.ABORTED]
        if not finished:
            return {
                "total_requests": self._total_requests,
                "finished": 0,
                "avg_ttft_ms": 0,
                "avg_throughput_tps": 0,
                "step_count": self._step_count,
            }

        ttfts = [r.ttft * 1000 for r in finished if r.ttft]
        throughputs = [r.throughput for r in finished if r.throughput]
        return {
            "total_requests": self._total_requests,
            "finished": len(finished),
            "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
            "avg_throughput_tps": sum(throughputs) / len(throughputs) if throughputs else 0,
            "step_count": self._step_count,
        }
