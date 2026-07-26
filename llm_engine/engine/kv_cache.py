"""
KV Cache Block 管理器 —— 受 PagedAttention 启发的设计。

核心思路：
  - 显存划分为等大的 Block（如 block_size=16 token 的 KV 对）
  - 每个 Sequence 有一个 block_table，记录它使用的物理 block 编号
  - 分配/释放以 block 为粒度，消除内碎片，支持灵活调度

与完整 vLLM PagedAttention 的区别（简化处理）：
  - 本实现 KV 仍以连续 tensor 存储（非真正的非连续物理块）
  - 包含 BlockAllocator 和 block_table 的调度逻辑
  - 真正非连续 block 需要自定义 CUDA kernel 才能高效访问
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class BlockAllocator:
    """
    管理所有物理 KV Cache block 的分配与释放。

    类比虚拟内存的页表管理器：
      - num_blocks: 总物理 block 数（相当于物理页框数）
      - free_blocks: 空闲 block 编号的栈
    """
    num_blocks: int
    free_blocks: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.free_blocks = list(range(self.num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def allocate(self) -> int:
        """分配一个空闲 block，返回 block_id。显存不足时抛出异常。"""
        if not self.free_blocks:
            raise RuntimeError(
                f"KV Cache OOM: no free blocks available "
                f"({self.num_blocks} blocks total)"
            )
        return self.free_blocks.pop()

    def free(self, block_id: int) -> None:
        """释放一个 block，归还给空闲池。"""
        self.free_blocks.append(block_id)

    def free_all(self, block_ids: List[int]) -> None:
        """批量释放（sequence 结束时调用）"""
        self.free_blocks.extend(block_ids)


@dataclass
class SequenceKVCache:
    """
    单个 Sequence 的 KV Cache 状态。

    block_table: 逻辑 block 编号 → 物理 block_id 的映射
    num_tokens: 当前已缓存的 token 数量
    """
    seq_id: int
    block_size: int
    block_table: List[int] = field(default_factory=list)  # 物理 block_id 列表
    num_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)

    @property
    def last_block_offset(self) -> int:
        """当前最后一个 block 已使用的位置数（0 表示 block 已满或未分配）"""
        return self.num_tokens % self.block_size

    def needs_new_block(self) -> bool:
        """检查是否需要分配新 block（当前 block 已满）"""
        return self.num_tokens % self.block_size == 0

    def get_token_position(self, token_idx: int) -> Tuple[int, int]:
        """
        将 token 的逻辑位置转换为 (block_id, block_offset)。
        用于从 KV Cache 中读写特定 token 的 KV 对。
        """
        logical_block = token_idx // self.block_size
        block_offset = token_idx % self.block_size
        physical_block = self.block_table[logical_block]
        return physical_block, block_offset


class KVCacheManager:
    """
    所有 Sequence 的 KV Cache 管理器。

    分层存储：
      Layer 0: K[num_blocks, block_size, n_heads, head_dim]
               V[num_blocks, block_size, n_heads, head_dim]
      Layer 1: 同上
      ...
      Layer n_layers-1: 同上

    接口：
      - allocate_sequence(seq_id): 为新 sequence 注册缓存空间
      - free_sequence(seq_id): 释放 sequence 的所有 blocks
      - extend(seq_id): 为 sequence 追加新 token 时扩展 block
      - write_kv(layer, seq_id, token_pos, k, v): 写入单个 token 的 KV
      - get_kv_tensor(layer, seq_id): 获取 sequence 的完整 KV 张量（用于 attention）
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device

        # 分配全部 KV Cache 显存（一次性分配，避免碎片化）
        # 每层: K/V 各 [num_blocks, block_size, n_heads, head_dim]
        block_shape = (num_blocks, block_size, n_heads, head_dim)
        self.k_cache = [torch.zeros(block_shape, dtype=dtype, device=device)
                        for _ in range(n_layers)]
        self.v_cache = [torch.zeros(block_shape, dtype=dtype, device=device)
                        for _ in range(n_layers)]

        self.allocator = BlockAllocator(num_blocks)
        self.sequences: Dict[int, SequenceKVCache] = {}

        mem_bytes = 2 * n_layers * block_shape[0] * block_shape[1] * block_shape[2] * block_shape[3]
        mem_bytes *= 2 if dtype == torch.float16 else 4
        print(f"[KVCacheManager] Allocated {mem_bytes / 1024**3:.2f} GB for KV Cache "
              f"({num_blocks} blocks × {block_size} tokens/block × {n_layers} layers)")

    def allocate_sequence(self, seq_id: int) -> None:
        """注册一个新 sequence，分配初始 block。"""
        assert seq_id not in self.sequences, f"Sequence {seq_id} already exists"
        seq_cache = SequenceKVCache(seq_id=seq_id, block_size=self.block_size)
        first_block = self.allocator.allocate()
        seq_cache.block_table.append(first_block)
        self.sequences[seq_id] = seq_cache

    def free_sequence(self, seq_id: int) -> None:
        """释放 sequence 使用的所有 blocks。"""
        if seq_id not in self.sequences:
            return
        seq_cache = self.sequences.pop(seq_id)
        self.allocator.free_all(seq_cache.block_table)

    def can_append(self, seq_id: int) -> bool:
        """检查是否有足够空间追加新 token（可能需要新 block）。"""
        seq_cache = self.sequences[seq_id]
        if seq_cache.needs_new_block():
            return self.allocator.num_free_blocks > 0
        return True

    def append_token(self, seq_id: int) -> None:
        """通知管理器该 sequence 又生成了一个新 token，必要时分配新 block。"""
        seq_cache = self.sequences[seq_id]
        if seq_cache.needs_new_block():
            new_block = self.allocator.allocate()
            seq_cache.block_table.append(new_block)
        seq_cache.num_tokens += 1

    def write_kv(
        self,
        layer_idx: int,
        seq_id: int,
        token_pos: int,
        k: torch.Tensor,    # [n_heads, head_dim]
        v: torch.Tensor,    # [n_heads, head_dim]
    ) -> None:
        """将单个 token 位置的 K/V 写入物理 block。"""
        seq_cache = self.sequences[seq_id]
        physical_block, block_offset = seq_cache.get_token_position(token_pos)
        self.k_cache[layer_idx][physical_block, block_offset] = k
        self.v_cache[layer_idx][physical_block, block_offset] = v

    def get_kv_tensor(
        self,
        layer_idx: int,
        seq_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 sequence 完整的 K/V 张量，用于 attention 计算。
        返回: K [1, n_heads, seq_len, head_dim], V [1, n_heads, seq_len, head_dim]

        注意: 这里做了物理 block → 连续张量的重组，真实 PagedAttention
        用自定义 kernel 直接从非连续 block 计算 attention，避免此拷贝。
        """
        seq_cache = self.sequences[seq_id]
        num_tokens = seq_cache.num_tokens
        if num_tokens == 0:
            return None, None

        # 从各个物理 block 收集 KV 数据
        k_parts = []
        v_parts = []
        tokens_remaining = num_tokens
        for logical_block_idx, physical_block_id in enumerate(seq_cache.block_table):
            tokens_in_block = min(self.block_size, tokens_remaining)
            k_parts.append(self.k_cache[layer_idx][physical_block_id, :tokens_in_block])
            v_parts.append(self.v_cache[layer_idx][physical_block_id, :tokens_in_block])
            tokens_remaining -= tokens_in_block
            if tokens_remaining <= 0:
                break

        # [seq_len, n_heads, head_dim] → [1, n_heads, seq_len, head_dim]
        k_full = torch.cat(k_parts, dim=0).permute(1, 0, 2).unsqueeze(0)
        v_full = torch.cat(v_parts, dim=0).permute(1, 0, 2).unsqueeze(0)
        return k_full, v_full

    def get_memory_stats(self) -> Dict[str, float]:
        """返回 KV Cache 内存使用统计。"""
        used = self.allocator.num_used_blocks
        total = self.allocator.num_blocks
        return {
            "used_blocks": used,
            "total_blocks": total,
            "utilization": used / total * 100,
            "tokens_cached": used * self.block_size,
        }
