"""PagedAttention 简化实现

模拟 vLLM 的分页 KV Cache 管理，对比传统连续分配 vs 分页分配的显存利用率。
运行方式: python paged_attention.py
"""

import math
import torch
import torch.nn.functional as F


class BlockManager:
    """KV Cache Block 管理器 (模拟 vLLM 的核心组件)"""

    def __init__(self, num_blocks, block_size, n_heads, head_dim, n_layers, device="cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.device = device

        # 物理 KV Block 池
        # shape: [n_layers, 2(K+V), num_blocks, block_size, n_heads, head_dim]
        self.kv_pool = torch.zeros(
            n_layers, 2, num_blocks, block_size, n_heads, head_dim,
            device=device, dtype=torch.float16
        )

        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # seq_id -> list of block indices

    def allocate_block(self):
        if not self.free_blocks:
            raise RuntimeError("No free blocks!")
        return self.free_blocks.pop(0)

    def free_sequence(self, seq_id):
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]

    def append_token_kv(self, seq_id, layer_idx, k, v):
        """为序列追加一个 token 的 KV

        Args:
            k, v: [1, n_heads, 1, head_dim]
        """
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []

        blocks = self.block_tables[seq_id]
        total_tokens = sum(1 for _ in range(len(blocks))) * self.block_size

        # 计算当前 token 应该写入哪个 block 的哪个 slot
        seq_len = self._get_seq_len(seq_id, layer_idx)
        block_idx_in_seq = seq_len // self.block_size
        slot_in_block = seq_len % self.block_size

        if slot_in_block == 0 and block_idx_in_seq >= len(blocks):
            new_block = self.allocate_block()
            blocks.append(new_block)

        physical_block = blocks[block_idx_in_seq]
        self.kv_pool[layer_idx, 0, physical_block, slot_in_block] = k.squeeze()
        self.kv_pool[layer_idx, 1, physical_block, slot_in_block] = v.squeeze()

    def _get_seq_len(self, seq_id, layer_idx):
        """通过检查非零 entries 推断已有 token 数"""
        blocks = self.block_tables.get(seq_id, [])
        count = 0
        for blk in blocks:
            for s in range(self.block_size):
                if self.kv_pool[layer_idx, 0, blk, s].abs().sum() > 0:
                    count += 1
                else:
                    return count
        return count

    def get_kv(self, seq_id, layer_idx):
        """收集序列的完整 KV Cache

        Returns: k, v both [1, n_heads, seq_len, head_dim]
        """
        blocks = self.block_tables.get(seq_id, [])
        if not blocks:
            return None, None

        k_parts, v_parts = [], []
        for blk in blocks:
            k_parts.append(self.kv_pool[layer_idx, 0, blk])  # [block_size, n_heads, head_dim]
            v_parts.append(self.kv_pool[layer_idx, 1, blk])

        k = torch.cat(k_parts, dim=0)  # [total_slots, n_heads, head_dim]
        v = torch.cat(v_parts, dim=0)

        # 只取实际有效的 token
        seq_len = self._get_seq_len(seq_id, layer_idx)
        k = k[:seq_len].unsqueeze(0).transpose(1, 2)  # [1, n_heads, seq_len, head_dim]
        v = v[:seq_len].unsqueeze(0).transpose(1, 2)
        return k, v

    @property
    def n_free(self):
        return len(self.free_blocks)

    @property
    def utilization(self):
        used = self.num_blocks - len(self.free_blocks)
        return used / self.num_blocks


def paged_attention(q, block_manager, seq_id, layer_idx):
    """使用 PagedAttention 计算 attention

    Args:
        q: [1, n_heads, 1, head_dim]  (当前 token 的 query)
    """
    k, v = block_manager.get_kv(seq_id, layer_idx)
    if k is None:
        return q  # 没有 cache

    scale = math.sqrt(q.size(-1))
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / scale
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.half()


# ==================== 显存利用率对比 ====================

def compare_memory_utilization():
    """对比传统预分配 vs PagedAttention 的显存利用率"""

    print("\n[1] 显存利用率对比\n")

    max_seq_len = 2048
    block_size = 16
    n_heads = 32
    head_dim = 128

    # 模拟不同长度的请求
    requests = [
        ("短对话", 50),
        ("中等问答", 300),
        ("长文档", 1500),
        ("接近上限", 1900),
    ]

    bytes_per_token = 2 * n_heads * head_dim * 2  # K+V, FP16

    print(f"  {'请求':<12} {'实际长度':>8} {'传统预分配':>12} {'Paged':>12} {'节省':>8}")
    print(f"  {'-' * 55}")

    for name, actual_len in requests:
        # 传统：预分配 max_seq_len
        traditional = max_seq_len * bytes_per_token
        # Paged：只分配需要的 blocks
        n_blocks = (actual_len + block_size - 1) // block_size
        paged = n_blocks * block_size * bytes_per_token
        saving = 1 - paged / traditional

        print(f"  {name:<12} {actual_len:>8} "
              f"{traditional / 1024:>10.0f} KB "
              f"{paged / 1024:>10.0f} KB "
              f"{saving:>7.0%}")

    # 多请求并发
    print(f"\n  --- 并发请求场景 ---")
    concurrent_requests = [50, 300, 150, 80, 500, 200, 100, 400]
    n_req = len(concurrent_requests)

    traditional_total = n_req * max_seq_len * bytes_per_token
    paged_total = sum(
        ((l + block_size - 1) // block_size) * block_size * bytes_per_token
        for l in concurrent_requests
    )

    print(f"  {n_req} 个并发请求，长度: {concurrent_requests}")
    print(f"  传统预分配: {traditional_total / 1024**2:.1f} MB")
    print(f"  PagedAttn:  {paged_total / 1024**2:.1f} MB")
    print(f"  节省: {1 - paged_total / traditional_total:.0%}")
    print(f"  → 可多服务 {traditional_total / paged_total:.1f}x 的请求！")


def demo_paged_attention():
    """在 GPU 上实际运行 PagedAttention"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[2] PagedAttention 实际运行 (device={device})\n")

    n_heads = 8
    head_dim = 64
    n_layers = 2
    block_size = 4
    num_blocks = 32

    manager = BlockManager(num_blocks, block_size, n_heads, head_dim, n_layers, device)

    print(f"  配置: {num_blocks} blocks × {block_size} tokens/block, "
          f"{n_heads} heads × {head_dim}d, {n_layers} layers")
    print(f"  总容量: {num_blocks * block_size} tokens\n")

    # 模拟两个并发请求
    for seq_id, seq_name, seq_len in [(0, "请求A", 10), (1, "请求B", 6)]:
        print(f"  {seq_name}: 生成 {seq_len} tokens")
        for t in range(seq_len):
            for layer in range(n_layers):
                k = torch.randn(1, n_heads, 1, head_dim, device=device, dtype=torch.float16)
                v = torch.randn(1, n_heads, 1, head_dim, device=device, dtype=torch.float16)
                manager.append_token_kv(seq_id, layer, k.squeeze(2), v.squeeze(2))

                q = torch.randn(1, n_heads, 1, head_dim, device=device, dtype=torch.float16)
                out = paged_attention(q, manager, seq_id, layer)

        blocks_used = len(manager.block_tables[seq_id])
        print(f"    → 使用 {blocks_used} blocks ({blocks_used * block_size} slots, "
              f"实际 {seq_len} tokens, 浪费 {blocks_used * block_size - seq_len})")

    print(f"\n  Block 使用情况:")
    print(f"    总 blocks: {num_blocks}")
    print(f"    已用: {num_blocks - manager.n_free}")
    print(f"    空闲: {manager.n_free}")
    print(f"    利用率: {manager.utilization:.0%}")

    # 释放请求 A
    manager.free_sequence(0)
    print(f"\n  释放请求A后:")
    print(f"    空闲 blocks: {manager.n_free}")
    print(f"    → 空闲 block 可立即分配给新请求（无碎片！）")


def main():
    print("=" * 60)
    print(" PagedAttention Demo")
    print("=" * 60)

    compare_memory_utilization()
    demo_paged_attention()

    print(f"\n{'=' * 60}")
    print(" 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
