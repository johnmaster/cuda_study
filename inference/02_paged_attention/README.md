# PagedAttention (vLLM 核心)

## 传统 KV Cache 的问题

传统 KV Cache 为每个请求**预分配连续的最大长度内存**：

```
请求 A (实际 100 tokens, 预分配 2048):
  [已用 100 tokens][=================== 浪费 1948 slots ===================]

请求 B (实际 500 tokens, 预分配 2048):
  [======== 已用 500 ========][=========== 浪费 1548 ============]
```

问题：
1. **内部碎片**：预分配远超实际使用量，显存浪费 60-80%
2. **外部碎片**：请求结束后留下不连续的空洞，无法被新请求使用
3. **无法动态增长**：如果生成超过预分配长度，需要复制整个 KV Cache

## PagedAttention 的解决方案

借鉴操作系统的**虚拟内存 + 分页**思想：

```
物理 KV Block 池 (显存中):
  [Block 0][Block 1][Block 2][Block 3][Block 4][Block 5]...

请求 A 的 Block Table:  [0, 3, 5]  → 物理块不需要连续！
请求 B 的 Block Table:  [1, 2, 4]

每个 Block 存固定数量的 token KV (如 16 tokens)
```

优势：
1. **无内部碎片**：最多浪费 1 个 block
2. **无外部碎片**：任何空闲 block 都可分配给新请求
3. **动态增长**：生成新 token 时只需分配新 block
4. **共享 KV**：相同 prefix 的请求可以共享 block（copy-on-write）

## 运行

```bash
python paged_attention.py
```
