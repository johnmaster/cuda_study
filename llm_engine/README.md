# LLM Inference Engine

支持 GPT-2 系列模型的 LLM 推理引擎实现，包含：
- KV Cache 管理（Block-based，参考 PagedAttention 设计）
- Continuous Batching 调度器（iteration-level scheduling）
- CUDA Fused Decode Attention Kernel

## 架构

```
llm_engine/
├── engine/
│   ├── model.py        # GPT-2 模型（手写前向，从 HuggingFace 加载权重）
│   ├── kv_cache.py     # KV Cache Block 管理器
│   ├── scheduler.py    # Continuous Batching 调度器
│   └── engine.py       # LLM Engine 主类
├── csrc/
│   ├── decode_attention.cu  # Fused Decode Attention CUDA Kernel
│   └── binding.cpp          # PyTorch C++ 绑定
├── serve.py            # CLI 推理服务入口
└── setup.py            # CUDA 扩展编译
```

## 核心设计

```
请求生命周期:

1. 用户提交 Request(prompt, max_new_tokens)
      ↓
2. Scheduler 将其加入 waiting 队列
      ↓
3. Prefill: 处理完整 prompt，生成初始 KV Cache
      ↓
4. Decode 循环 (Continuous Batching):
   每一步:
   a. Scheduler 检查是否有新请求可加入
   b. 所有 running 请求各生成 1 token
   c. 完成的请求离队，释放 KV Cache block
   d. 新请求立即填充空位
      ↓
5. 返回生成结果
```

### KV Cache 设计

```
每层 KV Cache:
  K: [max_blocks, block_size, n_heads, head_dim]
  V: [max_blocks, block_size, n_heads, head_dim]

BlockAllocator 维护 free_block_ids 列表
每个 Sequence 有自己的 block_table: List[block_id]
新 token 写入当前 block，block 满了分配新 block
```

### Continuous Batching 关键点

```
传统 Static Batching:
  [req_A (200 tokens)] [req_B (50 tokens)]
  B 结束后还得等 A 才能处理 C → GPU 空转

Continuous Batching:
  每个 iteration 检查:
  - 谁完成了? → 移出，释放 KV block
  - 有等待的? → 立即加入，填满 batch
  → GPU 始终满载
```

### Fused Decode Attention

Decode 阶段每步只有 1 个新 token，是 GEMV 操作（memory-bound）：

```
普通实现:
  Q [1, heads, 1, head_dim] × K^T [1, heads, seq_len, head_dim]
  → scores [1, heads, 1, seq_len]  写 global memory
  → softmax → × V                  读 global memory
  两次访问 global memory

Fused Kernel:
  一个 kernel 完成: QK^T → online softmax → ×V
  scores 只存在 register/shared memory，不写 global memory
  → 节省 1 次 global memory 读写
```

## 安装

```bash
pip install -r requirements.txt

# 编译 CUDA 扩展
python setup.py build_ext --inplace
```

## 运行

```bash
# 单次推理
python serve.py --prompt "The future of AI is" --max-tokens 100

# 批量测试（演示 Continuous Batching）
python serve.py --benchmark

# 对比测试（手写引擎 vs HuggingFace pipeline）
python serve.py --compare
```

## 性能说明

当前实现在 RTX 2080 Ti 上：
- GPT-2 small (117M): ~500 tokens/s (decode)
- GPT-2 medium (345M): ~200 tokens/s (decode)
- Fused kernel 相比 PyTorch 原生约提升 1.2-1.5x（decode 阶段）
