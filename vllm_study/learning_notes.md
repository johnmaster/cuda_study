# vLLM 与 LLM 推理学习笔记

> **配套文件**：架构层面的类关系图见 [v1_class_diagram.md](v1_class_diagram.md)
> **本文件内容**：概念解释、机制讲解、性能算账方法
> **基于版本**：vLLM v0.8.5
> **范围**：单机单卡 + 纯文本推理

## 目录

- [一、V0 vs V1 引擎的关系](#一v0-vs-v1-引擎的关系)
- [二、`compute_encoder_budget` 对单机推理是不是没用？](#二compute_encoder_budget-对单机推理是不是没用)
- [三、`InputBatch` 是干什么的](#三inputbatch-是干什么的)
- [四、PagedAttention 核心概念](#四pagedattention-核心概念)
- [五、PagedAttention 进阶机制](#五pagedattention-进阶机制)
- [六、FlashAttention 简介](#六flashattention-简介)
- [七、Continuous Batching](#七continuous-batching)
- [八、LLM 推理性能算账方法](#八llm-推理性能算账方法)
- [九、维度澄清：seq_len vs hidden_dim](#九维度澄清seq_len-vs-hidden_dim)
- [十、Attention FLOPs 公式详解](#十attention-flops-公式详解)
- [十一、Throughput 公式的本质](#十一throughput-公式的本质)

---

## 一、V0 vs V1 引擎的关系

### 一句话总结

**V1 是 V0 的"重写版"**，不是平行替代。共存于同一仓库里，`VLLM_USE_V1=1` 默认开启走 V1 路径，`VLLM_USE_V1=0` 切回 V0。模型/kernel 这层共用，**重写的是上层架构**：scheduler、KV cache manager、worker、sampler、API server。

### 时间线

- **V0**：2023 年初~2024 年底
- **V1 Alpha 发布**：2025-01-27
- **2025 年中起**：V1 成为默认（[envs.py:76](../vllm/envs.py#L76) `VLLM_USE_V1: bool = True`）

### 仓库共存

```
vllm/
├── core/      ← V0 调度 / KV cache
├── engine/    ← V0 引擎 (LLMEngine, AsyncLLMEngine)
├── worker/    ← V0 worker
│
└── v1/        ← V1 全套
    ├── core/sched/
    ├── core/
    ├── engine/
    ├── executor/
    ├── worker/
    └── sample/
```

### 关键差异

| 维度 | V0 | V1 |
|---|---|---|
| **进程模型** | 单进程 | EngineCore 跑独立子进程，前端 ZMQ IPC |
| **Scheduler** | prefill / decode 分阶段调度 | 统一调度（`{req_id: num_tokens}`） |
| **Chunked Prefill** | 后加 feature flag | 默认始终开启 |
| **Prefix Caching** | 后加 flag | 默认开 |
| **KV Cache 管理** | `BlockManager` + `BlockSpaceManager` + `Evictor`（多层） | `KVCacheManager` + `BlockPool`（两层） |
| **GPU↔CPU swap** | 支持 | 已弃用 |
| **`best_of`** | 支持 | 已弃用 |
| **CPU 开销** | 较高 | "near-zero" |

### 为什么重写而不是改

V0 的 chunked_prefill、prefix_caching、spec_decode、multi_modal 各自加 flag、各自打 patch，代码路径越来越乱。V1 是从头按"统一调度 + 子进程隔离"重新设计。

底层共用部分（不分 V0/V1）：模型实现、attention kernels、量化、distributed primitives、tokenizer。

---

## 二、`compute_encoder_budget` 对单机推理是不是没用？

### 跟"单机/多机"无关，跟**模型是不是多模态**有关

`compute_encoder_budget` 算的是**多模态模型（vision/audio encoder）的 encoder cache 预算**。看 [encoder_cache_manager.py:87-88](../vllm/v1/core/encoder_cache_manager.py#L87-L88)：

```python
if not model_config.is_multimodal_model:
    return 0, 0
```

只要不是多模态模型，直接返回 `(0, 0)`，整个 encoder cache 路径空转，**没有任何实际开销**。

### 适用矩阵

| 场景 | 是否有用 |
|---|---|
| 单机 + 纯文本 LLM | ❌ 没用，`return 0, 0` |
| 单机 + 多模态（Llava / Qwen2-VL） | ✅ 必须有 |
| 多机 + 纯文本 | ❌ 没用 |
| 多机 + 多模态 | ✅ 有用 |

### 不能删的原因

`Scheduler` 是通用的，要兼容多模态 path。后续 `schedule()` 里那些 `encoder_cache_manager.has_cache(...)` 等调用，在纯文本路径上都会立刻短路返回（请求里没有 `mm_inputs`），零开销。

---

## 三、`InputBatch` 是干什么的

### 定位

**`InputBatch` 是 worker 侧的"批处理状态总账"**——把当前在 GPU 上一起跑的所有请求的状态，铺平成**预分配好的固定大小张量**，让 GPU forward 和 sampler 用纯 index lookup 就能读到所有需要的东西。

类比：
- `Scheduler` 那边的 `dict[str, Request]` = **AoS**（Array of Structs，对人友好）
- `InputBatch` 这边 = **SoA**（Struct of Arrays，对 GPU 友好）

每个请求被分配一个 `req_index ∈ [0, max_num_reqs)`，**所有跟它相关的状态都按这个 index 横着放在一组等长 tensor 的同一行**。

### 持有的状态（按目的分组）

参考 [vllm/v1/worker/gpu_input_batch.py:55-223](../vllm/v1/worker/gpu_input_batch.py#L55-L223)：

```python
# 1. 请求身份映射
self._req_ids: list[str | None]
self.req_id_to_index: dict[str, int]

# 2. Token 状态（输入侧）
self.token_ids_cpu_tensor: (max_num_reqs, max_model_len) int32
self.num_tokens, num_prompt_tokens, num_computed_tokens_cpu

# 3. Block table（attention 用）
self.block_table: BlockTable  # (max_num_reqs, max_num_blocks_per_req)

# 4. 采样参数（全部预分配 tensor）
self.temperature, top_p, top_k, min_p   # (max_num_reqs,) GPU + CPU 两份
self.frequency_penalties, presence_penalties, repetition_penalties
self.greedy_reqs, random_reqs, top_p_reqs ...   # 集合用来快速判断

# 5. LoRA / Logprobs / Generator
self.request_lora_mapping
self.num_logprobs, num_prompt_logprobs
self.generators: dict[int, torch.Generator]

# 6. 衍生品
self.sampling_metadata    # 上面所有采样字段打包成的对象
```

### 5 个关键操作

| 方法 | 干什么 | 触发时机 |
|---|---|---|
| `add_request(state, req_index)` | 写一行 token、block_ids、sampling params | scheduler 调度了一个新 req |
| `remove_request(req_id)` | 把槽位标空 | scheduler 标了 finish |
| `swap_states(i1, i2)` | 交换两行 | preemption / 重排时 |
| `condense(empty_indices)` | 把空槽位压实 | finished 之后避免 batch 中间空洞 |
| `refresh_sampling_metadata()` | 重建 `SamplingMetadata` | batch 成员变化后 |

### 为什么这么设计——3 个核心动机

#### ① 避免每步重新分配 GPU 张量

如果每次 `execute_model` 都用当前请求重新构造一个 `(N, max_len)` 张量，会有 malloc + cudaMalloc 开销。`InputBatch` 在 `__init__` 一次性把 `(max_num_reqs, ...)` 全开好，**之后只改值，不重分配**。

#### ② AoS → SoA 转换发生在这里

`SchedulerOutput` 给的还是 list/dict 形式（每个请求一个 `NewRequestData` / `CachedRequestData`）。GPU 上要的是 `temperature: Tensor[N]` 这种平铺张量。`InputBatch.add_request()` 就是这个**转置点**。

#### ③ 跨 step 持久化

请求一旦进入 InputBatch，**只要它还活着，就一直留在那个槽位**：
- 第 1 步：`add_request` 写入 prompt + sampling params
- 第 2~N 步：只把新生成的 token 追加到尾部，sampling params 一动不动
- 完成：`remove_request` 释放槽位

**只有第 1 步是"全量上传"，后面只 copy 增量数据**。

### 在 step 中的位置

```
Scheduler.schedule()                  ─→ SchedulerOutput
                                          (list/dict of req data)
                                                │
                                                ▼
                                       GPUModelRunner.execute_model
                                                │
                       ┌────────────────────────┤
                       │ 1. 用 SchedulerOutput 更新 InputBatch
                       │ 2. CPU tensor → 异步 H2D 拷贝
                       │ 3. self.model(...)             ← 用 InputBatch 字段
                       │ 4. self.sampler(logits, sampling_metadata)
                       │
                       └─→ ModelRunnerOutput
```

`SchedulerOutput` 是**消息**（一次性、跨进程），`InputBatch` 是**状态**（持久、本进程）。

### 为什么需要 `condense`——把槽位压实

这是 `InputBatch` 里最容易困惑的一个操作。要先理解一个事实：**`remove_request` 只是把槽位"标空"，并不会把后面的请求往前挪**。

#### 问题：删除会留下"空洞"

假设 batch 有 4 个请求，占据 0~3 号槽位：

```
槽位:   0      1      2      3
       [reqA] [reqB] [reqC] [reqD]
```

这一步 B 和 C 完成了，`remove_request` 把它俩标空：

```
槽位:   0      1      2      3
       [reqA] [空]   [空]   [reqD]
                ↑      ↑
              中间出现两个空洞
```

现在 active 请求是 A（槽位 0）和 D（槽位 3），但它们**不连续**了，中间夹着空洞。

#### 这个空洞为什么是大问题

GPU forward 时，**所有 tensor 都是按 `[0, num_reqs)` 这段连续区间切片送进 kernel 的**。如果中间有空洞，会出三个问题：

1. **kernel 会算到垃圾数据**
   `model(input_batch.token_ids[:num_reqs])` 这种切片会把空洞槽位（槽位 1、2 里的残留旧数据）也一起送进 GPU。这些位置的 token、block_table 都是上一个请求 B/C 的残骸——算出来是垃圾，浪费算力，甚至可能读到已经被释放的 block 导致错误。

2. **没法用紧凑的 batch size**
   如果不压实，要表示"现在有 2 个 active 请求"，你被迫送 4 行（`[0,4)`）进 GPU，其中 2 行是浪费的。压实后只送 2 行（`[0,2)`），GPU 干的是实打实的活。

3. **破坏 SoA 的连续性优势**
   回顾 [AoS vs SoA](#三inputbatch-是干什么的)——SoA 的全部价值在于"同一字段的所有 active 值在内存里连续"。空洞会在连续区间里插入无效值，coalesced 读取又被破坏了。

#### 解决：`condense` 把尾部请求搬到前面的空洞

`condense` 的策略很简单——**拿最后面的 active 请求去填最前面的空洞**：

```
压实前:  0      1      2      3
        [reqA] [空]   [空]   [reqD]

condense: 把槽位 3 的 reqD 搬到槽位 1 (最小的空洞)
                ↓

压实后:  0      1      2      3
        [reqA] [reqD] [空]   [空]
        └─ active ─┘  └── 尾部空闲 ──┘
        num_reqs = 2
```

现在 active 请求紧凑地占据 `[0, 2)`，尾部是连续的空闲区。GPU 只需处理前 2 行，干净利落。

#### 源码对应

看 [gpu_input_batch.py:451-531](../vllm/v1/worker/gpu_input_batch.py#L451-L531)，核心逻辑：

```python
def condense(self, empty_req_indices):  # empty_req_indices 降序排列
    last_req_index = num_reqs + len(empty_req_indices) - 1
    while empty_req_indices:
        # 找最大的非空槽位（尾部的 active 请求）
        while last_req_index in empty_req_indices:
            last_req_index -= 1
        # 找最小的空槽位
        empty_index = empty_req_indices.pop()
        if empty_index >= last_req_index:
            break    # 空洞已经都在尾部了，不用再搬

        # 把 last_req_index 的所有字段搬到 empty_index
        self._req_ids[empty_index] = self._req_ids[last_req_index]
        self.req_id_to_index[req_id] = empty_index        # ← 更新映射！
        self.token_ids_cpu[empty_index, :n] = self.token_ids_cpu[last_req_index, :n]
        self.block_table.move_row(last_req_index, empty_index)
        self.temperature_cpu[empty_index] = self.temperature_cpu[last_req_index]
        ...  # 把这个请求的每一个字段都搬过去
        last_req_index -= 1

    del self._req_ids[self.num_reqs:]   # 裁掉尾部
```

注意第 480 行 `self.req_id_to_index[req_id] = empty_index`——**搬家后必须更新 `req_id → 槽位` 的反向映射**，否则后面用 request_id 查这个请求会查到旧的空槽位。

#### 为什么是"搬尾部填头部"而不是整体左移

你可能想：为什么不像数组删除一样，把后面所有元素整体往前挪一格？

因为**整体左移要搬动的数据远多于"尾部填空洞"**：
- 整体左移：空洞后面**每一个**请求都要搬一次
- 尾部填空洞：**只有**被选中去填洞的那几个请求要搬

`condense` 是后者——有几个空洞就最多搬几个请求，每个请求只搬一次。搬动 `token_ids`（可能是 `(max_model_len,)` 的一整行）和 `block_table` 行是有成本的，搬得越少越好。

#### 跟 `swap_states` 的关系

`swap_states(i1, i2)` 是"交换两个槽位的全部状态"，是 `condense` 用到的底层动作的近亲。不过 `condense` 里是**单向搬运**（把 last 搬到 empty，last 标空），不是双向交换——因为目标位置本来就是空的，不需要把空数据换回去。

#### 一句话总结

> **`remove_request` 制造空洞，`condense` 消除空洞**。压实是为了让 active 请求始终连续占据 `[0, num_reqs)`，这样 GPU 能用最小的 batch size 跑、且不碰到无效数据——本质上是在维护 SoA 布局"连续即高效"的前提。

---

## 四、PagedAttention 核心概念

### 解决的问题

PagedAttention 之前的系统（如 ORCA、FasterTransformer）给每个请求**预分配 `max_model_len` 长度的连续显存**。问题在于：

| 问题 | 说明 | 严重程度 |
|---|---|---|
| **内部碎片** | 实际生成长度 << 预分配长度 | 主要 |
| **外部碎片** | 请求来去后凑不出连续大块 | 次要 |
| **保留浪费** | 必须按最坏情况分配 | 主要 |

旧系统**有效利用率只有 20%~40%**，PagedAttention 能做到 **96%+**。这个 2~4 倍提升直接转化成 batch size 翻倍、吞吐翻倍。

> ⚠️ 注意：PagedAttention 解决的是**显存利用率**问题，**不是显存带宽**问题。它本身不降低内存读取量。

### OS 类比（正确版本）

| OS 概念 | PagedAttention |
|---|---|
| 进程 | `Request` |
| 进程的虚拟地址空间 | 该请求的 KV token 序列 |
| 虚拟页 | 一段连续的 logical KV slots（隐式概念） |
| 物理页 / page frame | `KVCacheBlock`（GPU 显存里一段 slot） |
| 页表 | `req_to_blocks[req_id]`（即 BlockTable） |
| OS 内存管理器 | `KVCacheManager` + `BlockPool` |
| 共享页 | 多请求共享的 prefix block（`ref_cnt > 1`） |

**关键纠正**：
- `KVCacheBlock` 是**物理页帧**的元数据，不是虚拟页
- `KVCacheManager` 是**OS 内核 / 内存管理子系统**，不是进程
- "进程"对应的是 `Request`

### block_size 的选择

vLLM 默认 `block_size = 16`。极端情况：

#### `block_size = 1` 的坏处

1. **GPU memory coalescing 失效** ⭐⭐⭐
   GPU 读显存时一次读 cache line（128B）。block_size=16, head_dim=128, fp16 → block 是 4KB，一段连续显存高效读。block_size=1 → 每个 token 256B 但**完全不同位置**，HBM 带宽利用率从 ~80% 跌到 ~10%。
2. **Block table 自身变巨大**：32K token 序列：block_size=16 → 2K 项；block_size=1 → 32K 项
3. **Prefix cache 哈希计算开销线性放大**

#### `block_size = 256` 的坏处

1. **Prefix cache 命中粒度变粗** ⭐⭐⭐
   prefix cache 只能在**完整 block 边界**命中。两个请求共享 200 个 token 前缀：
   - block_size=16 → 12 个完整 block 命中（前 192 token），节省 12 倍
   - block_size=256 → **0 个完整 block 命中**（200 < 256），完全没省
2. **内部碎片**：每请求最后一个 block 平均浪费 `block_size/2` token

block_size=16 是 GPU 访存效率（要够大）和 prefix cache 命中粒度（要够小）的折中。

### Block 状态表

| `ref_cnt` | `_block_hash` | 在 free_queue？ | 在 cached_dict？ | 含义 |
|---|---|---|---|---|
| > 0 | 有 | ❌ | ✅ | 在用，是个完整缓存 block |
| > 0 | 无 | ❌ | ❌ | 在用，正在写入还没满 |
| 0 | 有 | ✅ | ✅ | 没人在用，可被复用或被驱逐 |
| 0 | 无 | ✅ | ❌ | 干净空闲块 |

### 双数据结构的必要性

`BlockPool` 同时维护两个结构：

| 操作 | 用哪个？ | 为什么必须是这个？ |
|---|---|---|
| "给我一个空闲 block" | `free_block_queue.popleft()` | 链表头是最久未用的，实现 LRU |
| "找 hash=X 的 block" | `cached_block_hash_to_block[X]` | 哈希字典 O(1) 查询 |
| "驱逐一个 block" | 从 queue 头取 + 从 dict 删 | 两边都要改 |
| "复用排队中的 block" | dict 命中后从 queue 中间 `remove(block)` | 双向链表支持 O(1) 中间删除 |

**只用一个不行**：单独 queue 失去 O(1) 查找；单独 dict 失去 LRU 顺序。这是**经典的 LRU cache 实现**——和 Python `OrderedDict` / `functools.lru_cache` 同样的套路。

### 链式哈希（Chained Hashing）——prefix cache 命中的底层原理

这是 prefix caching 最核心的机制。理解它，就能明白"为什么命中要求**整段前缀**一致，而不只是单个 block 内容一致"。

#### 问题：怎么判断两个请求"共享前缀"

prefix cache 的目标是：如果请求 B 的开头一段 token 和某个已缓存的内容**完全一样**，就复用那段 KV。

但是"一样"必须是**从头到这里整段都一样**，不能只看单个 block。看这个反例（block_size=4）：

```
请求 A: [The, cat, sat, on][the, mat, ...]
请求 B: [A, dog, ran, on][the, mat, ...]
                            ↑
              第二个 block 内容都是 [the, mat, ...]，单看是一样的
```

如果只按"单个 block 内容"算哈希，A 和 B 的第二个 block 哈希会相同 → B 会错误地复用 A 的第二个 block 的 KV。但这是**错的**！因为：

> Transformer 的 attention 是**因果的**——第二个 block 的 KV 是在"看过前面所有 token"的基础上算出来的。A 的第二个 block 看过 `[The cat sat on]`，B 的看过 `[A dog ran on]`，**前缀不同 → KV 数值不同 → 不能复用**。

所以判断必须是："从第 0 个 token 到当前 block 结尾，**整段前缀**完全一致"才能复用。

#### 解法：把"前一个 block 的哈希"也喂进当前 block 的哈希

vLLM 用**链式哈希**解决——每个 block 的哈希不只取决于自己的 token，还取决于**前一个 block 的哈希**：

```
block_0 哈希 = hash(NONE_HASH,    block_0 的 token)
block_1 哈希 = hash(block_0 哈希,  block_1 的 token)   ← 包含了 block_0 的哈希
block_2 哈希 = hash(block_1 哈希,  block_2 的 token)   ← 间接包含了 block_0、block_1
```

像区块链一样，每一环都"封装"了前面所有环的信息。这样：

```
请求 A: block_1 哈希 = hash( hash(NONE, [The cat sat on]), [the mat ...] )
请求 B: block_1 哈希 = hash( hash(NONE, [A dog ran on]),   [the mat ...] )
                                  ↑ 不同              ↑ 相同
              → 两个 block_1 哈希【不同】 → 不会错误复用 ✓
```

即使两个 block 自身 token 相同，只要前缀不同，链式哈希就不同——**自动保证了"前缀一致性"**。

#### 源码：`hash_block_tokens`

见 [kv_cache_utils.py:392-420](../vllm/v1/core/kv_cache_utils.py#L392-L420)：

```python
def hash_block_tokens(hash_function, parent_block_hash,
                      curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH          # 第一个 block 用一个固定的"种子"

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash_function(
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)),  # ← 三元组一起哈希
        curr_block_token_ids_tuple, extra_keys)
```

哈希的输入是个三元组 `(前一个block哈希, 本block的token, extra_keys)`：
- **`parent_block_hash`**：前一个 block 的哈希值 → 这就是"链"
- **`curr_block_token_ids`**：本 block 的 token → 本块内容
- **`extra_keys`**：额外区分键（多模态 / LoRA 用，纯文本是 None）

#### 源码：`hash_request_tokens` 把整条链串起来

见 [kv_cache_utils.py:423-459](../vllm/v1/core/kv_cache_utils.py#L423-L459)：

```python
def hash_request_tokens(hash_function, block_size, request):
    token_ids = request.all_token_ids
    ret = []
    parent_block_hash_value = None              # 链条起点
    for start in range(0, len(token_ids), block_size):
        block_token_ids = token_ids[start:start + block_size]
        if len(block_token_ids) < block_size:
            break                               # 不满的 block 不算哈希（不能缓存）

        block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
                                       block_token_ids, ...)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value   # ← 把本环哈希传给下一环
    return ret
```

关键就是最后一行——**每算完一个 block，就把它的哈希设为下一个 block 的 `parent`**，链条这样一节一节往下接。

#### `BlockHashType` 为什么是三元组

回顾我们最开始那条 MYLOG 日志里的 `hash`：

```python
[4389094542432185936, [2, 642, 4628, 181, ...], null]
 ↑ hash_value           ↑ token_ids             ↑ extra_keys
```

`BlockHashType` 是个 NamedTuple，存了三样东西：

| 字段 | 作用 |
|---|---|
| `hash_value` | 链式哈希算出的整数值，用来做字典 key 快速查找 |
| `token_ids` | 本 block 的 token，用来**防哈希碰撞**（哈希值相同时再比 token 确认） |
| `extra_keys` | 多模态/LoRA 的额外区分键 |

为什么要存 `token_ids`？因为哈希值理论上可能碰撞（两段不同内容算出同样的 hash_value）。命中时再比对 `token_ids` 确认内容真的一样，避免用错 KV。这是"安全网"。

#### `NONE_HASH`——链条的起点种子

见 [kv_cache_utils.py:45-46](../vllm/v1/core/kv_cache_utils.py#L45-L46)：

```python
NONE_HASH = int.from_bytes(os.urandom(32), byteorder="big") if os.getenv(
    'PYTHONHASHSEED') is None else sha256(os.getenv('PYTHONHASHSEED'))
```

第一个 block 没有"前一个 block"，需要一个固定的种子值作为链条起点。

- 默认用 `os.urandom(32)`——**每次进程启动随机生成**。好处：不同 vLLM 实例的哈希空间不同，避免跨实例的意外碰撞/缓存污染。
- 如果设了 `PYTHONHASHSEED` 环境变量，则用它的 SHA256——**可复现**（debug / 测试用）。

#### 一图总结链式哈希

```
token 序列: [t0 t1 t2 t3 | t4 t5 t6 t7 | t8 t9 t10 t11 | t12 t13(没满)]
              block 0        block 1        block 2          (跳过)

NONE_HASH ──┐
            ▼
       hash(NONE_HASH, [t0..t3])  = H0 ──┐
                                          ▼
                       hash(H0, [t4..t7]) = H1 ──┐
                                                  ▼
                              hash(H1, [t8..t11]) = H2

block_hashes = [H0, H1, H2]   (block 3 没满，不算)

命中逻辑: 新请求算出自己的 [H0', H1', H2', ...]，
         从头比对，能匹配多少个连续的 H，就复用多少个 block 的 KV
```

#### 串起来的知识点

| 之前学的 | 在链式哈希里的体现 |
|---|---|
| 只缓存 full block | `hash_request_tokens` 里 `if len < block_size: break` |
| `cache_full_blocks` 的 `prev_block_hash_value` | 就是在手动维护这条链 |
| `get_computed_blocks` 的"全命中重算最后一块" | 链式哈希让"最长公共前缀"可被逐 block 匹配 |
| `BlockHashType` 三元组（MYLOG 日志里见过） | hash_value 查找 + token_ids 防碰撞 |

#### 一句话总结

> **链式哈希 = 每个 block 的哈希 `= hash(前一个block的哈希, 本block的token)`**。像区块链一样层层封装，使得"哈希相同"等价于"从头到此的整段前缀完全相同"——这正是 **prefix** cache 能安全复用 KV 的数学保证。

---

## 五、PagedAttention 进阶机制

### Kernel 伪码

```python
# 输入：
#   query: Tensor[head_dim]
#   block_table: list[int]                   # [phys_block_0, phys_block_1, ...]
#   kv_pool: Tensor[total_blocks, BLOCK_SIZE, head_dim]
#   num_keys: int
#   BLOCK_SIZE: int

scores = []
for k in range(num_keys):
    block_idx = k // BLOCK_SIZE              # 逻辑 block 编号
    block_offset = k % BLOCK_SIZE            # 块内偏移
    physical_block = block_table[block_idx]  # 一次间接：逻辑 → 物理
    K_k = kv_pool[physical_block, block_offset]  # 二次间接：取 K 向量
    score_k = dot(query, K_k)
    scores.append(score_k)

attn_weights = softmax(scores)

output = zeros(head_dim)
for k in range(num_keys):
    block_idx = k // BLOCK_SIZE
    block_offset = k % BLOCK_SIZE
    physical_block = block_table[block_idx]
    V_k = kv_pool[physical_block, block_offset]
    output += attn_weights[k] * V_k
```

**关键两行**：

```python
block_idx = k // BLOCK_SIZE
physical_block = block_table[block_idx]   # ← 普通 attention 没有的步骤
```

### 真实 CUDA 中的优化

```cuda
// 1. 把 block_table 加载到 shared memory（每个 SM 一次）
__shared__ int block_table_smem[MAX_BLOCKS_PER_SEQ];

// 2. 以 BLOCK_SIZE 为粒度并行
for (block_idx = 0; block_idx < num_blocks; block_idx++) {
    int phys = block_table_smem[block_idx];           // shared mem 查找，几乎免费
    half* K_block = &kv_pool[phys * BLOCK_SIZE * head_dim];

    // 一段 BLOCK_SIZE × head_dim 是连续的 → coalesced HBM read ✓
    for (offset = threadIdx.x; offset < BLOCK_SIZE; offset += blockDim.x) {
        ...
    }
}
```

**为什么 PagedAttention 几乎没有性能损失**：
- block_table 查找在 shared memory，零开销
- block 内部连续（4KB），HBM 读取仍 coalesced
- 跨 block 跳跃反正都得从 HBM 拿，无所谓物理是不是连续

block_size **小于 GPU cache line（128B）**才会真正出问题，所以 16 是安全选择。

### Copy-on-Write 与 Prefix Caching

#### "只存一份"靠 `ref_cnt`（引用计数）

```python
class KVCacheBlock:
    block_id: int
    ref_cnt: int          # ← "几个请求在用我"
    _block_hash: BlockHashType
```

- A 进来，prefix 5 个 block，每个 `ref_cnt=1`
- B 进来，命中相同的 5 个 block：`incr_ref()` 让 `ref_cnt=2`
- A 完成 → 5 个 block 都 `decr_ref()`，`ref_cnt=1`
- B 完成 → `ref_cnt=0`，进 free queue

**没有真正的"copy"——只是指针复用 + 引用计数加一**。

#### 不满的 last block 怎么处理

vLLM 的解法：**只有 full block 才进入 prefix cache**（见 [BlockPool.cache_full_blocks](../vllm/v1/core/block_pool.py)）。

例：A 的 prompt 50 token，block_size=16 → 前 3 block 各 16 token（满），第 4 block 只有 2 token（不满）
- 进 cache 的：**只有前 3 个**
- B 来时同样 50 token：命中前 3 block（48 token），**第 49、50 这 2 token 重新分配新 block**——这部分会重复存，但很小

#### CoW 的真正用武之地（v0 时代）

CoW 在 v0 时代是给 `n > 1` 的 parallel sampling 用的。4 个候选共享 prompt block，decode 分叉时：
- 共享 block `ref_cnt = 4`
- 候选 1 要往未满 block 写新 token → `ref_cnt > 1` 触发 CoW：分配新 block，拷过去

**v1 几乎不用 CoW**：cache 里的 block 都是 full 的（不可变），partial block 只属于一个 request。

### OS 类比追问

#### TLB？

**vLLM 没有显式 TLB**。原因：
- vLLM 的 block_table < 10 KB，直接整个 load 到 shared memory → 整张表都"热"
- OS 有 TLB 是因为页表巨大（GB 级），TLB 缓存热映射

可以理解为：**block_table 本身就在执行 TLB 的角色**。

#### Page fault？

**vLLM 里这个概念不存在**：

| | OS | vLLM |
|---|---|---|
| 内存分配模式 | **lazy**：访问才分配 | **eager**：调度前就 `allocate_slots()` |
| 物理内存不够时 | page fault → 从磁盘换入 | 调度时拒绝 → preempt 整个请求 |
| 触发时机 | 运行时（kernel 中） | 调度时（kernel 启动前） |

**GPU 上 lazy allocation 不可行**——CUDA kernel 一旦启动不能"中途暂停去分配显存"。

#### Swap？

**V0 有，V1 弃用**。

V0：GPU 显存满 → 拷贝 KV 到 CPU 内存（PCIe）→ 后面拷回。

V1 弃用原因：
1. PCIe 64 GB/s 远慢于 HBM 3.35 TB/s——swap 长 context KV 要几百 ms
2. **重算往往更便宜**：重新 prefill + prefix cache 命中其他请求共享前缀
3. 代码复杂度高
4. FlashAttention prefill 速度远比 swap 快

V1 哲学：**与其 swap，不如 preempt + 后续重算**。

### 概念区分

| 优化 | 做什么 | 层次 | 节省 |
|---|---|---|---|
| **(a) Prefix Caching** | 跨请求复用相同前缀 KV | 运行时 | 重算 + 显存 |
| **(b) Sliding Window Attention** | 只注意最近 W 个 token | 模型架构 | KV cache 显存 |
| **(c) FP8 KV cache** | 8 bit 存 KV | 运行时 | KV cache ×2 |
| **(d) MQA / GQA** | 多个 Q head 共享 K/V head | 模型架构 | KV cache（按 head 比） |

- (b) 和 (d) 是**模型架构属性**——训练时定的，serving 系统只能尊重
- (a) 和 (c) 是**运行时选择**——同模型可选

**全部可以叠加**——SWA + GQA + FP8 + prefix cache 同时启用，显存可以是原来的 1/8 甚至更低。

---

## 六、FlashAttention 简介

### 解决的问题

标准 attention 算 `softmax(Q @ K^T / √d) @ V`，中间会**实化**一个 `(N, N)` attention 矩阵。N=8K → 64M fp16 = 128 MB——而且要**写到 HBM 再读回来**：

1. 算 `S = Q @ K^T` → 写 HBM
2. 读 S → softmax → 写 HBM
3. 读 softmax(S) → @ V → 写 HBM

整个 attention matrix 读写 3-4 次 HBM。带宽浪费严重。

### 核心 idea：tiling + online softmax

**Tiling**：把 Q、K、V 切成小 tile，每个 tile 加载到 **shared memory**，所有计算在 SRAM 内完成，**不实化全 attention matrix**。

**Online softmax**：通过数值校正 `exp(m_old - m_new)` 流式累加：

```python
# 处理 K 的第一个 tile
m_1, l_1 = max(scores_1), sum(exp(scores_1 - m_1))
o_1 = attn_1 @ V_1

# 第二个 tile
m_new = max(m_1, max(scores_2))
correction = exp(m_1 - m_new)
l_new = correction * l_1 + sum(exp(scores_2 - m_new))
o_new = correction * o_1 + attn_2 @ V_2 (with new normalization)
```

每个 tile 完成后立刻丢弃中间结果，最终只输出 `(N, head_dim)` 大小的结果。

### 性能效果

- HBM 访问：从 O(N²) 降到 O(N²/M)，M 是 SRAM 能放下的 tile 大小
- 显存：从 O(N²) 降到 O(N)
- 速度：典型 2-4× 提升

### 跟 PagedAttention 的关系

**两者完全正交，可以叠加**：

| | 优化目标 | 优化对象 |
|---|---|---|
| **FlashAttention** | attention 计算的 HBM 流量 | 计算时的访存模式 |
| **PagedAttention** | KV cache 的显存利用率 | 数据存储的物理布局 |

| 组合 | 说明 |
|---|---|
| 标准 attn + 连续 KV | 原始 Transformer |
| FlashAttention + 连续 KV | Tri Dao 原版 |
| 标准 attn + paged KV | vLLM 早期 |
| **FlashAttention + paged KV** | **vLLM 现在主力** ✓ |

vLLM 的 attention backend 用 `flash_attn_with_kvcache`——**输入接受 block_table，内部用 tiling**。

> 一句话：FlashAttention 改的是"算 attention 时数据怎么从 HBM 流"，PagedAttention 改的是"KV cache 在 HBM 里怎么摆"。一个改计算，一个改存储——天然互补。

---

## 七、Continuous Batching

### Static vs Continuous

**Static batching**：凑齐 N 个请求 → 一起 forward → 等所有完成 → 才放下一批。问题：早完成的请求空占 batch 位置，长生成的请求拖累所有人。

**Continuous batching**（vLLM v1）：每 step 之间重新调度
- 完成的请求立刻返回结果
- 它们的 batch 位置立刻空出
- 新请求立刻填进来
- 没有？batch size 动态变小

**关键**：**调度发生在 token 级别**（每生成 1 个 token 就重新调度），叫 **iteration-level scheduling**。

### vLLM v1 的实现

`Scheduler.schedule()` 每次调用 = continuous batching 的一次决策：

```python
def step(self):
    scheduler_output = self.scheduler.schedule()      # 决定本 step batch
    model_output = self.model_executor.execute_model(scheduler_output)
    return self.scheduler.update_from_output(scheduler_output, model_output)
```

每次 `step()` 对应**生成 1 个 token**（decode）或**处理 1 段 prefill chunk**。

### 混合 prefill + decode 的挑战

一个 step 里：
- A：prefill 64 token
- B：decode 1 token
- C：prefill 64 token
- D：decode 1 token

总 query token = 130

**挑战**：
1. **Variable-length attention**：不同请求 K 长度不同，需用 **varlen attention**（FlashAttention varlen / FlashInfer），接受 `cu_seqlens` 数组
2. **Causal mask 不同**：prefill 内部有因果 mask，decode 只 1 个 query
3. **Token budget 控制**：prefill 烧 budget 快（64/step），decode 慢（1/step），scheduler 必须混合
4. **Block table 复杂化**：每个请求 block_table 长度不同
5. **CUDA Graph 形状变化**：用 piecewise CUDA Graph 或动态形状

**这就是为什么 V1 能默认开 chunked prefill + continuous batching，V0 这两者都要 hack-y 加 flag**——V1 一开始就按"每步混合任意请求"设计的。

---

## 八、LLM 推理性能算账方法

### Roofline 思维

```
理论时间 = max(memory_time, compute_time)
              ↑                    ↑
       从 HBM 流完所有数据    做完所有 FLOPs
```

哪个大听哪个的。**永远是 max，不是 sum**——因为 GPU 上读内存和计算流水线并行。

```
memory_time = total_bytes_read_from_HBM / HBM_bandwidth
compute_time = total_FLOPs / GPU_compute_throughput
```

经验法则：
- **Decode 永远 memory-bound**（除非超长 context）
- **Prefill 永远 compute-bound**（除非 prompt 极短）

### 四件套数字

#### 1. GPU 性能（H100 SXM）

| 指标 | 数值 |
|---|---|
| HBM3 带宽 | **3.35 TB/s** = 3350 GB/s |
| fp16 / bf16 算力（TensorCore） | **989 TFLOPS** |
| HBM 容量 | **80 GB** |

实际能达到峰值的 60-80% 是常态。

#### 2. 模型规格（Llama-7B）

| 参数 | 值 | 含义 |
|---|---|---|
| `num_layers` | 32 | Transformer 层数 |
| `hidden_dim` | 4096 | embedding 维度 |
| `num_heads` | 32 | attention head 数 |
| `head_dim` | 128 | = hidden_dim / num_heads |
| `num_params` | 6.74B | 总参数量 |

#### 3. 推导出来的"每 token 大小"

**模型权重**：

```
模型大小 = num_params × bytes_per_param
       = 6.74e9 × 2 (fp16)
       ≈ 13.5 GB
```

**每 token 的 KV cache**：

```
KV per token = 2 × num_layers × num_heads × head_dim × 2 bytes
              ↑   ↑                                    ↑
            K和V  每层都要存                          fp16

           = 2 × 32 × 32 × 128 × 2
           ≈ 0.5 MB / token
```

**每 token 的 FLOPs**（forward 一次）：

```
FLOPs per token ≈ 2 × num_params
                ≈ 13.5 GFLOPs
```

经验公式 `2 × num_params`：每个参数一次乘 + 一次加 = 2 FLOPs。

### 实战：batch=1, decode, Llama-7B, seq=2048

```
内存读取 = 13.5 GB (权重) + 1 GB (KV) = 14.5 GB
memory_time = 14.5 / 3350 ≈ 4.33 ms / step

compute_time = 1 × 13.5e9 / 989e12 ≈ 0.014 ms

step_time = max(4.33, 0.014) = 4.33 ms

throughput = 1 token / 4.33 ms ≈ 230 tokens/s
```

✅ memory_time >> compute_time → memory-bound。

### 实战：batch=64, decode, Llama-7B, seq=2048

**关键**：模型权重读取量**不变**！每 step 还是只读一次 13.5 GB。

```
内存读取 = 13.5 GB (权重) + 64 × 1 GB (KV) = 77.5 GB
memory_time = 77.5 / 3350 ≈ 23.1 ms

每 step token 数 = 64
throughput = 64 / 23.1 ms ≈ 2770 tokens/s
```

**比例 = 2770/230 ≈ 12 倍**（不是 64 倍）

为什么不是 64 倍？

```
batch=1  内存: 13.5 + 1 = 14.5 GB → 1 token   → 14.5 GB/token
batch=64 内存: 13.5 + 64 = 77.5 GB → 64 token  → 1.2 GB/token (12x 节省)
```

**KV cache 跟 batch 1:1 增长**——所以收益不是 64 倍。短 context 时（KV 很小）batch 收益接近线性。

### 实战：prefill 4096 token, batch=1

**Compute-bound** 场景：

```
内存读取 = 13.5 GB（权重）
memory_time = 13.5 / 3350 ≈ 4 ms

compute FLOPs:
  权重相关 = 4096 × 13.5 GFLOPs ≈ 55 TFLOPs
  attention (N²) ≈ 4.4 TFLOPs
  总计 ≈ 60 TFLOPs

compute_time = 60 / 989 ≈ 60 ms

step_time = max(4, 60) = 60 ms

throughput during prefill = 4096 / 0.06 s ≈ 68000 tokens/s
```

**对比 decode 230 tokens/s** → prefill 比 decode 快 **200-300 倍**。

### 速查表

```
LLM 推理理论性能速查（单卡 H100）

输入 4 个数：num_params, num_layers, num_heads × head_dim, seq_len

衍生量：
  W = 模型权重 (GB)        = num_params × 2 / 1e9
  K = KV per token (MB)    = 2 × layers × heads × head_dim × 2 / 1e6
  F = FLOPs per token (G)  = 2 × num_params / 1e9

每 step 内存 (GB):
  Decode:  Memory = W + B × seq × K / 1000
  Prefill: Memory ≈ W

每 step 计算 (TFLOPs):
  Decode:  Compute = B × F / 1000
  Prefill: Compute = B × N × F / 1000 + Attn(N²)

理论时间:
  time = max(Memory / 3.35, Compute / 989) ms

吞吐:
  Decode tokens/s = B / time × 1000
  Prefill tokens/s = B × N / time × 1000
```

---

## 九、维度澄清：seq_len vs hidden_dim

**最容易混淆**的两个数字：

> - **`hidden_dim = 4096`** 是"每个 token 的向量长度"
> - **`seq_len = 2048`** 是"有多少个 token"
> - **它们是两个完全不同的轴**

### 二维图

想象一个请求的输入是**矩阵**：

```
                      hidden_dim = 4096
                  ◄──────────────────────────────────►

         token 0   [0.12, -0.5, 0.33, ..., 0.07, 0.91]
         token 1   [0.05,  0.7, -0.2, ..., 0.44, -0.1]
seq_len   ...      [...]
 = 2048   token 2046 [...]
         token 2047 [...]
```

- **横向**：每个 token 是 **4096 维向量** ← `hidden_dim`
- **纵向**：一共 **2048 个这样的 token** ← `seq_len`

具体例子：
- "你好" 这两个字 → 2 个 token，每个 4096 维向量
- 一篇 2048 token 长文 → 2048 个 token，每个还是 4096 维

`hidden_dim` 是模型架构定的（写死在 `config.json`），`seq_len` 是请求长度（用户决定）。

### hidden_dim 在 KV 计算里藏在哪

```
每 token 一层的 K = num_heads × head_dim × 2 bytes
                 = 32 × 128 × 2
                 = 4096 × 2 bytes        ← hidden_dim 在这里
                 = 8 KB
```

注意 `num_heads × head_dim = 32 × 128 = 4096 = hidden_dim`——这是 Transformer 设计约束。

### KV cache 立体形状

```
KV cache shape: [num_layers, 2, seq_len, num_heads, head_dim]
                    32        2  2048      32         128

= 32 × 2 × 2048 × 32 × 128 × 2 bytes (fp16)
= 1 GB
```

### 四个维度全表

| 维度 | 名字 | Llama-7B 值 | 谁定的 | 含义 |
|---|---|---|---|---|
| **token 数** | `seq_len` | 0~max_model_len（变化） | 用户请求 | 这次请求多长 |
| **每 token 特征长度** | `hidden_dim` | 4096 | 模型架构 | 每 token 怎么表示 |
| **head 数** | `num_heads` | 32 | 模型架构 | hidden_dim 切成几份 |
| **每 head 维度** | `head_dim` | 128 | 模型架构 | hidden_dim / num_heads |

### 自检题

| 问题 | 答案 |
|---|---|
| "Hello world"（2 token）的 hidden state 形状 | `[2, 4096]` |
| 同一请求的 KV cache 形状 | `[32, 2, 2, 32, 128]` |
| batch=4 各 2 token 的 hidden state 形状 | `[4, 2, 4096]` |

---

## 十、Attention FLOPs 公式详解

### 公式

```
attention FLOPs = 2 × N² × head_dim × num_heads × num_layers × 2
```

### 从最里面一层开始推

**单 head、单 layer 的 attention**：

```
1. scores = Q @ K^T          (Q × K^T)
2. weights = softmax(scores)
3. output = weights @ V
```

#### 矩阵乘 1：Q @ K^T

形状：`[N, head_dim] @ [head_dim, N] = [N, N]`

矩阵乘 `(a, k) @ (k, b) = (a, b)` 的 FLOPs = `2 × a × k × b`（每个输出元素需要 k 次乘 + k 次加 = 2k FLOPs）。

```
Q @ K^T FLOPs = 2 × N × head_dim × N = 2 × N² × head_dim
```

#### 矩阵乘 2：weights @ V

形状：`[N, N] @ [N, head_dim] = [N, head_dim]`

```
weights @ V FLOPs = 2 × N × N × head_dim = 2 × N² × head_dim
```

#### 单 head、单 layer 合计

```
2 × N² × head_dim + 2 × N² × head_dim
= 2 × (2 × N² × head_dim)
= 2 × N² × head_dim × 2
                       ↑
              这个 2 = "两个矩阵乘"
```

### 加上 num_heads 和 num_layers

```
2 × N² × head_dim × num_heads × num_layers × 2
↑    ↑      ↑           ↑           ↑       ↑
①    ②      ③           ④           ⑤       ⑥
```

| 位置 | 因子 | 含义 |
|---|---|---|
| ① | `2` | 每次乘加 = 2 FLOPs（数值约定） |
| ② | `N²` | attention 矩阵大小 |
| ③ | `head_dim` | 每个 Q/K/V 向量长度 |
| ④ | `num_heads` | 多 head 并行 |
| ⑤ | `num_layers` | 每层都做一次 attention |
| ⑥ | `2` | 两个矩阵乘（结构层面） |

**两个 `2` 来源不同**：① 是 FLOPs 约定，⑥ 是 matmul 个数。

### 代入 Llama-7B（N=4096）

```
= 2 × 4096² × 128 × 32 × 32 × 2
≈ 8.8 TFLOPs
```

### Causal mask 让计算量减半

Prefill 时第 i 个 token 只看前 i 个 token，attention 矩阵下三角才有效：

```
       k₀  k₁  k₂  k₃  k₄  k₅
   q₀  ✓   .   .   .   .   .
   q₁  ✓   ✓   .   .   .   .
   q₂  ✓   ✓   ✓   .   .   .
   ...
   q₅  ✓   ✓   ✓   ✓   ✓   ✓

实际计算量 ≈ N² / 2
```

**现代 kernel（FlashAttention）真的省下这一半**——所以 Llama-7B prefill N=4096 attention 实际 ~4.4 TFLOPs（不是 8.8）。

### Attention 占比随 N 变化

| N | 权重 FLOPs | Attention FLOPs | Attention 占比 |
|---|---|---|---|
| 512 | 6.9 TF | 0.07 TF | 1% |
| 2048 | 27.6 TF | 1.1 TF | 4% |
| 4096 | 55 TF | 4.4 TF | 7% |
| 16384 | 220 TF | 70 TF | **24%** |
| 65536 | 880 TF | 1130 TF | **56%** |

**N 翻倍，权重 FLOPs 翻倍，attention FLOPs 翻 4 倍**（因为 N²）。

→ 这就是 **SWA、稀疏 attention、FlashAttention** 在长 context 才显现价值的原因。

---

## 十一、Throughput 公式的本质

### 核心问题

> 为什么 `throughput = 1 token / memory_time` 而不是 `1 token / compute_time`？

### 答案

**用 `max(memory_time, compute_time)`，谁大用谁**。

### Decode batch=1 的情况

```
memory_time = 4.33 ms   ← 把 14.5 GB 流过来
compute_time = 0.014 ms ← 算完所有 FLOPs

step_time = max(4.33, 0.014) = 4.33 ms   ← 听 memory 的
                                 ↑
                          所以叫 memory-bound

throughput = 1 token / 4.33 ms = 230 tokens/s
```

### 为什么是 max 不是 sum

GPU 上"读数据"和"做计算"是**流水线并行**，不是排队的。

> 类比：你和朋友做饭。
> - 你洗菜（30 分钟）
> - 朋友切菜炒菜（5 分钟）
>
> 总耗时是多少？
> - 不是 35 分钟（sum）—— 因为同时干
> - 不是 5 分钟（min）—— 因为朋友等你
> - **是 30 分钟（max）—— 听慢的那个**

GPU 同理：搬运数据和矩阵乘**同时干**，谁慢卡谁。

### Prefill 用 compute_time

```
prefill 4096 token:
  memory_time = 4 ms
  compute_time = 60 ms

step_time = max(4, 60) = 60 ms  ← 听 compute 的
                          ↑
                  所以叫 compute-bound
```

### 单位换算

```
"每 4.33 ms 产出 1 token" → "1 秒能产出几个 token"

1 秒 = 1000 ms
1000 ms ÷ 4.33 ms/token ≈ 231 tokens

→ 231 tokens/s
```

跟"跑 1 km 用 5 分钟，1 小时跑几 km"是同一个除法。

### Decode 循环的真实结构

LLM 生成是**自回归**——每生成 1 token 都要重新 forward 一遍：

```
0ms          4.33ms        8.66ms        12.99ms
 │              │              │              │
 ▼              ▼              ▼              ▼
[step 1]    [step 2]    [step 3]    ...
读 14.5 GB   读 14.5 GB   读 14.5 GB
产 1 token   产 1 token   产 1 token
```

每 4.33 ms 产出 1 token → 1 秒 ≈ 231 token。

### 为什么每 step 都要重读 13.5 GB 权重

GPU 计算单元（SM）不直接从 HBM 读数据——计算发生在 SM 内部 register 和 shared memory：

```
HBM (13.5 GB 权重)
   │
   ▼
L2 cache (50 MB)
   │
   ▼
SM shared memory (256 KB / SM)
   │
   ▼
SM register (做 matmul)
```

**整个 H100 片上 SRAM 加起来都只有几十 MB**——而 Llama-7B 权重 13.5 GB。

唯一能装下的地方就是 HBM。所以**每次 forward 都必须把 13.5 GB 整个从 HBM 流到 SM 用一次再扔掉**。

### 关键数字 = 关键比例

记住：
- 13.5 GB / 3.35 TB/s = **4 ms** 是 Llama-7B 在 H100 上的物理下限
- 不管你 batch 怎么变、kernel 怎么优化、模型怎么调，这个下限躲不过
- 所有"提高 decode 吞吐"的招数本质都是想办法让这 4 ms 内多产几个 token：
  - **加 batch**：4ms 产 N 个（N 个请求共享一次权重读取）
  - **GQA / MQA**：减小 KV，让 batch 能更大
  - **量化**：减小权重大小（fp8 → 7 GB → 2ms 下限）
  - **Speculative decoding**：4ms 产 k 个（一次读权重验证多个候选）

---

## 核心结论

1. **永远画图**：脑子里没有 `[batch, seq, hidden]` 的形状，看到代码会两眼一抹黑
2. **永远算数**：`max(memory, compute)` + "每 step 多少 GB / TFLOPs" + 4 件套硬件参数 = 任何模型任何卡的理论下限
3. **永远问"why"**：每个优化都是治某种病——prefix cache 治重算，paged 治碎片，flash 治 HBM 流量，GQA 治 KV 太大
4. **永远区分 V0/V1**：见到 `vllm/core/` 是 V0，见到 `vllm/v1/` 是 V1，机制完全不同

---

# 附录 A：PagedAttention 自测题（基础概念）

> 用法：先盖住答案自答，再看下方答案对照。难度标记：🟢 基础 / 🟡 中等 / 🔴 进阶

## A.1 动机题 🟢

**问**：PagedAttention 要解决的**核心问题**是什么？在它之前，传统 LLM serving 系统管理 KV cache 的方式有什么毛病？

**答**：

PagedAttention 之前的系统给每个请求**预分配 `max_model_len` 长度的连续显存**。三大问题：
- **内部碎片**：实际生成长度 << 预分配长度，剩下全浪费（主要矛盾）
- **外部碎片**：请求来去后凑不出连续大块
- **保留浪费**：必须按最坏情况分配

旧系统**有效利用率 20%~40%**，PagedAttention 能做到 **96%+**。这个 2~4 倍提升直接转化成 batch size 翻倍 → 吞吐翻倍。

> ⚠️ 注意：PagedAttention 解决的是**显存利用率**问题，**不是显存带宽**——它本身不降低内存读取量，甚至因间接寻址略增 overhead。

## A.2 OS 类比题 🟢

**问**：把下面 4 个 OS 概念分别对应到 PagedAttention 里的什么东西：

| OS 概念 | PagedAttention 对应 |
|---|---|
| 物理页（physical page） | ? |
| 虚拟页（virtual page）/ 进程地址空间 | ? |
| 页表（page table） | ? |
| 进程（process） | ? |

**答**：

| OS | PagedAttention |
|---|---|
| 进程 | **`Request`** |
| 进程的虚拟地址空间 | 该请求的 KV token 序列 |
| 虚拟页 | 一段连续的 logical KV slots（**隐式概念，无对应类**） |
| 物理页 / page frame | **`KVCacheBlock`**（GPU 显存里一段 slot） |
| 页表 | **`req_to_blocks[req_id]`**（即 BlockTable） |
| OS 内存管理器 | **`KVCacheManager` + `BlockPool`** |
| 共享页 | 多请求共享的 prefix block（`ref_cnt > 1`） |

**新人最容易答错的**：
- ❌ "虚拟页 = `KVCacheBlock`" → 错。`KVCacheBlock` 是**物理页帧描述符**
- ❌ "进程 = `KVCacheManager`" → 错。`KVCacheManager` 是**OS 内核**

## A.3 block_size 设计题 🟡

**问**：vLLM 默认 `block_size = 16`。如果改成 **1**（每 token 一个 block）或 **256**（很大），分别有什么好处和坏处？

提示：从内存碎片、attention kernel 性能、prefix cache 命中粒度三个角度想。

**答**：

### `block_size = 1` 的影响

| 影响 | 详情 | 严重程度 |
|---|---|---|
| **内部碎片** | ✅ 完全消失（每 token 一个 block） | 好处 |
| **GPU memory coalescing 失效** | HBM 一次读 cache line（128B）。每 token 256B 但**位置完全分散**，HBM 带宽利用率从 ~80% 跌到 ~10% | ⭐⭐⭐ 致命 |
| **Block table 自身变巨大** | 32K seq → 32K 项 block_table，污染 L1/L2 cache | ⭐⭐ |
| **Prefix cache 哈希计算开销线性放大** | 每 token 一次 hash | ⭐ |

### `block_size = 256` 的影响

| 影响 | 详情 | 严重程度 |
|---|---|---|
| **Prefix cache 命中粒度变粗** | 200 token 共享前缀：block_size=16 → 命中 192 token；block_size=256 → 命中 0 token | ⭐⭐⭐ 致命 |
| **内部碎片** | 平均每请求浪费 128 token | ⭐⭐ |
| **GPU 访存效率** | ✅ 反而更好 | 好处 |

### 为什么默认 16

是 **GPU 访存效率**（要够大，块内连续读）和 **prefix cache 命中粒度 + 内部碎片**（要够小）的折中。论文实验显示 16~32 是甜区。

具体计算：block_size=16, head_dim=128, fp16 → 一个 block = 4 KB，正好一段连续显存能高效读。

## A.4 Kernel 机制题 🟡

**问**：一个请求生成到第 100 个 token，KV cache 物理上散在 7 个不同 block 里。GPU 跑 attention kernel、要算第 101 个 token 的 query 跟前面 100 个 token 的 KV 做点积时，**kernel 是怎么找到那 100 个 token 的 KV 的**？普通的 attention kernel 能不能直接做这件事？

**答**：

### PagedAttention kernel 的核心：间接寻址

```python
for k in range(100):
    block_idx = k // BLOCK_SIZE              # 逻辑 block 编号
    block_offset = k % BLOCK_SIZE            # 块内偏移
    physical_block = block_table[block_idx]  # 一次间接：逻辑 → 物理
    K_k = kv_pool[physical_block, block_offset]  # 二次间接：取 K 向量
    score_k = dot(query, K_k)
```

**关键两行**：
```python
block_idx = k // BLOCK_SIZE
physical_block = block_table[block_idx]   # ← 普通 attention 没有的步骤
```

### 真实 CUDA 中

- `block_table` 整个 load 到 **shared memory**（几 KB），查找几乎免费
- 一个 block 内部连续（4KB），HBM 读取仍 coalesced
- 跨 block 跳跃反正都得从 HBM 拿，无所谓物理是不是连续

### 普通 attention kernel 不能做

普通 kernel 假设 K/V 是**连续张量**，用 `K[k]` 这种线性下标。当 K 散在不同 block 时，`K[k]` 这种访问会读到错误数据。所以 PagedAttention **必须配一个改造过的 attention kernel**——这是它从"优化"变成"必须有专用 kernel 支持的东西"的根本原因。

## A.5 Copy-on-Write 题 🟡

**问**：两个请求 A 和 B 共享同一段长 prompt 前缀（如同一个 system prompt）。具体说：

- 这段共享前缀的 KV 在显存里存几份？
- 如果 A 已经 decode 几个 token，B 才进来，B 怎么"接上" A 算过的 KV？
- 如果共享前缀的最后一个 block **没存满**（prompt 50 token，block_size=16，最后 block 只 2 token），怎么处理？

**答**：

### "只存一份"靠 `ref_cnt`（引用计数）

```python
class KVCacheBlock:
    block_id: int
    ref_cnt: int          # ← "几个请求在用我"
    _block_hash: BlockHashType
```

- A 进来，prefix 5 个 block，每个 `ref_cnt=1`
- B 进来，命中相同的 5 个 block：`incr_ref()` 让 `ref_cnt=2`
- A 完成 → 5 个 block `decr_ref()`，`ref_cnt=1`，B 还在用
- B 完成 → `ref_cnt=0`，进 free queue

**没有真正的"copy"——只是指针/id 复用 + 引用计数加一**。

### 不满的 last block：vLLM 干脆不共享

**只有 full block 才进入 prefix cache**（见 [BlockPool.cache_full_blocks](../vllm/v1/core/block_pool.py)，注意方法名就是 `cache_full_blocks`）。

例：A 的 prompt 50 token，前 3 block 满（48 token），第 4 block 只 2 token：
- 进 cache：**只前 3 个**
- B 同样 50 token：命中前 3 block（48 token），第 49、50 这 2 token **重新分配新 block 自己写**
- 这部分会重复存（A 一份 B 一份），但只 2 token，浪费很小

### Copy-on-Write 的真正用武之地（v0 时代）

CoW 主要给 `n > 1` 的 parallel sampling 用：4 个候选共享 prompt block，decode 分叉时若 `ref_cnt > 1` 触发 CoW（拷贝出新 block）。

**v1 几乎不用 CoW**：cache 里 block 都是 full 的（不可变），partial block 只属于一个 request。

## A.6 BlockPool 双数据结构题 🔴

**问**：[BlockPool](../vllm/v1/core/block_pool.py) 同时维护两个数据结构：
- `free_block_queue: FreeKVCacheBlockQueue`（双向链表）
- `cached_block_hash_to_block: dict[hash → {id → block}]`（哈希字典）

为什么需要两个？只用其中一个能不能实现 PagedAttention + prefix cache？

进一步：一个 block 在什么状态下**同时出现在两个数据结构里**？什么状态下**只在一个里**？

**答**：

### 服务两种不同的访问模式

| 操作 | 用哪个？ | 为什么必须是这个？ |
|---|---|---|
| "给我一个空闲 block" | `free_block_queue.popleft()` | 链表头是最久未用的，实现 LRU |
| "找 hash=X 的 block" | `cached_block_hash_to_block[X]` | 哈希字典 O(1) 查询 |
| "驱逐一个 block" | 从 queue 头取 + 从 dict 删 | 两边都要改 |
| "复用排队中的 block" | dict 命中后从 queue 中间 `remove(block)` | **双向**链表支持 O(1) 中间删除 |

### 单独用一个不行

- **只用 queue**：失去 O(1) 缓存查找，每次问 hash 在不在都要遍历
- **只用 dict**：失去 LRU 顺序，不知道驱逐谁

这是**经典 LRU cache 实现**——和 Python `OrderedDict` / `functools.lru_cache` 内部一样的套路。

### Block 状态表

| `ref_cnt` | `_block_hash` | 在 free_queue？ | 在 cached_dict？ | 含义 |
|---|---|---|---|---|
| > 0 | 有 | ❌ | ✅ | 在用，是个完整缓存 block |
| > 0 | 无 | ❌ | ❌ | 在用，正在写入还没满 |
| 0 | 有 | ✅ | ✅ | **没人在用，可被复用 or 被驱逐**（同时存在态） |
| 0 | 无 | ✅ | ❌ | 干净空闲块 |

## A.7 PagedAttention 的代价 🔴

**问**：PagedAttention 不是免费午餐。它相比"按 sequence 分配连续 KV cache"，**牺牲了什么**？

**答**：

主要代价：

1. **间接寻址开销**：每次访 K/V 多一次 `block_table[block_idx]` 查找。Shared memory 里查所以很小，但不是零。
2. **Kernel 复杂度**：必须写专用 attention kernel，普通 attention 实现不能直接用。生态成本高（每个新硬件平台都要适配）。
3. **跨 block 边界访问无法 coalesce**：标准连续 KV 时 GPU 可以一次性 load 大块；分页后跨 block 跳跃必须分多次 HBM 请求。
4. **CUDA Graph 兼容性**：block_table 内容动态变化，让 CUDA Graph capture 更复杂。
5. **小 block_size 的访存惩罚**：必须保持 block_size 够大（≥16），否则 HBM 带宽利用率断崖下跌——这反过来限制了 prefix cache 命中粒度。

但综合来看，**PagedAttention 节省的显存换来的 batch size 提升远大于这些 overhead**，所以是净赚的设计。

---

# 附录 B：进阶机制题

## B.1 GPU 显存层级题 🔴

**问**：把下面这些显存/存储的**带宽**从快到慢排序，并大致估个量级（H100 SXM）。为什么这个排序对 PagedAttention 设计**重要**？

- HBM3 显存（KV cache 存这里）
- SM 内的 register file
- SM 内的 shared memory
- L2 cache
- PCIe 5.0（GPU↔CPU 内存）

**答**：

### 从快到慢

| 层级 | 带宽 | 容量 |
|---|---|---|
| Register file | ~150 TB/s 每 SM | 64 KB / SM |
| Shared memory / L1 | ~20 TB/s 每 SM | 256 KB / SM |
| L2 cache | ~7 TB/s | 50 MB |
| **HBM3** | **~3.35 TB/s** | **80 GB** |
| PCIe 5.0 x16 | ~64 GB/s | 主机内存 |

### 对 PagedAttention 设计的重要性

1. **KV cache 住在 HBM**——所有 KV 数据每次 attention 都得从 HBM 读。**HBM 带宽是真正瓶颈**。
2. **PagedAttention 间接寻址额外开销在哪**：block_table 很小（几 KB）能全塞 shared memory，所以"查表"几乎免费；真正的成本只是 KV 数据从 HBM 读这一步，和不分页时一样。
3. **block_size 不能太小的根本原因**：HBM 读取以 cache line（128B）为单位。block_size=16, head_dim=128, fp16 → 一个 block 4KB，连续读非常划算。block_size=1 → 每 token 256B 但**位置完全分散**，带宽利用率从 ~80% 跌到 ~10%。

> 一句话：分页本身不增加 HBM 读取量，关键是**保证读 KV 时仍是大块连续读**——这就是 block_size 必须够大的根本原因。

## B.2 Compute-bound vs Memory-bound 🔴

**问**：(a) 在 H100 + Llama-7B 上，prefill 一秒能 forward 几十万 token，但 decode 一秒只能 几百到几千。**为什么差几百倍**？

(b) "Decode 是 memory-bound" 是什么意思？瓶颈卡在哪？

(c) Continuous batching 把 batch size 从 1 提到 64，decode 的**每秒 token 数**会涨多少倍？为什么？

**答**：

### (a) 表象 vs 本质

**表象**："prefill 一次过 N 个 token，decode 一次只过 1 个"——但这只是 What，不是 Why。

**本质**：每 step 都要把 **13.5 GB 模型权重**从 HBM 流过来一次。

- **Prefill 一个 step（N=512）**：权重 13.5 GB 流一次 → 算 512 个 token 的 forward → amortize 给 512 个 token
- **Decode 一个 step（N=1）**：权重 13.5 GB 流一次 → 算 1 个 token → amortize 给 1 个 token

每 token 的内存成本差 ~512 倍 → throughput 差几百倍。

### (b) Memory-bound 含义

物理瓶颈：**13.5 GB ÷ 3.35 TB/s ≈ 4 ms**——不管做多少 FLOPs，光把权重读一遍就要 4 ms。这是 H100 上 Llama-7B 的物理下限。

具体到 batch=1 decode：
- memory_time = 4.33 ms（包含 KV）
- compute_time = 0.014 ms
- step_time = max(...) = 4.33 ms ← memory 占主导 = "memory-bound"

### (c) Batch size 1→64 涨多少倍

```
batch=1, seq=2048: 14.5 GB / 3350 = 4.33 ms → 230 tok/s
batch=64, seq=2048: 77.5 GB / 3350 = 23.1 ms → 64/23.1ms = 2770 tok/s

比例: 2770/230 ≈ 12 倍  (不是 64 倍)
```

**为什么不是 64 倍**：模型权重读取**一次给所有请求复用**，不变；但 KV cache **跟 batch 1:1 涨**。所以收益只有 12 倍。

如果 context 短（seq=128，KV 很小），收益会接近 50 倍。

## B.3 PagedAttention kernel 伪码题 🟡

**问**：用伪码写出 PagedAttention kernel 算"第 q 个 query 跟前面 100 个 token 做 attention"的逻辑。

**答**：

```python
# 输入：
#   query: Tensor[head_dim]
#   block_table: list[int]                   # [phys_block_0, phys_block_1, ...]
#   kv_pool: Tensor[total_blocks, BLOCK_SIZE, head_dim]
#   num_keys: int = 100
#   BLOCK_SIZE: int = 16

scores = []
for k in range(num_keys):
    block_idx = k // BLOCK_SIZE              # 逻辑 block 编号
    block_offset = k % BLOCK_SIZE            # 块内偏移
    physical_block = block_table[block_idx]  # 一次间接：逻辑 → 物理
    K_k = kv_pool[physical_block, block_offset]  # 二次间接：取 K 向量
    score_k = dot(query, K_k)
    scores.append(score_k)

attn_weights = softmax(scores)

output = zeros(head_dim)
for k in range(num_keys):
    block_idx = k // BLOCK_SIZE
    block_offset = k % BLOCK_SIZE
    physical_block = block_table[block_idx]
    V_k = kv_pool[physical_block, block_offset]
    output += attn_weights[k] * V_k
```

**关键就是这两行**：

```python
block_idx = k // BLOCK_SIZE
physical_block = block_table[block_idx]
```

普通 attention 没有这两步——它假设 K/V 是连续张量，用 `K[k]` 直接索引。

## B.4 OS 类比追问 🟡

**问**：(a) OS 里有 **TLB**，缓存"虚拟页 → 物理页"映射避免每次访存查页表。**vLLM 里有没有类似的东西**？

(b) OS 里的 **page fault**（访问的虚拟页不在物理内存 → 中断 → 从磁盘换入）对应到 vLLM 是什么？

(c) OS 里的 **swap**（换出到磁盘）对应到 vLLM 是什么？V0 有 V1 没了，为什么？

**答**：

### (a) TLB？

**vLLM 没有显式 TLB**。原因：

OS 要 TLB 是因为页表巨大（GB 级），TLB 缓存热映射。

vLLM 的 block_table：
- 每个请求最多几千 block
- 整个 block_table < 10 KB
- **直接整个 load 到 shared memory** → 整张表都"热"
- 不需要专门的小缓存

可以理解为：**block_table 本身就在执行 TLB 的角色**。

### (b) Page fault？

**vLLM 里这个概念不存在**：

| | OS | vLLM |
|---|---|---|
| 内存分配模式 | **lazy**：访问才分配 | **eager**：调度前 `allocate_slots()` |
| 物理内存不够时 | page fault → 从磁盘换入 | 调度时拒绝 → preempt 整个请求 |
| 触发时机 | 运行时（kernel 中） | 调度时（kernel 启动前） |

**GPU 上 lazy allocation 不可行**——CUDA kernel 一旦启动不能"中途暂停去分配显存"。

### (c) Swap？

**V0 有，V1 弃用**。

V0：GPU 显存满 → 把 KV 拷贝到 CPU 内存（PCIe）→ 后面拷回。

V1 弃用原因：
1. **PCIe 64 GB/s 远慢于 HBM 3.35 TB/s**——swap 长 context KV 要几百 ms
2. **重算往往更便宜**：重新 prefill + prefix cache 命中其他请求共享前缀
3. **代码复杂度高**：维护 CPU 端 KV blocks 麻烦
4. **FlashAttention prefill 速度远比 swap 快**

V1 哲学：**与其 swap，不如 preempt + 后续重算**。

## B.5 概念区分题 🟡

**问**：下面 4 个名词都跟"减少 KV cache 显存"有关。**分别说清楚它们各自做的事**，并指出它们**互相之间能不能同时启用**：

(a) Prefix Caching
(b) Sliding Window Attention（如 Mistral 用的）
(c) FP8 KV cache
(d) Multi-Query Attention / Grouped-Query Attention

特别地：(d) 跟 (a)(b)(c) 是同一个层次的优化吗？

**答**：

| 优化 | 做什么 | 层次 | 节省 |
|---|---|---|---|
| **(a) Prefix Caching** | 跨请求复用相同前缀的 KV | **运行时**（serving 系统） | 重算 + 显存（多请求场景） |
| **(b) SWA** | 只注意最近 W 个 token | **模型架构**（训练时定的） | KV cache 显存（窗口外不存） |
| **(c) FP8 KV cache** | 8 bit 存 KV 而不是 16 bit | **运行时**（数值精度） | KV cache ×2 |
| **(d) MQA / GQA** | 多个 Q head 共享 K/V head | **模型架构**（训练时定的） | KV cache 按 head 比例 |

### 核心区分

- **(b) 和 (d) 是模型本身属性**——Mistral 是 SWA、Llama-3 是 GQA，**vLLM 跑时只能尊重模型已有设计**
- **(a) 和 (c) 是 vLLM 的运行时选择**——同模型可选

### 能否同时启用？全部都能并存

- Prefix caching + GQA：完全正交
- Prefix caching + SWA：cache 只能存窗口内的 block，窗口外的必须 evict
- Prefix caching + FP8：完全正交
- **SWA + GQA + FP8 + prefix cache**：可以同时（vLLM 实际有这种 stacking）

实际工程例：Mistral 7B（SWA）+ vLLM（prefix caching）+ FP8 KV cache → 显存占用 1/8 甚至更低。

## B.6 FlashAttention 题 🔴

**问**：FlashAttention 跟 PagedAttention 是什么关系？vLLM 怎么把两者结合？

**答**：

### FlashAttention 的核心 idea

标准 attention 把 `(N, N)` attention matrix **实化到 HBM**，读写 3-4 次。FlashAttention：
- **Tiling**：把 Q/K/V 切成小 tile，加载到 **shared memory**，计算在 SRAM 内完成
- **Online softmax**：流式累加（数值校正 `exp(m_old - m_new)`），不实化全 attention matrix

效果：HBM 访问从 O(N²) 降到 O(N²/M)；显存从 O(N²) 降到 O(N)；速度 2-4× 提升。

### 跟 PagedAttention 的关系

**两者完全正交，可以叠加**：

| | 优化目标 | 优化对象 |
|---|---|---|
| **FlashAttention** | attention 计算的 HBM 流量 | 计算时的访存模式 |
| **PagedAttention** | KV cache 的显存利用率 | 数据存储的物理布局 |

| 组合 | 说明 |
|---|---|
| 标准 attn + 连续 KV | 原始 Transformer |
| FlashAttention + 连续 KV | Tri Dao 原版 |
| 标准 attn + paged KV | vLLM 早期 |
| **FlashAttention + paged KV** | **vLLM 现在主力** ✓ |

vLLM 用 `flash_attn_with_kvcache`——**输入接受 block_table，内部用 tiling**。两者在 kernel 层面已融合。

> 一句话：FlashAttention 改"算 attention 时数据怎么从 HBM 流"，PagedAttention 改"KV cache 在 HBM 里怎么摆"。一个改计算，一个改存储——天然互补。

## B.7 Continuous Batching 题 🟡

**问**：(a) 一个 batch 8 个请求，3 个完成、5 个还在生成。下一个 step 怎么处理？continuous batching 这时做了什么 static batching 做不到的事？

(b) `Scheduler.schedule()` 跟 continuous batching 是什么关系？

(c) Continuous batching + chunked prefill 一起用，一个 step 里**可以同时混合 prefill 和 decode**。这件事在 GPU kernel 层面带来什么挑战？

**答**：

### (a) Static vs Continuous

**Static batching**：等 8 个全完成再开新 batch。3 个完成的请求**已结束但还占着 batch 位置**——空转或补 padding，浪费算力。

**Continuous batching**（vLLM v1）每 step 之间重新调度：
- 3 个完成的立刻返回结果
- batch 位置**立刻空出**
- waiting 队列里有新请求？立刻填进来开始 prefill
- 没有？这个 step batch size 就是 5

**关键**：调度发生在 **token 级别**，叫 **iteration-level scheduling**。

### (b) Scheduler.schedule() 与 continuous batching 关系

**`Scheduler.schedule()` 的每次调用 = continuous batching 的一次决策**。

vLLM v1 的 `EngineCore.step()` 主循环：
```python
def step(self):
    scheduler_output = self.scheduler.schedule()      # 决定本 step batch
    model_output = self.model_executor.execute_model(scheduler_output)
    return self.scheduler.update_from_output(scheduler_output, model_output)
```

每次 `step()` 对应**生成 1 个 token**或**处理 1 段 prefill chunk**。

**continuous batching 不是 vLLM 的某个独立模块**，它就是**主循环的形状**——每个 forward step 之间都过一遍 scheduler。

### (c) 混合 prefill + decode 的 GPU 挑战

设想一个 step：
- A：prefill 64 token（attend 前 100 个 K）
- B：decode 1 token（attend 前 199 个 K）
- C：prefill 64 token（attend 前 50 个 K）
- D：decode 1 token（attend 前 49 个 K）

总 query token = 130

挑战：

1. **Variable-length attention**：不同请求 K 长度不同，普通 kernel 要求 batch 内序列等长。**解决**：用 varlen attention（FlashAttention varlen / FlashInfer），接受 `cu_seqlens`
2. **Causal mask 不同**：prefill 内部因果 mask；decode 只 1 个 query
3. **Token budget 控制**：prefill 烧 budget 快（64/step），decode 慢（1/step），scheduler 必须混合让总 token 不超算力上限
4. **Block table 复杂化**：每个请求 block_table 长度不同
5. **CUDA Graph 形状变化**：用 piecewise CUDA Graph 或动态形状

**这就是为什么 V1 能默认开 chunked prefill + continuous batching，V0 这两者都要 hack-y 加 flag**——V1 一开始就按"每步混合任意请求"设计的。

---

# 附录 C：算账练习题

> 全部题目都有具体数值答案。先盖住答案自己算，再对照。

## C.1 基础：Llama-7B 在不同配置下的 throughput

**模型常数**：13.5 GB 权重，0.5 MB/token KV，13.5 GFLOPs/token
**硬件**：H100 SXM，3.35 TB/s HBM，989 TFLOPS

### Q1：batch=1, decode, seq=2048

```
Memory = 13.5 + 1 = 14.5 GB
time = 14.5 / 3350 = 4.33 ms
throughput = 1 / 4.33ms ≈ 230 tok/s
```

### Q2：batch=64, decode, seq=2048

```
Memory = 13.5 + 64 × 1 = 77.5 GB
time = 77.5 / 3350 = 23.1 ms
throughput = 64 / 23.1ms ≈ 2770 tok/s
```

提升 12 倍（不是 64 倍，因为 KV 跟 batch 1:1 涨）

### Q3：batch=128, decode, seq=2048

```
KV 总量 = 128 × 1 = 128 GB
权重 + KV = 13.5 + 128 = 141.5 GB
但 H100 只有 80 GB → ❌ 装不下！
```

**结论**：batch=128 + seq=2048 + Llama-7B fp16 在单卡 H100 上**不可行**。需要：
- 降 seq_len（短上下文）
- 用 GQA / MQA 模型
- 用 FP8 KV cache
- 或多卡 TP

### Q4：batch=128, decode, seq=512（短 context）

```
KV/req = 512 × 0.5 MB = 0.25 GB
Memory = 13.5 + 128 × 0.25 = 45.5 GB ✓ 装得下
time = 45.5 / 3350 = 13.6 ms
throughput = 128 / 13.6ms ≈ 9400 tok/s
```

### Q5：prefill 4096 token, batch=1

```
Memory = 13.5 GB → 4 ms
Compute:
  权重相关 = 4096 × 13.5 GFLOPs = 55 TFLOPs
  attention (N²) ≈ 4.4 TFLOPs (含 causal mask)
  总计 ≈ 60 TFLOPs
Compute time = 60 / 989 ≈ 60 ms

step_time = max(4, 60) = 60 ms ← compute-bound
throughput = 4096 / 60ms ≈ 68000 tok/s
```

prefill 比 decode 快 **~300 倍**。

## C.2 综合：用户问答总耗时

**问**：用户输入 prompt 1024 token，期望 LLM 回答 500 token。在 batch=1, Llama-7B, H100 上，**总耗时**多少？

**答**：

### 第一段：Prefill 1024 token

```
Compute:
  权重相关 = 1024 × 13.5 GFLOPs = 13.8 TFLOPs
  attention = 2 × 1024² × 128 × 32 × 32 × 2 / 2 (causal) ≈ 0.27 TFLOPs
  总计 ≈ 14 TFLOPs
Compute time = 14 / 989 ≈ 14 ms
```

### 第二段：Decode 500 token

```
平均 KV = (1024 + 1024+500) / 2 × 0.5 MB ≈ 0.75 GB
Memory per step = 13.5 + 0.75 = 14.25 GB
time per step = 14.25 / 3350 ≈ 4.25 ms

500 token × 4.25 ms ≈ 2125 ms = 2.1 秒
```

### 总耗时

```
Prefill (14 ms) + Decode (2125 ms) ≈ 2.14 秒
```

**大头在 decode**——这是为什么 decode 优化（quantization、speculative decoding、GQA）那么重要。

## C.3 GQA 的影响

**问**：Llama-7B 用 GQA（num_kv_heads=4 而不是 32）后，KV cache 缩小几倍？同样 batch=64, seq=2048 throughput 涨多少？

**答**：

```
KV 缩小: 32/4 = 8 倍
新 KV per token = 0.5 / 8 = 0.0625 MB
新 KV per req @ seq=2048: 128 MB

batch=64 总 KV = 64 × 128 MB = 8 GB (之前 64 GB)
Memory = 13.5 + 8 = 21.5 GB
time = 21.5 / 3350 = 6.42 ms
throughput = 64 / 6.42ms ≈ 9970 tok/s
```

比之前 2770 提升 **~3.6 倍**。

## C.4 大模型单卡能否跑

**问**：Llama-70B fp16 能不能在 H100 单卡 batch=1 decode？

**答**：

```
权重 = 70B × 2 = 140 GB
H100 HBM = 80 GB
140 > 80 → ❌ 单卡装不下，必须 TP=2 或更多
```

如果用 INT8 量化：70 GB，加上 KV 仍可能勉强。
如果用 INT4 量化：35 GB，单卡可行，留 45 GB 给 KV+激活值。

## C.5 极限 batch 与 KV 量化组合

**问**：想跑 batch=256, seq=2048 用什么 KV 精度？

**答**：

```
fp16 KV: 256 × 1 GB = 256 GB → 必爆
FP8 KV: 256 × 0.5 GB = 128 GB → 还是爆

加上 GQA (num_kv_heads=4):
  fp16 KV: 256 × 0.125 GB = 32 GB → 13.5+32=45.5 ✓ 可行
  FP8 KV: 256 × 0.0625 = 16 GB → 13.5+16=29.5 ✓ 很宽松
```

**结论**：高 batch 长 context 的关键是 **GQA + 量化双管齐下**。

## C.6 Llama-13B 自测

**问**：Llama-13B（40 layers, 40 heads, head_dim=128, 13B params）在 H100 上：

(a) batch=1, seq=4096 decode 理论 tokens/s？
(b) prefill 8192 tokens 理论时间？

**答**：

### 衍生量

```
权重 = 13e9 × 2 = 26 GB
KV/token = 2 × 40 × 40 × 128 × 2 = 819,200 byte ≈ 0.82 MB
FLOPs/token = 2 × 13e9 = 26 GFLOPs
```

### (a) batch=1, seq=4096 decode

```
KV/req = 4096 × 0.82 MB = 3.36 GB
Memory = 26 + 3.36 = 29.4 GB
time = 29.4 / 3350 = 8.78 ms
throughput = 1 / 8.78ms ≈ 114 tok/s
```

（对比 Llama-7B batch=1 seq=2048 的 230 tok/s，模型大一倍约慢一半）

### (b) prefill 8192 tokens

```
权重相关 FLOPs = 8192 × 26 GFLOPs = 213 TFLOPs
Attention FLOPs = 2 × 8192² × 128 × 40 × 40 × 2 / 2 (causal)
                ≈ 27.5 TFLOPs
总 = 240 TFLOPs

Compute time = 240 / 989 ≈ 243 ms
Memory time = 26 / 3350 = 7.8 ms
step_time = max(243, 7.8) = 243 ms
```

实际 70% 效率：~350 ms 完成 8192 token prefill。

---

# 附录 D：术语速查

| 中文 | 英文 | 含义 |
|---|---|---|
| 显存 | HBM (High Bandwidth Memory) | GPU 上的主显存，KV cache 和模型权重住这里 |
| 算力 | TFLOPS | Tera FLoating-point Operations Per Second，每秒万亿次浮点运算 |
| 块 | block | KV cache 分页的最小单位，vLLM 默认 16 token |
| 引用计数 | ref_cnt | 几个请求在用同一个 block |
| 前缀缓存 | prefix caching | 跨请求复用相同前缀的 KV |
| 分块预填 | chunked prefill | 把长 prefill 切成小块跟 decode 混跑 |
| 连续批处理 | continuous batching | 每生成 1 token 就重新调度的批处理方式 |
| 预填阶段 | prefill | 处理 prompt 的阶段，compute-bound |
| 解码阶段 | decode | 自回归生成的阶段，memory-bound |
| 推测解码 | speculative decoding | 用小模型先猜，大模型并行验证多个候选 |
| 量化 | quantization | 用更低精度（FP8/INT8/INT4）存权重或 KV |
| 张量并行 | tensor parallel (TP) | 把权重切到多卡上 |
| 流水线并行 | pipeline parallel (PP) | 把不同层放到不同卡上 |
| 数据并行 | data parallel (DP) | 多副本模型处理不同请求 |
