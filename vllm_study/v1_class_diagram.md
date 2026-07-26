# vLLM V1 引擎核心类关系图

> **适用版本**：vLLM **v0.8.5**（仓库当前 detached HEAD）
> **范围**：仅覆盖**单机单卡 + 纯文本模型**的主线代码路径
> **不包含**（已主动剔除，避免初学时干扰）：
> - 多模态（vision/audio encoder、`EncoderCacheManager`、`mm_*` 字段）
> - 分布式推理（DP / TP / PP、Ray、`MultiprocExecutor`、`KVConnector`、pipeline batch queue）
>
> 后续要复习这些主题时再补一份单独的文档。

聚焦 `vllm/v1/`，覆盖 **从一次 `generate()` 调用到 GPU forward 一遍** 涉及的核心类。
每张图后面都有"成员/方法速查表"，便于复习时快速定位源码。

## 目录

- [一、总览：一次 `step()` 经过哪些对象](#一总览一次-step-经过哪些对象)
- [二、Engine 层 — 谁调谁](#二engine-层--谁调谁)
- [三、Scheduling 层 — Scheduler / SchedulerOutput](#三scheduling-层--scheduler--scheduleroutput)
- [四、KV Cache 层 — KVCacheManager / BlockPool / KVCacheBlock](#四kv-cache-层--kvcachemanager--blockpool--kvcacheblock)
- [五、Request 层 — Request / RequestStatus](#五request-层--request--requeststatus)
- [六、Executor / Worker 层](#六executor--worker-层)
- [七、Sampler / Output 数据结构](#七sampler--output-数据结构)
- [八、调用时序：一次 `EngineCore.step()`](#八调用时序一次-enginecorestep)
- [九、速查表 — 类与文件位置](#九速查表--类与文件位置)

---

## 一、总览：一次 `step()` 经过哪些对象

```mermaid
flowchart TB
    subgraph Frontend["Frontend 进程"]
        AsyncLLM
    end

    subgraph EngineProc["EngineCore 子进程（ZMQ IPC）"]
        EngineCore
        Scheduler
        KVCacheManager
        BlockPool
    end

    subgraph WorkerProc["Worker 进程（GPU）"]
        Executor
        Worker
        GPUModelRunner
        InputBatch
        Sampler
    end

    AsyncLLM -- "EngineCoreRequest" --> EngineCore
    EngineCore --> Scheduler
    Scheduler --> KVCacheManager
    KVCacheManager --> BlockPool
    EngineCore -- "SchedulerOutput" --> Executor
    Executor --> Worker
    Worker --> GPUModelRunner
    GPUModelRunner --> InputBatch
    GPUModelRunner --> Sampler
    GPUModelRunner -- "ModelRunnerOutput" --> EngineCore
    EngineCore -- "EngineCoreOutputs" --> AsyncLLM
```

**记忆口诀**：`Frontend → EngineCore → Scheduler ⇄ KVCacheManager ⇄ BlockPool` + `Executor → Worker → ModelRunner → Sampler`。

---

## 二、Engine 层 — 谁调谁

```mermaid
classDiagram
    class EngineClient {
        <<abstract>>
    }

    class AsyncLLM {
        +VllmConfig vllm_config
        +Processor processor
        +OutputProcessor output_processor
        +AsyncMPClient engine_core
        +AnyTokenizer tokenizer
        +asyncio.Task output_handler
        +generate() async
        +add_request() async
        +abort() async
    }

    class EngineCore {
        +Executor model_executor
        +SchedulerInterface scheduler
        +step() EngineCoreOutputs
        +add_request(EngineCoreRequest)
        +abort_requests(ids)
    }

    class EngineCoreProc {
        +Queue input_queue
        +Queue output_queue
        +bool engines_running
        +run_busy_loop()
        +process_input_socket()
        +process_output_socket()
    }

    EngineClient <|-- AsyncLLM
    EngineCore <|-- EngineCoreProc
    AsyncLLM "1" o-- "1 (远程)" EngineCore : ZMQ IPC
    EngineCore "1" *-- "1" Executor
    EngineCore "1" *-- "1" SchedulerInterface
```

**关键点**

- [AsyncLLM](../vllm/v1/engine/async_llm.py) 跑在前端进程，**不直接持有 `EngineCore`**，而是通过 `AsyncMPClient`（ZMQ）跟它通信。
- [EngineCore](../vllm/v1/engine/core.py) 跑在子进程，是**真正的 inner loop**。`step()` = schedule → execute → update。
- `EngineCoreProc` 在 `EngineCore` 上加了 ZMQ socket 处理，把它包装成可以被 spawn 的子进程入口。

---

## 三、Scheduling 层 — Scheduler / SchedulerOutput

```mermaid
classDiagram
    class SchedulerInterface {
        <<abstract>>
        +schedule() SchedulerOutput
        +update_from_output(out, model_out) EngineCoreOutputs
        +add_request(Request)
        +finish_requests(ids, status)
        +has_requests() bool
        +make_stats() SchedulerStats
    }

    class Scheduler {
        +VllmConfig vllm_config
        +KVCacheConfig kv_cache_config
        +KVCacheManager kv_cache_manager
        +dict~str,Request~ requests
        +deque~Request~ waiting
        +list~Request~ running
        +set~str~ finished_req_ids
        +int num_spec_tokens
        +schedule() SchedulerOutput
        +update_from_output(...)
        +add_request(Request)
        +finish_requests(ids, status)
    }

    class SchedulerOutput {
        +list~NewRequestData~ scheduled_new_reqs
        +list~CachedRequestData~ scheduled_cached_reqs
        +dict~str,int~ num_scheduled_tokens
        +int total_num_scheduled_tokens
        +dict~str,list~ scheduled_spec_decode_tokens
        +int num_common_prefix_blocks
        +set~str~ finished_req_ids
    }

    class NewRequestData {
        +str req_id
        +list~int~ prompt_token_ids
        +list~int~ block_ids
        +SamplingParams sampling_params
    }

    class CachedRequestData {
        +str req_id
        +int num_computed_tokens
        +list~int~ new_block_ids
    }

    SchedulerInterface <|.. Scheduler
    Scheduler "1" *-- "1" KVCacheManager
    Scheduler --> SchedulerOutput : 产出
    SchedulerOutput "1" *-- "*" NewRequestData
    SchedulerOutput "1" *-- "*" CachedRequestData
    Scheduler "1" o-- "*" Request : waiting/running/requests
```

**`schedule()` 三件事**：

1. **挑请求**：从 `waiting` 队头按 FCFS 取，按 token budget 决定本步处理多少 token。
2. **要内存**：调 `kv_cache_manager.allocate_slots(...)`，没空间就把 `running` 末尾的请求 **preempt** 回 `waiting`。
3. **打包**：把"新请求完整状态"（`NewRequestData`）和"老请求 delta"（`CachedRequestData`）打成 `SchedulerOutput` 发给 worker。

> **新 vs 缓存**：第一次进来的请求需要发完整 prompt token + block ids（`NewRequestData`）；后续步的请求只发增量（`CachedRequestData`），节约 IPC 带宽。

---

## 四、KV Cache 层 — KVCacheManager / BlockPool / KVCacheBlock

```mermaid
classDiagram
    class KVCacheManager {
        +KVCacheConfig kv_cache_config
        +BlockPool block_pool
        +dict~str,list~ req_to_blocks
        +dict~str,list~ req_to_block_hashes
        +dict~str,int~ num_cached_block
        +PrefixCacheStats prefix_cache_stats
        +int block_size
        +int num_gpu_blocks
        +allocate_slots(req, n_tokens) list~KVCacheBlock~
        +get_computed_blocks(req) tuple
        +free(req)
        +reset_prefix_cache() bool
    }

    class BlockPool {
        +list~KVCacheBlock~ blocks
        +FreeKVCacheBlockQueue free_block_queue
        +dict cached_block_hash_to_block
        +bool enable_caching
        +KVCacheBlock null_block
        +get_cached_block(hash) KVCacheBlock
        +cache_full_blocks(...)
        +allocate() KVCacheBlock
        +free(block)
    }

    class KVCacheBlock {
        +int block_id
        +int ref_cnt
        +BlockHashType _block_hash
        +KVCacheBlock prev_free_block
        +KVCacheBlock next_free_block
        +incr_ref()
        +decr_ref()
    }

    class BlockHashType {
        <<NamedTuple>>
        +int hash_value
        +tuple~int~ token_ids
        +Any extra_keys
    }

    class FreeKVCacheBlockQueue {
        +KVCacheBlock head
        +KVCacheBlock tail
        +popleft()
        +remove(block)
        +append(block)
    }

    class KVCacheConfig {
        +int num_blocks
        +list~KVCacheGroupSpec~ kv_cache_groups
    }

    KVCacheManager "1" *-- "1" BlockPool
    KVCacheManager --> KVCacheConfig : uses
    BlockPool "1" *-- "*" KVCacheBlock
    BlockPool "1" *-- "1" FreeKVCacheBlockQueue
    BlockPool ..> BlockHashType : keys cache
    KVCacheBlock --> BlockHashType : 持有 hash
    FreeKVCacheBlockQueue "1" o-- "*" KVCacheBlock : 双向链表
```

**两套数据结构同时维护一个 block**：

| 视角 | 数据结构 | 作用 |
|---|---|---|
| **空闲表** | `FreeKVCacheBlockQueue`（双向链表） | 决定 LRU 驱逐顺序 |
| **缓存表** | `cached_block_hash_to_block: dict[hash → {id → block}]` | prefix cache 命中查找 |

block 在两边的状态：
- `ref_cnt > 0`：在用，**只在缓存表里**。
- `ref_cnt == 0` 且 `_block_hash != None`：**两边都在**（既能被复用，也能被驱逐）。
- `ref_cnt == 0` 且 `_block_hash == None`：纯空闲，**只在空闲表里**。

---

## 五、Request 层 — Request / RequestStatus

```mermaid
classDiagram
    class Request {
        +str request_id
        +RequestStatus status
        +SamplingParams sampling_params
        +list~int~ prompt_token_ids
        +list~int~ _output_token_ids
        +list~int~ _all_token_ids
        +list~int~ spec_token_ids
        +int num_computed_tokens
        +int num_prompt_tokens
        +LoRARequest lora_request
        +list~EngineCoreEvent~ events
        +append_output_token_ids(ids)
        +is_finished() bool
        +get_finished_reason() FinishReason
        +record_event(type, ts)
        +from_engine_core_request(req)$ Request
    }

    class RequestStatus {
        <<enum>>
        WAITING
        WAITING_FOR_FSM
        RUNNING
        PREEMPTED
        FINISHED_STOPPED
        FINISHED_LENGTH_CAPPED
        FINISHED_ABORTED
        FINISHED_IGNORED
    }

    class EngineCoreRequest {
        <<msgspec.Struct>>
        +str request_id
        +list~int~ prompt_token_ids
        +SamplingParams sampling_params
        +int eos_token_id
        +float arrival_time
        +LoRARequest lora_request
    }

    Request --> RequestStatus
    EngineCoreRequest ..> Request : from_engine_core_request()
```

**两个 Request 别搞混**：

- [EngineCoreRequest](../vllm/v1/engine/__init__.py)：**前端 → EngineCore 的 IPC 消息**，纯 msgspec.Struct，可序列化。
- [Request](../vllm/v1/request.py)：**Scheduler 内部的工作对象**，可变状态，跟 KV blocks / 事件绑定。

`EngineCore.add_request()` 收到前者，调用 `Request.from_engine_core_request(...)` 转成后者，扔进 `scheduler.waiting`。

### Request 状态机

```
                add_request()
                     │
                     ▼
                ┌─────────┐
                │ WAITING │
                └────┬────┘
                     │ schedule()
                     ▼
                ┌─────────┐ ◄─── 内存不够时 ────┐
                │ RUNNING │                    │
                └────┬────┘                    │
        ┌────────────┼─────────────┐           │
        │            │             │           │
        │            ▼             ▼           │
        │  ┌───────────────┐  ┌───────────┐   │
        │  │FINISHED_STOPPED│  │ PREEMPTED │ ──┘
        │  │FINISHED_LENGTH│   └───────────┘
        │  └───────────────┘
        ▼
  FINISHED_ABORTED  (client 断开)
```

---

## 六、Executor / Worker 层

> 本节只画 **`UniProcExecutor`（单进程单卡）** 这条路径。
> 多卡场景的 `MultiprocExecutor` / Ray Executor / 跨机 Worker 已剔除，单独看分布式文档。

```mermaid
classDiagram
    class Executor {
        <<abstract>>
        +get_kv_cache_specs() list
        +determine_available_memory() list~int~
        +initialize_from_config(configs)
        +execute_model(SchedulerOutput) ModelRunnerOutput
        +profile(is_start)
        +get_class(config)$ type
    }

    class UniProcExecutor {
        +Worker driver_worker
    }

    class Worker {
        +GPUModelRunner model_runner
        +VllmConfig vllm_config
        +int rank
        +int local_rank
        +execute_model(SchedulerOutput) ModelRunnerOutput
        +initialize_from_config(configs)
        +get_kv_cache_spec() dict
        +compile_or_warm_up_model()
    }

    class GPUModelRunner {
        +VllmConfig vllm_config
        +nn.Module model
        +Sampler sampler
        +RejectionSampler rejection_sampler
        +KVCacheTensor kv_cache
        +InputBatch input_batch
        +torch.device device
        +execute_model(SchedulerOutput) ModelRunnerOutput
        +prepare_input_batch(out) InputBatch
        +forward()
        +sample(logits) SamplerOutput
    }

    class InputBatch {
        +list~str~ _req_ids
        +dict req_id_to_index
        +Tensor token_ids_cpu_tensor
        +int max_num_reqs
        +int max_model_len
        +add_request(NewRequestData)
        +update_request(CachedRequestData)
        +prepare_input_tokens() Tensor
        +get_sampling_metadata() SamplingMetadata
    }

    class CachedRequestState {
        +list~int~ block_ids
        +int num_computed_tokens
    }

    Executor <|-- UniProcExecutor
    Executor "1" *-- "1" Worker
    Worker "1" *-- "1" GPUModelRunner
    GPUModelRunner "1" *-- "1" InputBatch
    GPUModelRunner "1" *-- "1" Sampler
    GPUModelRunner "1" *-- "0..1" RejectionSampler
    InputBatch "1" *-- "*" CachedRequestState
```

**`execute_model()` 的执行链**（一次 forward 经过的对象）：

```
Executor.execute_model(SchedulerOutput)
  → Worker.execute_model(SchedulerOutput)
      → GPUModelRunner.execute_model(SchedulerOutput)
          1. input_batch.add_request / update_request    (按 SchedulerOutput 改 batch)
          2. input_batch.prepare_input_tokens()          (拼输入张量)
          3. self.model(...)                              (forward)
          4. self.sampler(logits, sampling_metadata)     (采样)
          5. 返回 ModelRunnerOutput
```

---

## 七、Sampler / Output 数据结构

```mermaid
classDiagram
    class Sampler {
        +TopKTopPSampler topk_topp_sampler
        +forward(logits, meta) SamplerOutput
        +apply_penalties()
        +apply_bad_words()
        +apply_temperature()
        +sample(logits, meta)
        +compute_logprobs(logits)
        +gather_logprobs(...)
    }

    class RejectionSampler {
        +forward(draft, target_logits) Tensor
    }

    class SamplerOutput {
        +Tensor sampled_token_ids
        +LogprobsTensors logprobs_tensors
    }

    class ModelRunnerOutput {
        +list~str~ req_ids
        +dict req_id_to_index
        +list~list~int~~ sampled_token_ids
        +list~list~int~~ spec_token_ids
        +LogprobsLists logprobs
        +dict prompt_logprobs_dict
    }

    class EngineCoreOutput {
        +str request_id
        +list~int~ new_token_ids
        +FinishReason finish_reason
        +LogprobsLists new_logprobs
        +int stop_reason
        +list events
    }

    class EngineCoreOutputs {
        +list~EngineCoreOutput~ outputs
        +SchedulerStats scheduler_stats
        +float timestamp
    }

    class FinishReason {
        <<enum>>
        STOP
        LENGTH
        ABORT
    }

    Sampler --> SamplerOutput
    EngineCoreOutput --> FinishReason
    EngineCoreOutputs "1" *-- "*" EngineCoreOutput
    ModelRunnerOutput ..> EngineCoreOutput : Scheduler.update_from_output() 转换
```

**输出数据流**：`SamplerOutput`（GPU tensor）→ `ModelRunnerOutput`（每请求 list）→ `EngineCoreOutput`（每请求增量）→ `EngineCoreOutputs`（一步全部）→ ZMQ 回 frontend → `RequestOutput`。

---

## 八、调用时序：一次 `EngineCore.step()`

```mermaid
sequenceDiagram
    participant F as AsyncLLM
    participant E as EngineCore
    participant S as Scheduler
    participant K as KVCacheManager
    participant B as BlockPool
    participant X as Executor
    participant W as Worker (GPUModelRunner)

    F->>E: add_request(EngineCoreRequest)  [ZMQ]
    E->>S: scheduler.add_request(Request)
    Note over S: req 进 waiting 队列

    loop 每一步
        E->>S: scheduler.schedule()
        S->>K: get_computed_blocks(req)         (prefix 命中查询)
        K->>B: get_cached_block(hash)
        B-->>K: KVCacheBlock or None
        K-->>S: (computed_blocks, n_cached_tokens)

        S->>K: allocate_slots(req, n_tokens)
        K->>B: allocate() / free()              (按需)
        B-->>K: blocks
        K-->>S: blocks

        S-->>E: SchedulerOutput

        E->>X: execute_model(SchedulerOutput)
        X->>W: Worker.execute_model
        Note over W: input_batch → forward → sampler
        W-->>X: ModelRunnerOutput
        X-->>E: ModelRunnerOutput

        E->>S: update_from_output(out, model_out)
        Note over S: 写回 token、判 stop、释放 block
        S->>K: free(req)  (若该 req 完成)
        K->>B: free(block) for each block
        S-->>E: EngineCoreOutputs
        E-->>F: EngineCoreOutputs  [ZMQ]
    end
```

---

## 九、速查表 — 类与文件位置

### Engine

| 类 | 位置 |
|---|---|
| [AsyncLLM](../vllm/v1/engine/async_llm.py) | `vllm/v1/engine/async_llm.py` |
| [EngineCore](../vllm/v1/engine/core.py) | `vllm/v1/engine/core.py` |
| [EngineCoreProc](../vllm/v1/engine/core.py) | `vllm/v1/engine/core.py` |
| [EngineCoreRequest](../vllm/v1/engine/__init__.py) | `vllm/v1/engine/__init__.py` |
| [EngineCoreOutput / Outputs / FinishReason](../vllm/v1/engine/__init__.py) | `vllm/v1/engine/__init__.py` |

### Scheduling

| 类 | 位置 |
|---|---|
| [SchedulerInterface](../vllm/v1/core/sched/interface.py) | `vllm/v1/core/sched/interface.py` |
| [Scheduler](../vllm/v1/core/sched/scheduler.py) | `vllm/v1/core/sched/scheduler.py` |
| [SchedulerOutput / NewRequestData / CachedRequestData](../vllm/v1/core/sched/output.py) | `vllm/v1/core/sched/output.py` |

### KV Cache

| 类 | 位置 |
|---|---|
| [KVCacheManager](../vllm/v1/core/kv_cache_manager.py) | `vllm/v1/core/kv_cache_manager.py` |
| [BlockPool / FreeKVCacheBlockQueue](../vllm/v1/core/block_pool.py) | `vllm/v1/core/block_pool.py` |
| [KVCacheBlock / BlockHashType](../vllm/v1/core/kv_cache_utils.py) | `vllm/v1/core/kv_cache_utils.py` |
| [KVCacheConfig / KVCacheSpec](../vllm/v1/kv_cache_interface.py) | `vllm/v1/kv_cache_interface.py` |

### Request

| 类 | 位置 |
|---|---|
| [Request / RequestStatus](../vllm/v1/request.py) | `vllm/v1/request.py` |

### Executor / Worker

| 类 | 位置 |
|---|---|
| [Executor (abstract)](../vllm/v1/executor/abstract.py) | `vllm/v1/executor/abstract.py` |
| `UniProcExecutor` | `vllm/v1/executor/uniproc_executor.py` |
| [Worker](../vllm/v1/worker/gpu_worker.py) | `vllm/v1/worker/gpu_worker.py` |
| [GPUModelRunner](../vllm/v1/worker/gpu_model_runner.py) | `vllm/v1/worker/gpu_model_runner.py` |
| [InputBatch / CachedRequestState](../vllm/v1/worker/gpu_input_batch.py) | `vllm/v1/worker/gpu_input_batch.py` |

### Sampler / Output

| 类 | 位置 |
|---|---|
| [Sampler](../vllm/v1/sample/sampler.py) | `vllm/v1/sample/sampler.py` |
| [RejectionSampler](../vllm/v1/sample/rejection_sampler.py) | `vllm/v1/sample/rejection_sampler.py` |
| [SamplingMetadata](../vllm/v1/sample/metadata.py) | `vllm/v1/sample/metadata.py` |
| [ModelRunnerOutput / SamplerOutput](../vllm/v1/outputs.py) | `vllm/v1/outputs.py` |

---

## 复习要点

1. **进程边界**：Frontend ⇄ EngineCore 走 ZMQ；EngineCore ⇄ Worker 在单卡 `UniProcExecutor` 里是同进程直接调用。跨边界要序列化（`EngineCoreRequest` / `SchedulerOutput`）。
2. **`Scheduler` 唯一持有 `KVCacheManager`**，`KVCacheManager` 唯一持有 `BlockPool`。所有 KV 分配/释放都要从 Scheduler 这条线走。
3. **block 状态由 `ref_cnt + _block_hash` 决定**，参考"四"节末尾那张表。
4. **`InputBatch` 是 GPU 端的对端**：scheduler 出 `SchedulerOutput`，runner 把它消费成 `InputBatch`，再喂模型。
5. **两个 Request 别混**：`EngineCoreRequest` 是 IPC 消息（msgspec），`Request` 是 Scheduler 内部可变工作对象。
