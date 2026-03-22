# DDP — 分布式数据并行

## 核心思想

每张 GPU 持有**模型的完整副本**，但处理**不同的数据 batch**。
反向传播后，通过 AllReduce 同步梯度，确保所有卡的模型参数保持一致。

```
步骤:
1. 每卡各取 batch 的 1/N
2. 各自前向传播
3. 各自反向传播，得到各自的梯度
4. AllReduce 梯度（求和后除以 N = 平均梯度）  ← NCCL 通信
5. 各卡用相同的平均梯度更新参数 → 参数保持一致
```

## DDP vs DataParallel

| 特性 | `nn.DataParallel` (DP) | `DistributedDataParallel` (DDP) |
|------|----------------------|-------------------------------|
| 进程模型 | 单进程多线程 | **多进程**（每卡一个进程） |
| GIL 影响 | 受 Python GIL 限制 | 不受影响 |
| 通信 | GPU 0 汇总梯度（瓶颈） | **Ring AllReduce**（均衡） |
| 速度 | 慢 | **快 1.5-2x** |
| 多机支持 | 不支持 | 支持 |
| 推荐 | **永远不要用** | **永远用这个** |

## DDP 关键机制

### 1. Gradient Bucketing（梯度分桶）

DDP 不是等所有梯度算完才 AllReduce，而是把参数分成多个 bucket：

```
反向传播:  param_N → param_N-1 → ... → param_1 → param_0
                                          ↓
Bucket 机制: [bucket_2: param_N ~ param_K] [bucket_1: ...] [bucket_0: ...]
             ↓ 算完就开始 AllReduce         ↓                ↓
             AllReduce bucket_2              AllReduce ...     AllReduce ...
```

好处：**计算和通信 overlap**，反向传播还没全部结束，前面的 bucket 已经在通信了。

### 2. Gradient Accumulation（梯度累积）

当 batch 太大放不进显存时，可以分多个 micro-batch 累积梯度：

```python
for i, micro_batch in enumerate(micro_batches):
    loss = model(micro_batch) / accumulation_steps
    # 最后一步才同步梯度
    if i < accumulation_steps - 1:
        with model.no_sync():  # 跳过 AllReduce
            loss.backward()
    else:
        loss.backward()  # 这一步触发 AllReduce
optimizer.step()
```

## 运行

```bash
torchrun --nproc_per_node=2 ddp_training.py
```
