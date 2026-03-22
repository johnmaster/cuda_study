# Continuous Batching（连续批处理）

## Static Batching 的问题

传统推理：一批请求必须**全部完成**才能处理下一批。

```
Static Batching (batch=4):

请求 A: [==================] 完成 (200 tokens)
请求 B: [======] 完成 (60 tokens)
         [       等等等等等等 ] ← B 完成了还得等 A！
请求 C: [====] 完成 (40 tokens)
         [         等等等等等 ] ← 更多浪费
请求 D: [============] 完成 (120 tokens)
         [      等等等 ]

时间 →→→→→→→→→→→→→→→→→→→→→
```

问题：短请求等长请求 → GPU 利用率低，吞吐量差。

## Continuous Batching

每个 **iteration** 检查：哪些请求完成了？有新请求等待吗？

```
Continuous Batching:

iter 1: [A, B, C, D] → 都在生成
iter 2: [A, B, C, D] → C 完成了！
iter 3: [A, B, E, D] → E 新加入，填上 C 的位置
iter 4: [A, F, E, D] → B 完成了，F 加入
...

GPU 永远满载，没有空等！
```

## 关键指标

| 指标 | Static | Continuous |
|------|--------|-----------|
| GPU 利用率 | 低（等最慢的） | 高（持续满载） |
| 吞吐量 | 低 | **高 2-10x** |
| 首 token 延迟 (TTFT) | 高（等上一批完成） | **低**（随时插入） |

## 运行

```bash
python batching_simulator.py
```
