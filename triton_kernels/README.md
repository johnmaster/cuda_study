# Triton 示例 (`triton_kernels/`)

| 目录 | 内容 |
|------|------|
| `gemm/` | 分块矩阵乘 + autotune |
| `flash_attention/` | Flash Attention |
| `elementwise/` | 向量加法、逐元素乘（1D grid + mask） |
| `reduce/` | 全局 `sum_all`（atomic）、二维 `row_sum` |

## 运行

需安装 PyTorch（CUDA）与 Triton（通常随 PyTorch 2.x 提供）。

```bash
cd triton_kernels/elementwise && python test_vector_ops.py
cd triton_kernels/reduce && python test_reduce_ops.py
```

## API 速览

- `elementwise.vector_ops.vector_add(x, y)` → `x + y`
- `elementwise.vector_ops.vector_mul(x, y)` → `x * y`
- `reduce.reduce_ops.sum_all(x)` → 标量，全元素和
- `reduce.reduce_ops.row_sum(x)` → `x` 为 `[M,N]` 时得到 `[M]`
