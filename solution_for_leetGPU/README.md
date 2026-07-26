# LeetGPU CUDA 题解

本目录收录可独立构建的 CUDA 算子题解，包括向量运算、归约、矩阵运算、归一化、Attention 与排序等主题。

## 构建方式

每道题都是独立 CMake 工程。以矩阵乘为例，在仓库根目录运行：

```bash
cmake -S solution_for_leetGPU/matrix_multiply \
      -B solution_for_leetGPU/matrix_multiply/build
cmake --build solution_for_leetGPU/matrix_multiply/build -j
```

可执行文件生成在对应题目的 `build/` 目录。构建目录和性能分析报告属于本地产物，不应提交到版本控制。

## 题目分类

- 基础访存：`matrix_transpose`、`histogram`
- 向量与归约：`dot_product`、`reduction`、`prefix_sum`
- 矩阵计算：`gemv`、`matrix_multiply`、`wmma_gemm`
- 深度学习算子：`gelu`、`softmax`、`layer_norm`、`rms_norm`
- Attention：`flash_attention`、`multi_head_attention`
- 排序与选择：`radix_sort`、`topk`
