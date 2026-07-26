# 基础示例

## GPU 设备查询

[`device_query.cu`](device_query.cu) 展示如何通过 CUDA Runtime 查询设备属性。

在仓库根目录运行：

```bash
nvcc -O2 examples/device_query.cu -o device_query
./device_query
```
