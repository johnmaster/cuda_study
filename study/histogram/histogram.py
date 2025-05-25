import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

lib = load(
    name="hist_lib",
    sources=["histogram.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

# 创建一个包含数字0-9各重复1000次的列表
# 转换为pytorch张量，指定数据类型为int32
# 使用.cuda()将张量移动到gpu
a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32).cuda();
h_i32 = lib.histogram_i32(a)
print("-" * 80)

# h_i32.shape[0]获取张量h_i32的第0维长度
for i in range(h_i32.shape[0]):
    print(f"h_i32   {i}: {h_i32[i]}")

print("-" * 80)
h_i32x4 = lib.histogram_i32x4(a)
for i in range(h_i32x4.shape[0]):
    print(f"h_i32x4 {i}: {h_i32x4[i]}")
print("-" * 80)
