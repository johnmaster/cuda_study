"""
PyTorch CUDA 扩展编译脚本。

编译命令:
    python setup.py build_ext --inplace

编译后会生成 decode_attention_cuda.so（Linux）或 .pyd（Windows），
可直接 import decode_attention_cuda 使用。

也可用 torch.utils.cpp_extension.load() 在运行时 JIT 编译（见 serve.py）。
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA 架构：RTX 2080 Ti = sm_75 (Turing)
# 如果有更新的卡，可以添加更多架构
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

setup(
    name="decode_attention_cuda",
    ext_modules=[
        CUDAExtension(
            name="decode_attention_cuda",
            sources=["csrc/decode_attention.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",          # 更快的数学函数（精度略降）
                    "-gencode=arch=compute_75,code=sm_75",  # RTX 2080 Ti
                    # 调试时可加: "-G", "-lineinfo"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
