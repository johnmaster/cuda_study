from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_custom",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_cuda",
            sources=[
                "csrc/binding.cpp",
                "csrc/flash_attn_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
