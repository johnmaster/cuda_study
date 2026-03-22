from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_ops",
    ext_modules=[
        CUDAExtension(
            name="custom_ops_cuda",
            sources=[
                "csrc/binding.cpp",
                "csrc/custom_gelu_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
