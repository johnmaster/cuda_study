"""cd triton_kernels/elementwise && python test_vector_ops.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from vector_ops import vector_add, vector_mul


def test_vector_add():
    torch.manual_seed(0)
    for shape in [(1024,), (3, 256), (7, 11, 13)]:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        y = torch.randn(shape, device="cuda", dtype=torch.float32)
        z = vector_add(x, y)
        ref = x + y
        assert torch.allclose(z, ref, rtol=0, atol=0), shape


def test_vector_mul():
    x = torch.randn(5000, device="cuda", dtype=torch.float16)
    y = torch.randn(5000, device="cuda", dtype=torch.float16)
    z = vector_mul(x, y)
    assert torch.allclose(z, x * y, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_vector_add()
    test_vector_mul()
    print("vector_ops OK")
