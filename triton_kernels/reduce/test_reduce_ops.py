"""cd triton_kernels/reduce && python test_reduce_ops.py"""

import sys
from pathlib import Path

# 允许从 reduce/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from reduce_ops import row_sum, sum_all


def test_sum_all():
    torch.manual_seed(1)
    for n in [1, 17, 4095, 100_000]:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        s = sum_all(x)
        ref = x.sum()
        assert torch.allclose(s, ref, rtol=1e-5, atol=1e-5), n


def test_row_sum():
    x = torch.randn(128, 777, device="cuda", dtype=torch.float16)
    r = row_sum(x)
    ref = x.float().sum(dim=-1)
    assert torch.allclose(r, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_sum_all()
    test_row_sum()
    print("reduce_ops OK")
