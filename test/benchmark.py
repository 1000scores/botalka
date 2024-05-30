import torch
import timeit


def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f'device={device}')
x = torch.randn(10000, 1024, device=device)

t0 = timeit.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

# Ran each twice to show difference before/after warm-up
n = 10
_mul_sum = 0
_bmm = 0
for _ in range(n):
    _mul_sum += t0.timeit(100) / 100 * 1e6
    _mul_sum += t0.timeit(100) / 100 * 1e6
    _bmm += t1.timeit(100) / 100 * 1e6
    _bmm += t1.timeit(100) / 100 * 1e6

_mul_sum /= 2. * n
_bmm /= 2. * n
print(f'mul_sum(x, x):  {_mul_sum:>5.1f} us')
print(f'bmm(x, x):      {_bmm:>5.1f} us')