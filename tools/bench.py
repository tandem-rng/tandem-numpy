"""Throughput in GiB/s of output at 2**22 elements, best of five, as a markdown table.

Each row draws a new array. The Tandem column calls the BitGenerator's own C fill and
TandemGenerator routes the sampler names through the fills. The Generator columns call NumPy's
own samplers over Tandem, over PCG64, and over Philox. Run it on a quiet machine.
"""

import time

import numpy as np
from numpy.random import PCG64, Generator, Philox

from tandem_rng import Tandem, TandemGenerator

N = 2**22


def best(fn, nbytes, runs=5):
    fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return nbytes / min(times) / 2**30


t = Tandem(42)
tg = TandemGenerator(42)
gt = Generator(Tandem(42))
gp = Generator(PCG64(42))
gx = Generator(Philox(42))
f32, f64 = np.float32, np.float64

# (label, bytes per element, Tandem fill, TandemGenerator, Generator over Tandem, Generator over PCG64,
# Generator over Philox)
rows = [
    ("random float64", 8, lambda: t.random(N), lambda: tg.random(N), lambda: gt.random(N), lambda: gp.random(N),
     lambda: gx.random(N)),
    ("random float32", 4, lambda: t.random(N, f32), lambda: tg.random(N, f32), lambda: gt.random(N, f32), lambda: gp.random(N, f32),
     lambda: gx.random(N, f32)),
    ("integers(0, 1000) int32", 4, lambda: t.below(1000, N, np.uint32),
     lambda: tg.integers(0, 1000, N, np.int32), lambda: gt.integers(0, 1000, N, np.int32), lambda: gp.integers(0, 1000, N, np.int32),
     lambda: gx.integers(0, 1000, N, np.int32)),
    ("integers(0, 1000) int64", 8, lambda: t.below(1000, N),
     lambda: tg.integers(0, 1000, N), lambda: gt.integers(0, 1000, N), lambda: gp.integers(0, 1000, N),
     lambda: gx.integers(0, 1000, N)),
    ("standard_normal float64", 8, lambda: t.normal(N), lambda: tg.standard_normal(N), lambda: gt.standard_normal(N),
     lambda: gp.standard_normal(N),
     lambda: gx.standard_normal(N)),
    ("standard_normal float32", 4, lambda: t.normal(N, f32), lambda: tg.standard_normal(N, f32), lambda: gt.standard_normal(N, f32),
     lambda: gp.standard_normal(N, f32),
     lambda: gx.standard_normal(N, f32)),
    ("standard_exponential float64", 8, lambda: t.exponential(N), lambda: tg.standard_exponential(N),
     lambda: gt.standard_exponential(N), lambda: gp.standard_exponential(N),
     lambda: gx.standard_exponential(N)),
    ("standard_exponential float32", 4, lambda: t.exponential(N, f32), lambda: tg.standard_exponential(N, f32),
     lambda: gt.standard_exponential(N, f32), lambda: gp.standard_exponential(N, f32),
     lambda: gx.standard_exponential(N, f32)),
    ("raw uint64 words", 8, lambda: t.raw(N), None, lambda: gt.bit_generator.random_raw(N),
     lambda: gp.bit_generator.random_raw(N),
     lambda: gx.bit_generator.random_raw(N)),
]
print("| 2^22 elements | Tandem fill | TandemGenerator | Generator(Tandem) | Generator(PCG64) | Generator(Philox) |")
print("|---|---|---|---|---|---|")
for name, width, *fns in rows:
    print(f"| {name} | " + " | ".join("-" if f is None else f"{best(f, N * width):.1f}" for f in fns) + " |")
