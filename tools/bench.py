"""Throughput of 2**24 Float64 draws: Tandem fills into a buffer and a new array, then Generator rows."""

import time

import numpy as np
from numpy.random import Generator, PCG64

from tandem_rng import Tandem

N = 2**24
BYTES = N * 8


def best(fn, runs=7):
    fn()
    t = min(timeit(fn) for _ in range(runs))
    return BYTES / t / 2**30


def timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


tandem = Tandem(42)
buf = np.empty(N)
gen_tandem = Generator(Tandem(42))
gen_pcg = Generator(PCG64(42))
rows = [
    ("Tandem(42).random(out=buf)", best(lambda: tandem.random(out=buf))),
    ("Tandem(42).random(n)", best(lambda: tandem.random(N))),
    ("Generator(Tandem(42)).random(n)", best(lambda: gen_tandem.random(N))),
    ("Generator(PCG64(42)).random(n)", best(lambda: gen_pcg.random(N))),
]
for name, gibs in rows:
    print(f"{name:36s} {gibs:6.2f} GiB/s")
