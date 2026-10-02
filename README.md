<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng"></p>

# tandem-numpy

NumPy `BitGenerator` for [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. It wraps the C
reference [tandem-c](https://github.com/tandem-rng/tandem-c) and produces the same stream, bit
for bit, as the Julia reference [TandemRNG.jl](https://github.com/tandem-rng/TandemRNG.jl).

## Use

```python
import numpy as np
from numpy.random import Generator
from tandem_rng import Tandem

rng = Generator(Tandem(42))            # any NumPy distribution
x = rng.normal(size=1000)

t = Tandem(42)
u = t.random(2**20)                    # the stream's Float64 draws, fast fill
t.random(out=u)                        # or fill a preallocated array in place
w = t.raw(2**20, np.uint32)            # unsigned words of the stream
worker = t.split(7)                    # by index, from the key alone
kids = t.fork(4)                       # from the current block, parent moves on
t.key, t.position, t.chunk_length      # transport form
```

`Tandem(seed, K=32)` accepts an integer seed in `[0, 2**128)`, a `SeedSequence`, or `None`
for OS entropy. An integer goes through the specification's seed whitening, so `Tandem(42)`
produces the stream of Julia `Tandem8x32(42)` and C `tandem_seed(42, 0, 32)`. A
`SeedSequence` or `None` reduces to 128 bits that are treated as the integer seed.
`Tandem.from_key(key, position, K)` takes the transport form directly.

NumPy's `Generator.random()` computes `(next_uint64 >> 11) * 2**-53` and
`random(dtype=np.float32)` computes `(next_uint32 >> 8) * 2**-24`. Both are the
specification's own mappings, so `Generator(Tandem(42)).random(n)` equals `Tandem(42).random(n)`
and the Julia `rand(Tandem8x32(42), Float64, n)`. The `state` property is a dict of
`key`, `position`, and `K`, and pickling goes through it.

## Install

```sh
pip install .
```

The build needs a C compiler, Cython 3, and NumPy 2. For development, `pixi install` creates
an environment with the package installed editable, and `pixi run test` runs the tests.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills and scalar draws with dumps written by
TandemRNG.jl (`tests/data`, shared with tandem-c). CI fails when the vendored C sources or
the vectors drift from their upstream repositories. `tools/sync_c.sh` refreshes the C sources.

## Speed

Apple M4, one thread, `pixi run bench`, 2^24 Float64 draws, minimum of seven runs, load
average 4.4 during the run:

| | GiB/s |
|---|---|
| `Tandem(42).random(out=buf)`, preallocated | 12.0 |
| `Tandem(42).random(n)`, new array each call | 7.4 |
| `Generator(Tandem(42)).random(n)` | 3.3 |
| `Generator(PCG64(42)).random(n)` | 2.1 |
| C `tandem_fill_f64`, from the tandem-c README | 11.3 |

The preallocated fill is the C fill with the GIL released. The allocating row pays for a
fresh 128 MiB array and its page faults on every call. The two `Generator` rows go through
NumPy's per-element `next_double` call, which is where the time goes for any BitGenerator.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
