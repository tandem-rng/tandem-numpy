<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .np"></p>

# tandem-numpy

NumPy `BitGenerator` for [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. It wraps a vendored
copy of the reference C implementation and produces the stream the specification defines, bit
for bit.

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
produces the specification's stream for seed 42. A
`SeedSequence` or `None` reduces to 128 bits that are treated as the integer seed.
`Tandem.from_key(key, position, K)` takes the transport form directly.

NumPy's `Generator.random()` computes `(next_uint64 >> 11) * 2**-53` and
`random(dtype=np.float32)` computes `(next_uint32 >> 8) * 2**-24`. Both are the
specification's own mappings, so `Generator(Tandem(42)).random(n)` equals `Tandem(42).random(n)`. The `state` property is a dict of
`key`, `position`, and `K`, and pickling goes through it.

## Install

```sh
pip install .
```

The build uses meson-python and needs a C compiler. For development, `pixi install` creates an
environment with the package installed editable, which rebuilds the extension on import when
the sources change, and `pixi run test` runs the tests.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills and scalar draws with reference stream dumps in `tests/data`.
CI fails when the vendored C sources or the vectors drift from upstream. `tools/sync_c.sh` refreshes the C sources.

## Speed

Apple M4, one thread, `pixi run bench`, 2^24 Float64 draws, minimum of seven runs:

| | GiB/s |
|---|---|
| `Tandem(42).random(out=buf)`, preallocated | 13.5 |
| `Tandem(42).random(n)`, new array each call | 9.1 |
| `Generator(Tandem(42)).random(n)` | 4.4 |
| `Generator(PCG64(42)).random(n)` | 2.5 |

The preallocated fill is the C fill with the GIL released. The allocating row pays for a
fresh 128 MiB array and its page faults on every call. The two `Generator` rows go through
NumPy's per-element `next_double` call, which bounds any BitGenerator. The hooks fill a
buffer of 1024 words at a time, with the stream's alignment rules kept for mixed widths.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
