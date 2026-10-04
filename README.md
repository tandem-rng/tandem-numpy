<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .np"></p>

# tandem-numpy

NumPy `BitGenerator` for [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator. It wraps the reference C implementation and produces the stream
the specification defines, bit for bit, fast on CPUs.

## Install

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-numpy
pip install ./tandem-numpy
```

Needs Python 3.11, NumPy 2.0, a C compiler, and meson-python. The C code is the `external/tandem-c` submodule, pinned at
tandem-c `b049384`. For development, `pixi install` then `pixi run test`.

## Use

```python
import numpy as np
from numpy.random import Generator
from tandem_rng import Tandem, TandemGenerator

rng = Generator(Tandem(42))            # any NumPy distribution
x = rng.normal(size=1000)

t = Tandem(42)
u = t.random(2**20)                    # the stream's Float64 draws, fast fill
worker = t.split(7)                    # by index, from the key alone
kids = t.fork(4)                       # from the current block, parent moves on
sub = t.sub(3)                         # by purpose identifier
t.normal(1000, np.float32)             # Tandem's Box-Muller normals

g = TandemGenerator(42)                # NumPy Generator with Tandem's own samplers, fast
g.standard_normal(10**6); g.integers(0, 1000, 10**6); g.standard_exponential(10**6)
```

## What it provides

- `Tandem(seed, K=32)`: the bit generator. The seed is an integer in `[0, 2**128)`, a
  `SeedSequence`, or `None`. `Tandem.from_key(key, position, K)` takes the transport form.
- `random`, `raw`: Float64 or Float32 draws and unsigned words, fast fill, `out=` supported.
- `fill(out)`: any spec type (`bool`, `int8` to `uint64`, `float16` to `float64`, `complex64`,
  `complex128`, `V16`) in any contiguous shape, GIL released.
- `u128`, `char`: 128-bit words as `(size, 2)` `uint64` rows, and Unicode scalar values.
- `below(n, size, dtype)`: bounded integers on `[0, n)`, one draw per element.
- `normal`, `exponential`: Box-Muller normals and `-ln(1 - u)` exponentials, `float64` or `float32`.
- `split`, `fork`, `sub`, `spawn`: child streams. `Generator.spawn` uses `split(0)`, `split(1)`, ...
- `at(dtype, i)`, `advance_to(p)`, `position`, `key`, `chunk_length`, `state`: random access
  and transport. Pickling goes through `state`.
- `TandemGenerator(seed, K=32)`: a `Generator` whose `random`, `uniform`, `standard_normal`,
  `normal`, `standard_exponential`, `exponential`, and `integers` use the C fills and give the
  cross-port values. Other methods fall through to NumPy.
- Parallel use: ranks, threads, or devices that start at their first element, or draw from
  `split(task)`, reproduce a serial run. See
  [Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative).

`Generator(Tandem(seed))` keeps NumPy's ziggurat and Lemire code over the same stream, so its
normals and integers differ from `TandemGenerator`.

## Tests

`pixi run test` runs `tests/test_tandem.py`. It checks:

- Every specification vector (`tests/vectors.json`) and the stream dumps in `tests/data`.
- `below`, `normal`, and `exponential` against the tandem-cuda fixtures in `external/tandem-c/tests`.
- The `standard_normal` and `standard_exponential` fills against hashes of the dumps from
  tandem-c's `tools/dump_normals.c` and `tools/dump_exponentials.c`.
- Fills cut at any element boundary equal the whole fill.

## Speed

Apple M4, one thread, `pixi run bench`, 2^22 elements per call, minimum of five runs, GiB/s of
output. Every call allocates its array.

| | Tandem fill | `TandemGenerator(42)` | `Generator(Tandem(42))` | `Generator(PCG64(42))` |
|---|---|---|---|---|
| `random` float64 | 16.7 | 16.5 | 4.9 | 2.5 |
| `random` float32 | 16.5 | 16.6 | 3.1 | 2.4 |
| `integers(0, 1000)` int32 | 7.8 | 7.7 | 3.1 | 2.3 |
| `integers(0, 1000)` int64 | 7.5 | 12.7 | 5.9 | 4.5 |
| `standard_normal` float64 | 4.9 | 4.9 | 2.2 | 2.0 |
| `standard_normal` float32 | 5.5 | 5.5 | 1.1 | 1.5 |
| `standard_exponential` float64 | 6.0 | 6.0 | 3.8 | 2.0 |
| `standard_exponential` float32 | 6.6 | 6.6 | 1.2 | 1.3 |
| raw `uint64` words | 18.6 | - | 4.9 | 2.4 |

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
