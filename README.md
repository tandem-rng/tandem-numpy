<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .np"></p>

# tandem-numpy

NumPy `BitGenerator` for [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. It wraps the reference C
implementation and produces the stream the specification defines, bit for bit.

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
t.fill(np.empty((1000, 3), np.complex64))   # any spec type, any contiguous shape
t.u128(1000)                           # (1000, 2) uint64 rows: low and high half
t.char(1000)                           # Unicode scalar values as uint32
t.below(10, 1000)                      # Tandem's bounded integers on [0, 10), uint64
t.normal(1000, np.float32)             # Tandem's Box-Muller normals
worker = t.split(7)                    # by index, from the key alone
kids = t.fork(4)                       # from the current block, parent moves on
sub = t.sub(3)                         # by purpose identifier
streams = Generator(t).spawn(4)        # Generators over t.split(0) .. t.split(3)
t.key, t.position, t.chunk_length      # transport form
t.at(np.float64, 10**12)               # random access: element i of the next fill
t.advance_to(2**40)                    # seek to a bit position, same as t.position = 2**40
```

`Tandem(seed, K=32)` accepts an integer seed in `[0, 2**128)`, a `SeedSequence`, or `None`
for OS entropy. An integer goes through the specification's seed whitening, so `Tandem(42)`
produces the specification's stream for seed 42. A
`SeedSequence` or `None` reduces to 128 bits that are treated as the integer seed.
`Tandem.from_key(key, position, K)` takes the transport form directly.

`split`, `sub`, and `fork` return children at position 0 with the parent's `K`. `spawn(n)`,
which `Generator.spawn` calls, overrides NumPy's `SeedSequence` spawning with the
specification's split: the first call returns `split(0)` to `split(n - 1)`, and later calls
continue the numbering. The children have no seed sequence, so their `_seed_seq` is `None`.
The count of earlier `spawn` calls is not part of `state` or of a pickle.

`fill(out)` fills a writeable, C-contiguous, native-endian array of any shape in place with the
specification's draws of its dtype, through the C fills with the GIL released. It accepts
`bool`, `int8` to `int64` and `uint8` to `uint64` (the unsigned draw's bits), `float16`,
`float32`, `float64`, `complex64`, `complex128`, and `V16` for 128-bit words (low half first).
`u128(size)` returns the same words as `(size, 2)` `uint64` rows and `char(size)` returns Unicode
scalar values as `uint32`. `random` and `raw` are unchanged.

`below(n, size, dtype)` and `normal(size, dtype)` are Tandem's own bounded-integer and normal
contracts, taken from tandem-c and tandem-cuda and not part of the specification. They are not
`Generator.integers` and `Generator.standard_normal`, which keep NumPy's algorithms over the bit
generator and give different values.

- `below` draws on `[0, n)` by Lemire's multiply-and-reject over `uint32` or `uint64` draws
  (`dtype`, default `uint64`). With `size` or `out` it draws element `i` from stream draw `i`
  and retries a rejected draw on a fallback generator, so it uses exactly one draw per element
  and the position advances by that. Without them it is one scalar draw, which takes as many
  draws as the rejection loop needs. `n = 0` returns 0.
- `normal` is Box-Muller on pairs of uniform draws `a`, `b` of the dtype: with
  `r = sqrt(-2 ln(1 - a))` the pair is `(r cos 2 pi b, r sin 2 pi b)`. Pair `j` gives elements
  `2j` (cos half) and `2j + 1` (sin half) from uniforms `2j` and `2j + 1`. An odd `size` keeps
  the cos half of its last pair and still consumes both uniforms. A scalar draw is the cos half
  and consumes two uniforms, so it equals element 0 of a fill. Values
  agree with other ports to about `1e-12` relative for `float64` and a few ulps for `float32`,
  because libm differs.

`at(dtype, i)` returns element `i` of the fill that would start at the current position, for
`uint32`, `uint64`, `float32`, and `float64`, without moving the generator. `advance_to(p)` and
the `position` setter move to bit position `p` in `[0, 2**63)`, forward or backward, and discard
the draws that `Generator` has buffered. `state` reports the new position.

NumPy's `Generator.random()` computes `(next_uint64 >> 11) * 2**-53` and
`random(dtype=np.float32)` computes `(next_uint32 >> 8) * 2**-24`. Both are the
specification's own mappings, so `Generator(Tandem(42)).random(n)` equals `Tandem(42).random(n)`. The `state` property is a dict of
`key`, `position`, and `K`, and pickling goes through it.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## Install

```sh
pip install .
```

The reference C implementation sits in the `external/tandem-c` git submodule. Clone with
`git clone --recurse-submodules`, or run `git submodule update --init` in an existing clone.
GitHub's ZIP download omits submodules and does not build.

The build uses meson-python and needs a C compiler. For development, `pixi install` creates an
environment with the package installed editable, which rebuilds the extension on import when
the sources change, and `pixi run test` runs the tests.

## Tests

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills and scalar draws with reference stream dumps in `tests/data`.
The new paths are checked the same way: `fill`, `u128`, and `char` against the dumps of every
specification type, `at` and `advance_to` against the dumps and against fills from the same
position, `spawn` against `split`, and `below` and `normal` against the tandem-cuda fixtures in
`external/tandem-c/tests` (`cross_below.h`, `cross_fill_below.h`, `cross_normal.h`), which the test
parses.
CI fails when the vectors drift from upstream or the tandem-c pin is not on tandem-c main.
`tools/bump.sh` moves the pin to the latest main.

## Speed

Apple M4, one thread, `pixi run bench`, 2^24 elements per call, minimum of seven runs, GiB/s of
output. Every call allocates its array.

| | Tandem fill | `Generator(Tandem(42))` | `Generator(PCG64(42))` |
|---|---|---|---|
| `random` float64 | 9.4 | 4.1 | 2.2 |
| `random` float32 | 16.4 | 2.9 | 2.3 |
| `integers(0, 1000)` int32 | 7.7 | 3.0 | 2.3 |
| `integers(0, 1000)` int64 | 5.6 | 4.8 | 3.7 |
| `standard_normal` float64 | 3.9 | 1.9 | 1.9 |
| `standard_normal` float32 | 5.4 | 1.1 | 1.5 |
| raw `uint64` words | 10.5 | 3.9 | 2.1 |

The Tandem column calls the C fills of the bit generator with the GIL released: `random`,
`below(1000, n, dtype)`, `normal`, and `raw`. The `Generator` columns are NumPy's own samplers.
`Generator` calls the bit generator one value at a time, through `next_double`, `next_uint32`,
and `next_uint64`, and runs its own ziggurat for normals and its own Lemire method for integers.
Its speed is therefore bounded by that call and does not use the C fills, and its integers and
normals are NumPy's values, not Tandem's. The hooks fill a buffer of 1024 words at a time, with
the stream's alignment rules kept for mixed widths.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
