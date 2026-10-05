# API

## Use

```python
import numpy as np
from numpy.random import Generator
from tandem_rng import Tandem, TandemGenerator

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
t.exponential(1000)                    # Tandem's standard exponentials, float64 or float32
worker = t.split(7)                    # by index, from the key alone
kids = t.fork(4)                       # from the current block, parent moves on
sub = t.sub(3)                         # by purpose identifier
streams = Generator(t).spawn(4)        # Generators over t.split(0) .. t.split(3)
g = TandemGenerator(42)                # NumPy Generator with Tandem's own samplers, fast
g.standard_normal(10**6); g.integers(0, 1000, 10**6); g.standard_exponential(10**6)
t.key, t.position, t.chunk_length      # transport form
t.at(np.float64, 10**12)               # random access: element i of the next fill
t.advance_to(2**40)                    # seek to a bit position, same as t.position = 2**40
```

## Reference

- `Tandem(seed, K=32)`: the bit generator. The seed is an integer in `[0, 2**128)`, a
  `SeedSequence`, or `None`. `Tandem.from_key(key, position, K)` takes the transport form.
- `random`, `raw`: Float64 or Float32 draws and unsigned words, fast fill, `out=` supported.
- `fill(out)`: any spec type (`bool`, `int8` to `uint64`, `float16` to `float64`, `complex64`,
  `complex128`, `V16`) in any contiguous shape, GIL released.
- `u128`, `char`: 128-bit words as `(size, 2)` `uint64` rows, and Unicode scalar values.
- `below(n, size, dtype)`: bounded integers on `[0, n)`, one draw per element.
- `normal`, `exponential`: normals (ziggurat for `float64`, Box-Muller for `float32`) and
  `-ln(1 - u)` exponentials, `float64` or `float32`.
- `ChoiceTable(weights)`, `choice(table, size)`: weighted indices on `[0, m)` by the alias
  table of Appendix C, one 64-bit draw each, `uint32`. Build the table once and reuse it.
  `TandemGenerator.choice(a, size, p=...)` takes `p` as an array or a `ChoiceTable`.
- `split`, `fork`, `sub`, `spawn`: child streams. `Generator.spawn` uses `split(0)`, `split(1)`, ...
- `at(dtype, i)`, `advance_to(p)`, `position`, `key`, `chunk_length`, `state`: random access
  and transport. Pickling goes through `state`.
- `TandemGenerator(seed, K=32)`: a `Generator` whose `random`, `uniform`, `standard_normal`,
  `normal`, `standard_exponential`, `exponential`, and `integers` use the C fills and give the
  cross-port values. Other methods fall through to NumPy.

`Generator(Tandem(seed))` keeps NumPy's ziggurat and Lemire code over the same stream, so its
normals and integers differ from `TandemGenerator`.

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

`below(n, size, dtype)`, `normal(size, dtype)`, and `exponential(size, dtype)` are Tandem's own
bounded-integer, normal, and exponential contracts, taken from tandem-c and tandem-cuda and not
part of the specification. They are not `Generator.integers`, `Generator.standard_normal`, and
`Generator.standard_exponential`, which keep NumPy's algorithms over the bit generator and give
different values. [Design](design.md) gives the three contracts.

`TandemGenerator(seed, K=32)` is a `numpy.random.Generator` over `Tandem` that overrides
`random`, `uniform`, `standard_normal`, `normal`, `standard_exponential`, `exponential`, and
`integers` with the C fills, with NumPy's
signatures, `dtype`, `size`, `out`, and `endpoint` handling. Its values are the cross-port ones
of Appendix A: Tandem's normals and Lemire integers with the fallback stream, equal in every
Tandem port, and they run at the speed of the fills. `Generator(Tandem(seed))` is the other
choice: it keeps NumPy's ziggurat and Lemire code over the same stream, so its normals differ.
Every method that `TandemGenerator` does not override falls through to NumPy and reads the bit
generator one value at a time. The overridden methods move the shared bit generator exactly as
the matching `Tandem` fills do, and `normal` and `uniform` are `loc + scale * standard_normal`
and `low + (high - low) * random`. `exponential(scale)` is `scale * standard_exponential`, and
`standard_exponential` ignores `method`, since its values are the inversion `-ln(1 - u)` whatever
the name. `integers` takes the draw width from the range, 32-bit words
when the range is at most `2**32` and 64-bit words otherwise, as the specification says, so the
dtype does not change the values: `int64` with a small range uses the `uint32` bounded fill and
widens. It supports `bool`, and array-valued bounds fall through to NumPy's own `integers`. NumPy's
integers need not match them.

`at(dtype, i)` returns element `i` of the fill that would start at the current position, for
`uint32`, `uint64`, `float32`, and `float64`, without moving the generator. `advance_to(p)` and
the `position` setter move to bit position `p` in `[0, 2**63)`, forward or backward, and discard
the draws that `Generator` has buffered. `state` reports the new position.

NumPy's `Generator.random()` computes `(next_uint64 >> 11) * 2**-53` and
`random(dtype=np.float32)` computes `(next_uint32 >> 8) * 2**-24`. Both are the
specification's own mappings, so `Generator(Tandem(42)).random(n)` equals `Tandem(42).random(n)`. The `state` property is a dict of
`key`, `position`, and `K`, and pickling goes through it.

## Parallel use

Element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.
