# Speed

`pixi run bench` produces the figures.

## CPU

Apple M4, one thread, `pixi run bench`, 2^22 elements per call, GiB/s of output, the median of
three passes of the minimum of five runs. Every call allocates its array.

| | Tandem fill | `TandemGenerator(42)` | `Generator(Tandem(42))` | `Generator(PCG64(42))` | `Generator(Philox(42))` |
|---|---|---|---|---|---|
| `random` float64 | 16.5 | 16.5 | 5.0 | 2.4 | 2.8 |
| `random` float32 | 16.6 | 16.4 | 3.0 | 2.3 | 1.9 |
| `integers(0, 1000)` int32 | 7.7 | 7.7 | 3.1 | 2.3 | 1.9 |
| `integers(0, 1000)` int64 | 7.5 | 12.7 | 5.8 | 4.5 | 3.8 |
| `standard_normal` float64 | 7.6 | 7.7 | 2.2 | 2.0 | 2.0 |
| `standard_normal` float32 | 5.5 | 5.5 | 1.1 | 1.5 | 1.0 |
| `standard_exponential` float64 | 6.1 | 6.1 | 3.9 | 2.1 | 2.5 |
| `standard_exponential` float32 | 6.7 | 6.6 | 1.2 | 1.3 | 1.0 |
| raw `uint64` words | 19.1 | - | 4.9 | 2.4 | 3.1 |

The Tandem column calls the C fills of the bit generator with the GIL released: `random`,
`below(1000, n, dtype)` at the dtype's width, `normal`, `exponential`, and `raw`. `TandemGenerator` routes the sampler names through
the same fills. The plain `Generator` columns are NumPy's own samplers, over Tandem, over the default PCG64,
and over Philox4x32-10, the counter-based generator of the GPU libraries: `Generator` calls the bit
generator one value at a time, through `next_double`, `next_uint32`, and `next_uint64`, and runs
its own ziggurat for normals and its own Lemire method for integers. Its speed is bounded by
that call, and its normals and large-range integers are NumPy's values, not Tandem's. The hooks
fill a buffer of 1024 words at a time, with the stream's alignment rules kept for mixed widths.
