# Speed

`pixi run bench` produces the figures.

## CPU

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

The Tandem column calls the C fills of the bit generator with the GIL released: `random`,
`below(1000, n, dtype)` at the dtype's width, `normal`, `exponential`, and `raw`. `TandemGenerator` routes the sampler names through
the same fills. The plain `Generator` columns are NumPy's own samplers: `Generator` calls the bit
generator one value at a time, through `next_double`, `next_uint32`, and `next_uint64`, and runs
its own ziggurat for normals and its own Lemire method for integers. Its speed is bounded by
that call, and its normals and large-range integers are NumPy's values, not Tandem's. The hooks
fill a buffer of 1024 words at a time, with the stream's alignment rules kept for mixed widths.
