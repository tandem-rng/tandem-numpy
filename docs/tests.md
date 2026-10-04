# Tests

```sh
pixi run test     # tests/test_tandem.py
```

## Suite

`tests/test_tandem.py` checks:

- Every specification vector (`tests/vectors.json`) and the stream dumps in `tests/data`.
- `below`, `normal`, and `exponential` against the tandem-cuda fixtures in
  `external/tandem-c/tests`.
- The `standard_normal` and `standard_exponential` fills against hashes of the dumps from
  tandem-c's `tools/dump_normals.c` and `tools/dump_exponentials.c`.
- Fills cut at any element boundary equal the whole fill.

`tests/test_tandem.py` checks every vector of the specification (`tests/vectors.json`, a copy
of the spec repository's file) and compares fills and scalar draws with reference stream dumps in `tests/data`.
The new paths are checked the same way: `fill`, `u128`, and `char` against the dumps of every
specification type, `at` and `advance_to` against the dumps and against fills from the same
position, `spawn` against `split`, and `below`, `normal`, and `exponential` against the tandem-cuda
fixtures in `external/tandem-c/tests` (`cross_below.h`, `cross_fill_below.h`, `cross_normal.h`,
`cross_exponential.h`), which the test parses. The normal and exponential fixtures must match bit for bit. `TandemGenerator` is checked against the same fills and fixtures, including `integers` calls of every dtype, with and without `endpoint`, cut at arbitrary
boundaries against the whole call. A hash test compares the `standard_normal` float64 fills with
the bytes of tandem-c's `tools/dump_normals.c` (SHA-256 `700ec4d2…`) and the float32 fills with
the bytes tandem-c's `tests/test_normal_bits.c` hashes (FNV-1a `0xaa1ea656ce73a4fb`, checked here
as SHA-256), so they are the same bits on every compiler. The `standard_exponential` fills have the
same test against `tools/dump_exponentials.c` (FNV-1a `0x47f8f98297d94ee2`). Fills cut at element
boundaries, `n = 0`, the Exp(1) moments to fourth order, and a Kolmogorov-Smirnov test on 10^7
draws cover the rest.

## Fixtures

`tools/bump.sh` moves the pin to the latest main.
