# tandem-numpy

NumPy `BitGenerator` for Tandem8x32. It wraps tandem-c and produces the stream of the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for bit.

- [API](api.md): `Tandem`, `TandemGenerator` and parallel use.
- [Design](design.md): the bounded integer, normal and exponential contracts.
- [Tests](tests.md): what the suite checks.
- [Speed](speed.md): fill and `Generator` figures on the Apple M4.

## Install

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-numpy
pip install ./tandem-numpy
```

Needs Python 3.11, NumPy 2.0, a C compiler, and meson-python.
The C code is the `external/tandem-c` submodule, pinned at tandem-c `ef67bd7`.
For development, `pixi install` then `pixi run test`.

The reference C implementation sits in the `external/tandem-c` git submodule. Clone with
`git clone --recurse-submodules`, or run `git submodule update --init` in an existing clone.
GitHub's ZIP download omits submodules and does not build.

The submodule is pinned at tandem-c `ef67bd7`. The extension is built with `-ffp-contract=off` and no
`-mfma`: the normal loop uses explicit fused multiply-adds, so its bits do not depend on the
compiler, and on x86 the AVX2 and FMA copy is chosen at run time, also in a wheel built for a
baseline x86-64.

The build uses meson-python and needs a C compiler. For development, `pixi install` creates an
environment with the package installed editable, which rebuilds the extension on import when
the sources change, and `pixi run test` runs the tests.


Normals are bit exact with tandem-c.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
