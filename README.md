<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .np"></p>

# tandem-numpy

[![CI](https://github.com/tandem-rng/tandem-numpy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-numpy/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-numpy/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

NumPy `BitGenerator` for [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator. It wraps tandem-c and produces the specified stream bit for bit.
Normals are bit exact with tandem-c since tandem-c `09615e0`.

Needs Python 3.11, NumPy 2.0, a C compiler, and meson-python. The `external/tandem-c` submodule
is pinned at tandem-c `b049384`.

```sh
git clone --recurse-submodules https://github.com/tandem-rng/tandem-numpy
pip install ./tandem-numpy
```

```python
from numpy.random import Generator
from tandem_rng import Tandem, TandemGenerator

t = Tandem(42)
u = t.random(2**20)                    # the stream's Float64 draws, fast fill
worker = t.split(7)                    # by index, from the key alone
g = TandemGenerator(42)                # NumPy Generator with Tandem's own samplers, fast
z = g.standard_normal(10**6)           # Box-Muller, bit identical to tandem-c
rng = Generator(Tandem(42))            # any NumPy distribution over the same stream
```

See [API](docs/api.md), [design](docs/design.md), [tests](docs/tests.md) and [speed](docs/speed.md).

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-numpy/) · [Apache 2.0 license](LICENSE)
