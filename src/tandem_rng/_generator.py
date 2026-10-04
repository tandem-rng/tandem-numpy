"""A NumPy Generator whose samplers are Tandem's C fills."""

import operator

import numpy as np
from numpy.random import Generator

from ._tandem import Tandem

_UNSIGNED = {32: np.uint32, 64: np.uint64}


def _shape(size):
    return (operator.index(size),) if np.ndim(size) == 0 else tuple(size)


class TandemGenerator(Generator):
    """NumPy ``Generator`` that draws through Tandem's C fills.

    ``TandemGenerator(seed, K=32)`` takes what ``Tandem`` takes, or a ``Tandem`` instance.
    ``random``, ``uniform``, ``standard_normal``, ``normal`` and ``integers`` return the
    values of Appendix A of the specification, which every Tandem port returns, and run at
    the speed of the C fills. ``standard_exponential`` and ``exponential`` return Tandem's
    inversion values, ``-ln(1 - u)`` with the same bits in every Tandem port, for float64 and
    float32, whatever ``method`` says. They are not NumPy's ziggurat and Lemire values:
    ``Generator(Tandem(seed))`` keeps NumPy's algorithms over the same stream. Every other
    method falls through to NumPy and reads the bit generator one value at a time.

    The overridden methods move the shared bit generator exactly as the matching
    ``Tandem`` fills do. ``integers`` draws 32-bit words when the range is at most 2**32 and
    64-bit words otherwise, whatever the dtype, so ``dtype`` does not change the values. Its
    argument handling, result types and errors are NumPy's. Array-valued bounds fall
    through to NumPy's own ``integers``, since Tandem's bounded fill takes one range.
    """

    def __init__(self, seed=None, K=32):
        super().__init__(seed if isinstance(seed, Tandem) else Tandem(seed, K))

    def __reduce__(self):
        return (type(self), (self.bit_generator,))

    def _fill_into(self, draw, size, dtype, out):
        """Run ``draw(size, dtype, out)``, with a flat view for a contiguous ``out``."""
        dtype = np.dtype(dtype)
        if out is None:
            return draw(size, dtype, None)
        if not isinstance(out, np.ndarray) or out.dtype != dtype:
            raise TypeError("out must be an ndarray of the requested dtype")
        if size is not None and out.shape != _shape(size):
            raise ValueError("size does not match the shape of out")
        if out.flags.c_contiguous and out.flags.writeable and out.dtype.isnative:
            draw(None, dtype, out.reshape(-1))
        else:
            out[...] = draw(out.size, dtype, None).reshape(out.shape)
        return out

    def random(self, size=None, dtype=np.float64, out=None):
        return self._fill_into(self.bit_generator.random, size, dtype, out)

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        return self._fill_into(self.bit_generator.normal, size, dtype, out)

    def standard_exponential(self, size=None, dtype=np.float64, method="zig", out=None):
        # The values are the inversion -ln(1 - u) whatever ``method`` says. NumPy accepts
        # any value too and takes every name but "zig" as inversion.
        return self._fill_into(self.bit_generator.exponential, size, dtype, out)

    def uniform(self, low=0.0, high=1.0, size=None):
        low, high = np.asarray(low, np.float64), np.asarray(high, np.float64)
        shape = np.broadcast(low, high).shape if size is None else size
        u = self.random(shape if shape != () else None)
        r = low + (high - low) * u
        return r[()] if size is None and np.ndim(r) == 0 else r

    def normal(self, loc=0.0, scale=1.0, size=None):
        loc, scale = np.asarray(loc, np.float64), np.asarray(scale, np.float64)
        if (scale < 0).any():
            raise ValueError("scale < 0")
        shape = np.broadcast(loc, scale).shape if size is None else size
        z = self.standard_normal(shape if shape != () else None)
        r = loc + scale * z
        return r[()] if size is None and np.ndim(r) == 0 else r

    def exponential(self, scale=1.0, size=None):
        scale = np.asarray(scale, np.float64)
        if (scale < 0).any():
            raise ValueError("scale < 0")
        shape = scale.shape if size is None else size
        e = self.standard_exponential(shape if shape != () else None)
        r = scale * e
        return r[()] if size is None and np.ndim(r) == 0 else r

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if np.ndim(low) or np.ndim(high):
            return super().integers(low, high, size, dtype, endpoint)
        python_type = dtype if dtype is bool or dtype is int else None
        dtype = np.dtype(dtype)
        if not dtype.isnative:
            raise ValueError(
                "Providing a dtype with a non-native byteorder is not supported. If you require "
                "platform-independent byteorder, call byteswap when required."
            )
        if dtype.kind not in "iub":
            raise TypeError(f"Unsupported dtype {dtype!r} for integers")
        if size is not None and np.prod(size) == 0:
            return np.empty(size, dtype)
        # NumPy truncates scalar bounds with int(), floats included.
        low = int(np.asarray(low))
        low, high = (0, low) if high is None else (low, int(np.asarray(high)))
        top = high if endpoint else high - 1
        dmin, dmax = (0, 1) if dtype.kind == "b" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if low < dmin:
            raise ValueError(f"low is out of bounds for {dtype}")
        if top > dmax:
            raise ValueError(f"high is out of bounds for {dtype}")
        if low > top:
            raise ValueError("low > high" if endpoint else "high <= 0" if low == 0 else "low >= high")
        span = top - low + 1
        # The range alone picks the draw width, so the dtype does not change the values.
        width = 32 if span <= 2**32 else 64
        word = _UNSIGNED[width]
        bg = self.bit_generator
        # A span of 2**width needs no rejection: every word is in range.
        r = bg.raw(size, word) if span == 2**width else bg.below(span, size, word)
        if size is None:
            # NumPy returns a Python scalar when dtype is the type bool or int.
            return (python_type or dtype.type)(int(r) + low)
        bits = 8 * dtype.itemsize
        if bits > width:
            r = r.astype(np.uint64)
            width = 64
        if low:
            r += r.dtype.type(low % 2**width)
        return r.view(dtype) if bits == width and dtype.kind != "b" else r.astype(dtype)
