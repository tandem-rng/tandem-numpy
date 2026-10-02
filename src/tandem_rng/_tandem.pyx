# cython: language_level=3, boundscheck=False, wraparound=False
"""Tandem8x32 as a NumPy BitGenerator over the reference C implementation."""

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

import numpy as np
cimport numpy as np
from numpy.random cimport bitgen_t
from numpy.random.bit_generator cimport BitGenerator
from numpy.random import SeedSequence

np.import_array()

cdef extern from "tandem.h":
    ctypedef struct tandem_rng:
        uint64_t pos
    tandem_rng tandem_from_key(const uint32_t key[4], uint64_t pos, uint32_t K) nogil
    tandem_rng tandem_seed(uint64_t seed_lo, uint64_t seed_hi, uint32_t K) nogil
    void tandem_key(const tandem_rng *rng, uint32_t key[4]) nogil
    uint64_t tandem_position(const tandem_rng *rng) nogil
    uint32_t tandem_chunk_length(const tandem_rng *rng) nogil
    uint32_t tandem_next_u32(tandem_rng *rng) nogil
    uint64_t tandem_next_u64(tandem_rng *rng) nogil
    double tandem_next_f64(tandem_rng *rng) nogil
    void tandem_fill_u8(tandem_rng *rng, uint8_t *out, size_t n) nogil
    void tandem_fill_u16(tandem_rng *rng, uint16_t *out, size_t n) nogil
    void tandem_fill_u32(tandem_rng *rng, uint32_t *out, size_t n) nogil
    void tandem_fill_u64(tandem_rng *rng, uint64_t *out, size_t n) nogil
    void tandem_fill_f32(tandem_rng *rng, float *out, size_t n) nogil
    void tandem_fill_f64(tandem_rng *rng, double *out, size_t n) nogil
    tandem_rng tandem_split(const tandem_rng *rng, uint64_t index) nogil
    tandem_rng tandem_sub(const tandem_rng *rng, uint64_t purpose) nogil
    void tandem_fork(tandem_rng *parent, tandem_rng *children, uint64_t n) nogil

# NumPy asks the bit generator for one value at a time through these hooks. A buffer of
# 64-bit words amortises the per-call cost of the generator. The values and their order
# are those of tandem_next_u64 and tandem_next_u32: a 32-bit draw takes the low half of a
# word and keeps the high half for the next 32-bit draw, which is the stream's alignment
# rule, and a 64-bit draw after a pending half discards it, as alignment to 64 bits does.
cdef enum:
    BUF_WORDS = 1024

cdef struct buffered:
    tandem_rng rng
    uint64_t buf[BUF_WORDS]
    size_t left
    uint32_t pending
    bint has_pending

cdef inline uint64_t next_u64(void *st) noexcept nogil:
    cdef buffered *b = <buffered *>st
    b.has_pending = False
    if b.left == 0:
        tandem_fill_u64(&b.rng, b.buf, BUF_WORDS)
        b.left = BUF_WORDS
    b.left -= 1
    return b.buf[BUF_WORDS - 1 - b.left]

cdef uint32_t next_u32(void *st) noexcept nogil:
    cdef buffered *b = <buffered *>st
    cdef uint64_t w
    if b.has_pending:
        b.has_pending = False
        return b.pending
    if b.left == 0 and (tandem_position(&b.rng) & 63) != 0:
        # Only from a 64-bit boundary does a buffered word read the same bits as the C draw.
        return tandem_next_u32(&b.rng)
    w = next_u64(st)
    b.pending = <uint32_t>(w >> 32)
    b.has_pending = True
    return <uint32_t>w

cdef double next_f64(void *st) noexcept nogil:
    return <double>(next_u64(st) >> 11) * (1.0 / 9007199254740992.0)

cdef inline uint64_t logical_position(buffered *b) noexcept nogil:
    return tandem_position(&b.rng) - 64 * b.left - (32 if b.has_pending else 0)

cdef inline void flush(buffered *b) noexcept nogil:
    # Put the C generator at the position the buffered draws have reached.
    b.rng.pos = logical_position(b)
    b.left = 0
    b.has_pending = False


cdef class Tandem(BitGenerator):
    """Tandem8x32 BitGenerator.

    ``seed`` is an integer in [0, 2**128), a ``SeedSequence``, or ``None`` for
    OS entropy. An integer goes through the specification's seed whitening, so
    ``Tandem(42)`` produces the stream of Julia ``Tandem8x32(42)`` and C
    ``tandem_seed(42, 0, K)``. ``K`` is the chunk length, a power of two in
    [1, 65536].
    """

    cdef buffered st

    def __init__(self, seed=None, K=32):
        BitGenerator.__init__(self, seed)
        if not isinstance(seed, int):
            # Entropy and SeedSequence both reduce to 128 bits treated as an integer seed.
            words = self._seed_seq.generate_state(4, np.uint32)
            seed = int.from_bytes(words.tobytes(), "little")
        if not 0 <= seed < 2**128:
            raise ValueError("seed must lie in [0, 2**128)")
        self.st.rng = tandem_seed(seed & (2**64 - 1), seed >> 64, _check_K(K))
        self.st.left = 0
        self.st.has_pending = False
        self._bitgen.state = &self.st
        self._bitgen.next_uint64 = &next_u64
        self._bitgen.next_uint32 = &next_u32
        self._bitgen.next_double = &next_f64
        self._bitgen.next_raw = &next_u64

    @classmethod
    def from_key(cls, key, position=0, K=32):
        """Generator with the given four 32-bit key words at a bit position."""
        rng = cls(0, K)
        rng.state = {"bit_generator": "Tandem", "key": list(key), "position": position, "K": K}
        return rng

    @staticmethod
    cdef Tandem _wrap(tandem_rng s):
        cdef Tandem rng = Tandem(0, tandem_chunk_length(&s))
        rng.st.rng = s
        return rng

    @property
    def key(self):
        cdef uint32_t k[4]
        tandem_key(&self.st.rng, k)
        return (k[0], k[1], k[2], k[3])

    @property
    def position(self):
        return logical_position(&self.st)

    @property
    def chunk_length(self):
        return tandem_chunk_length(&self.st.rng)

    @property
    def state(self):
        return {
            "bit_generator": "Tandem",
            "key": list(self.key),
            "position": self.position,
            "K": self.chunk_length,
        }

    @state.setter
    def state(self, value):
        if value.get("bit_generator") != "Tandem":
            raise ValueError("state must come from a Tandem generator")
        cdef uint32_t k[4]
        key = value["key"]
        if len(key) != 4:
            raise ValueError("key must have four 32-bit words")
        for i in range(4):
            k[i] = _check_u32(key[i])
        pos = int(value["position"])
        if not 0 <= pos < 2**63:
            raise ValueError("position must lie in [0, 2**63)")
        self.st.rng = tandem_from_key(k, pos, _check_K(value["K"]))
        self.st.left = 0
        self.st.has_pending = False

    def split(self, index):
        """Child by index, from the key alone. Position is preserved."""
        return Tandem._wrap(tandem_split(&self.st.rng, _check_u64(index)))

    def sub(self, purpose):
        """Child for a purpose identifier. Position is preserved."""
        return Tandem._wrap(tandem_sub(&self.st.rng, _check_u64(purpose)))

    def fork(self, n):
        """``n`` children from the current block. The parent moves past the block."""
        n = int(n)
        if not 0 <= n <= 2**33:
            raise ValueError("fork size must lie in [0, 2**33]")
        cdef np.ndarray buf = np.empty(n * sizeof(tandem_rng), dtype=np.uint8)
        cdef tandem_rng *kids = <tandem_rng *>np.PyArray_DATA(buf)
        flush(&self.st)
        tandem_fork(&self.st.rng, kids, n)
        return [Tandem._wrap(kids[i]) for i in range(n)]

    def random(self, size=None, dtype=np.float64, out=None):
        """Uniform draws in [0, 1), the stream's own Float64 or Float32 mapping.

        ``out`` is a C-contiguous one-dimensional array of ``dtype``, filled in place.
        """
        cdef np.ndarray a = _buffer(size, dtype, out, (np.float64, np.float32))
        cdef size_t n = a.size
        cdef void *p = np.PyArray_DATA(a)
        flush(&self.st)
        if a.dtype == np.float64:
            with nogil:
                tandem_fill_f64(&self.st.rng, <double *>p, n)
        else:
            with nogil:
                tandem_fill_f32(&self.st.rng, <float *>p, n)
        return a[()] if size is None and out is None else a

    def raw(self, size=None, dtype=np.uint64, out=None):
        """Unsigned integers of the stream at the dtype's width.

        ``out`` is a C-contiguous one-dimensional array of ``dtype``, filled in place.
        """
        cdef np.ndarray a = _buffer(size, dtype, out, (np.uint64, np.uint32, np.uint16, np.uint8))
        cdef size_t n = a.size
        cdef void *p = np.PyArray_DATA(a)
        cdef int w = a.dtype.itemsize
        flush(&self.st)
        with nogil:
            if w == 8:
                tandem_fill_u64(&self.st.rng, <uint64_t *>p, n)
            elif w == 4:
                tandem_fill_u32(&self.st.rng, <uint32_t *>p, n)
            elif w == 2:
                tandem_fill_u16(&self.st.rng, <uint16_t *>p, n)
            else:
                tandem_fill_u8(&self.st.rng, <uint8_t *>p, n)
        return a[()] if size is None and out is None else a


cdef np.ndarray _buffer(size, dtype, out, allowed):
    dtype = np.dtype(dtype)
    if dtype not in allowed:
        raise TypeError("dtype must be one of " + ", ".join(np.dtype(d).name for d in allowed))
    if out is None:
        return np.empty(size if size is not None else (), dtype=dtype)
    if not isinstance(out, np.ndarray) or out.dtype != dtype:
        raise TypeError("out must be an ndarray of the requested dtype")
    if out.ndim != 1 or not out.flags.c_contiguous or not out.flags.writeable:
        raise ValueError("out must be a writeable C-contiguous one-dimensional array")
    if size is not None and out.shape != (size,):
        raise ValueError("size does not match the shape of out")
    return out


cdef uint32_t _check_K(K) except? 0:
    K = int(K)
    if K < 1 or K > 65536 or K & (K - 1):
        raise ValueError("K must be a power of two in [1, 65536]")
    return K

cdef uint32_t _check_u32(x) except? 0:
    x = int(x)
    if not 0 <= x < 2**32:
        raise ValueError("key words must lie in [0, 2**32)")
    return x

cdef uint64_t _check_u64(x) except? 0:
    x = int(x)
    if not 0 <= x < 2**64:
        raise ValueError("index must lie in [0, 2**64)")
    return x
