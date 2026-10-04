# cython: language_level=3, boundscheck=False, wraparound=False
"""Tandem8x32 as a NumPy BitGenerator over the reference C implementation."""

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

import numpy as np
cimport numpy as np
from numpy.random cimport bitgen_t
from numpy.random.bit_generator cimport BitGenerator
from numpy.random import SeedSequence

np.import_array()

cdef extern from "stdbool.h":
    ctypedef unsigned char c_bool "bool"

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
    void tandem_fill_bool(tandem_rng *rng, c_bool *out, size_t n) nogil
    ctypedef struct tandem_u128:
        uint64_t lo, hi
    void tandem_fill_u128(tandem_rng *rng, tandem_u128 *out, size_t n) nogil
    void tandem_fill_f16_bits(tandem_rng *rng, uint16_t *out, size_t n) nogil
    void tandem_fill_char(tandem_rng *rng, uint32_t *out, size_t n) nogil
    void tandem_fill_c32(tandem_rng *rng, float *out, size_t n) nogil
    void tandem_fill_c64(tandem_rng *rng, double *out, size_t n) nogil
    uint32_t tandem_u32_below(tandem_rng *rng, uint32_t n) nogil
    uint64_t tandem_u64_below(tandem_rng *rng, uint64_t n) nogil
    double tandem_normal_f64(tandem_rng *rng) nogil
    float tandem_normal_f32(tandem_rng *rng) nogil
    void tandem_fill_u32_below(tandem_rng *rng, uint32_t *out, size_t len, uint32_t n) nogil
    void tandem_fill_u64_below(tandem_rng *rng, uint64_t *out, size_t len, uint64_t n) nogil
    void tandem_fill_normal_f64(tandem_rng *rng, double *out, size_t n) nogil
    void tandem_fill_normal_f32(tandem_rng *rng, float *out, size_t n) nogil
    void tandem_fill_u8(tandem_rng *rng, uint8_t *out, size_t n) nogil
    void tandem_fill_u16(tandem_rng *rng, uint16_t *out, size_t n) nogil
    void tandem_fill_u32(tandem_rng *rng, uint32_t *out, size_t n) nogil
    void tandem_fill_u64(tandem_rng *rng, uint64_t *out, size_t n) nogil
    void tandem_fill_f32(tandem_rng *rng, float *out, size_t n) nogil
    void tandem_fill_f64(tandem_rng *rng, double *out, size_t n) nogil
    bint tandem_set_position(tandem_rng *rng, uint64_t pos) nogil
    uint32_t tandem_at_u32(const tandem_rng *rng, uint64_t i) nogil
    uint64_t tandem_at_u64(const tandem_rng *rng, uint64_t i) nogil
    float tandem_at_f32(const tandem_rng *rng, uint64_t i) nogil
    double tandem_at_f64(const tandem_rng *rng, uint64_t i) nogil
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


# Fill kinds, one per C fill.
cdef enum:
    CODE_BOOL, CODE_U8, CODE_U16, CODE_U32, CODE_U64, CODE_F16, CODE_F32, CODE_F64
    CODE_C32, CODE_C64, CODE_U128, CODE_CHAR


cdef class Tandem(BitGenerator):
    """Tandem8x32 BitGenerator.

    ``seed`` is an integer in [0, 2**128), a ``SeedSequence``, or ``None`` for
    OS entropy. An integer goes through the specification's seed whitening, so
    ``Tandem(42)`` produces the stream of Julia ``Tandem8x32(42)`` and C
    ``tandem_seed(42, 0, K)``. ``K`` is the chunk length, a power of two in
    [1, 65536].
    """

    cdef buffered st
    cdef object _spawned

    def __init__(self, seed=None, K=32):
        BitGenerator.__init__(self, seed)
        self._spawned = 0
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
        # The key came from the specification's split, not from a seed sequence.
        rng._seed_seq = None
        return rng

    @property
    def key(self):
        cdef uint32_t k[4]
        tandem_key(&self.st.rng, k)
        return (k[0], k[1], k[2], k[3])

    @property
    def position(self):
        return logical_position(&self.st)

    @position.setter
    def position(self, value):
        self.advance_to(value)

    def advance_to(self, position):
        """Move to a bit position in [0, 2**63), forward or backward."""
        position = int(position)
        if not 0 <= position < 2**63:
            raise ValueError("position must lie in [0, 2**63)")
        flush(&self.st)
        tandem_set_position(&self.st.rng, position)

    def at(self, dtype, i):
        """Element ``i`` of the fill that would start at the current position.

        ``dtype`` is uint32, uint64, float32, or float64. The generator does not move.
        """
        cdef uint64_t idx = _check_u64(i)
        cdef tandem_rng s = self.st.rng
        dtype = np.dtype(dtype)
        # The buffered generator runs ahead of the C one, so read from its logical position.
        tandem_set_position(&s, logical_position(&self.st))
        if dtype == np.uint32:
            return np.uint32(tandem_at_u32(&s, idx))
        if dtype == np.uint64:
            return np.uint64(tandem_at_u64(&s, idx))
        if dtype == np.float32:
            return np.float32(tandem_at_f32(&s, idx))
        if dtype == np.float64:
            return np.float64(tandem_at_f64(&s, idx))
        raise TypeError("dtype must be one of uint32, uint64, float32, float64")

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
        """Child by index, from the key alone. It starts at position 0 with the same ``K``."""
        return Tandem._wrap(tandem_split(&self.st.rng, _check_u64(index)))

    def sub(self, purpose):
        """Child for a purpose identifier. It starts at position 0 with the same ``K``."""
        return Tandem._wrap(tandem_sub(&self.st.rng, _check_u64(purpose)))

    def spawn(self, n_children):
        """``n_children`` independent generators by the specification's split.

        Children are ``split(i)``, numbered on from the earlier calls of ``spawn``, so
        repeated calls never return the same stream. This replaces NumPy's ``SeedSequence``
        spawning: the children have no seed sequence and ``_seed_seq`` is ``None``. The
        count of earlier calls is not part of ``state`` or of a pickle. It restarts at 0
        in a restored generator.
        """
        n = int(n_children)
        if n < 0 or self._spawned + n > 2**64:
            raise ValueError("n_children must be non-negative and keep the index below 2**64")
        first = self._spawned
        self._spawned += n
        return [self.split(first + i) for i in range(n)]

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


    def fill(self, out):
        """Fill a contiguous array in place with the stream's draws of its dtype.

        Supported dtypes: bool, int8 to int64 and uint8 to uint64 (the unsigned draw's
        bits), float16, float32, float64, complex64, complex128, and ``V16`` for 128-bit
        words stored as low then high 64-bit half. The array may have any shape and is
        filled in C order. It must be writeable and native-endian. Returns ``out``.
        """
        cdef np.ndarray a = _contiguous(out)
        self._fill(a, _code(a.dtype), a.size)
        return out

    def u128(self, size=None, out=None):
        """Unsigned 128-bit draws as rows of (low, high) uint64 halves.

        ``out`` is a C-contiguous uint64 array of shape (n, 2), filled in place. Without
        ``size`` and ``out`` the result has shape (2,).
        """
        cdef np.ndarray a
        if out is None:
            a = np.empty((1 if size is None else size, 2), dtype=np.uint64)
        else:
            a = _contiguous(out)
            if a.dtype != np.uint64 or a.ndim != 2 or a.shape[1] != 2:
                raise TypeError("out must be a uint64 array of shape (n, 2)")
            if size is not None and a.shape[0] != size:
                raise ValueError("size does not match the shape of out")
        self._fill(a, CODE_U128, a.shape[0])
        return a[0] if size is None and out is None else a

    def char(self, size=None, out=None):
        """Unicode scalar values as uint32, 64 stream bits per draw.

        ``out`` is a C-contiguous one-dimensional uint32 array, filled in place.
        """
        cdef np.ndarray a = _buffer(size, np.uint32, out, (np.uint32,))
        self._fill(a, CODE_CHAR, a.size)
        return a[()] if size is None and out is None else a

    def below(self, n, size=None, dtype=np.uint64, out=None):
        """Uniform integers on [0, n) by Tandem's own bounded contract.

        This is Lemire's multiply-and-reject over the stream's uint32 or uint64 draws, the
        algorithm of tandem-cuda, and is not part of the specification. It is not the
        algorithm of ``numpy.random.Generator.integers``, which keeps its own method over
        this bit generator. ``dtype`` is uint32 or uint64 and fixes the draw width.

        With ``size`` or ``out`` the fill draws element i from stream draw i and retries a
        rejected draw on a fallback generator keyed by the draw's global index, so a fill cut
        at any element boundary equals the whole fill. It consumes exactly one draw per
        element and equals the scalar calls except where a draw is rejected. Without them one scalar
        draw takes as many draws as it needs. ``n = 0`` returns 0.
        """
        cdef np.ndarray a
        cdef uint64_t bound
        dtype = np.dtype(dtype)
        if dtype not in (np.uint32, np.uint64):
            raise TypeError("dtype must be uint32 or uint64")
        n = int(n)
        if not 0 <= n < 2 ** (8 * dtype.itemsize):
            raise ValueError("n must lie in [0, 2**32) for uint32 or [0, 2**64) for uint64")
        bound = n
        flush(&self.st)
        if size is None and out is None:
            if dtype == np.uint32:
                return np.uint32(tandem_u32_below(&self.st.rng, <uint32_t>bound))
            return np.uint64(tandem_u64_below(&self.st.rng, bound))
        a = _buffer(size, dtype, out, (np.uint32, np.uint64))
        cdef void *p = np.PyArray_DATA(a)
        cdef size_t count = a.size
        if dtype == np.uint32:
            with nogil:
                tandem_fill_u32_below(&self.st.rng, <uint32_t *>p, count, <uint32_t>bound)
        else:
            with nogil:
                tandem_fill_u64_below(&self.st.rng, <uint64_t *>p, count, bound)
        return a

    def normal(self, size=None, dtype=np.float64, out=None):
        """Standard normal draws by Tandem's own Box-Muller contract.

        Box-Muller on pairs of the stream's uniform draws a, b of ``dtype``: with
        r = sqrt(-2 ln(1 - a)) the pair is (r cos 2 pi b, r sin 2 pi b). Pair j makes
        elements 2j (cos half) and 2j + 1 (sin half) from uniforms 2j and 2j + 1. An odd
        ``size`` keeps the cos half of its last pair and still consumes both uniforms. A
        scalar draw is the cos half and consumes two uniforms, so it equals element 0 of a
        fill. This matches tandem-c and tandem-cuda and is not part of the specification.
        values agree across ports to about 1e-12 relative for float64 and a few ulps for
        float32, since libm differs. It is not ``numpy.random.Generator.standard_normal``, which
        keeps its own ziggurat over this bit generator.
        """
        cdef np.ndarray a = _buffer(size, dtype, out, (np.float64, np.float32))
        cdef void *p = np.PyArray_DATA(a)
        cdef size_t n = a.size
        flush(&self.st)
        if a.dtype == np.float64:
            with nogil:
                tandem_fill_normal_f64(&self.st.rng, <double *>p, n)
        else:
            with nogil:
                tandem_fill_normal_f32(&self.st.rng, <float *>p, n)
        return a[()] if size is None and out is None else a

    cdef void _fill(self, np.ndarray a, int code, size_t n):
        cdef void *p = np.PyArray_DATA(a)
        flush(&self.st)
        with nogil:
            fill_code(&self.st.rng, code, p, n)


cdef void fill_code(tandem_rng *rng, int code, void *p, size_t n) noexcept nogil:
    if code == CODE_BOOL: tandem_fill_bool(rng, <c_bool *>p, n)
    elif code == CODE_U8: tandem_fill_u8(rng, <uint8_t *>p, n)
    elif code == CODE_U16: tandem_fill_u16(rng, <uint16_t *>p, n)
    elif code == CODE_U32: tandem_fill_u32(rng, <uint32_t *>p, n)
    elif code == CODE_U64: tandem_fill_u64(rng, <uint64_t *>p, n)
    elif code == CODE_F16: tandem_fill_f16_bits(rng, <uint16_t *>p, n)
    elif code == CODE_F32: tandem_fill_f32(rng, <float *>p, n)
    elif code == CODE_F64: tandem_fill_f64(rng, <double *>p, n)
    elif code == CODE_C32: tandem_fill_c32(rng, <float *>p, n)
    elif code == CODE_C64: tandem_fill_c64(rng, <double *>p, n)
    elif code == CODE_U128: tandem_fill_u128(rng, <tandem_u128 *>p, n)
    else: tandem_fill_char(rng, <uint32_t *>p, n)


cdef int _code(dtype) except -1:
    kind, w = dtype.kind, dtype.itemsize
    if kind == "b":
        return CODE_BOOL
    if kind in "iu":
        # Signed draws are the unsigned draw's bits.
        codes = {1: CODE_U8, 2: CODE_U16, 4: CODE_U32, 8: CODE_U64}
    elif kind == "f":
        codes = {2: CODE_F16, 4: CODE_F32, 8: CODE_F64}
    elif kind == "c":
        codes = {8: CODE_C32, 16: CODE_C64}
    elif kind == "V" and dtype.names is None:
        codes = {16: CODE_U128}
    else:
        codes = {}
    if w not in codes:
        raise TypeError(f"unsupported dtype {dtype}")
    return codes[w]


cdef np.ndarray _contiguous(out):
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be an ndarray")
    if not out.flags.c_contiguous or not out.flags.writeable or not out.dtype.isnative:
        raise ValueError("out must be a writeable, C-contiguous, native-endian array")
    return out


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
