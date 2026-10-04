"""Agreement with the specification vectors and with dumps written by TandemRNG.jl."""

import hashlib
import json
import re
import pickle
from pathlib import Path

import numpy as np
import pytest
from numpy.random import Generator

from tandem_rng import Tandem

DATA = Path(__file__).parent / "data"
VECTORS = json.loads((Path(__file__).parent / "vectors.json").read_text())


def words(hexes):
    return tuple(int(w, 16) for w in hexes)


def load(name, dtype):
    return np.fromfile(DATA / name, dtype=dtype)


def make(name):
    if name.startswith("k1234_K8_"):
        return Tandem.from_key((1, 2, 3, 4), K=8)
    if name.startswith("k1234_"):
        return Tandem.from_key((1, 2, 3, 4))
    return Tandem(42)


@pytest.mark.parametrize(
    "name, dtype",
    [
        ("k1234_K32_u32.bin", np.uint32),
        ("k1234_K32_u64.bin", np.uint64),
        ("k1234_K8_u32.bin", np.uint32),
        ("seed42_K32_u8.bin", np.uint8),
    ],
)
def test_raw_matches_julia(name, dtype):
    want = load(name, dtype)
    rng = make(name)
    got = rng.raw(want.size, dtype)
    assert np.array_equal(got, want)
    assert rng.position == want.size * want.itemsize * 8
    rng = make(name)
    assert all(rng.raw(dtype=dtype) == w for w in want[:300])


@pytest.mark.parametrize(
    "name, dtype", [("seed42_K32_f64.bin", np.float64), ("seed42_K32_f32.bin", np.float32)]
)
def test_random_matches_julia(name, dtype):
    want = load(name, dtype)
    got = make(name).random(want.size, dtype)
    assert np.array_equal(got, want)
    # NumPy's own conversions are the specification's mappings.
    assert np.array_equal(Generator(make(name)).random(want.size, dtype=dtype), want)


def test_generator_uint16_matches_stream():
    # f16 bits are a function of the u16 stream; check the u16 stream through raw.
    want = load("seed42_K32_f16bits.bin", np.uint16)
    raw = Tandem(42).raw(want.size, np.uint16)
    k = raw >> 5
    m = np.where(k > 0, np.floor(np.log2(np.maximum(k, 1))).astype(np.uint16), 0)
    got = np.where(k > 0, ((m + 4) << 10) | ((k << (10 - m)) & 0x3FF), 0).astype(np.uint16)
    assert np.array_equal(got, want)


def test_spec_vectors():
    key = words(VECTORS["key"])
    rng = Tandem.from_key(key)
    stream = rng.raw(36, np.uint32)
    for s in VECTORS["stream_words"]:
        f = s["first_word"]
        assert tuple(stream[f : f + 4]) == words(s["words"])

    draws = VECTORS["draws_from_position_0"]
    f64 = Tandem.from_key(key).random(32)
    for i, x in draws["Float64"].items():
        assert f64[int(i)] == x
    f32 = Tandem.from_key(key).random(32, np.float32)
    for i, x in draws["Float32"].items():
        assert f32[int(i)] == np.float32(x)

    derived = VECTORS["derived_keys"]
    assert rng.split(0).key == words(derived["split_child_0"])
    assert rng.split(1).key == words(derived["split_child_1"])
    assert rng.sub(7).key == words(derived["purpose_7"])
    parent = Tandem.from_key(key)
    kids = parent.fork(2)
    assert kids[0].key == words(derived["fork_child_0_at_block_0"])
    assert kids[0].position == 0 and kids[1].position == 0
    assert parent.position == 128

    seed = VECTORS["seed_whitening"]
    rng = Tandem(seed["seed"])
    assert rng.key == words(seed["key"])
    f64 = Tandem(seed["seed"]).random(32)
    for i, x in seed["Float64"].items():
        assert f64[int(i)] == x
    u32 = Tandem(seed["seed"]).raw(8, np.uint32)
    for i, x in seed["UInt32"].items():
        assert u32[int(i)] == int(x, 16)


def test_state_and_pickle_roundtrip():
    rng = Tandem(7, K=8)
    rng.random(1000)
    state = rng.state
    assert state == {"bit_generator": "Tandem", "key": list(rng.key), "position": 64000, "K": 8}
    copy = pickle.loads(pickle.dumps(rng))
    assert copy.state == state
    other = Tandem(1)
    other.state = state
    want = rng.random(100)
    assert np.array_equal(copy.random(100), want)
    assert np.array_equal(other.random(100), want)


def test_seed_handling():
    a, b = Tandem(), Tandem()
    assert a.key != b.key
    with pytest.raises(ValueError):
        Tandem(-1)
    with pytest.raises(ValueError):
        Tandem(2**128)
    with pytest.raises(ValueError):
        Tandem(1, K=3)
    ss = np.random.SeedSequence(5)
    want = int.from_bytes(ss.generate_state(4, np.uint32).tobytes(), "little")
    assert Tandem(np.random.SeedSequence(5)).key == Tandem(want).key


def test_scalar_and_shape():
    rng = Tandem(3)
    assert isinstance(rng.random(), float)
    assert rng.random((2, 3)).shape == (2, 3)
    assert rng.raw((4,), np.uint8).dtype == np.uint8
    with pytest.raises(TypeError):
        rng.random(2, np.int32)
    with pytest.raises(TypeError):
        rng.raw(2, np.int64)


def test_out_argument():
    want = Tandem(9).random(1000)
    buf = np.empty(1000)
    got = Tandem(9).random(out=buf)
    assert got is buf and np.array_equal(buf, want)
    want32 = Tandem(9).raw(500, np.uint32)
    buf32 = np.empty(500, np.uint32)
    assert Tandem(9).raw(500, np.uint32, out=buf32) is buf32
    assert np.array_equal(buf32, want32)
    rng = Tandem(9)
    with pytest.raises(TypeError):
        rng.random(out=np.empty(10, np.float32))
    with pytest.raises(TypeError):
        rng.raw(out=np.empty(10, np.int64))
    with pytest.raises(ValueError):
        rng.random(out=np.empty(20)[::2])
    with pytest.raises(ValueError):
        rng.random(out=np.empty((2, 5)))
    with pytest.raises(ValueError):
        rng.random(5, out=np.empty(6))


def test_generator_draws_match_spec_alignment():
    # The buffered hooks must read the same bits as the fills for any mix of widths.
    ref = Tandem(42)
    words = ref.raw(4096, np.uint32)
    gen = Generator(Tandem(42))
    got = gen.integers(0, 2**32, size=4096, dtype=np.uint32, endpoint=False)
    assert np.array_equal(got, words)

    t = Tandem(42)
    gen = Generator(t)
    a = gen.integers(0, 2**32, size=3, dtype=np.uint32)  # words 0, 1, 2: one pending half
    assert t.position == 96
    b = t.raw(1, np.uint64)  # aligns to 128, so word 3 is skipped
    assert t.position == 192
    assert np.array_equal(a, words[:3]) and b[0] == ref_u64(words, 4)
    c = gen.random()  # next_double: aligned read at 192
    assert c == (ref_u64(words, 6) >> 11) * 2.0**-53
    assert t.position == 256
    d = gen.integers(0, 2**32, size=1, dtype=np.uint32)[0]  # word 8
    assert d == words[8]
    t.raw(1, np.uint8)  # byte at 288, position 296
    e = gen.integers(0, 2**32, size=2, dtype=np.uint32)  # aligns to 320: words 10 and 11
    assert np.array_equal(e, words[10:12])
    assert t.position == 384


def ref_u64(words, i):
    return int(words[i]) | int(words[i + 1]) << 32


def test_spawn_is_split_by_index():
    # Spawned children follow the specification's split, not SeedSequence spawning.
    parent = Tandem(11, K=8)
    kids = parent.spawn(3)
    for i, kid in enumerate(kids):
        want = Tandem(11, K=8).split(i)
        assert kid.key == want.key and kid.position == 0 and kid.chunk_length == 8
        assert kid._seed_seq is None
        assert np.array_equal(kid.random(50), want.random(50))
    assert [k.key for k in parent.spawn(2)] == [Tandem(11, K=8).split(i).key for i in (3, 4)]
    assert parent.position == 0


def test_generator_spawn_uses_split():
    gen = Generator(Tandem(5))
    kids = gen.spawn(2)
    assert all(isinstance(k, Generator) for k in kids)
    assert [k.bit_generator.key for k in kids] == [Tandem(5).split(i).key for i in range(2)]
    assert np.array_equal(kids[1].random(10), Tandem(5).split(1).random(10))
    kid = pickle.loads(pickle.dumps(kids[0].bit_generator))
    assert kid.key == kids[0].bit_generator.key


@pytest.mark.parametrize(
    "dtype, name",
    [(np.uint32, "k1234_K32_u32.bin"), (np.uint64, "k1234_K32_u64.bin"),
     (np.float64, "seed42_K32_f64.bin"), (np.float32, "seed42_K32_f32.bin")],
)
def test_at_matches_dumps(dtype, name):
    want = load(name, dtype)
    rng = make(name)
    for i in (0, 1, 7, 100, want.size - 1):
        assert rng.at(dtype, i) == want[i] and rng.at(dtype, i).dtype == dtype
    assert rng.position == 0


@pytest.mark.parametrize("dtype, fill", [(np.uint32, lambda t, n: t.raw(n, np.uint32)),
                                         (np.uint64, lambda t, n: t.raw(n, np.uint64)),
                                         (np.float32, lambda t, n: t.random(n, np.float32)),
                                         (np.float64, lambda t, n: t.random(n))])
def test_at_from_unaligned_buffered_position(dtype, fill):
    # An odd 32-bit draw through Generator leaves the buffered generator ahead of the C one.
    t = Tandem(42)
    Generator(t).integers(0, 2**32, size=3, dtype=np.uint32)
    pos = t.position
    got = [t.at(dtype, i) for i in (0, 5, 1000)]
    assert t.position == pos
    ref = Tandem(42)
    ref.position = pos
    want = fill(ref, 1001)
    assert got == [want[0], want[5], want[1000]]


def test_position_setter_and_advance_to():
    ref = Tandem(8).raw(500, np.uint32)
    rng = Tandem(8)
    rng.random(10)
    rng.position = 32 * 100
    assert np.array_equal(rng.raw(50, np.uint32), ref[100:150])
    rng.advance_to(32 * 7)
    assert rng.raw(1, np.uint32)[0] == ref[7]
    # Buffered Generator draws are discarded by a move.
    gen = Generator(rng)
    gen.integers(0, 2**32, size=3, dtype=np.uint32)
    rng.advance_to(32 * 20)
    assert gen.integers(0, 2**32, dtype=np.uint32) == ref[20]
    assert rng.state["position"] == 32 * 21
    for bad in (-1, 2**63):
        with pytest.raises(ValueError):
            rng.advance_to(bad)


def filled(dtype, n):
    out = np.empty(n, dtype)
    assert Tandem(42).fill(out) is out
    return out


def test_fill_bool_u8_match_dumps():
    want = load("seed42_K32_bool.bin", np.bool_)
    assert np.array_equal(filled(np.bool_, want.size), want)
    want = load("seed42_K32_u8.bin", np.uint8)
    assert np.array_equal(filled(np.uint8, want.size), want)
    # Signed integers carry the unsigned draw's bits.
    assert np.array_equal(filled(np.int8, want.size).view(np.uint8), want)


@pytest.mark.parametrize("signed, unsigned", [(np.int16, np.uint16), (np.int32, np.uint32), (np.int64, np.uint64)])
def test_fill_signed_reinterprets_unsigned(signed, unsigned):
    assert np.array_equal(filled(signed, 777).view(unsigned), Tandem(42).raw(777, unsigned))


def test_fill_f16_matches_dump():
    want = load("seed42_K32_f16bits.bin", np.uint16)
    got = filled(np.float16, want.size)
    assert np.array_equal(got.view(np.uint16), want)
    assert (got >= 0).all() and (got < 1).all()


def test_fill_complex_matches_dumps():
    want = load("seed42_K32_c32.bin", np.complex64)
    assert np.array_equal(filled(np.complex64, want.size), want)
    want = load("seed42_K32_c64.bin", np.complex128)
    assert np.array_equal(filled(np.complex128, want.size), want)


def test_u128_matches_dump_and_bytes_fill():
    want = load("seed42_K32_u128.bin", np.uint64).reshape(-1, 2)
    assert np.array_equal(Tandem(42).u128(want.shape[0]), want)
    assert np.array_equal(Tandem(42).u128(), want[0])
    buf = np.empty((want.shape[0], 2), np.uint64)
    assert Tandem(42).u128(out=buf) is buf and np.array_equal(buf, want)
    v = np.empty(want.shape[0], "V16")
    Tandem(42).fill(v)
    assert v.tobytes() == want.tobytes()


def test_char_matches_dump():
    want = load("seed42_K32_char.bin", np.uint32)
    assert np.array_equal(Tandem(42).char(want.size), want)
    assert int(Tandem(42).char()) == want[0]


def test_fill_matches_typed_fills_and_advances():
    # Any shape fills in C order, from an unaligned start, and leaves the position right.
    t, ref = Tandem(3), Tandem(3)
    t.raw(1, np.uint8)
    ref.raw(1, np.uint8)
    a = np.empty((4, 5), np.float32)
    t.fill(a)
    assert np.array_equal(a.ravel(), ref.random(20, np.float32))
    assert t.position == ref.position
    b = np.empty(9, np.bool_)
    t.fill(b)
    assert np.array_equal(b, np.asarray([ref.fill(np.empty(1, np.bool_))[0] for _ in range(9)]))
    assert t.position == ref.position


def test_fill_rejects_bad_buffers():
    t = Tandem(1)
    with pytest.raises(TypeError):
        t.fill(np.empty(4, "M8[s]"))
    with pytest.raises(TypeError):
        t.fill([0.0] * 4)
    with pytest.raises(ValueError):
        t.fill(np.empty(8)[::2])
    with pytest.raises(ValueError):
        t.fill(np.empty(4, ">u4"))
    with pytest.raises(TypeError):
        t.u128(out=np.empty(4, np.uint64))


C_TESTS = Path(__file__).parent.parent / "external" / "tandem-c" / "tests"


def cross_cases(header, name):
    """(start, n, want, end_pos) per case of one integer table in a tandem-c cross header.

    ``start`` is the fill's start position, or None for the scalar tables."""
    text = (C_TESTS / header).read_text()
    table = text.split(name + "[] = {")[1].split("\n};")[0]
    return [
        (int(start) if start else None, int(n), [int(x) for x in re.findall(r"(\d+)u", want)], int(end))
        for start, n, want, end in re.findall(
            r"\{(?:(\d+)ull, )?(\d+)u(?:ll)?,\s*\{([^}]*)\},\s*(\d+)u\}", table
        )
    ]


def cross_floats(header, name):
    text = (C_TESTS / header).read_text()
    body = text.split(name + "[2 * CROSS_NORMAL_COUNT] = {")[1].split("};")[0]
    return np.array([float(x) for x in re.findall(r"[-+]?\d[\d.]*(?:e[-+]?\d+)?", body)])


def cross_position(header, name):
    return int(re.search(name + r" = (\d+)u", (C_TESTS / header).read_text()).group(1))


def at_position(pos, seed=42):
    rng = Tandem(seed)
    rng.position = pos
    return rng


def unaligned(seed=42):
    rng = Tandem(seed)
    rng.fill(np.empty(1, np.bool_))
    return rng


@pytest.mark.parametrize("dtype, table", [(np.uint32, "CROSS_U32"), (np.uint64, "CROSS_U64")])
def test_below_scalar_matches_tandem_cuda(dtype, table):
    cases = cross_cases("cross_below.h", table)
    assert len(cases) >= 5
    for _, n, want, end in cases:
        rng = unaligned()
        got = [rng.below(n, dtype=dtype) for _ in range(len(want))]
        assert got == want and rng.position == end


@pytest.mark.parametrize("dtype, table", [(np.uint32, "CROSS_FILL_U32"), (np.uint64, "CROSS_FILL_U64")])
def test_below_fill_matches_tandem_cuda(dtype, table):
    cases = cross_cases("cross_fill_below.h", table)
    assert len(cases) >= 5
    assert len({c[0] for c in cases}) > 1
    for start, n, want, end in cases:
        rng = at_position(start)
        got = rng.below(n, len(want), dtype)
        assert got.dtype == dtype and got.tolist() == want and rng.position == end
        buf = np.empty(len(want), dtype)
        assert at_position(start).below(n, out=buf, dtype=dtype) is buf and buf.tolist() == want


def test_below_edges():
    rng = Tandem(1)
    assert rng.below(0, dtype=np.uint32) == 0 and rng.position == 32
    assert rng.below(0) == 0 and rng.position == 128
    with pytest.raises(ValueError):
        rng.below(2**32, dtype=np.uint32)
    with pytest.raises(TypeError):
        rng.below(5, dtype=np.int64)


def test_normal_matches_tandem_cuda():
    # The fixtures are pairs, cos half first. A fill is the flattened pairs and a scalar draw
    # is the cos half, so the scalars equal the even elements.
    want = cross_floats("cross_normal.h", "CROSS_NORMAL")
    end = cross_position("cross_normal.h", "CROSS_NORMAL_END_POS")
    rng = unaligned()
    assert np.allclose(rng.normal(want.size), want, rtol=1e-12, atol=0)
    assert rng.position == end
    rng = unaligned()
    got = np.array([rng.normal() for _ in want[::2]])
    assert np.allclose(got, want[::2], rtol=1e-12, atol=0) and rng.position == end

    want = cross_floats("cross_normal.h", "CROSS_NORMALF")
    end = cross_position("cross_normal.h", "CROSS_NORMALF_END_POS")
    # Float libm differs between platforms: 8 ulps and a floor near the zeros of cos and sin.
    close = lambda got: np.all(np.abs(got - want[:len(got)]) <= 8 * 2.0**-23 * np.abs(want[:len(got)]) + 1e-6)
    rng = unaligned()
    assert close(rng.normal(want.size, np.float32)) and rng.position == end
    rng = unaligned()
    got = np.array([rng.normal(dtype=np.float32) for _ in want[::2]])
    assert np.all(np.abs(got - want[::2]) <= 8 * 2.0**-23 * np.abs(want[::2]) + 1e-6)
    assert rng.position == end


def test_normal_fill_pairs_and_out():
    for dtype in (np.float64, np.float32):
        a, b = unaligned(7), unaligned(7)
        got = a.normal(1000, dtype)
        assert got.dtype == dtype
        # Scalar draws are the cos halves, the even elements, two uniforms each.
        assert np.array_equal(got[::2], [b.normal(dtype=dtype) for _ in range(500)])
        assert a.position == b.position
        # An odd n drops the last sin half but still consumes both uniforms of its pair.
        odd, even = unaligned(7), unaligned(7)
        assert np.array_equal(odd.normal(7, dtype), even.normal(8, dtype)[:7])
        assert odd.position == even.position
        buf = np.empty(10, dtype)
        assert Tandem(2).normal(out=buf, dtype=dtype) is buf
    z = Tandem(5).normal(200_000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_below_empty_fill_keeps_position(dtype):
    rng = unaligned()
    assert rng.below(5, 0, dtype).size == 0
    assert rng.position == 1


@pytest.mark.parametrize("dtype, n", [(np.uint32, 0xC0000001), (np.uint64, 0xC000000000000001)])
def test_below_fill_cut_equals_whole(dtype, n):
    # The fallback stream is keyed by the global draw index, so a cut at any element
    # boundary reproduces the whole fill. These ranges reject about a quarter of the draws.
    cuts = (37, 1, 100, 62)
    whole = unaligned(5).below(n, sum(cuts), dtype)
    rng = unaligned(5)
    parts = [rng.below(n, c, dtype) for c in cuts]
    assert np.array_equal(np.concatenate(parts), whole)
    scalar = unaligned(5)
    assert not np.array_equal([scalar.below(n, dtype=dtype) for _ in whole], whole)


# TandemGenerator: the overridden samplers are the C fills, and the rest is NumPy's.

from tandem_rng import TandemGenerator


def pair(seed, pos):
    g = TandemGenerator(seed)
    g.bit_generator.position = pos
    return g, at_position(pos, seed)


@pytest.mark.parametrize("n, pos", [(1, 0), (7, 33), (1000, 12345), (4097, 999)])
def test_generator_equals_fills_at_random_positions(n, pos):
    g, ref = pair(11, pos)
    assert np.array_equal(g.random(n), ref.random(n))
    assert np.array_equal(g.random(n, np.float32), ref.random(n, np.float32))
    assert np.array_equal(g.standard_normal(n), ref.normal(n))
    assert np.array_equal(g.standard_normal(n, np.float32), ref.normal(n, np.float32))
    assert g.bit_generator.position == ref.position


def test_generator_derived_samplers():
    g, ref = pair(3, 77)
    assert np.array_equal(g.normal(2.5, 3.0, 100), 2.5 + 3.0 * ref.normal(100))
    assert np.array_equal(g.uniform(-2, 6, 100), -2 + 8.0 * ref.random(100))
    assert g.bit_generator.position == ref.position
    # Broadcast parameters take their shape from loc and scale.
    assert g.normal(np.zeros(4), np.ones((3, 1))).shape == (3, 4)
    assert isinstance(g.normal(), float) and isinstance(g.uniform(), float)
    with pytest.raises(ValueError):
        g.normal(0, -1)


@pytest.mark.parametrize("dtype, width", [(np.int8, 32), (np.uint16, 32), (np.int32, 32), (np.uint32, 32),
                                          (np.int64, 64), (np.uint64, 64)])
def test_integers_are_below_plus_low(dtype, width):
    # Every range here is at most 2**32, so the draws are 32-bit whatever the dtype.
    word = np.uint32
    info = np.iinfo(dtype)
    low, high = max(info.min, -1000), min(info.max, 1000)
    g, ref = pair(21, 65)
    got = g.integers(low, high, 3000, dtype)
    want = ref.below(high - low, 3000, word).astype(np.int64) + low
    assert got.dtype == dtype and np.array_equal(got, want)
    assert g.bit_generator.position == ref.position
    assert (got >= low).all() and (got < high).all()
    # endpoint includes high, and one argument means [0, low).
    g, ref = pair(21, 65)
    assert np.array_equal(g.integers(low, high, 50, dtype, endpoint=True),
                          ref.below(high - low + 1, 50, word).astype(np.int64) + low)
    assert 0 <= g.integers(9, dtype=dtype) < 9


def test_integers_scalar_and_full_range():
    g, ref = pair(4, 5)
    assert g.integers(1000, dtype=np.uint32) == ref.below(1000, dtype=np.uint32)
    assert isinstance(g.integers(1000), np.int64)
    # A span of 2**64 is the raw words, with no rejection to apply.
    g, ref = pair(4, 5)
    full = g.integers(0, 2**64 - 1, 20, np.uint64, endpoint=True)
    assert np.array_equal(full, ref.raw(20, np.uint64))
    g, ref = pair(4, 5)
    assert np.array_equal(g.integers(-2**31, 2**31, 20, np.int32),
                          (ref.raw(20, np.uint32) + np.uint32(2**31)).view(np.int32))
    for args in [(5, 5), (5, 4)]:
        with pytest.raises(ValueError):
            g.integers(*args)
    with pytest.raises(ValueError):
        g.integers(0, 300, dtype=np.uint8)
    with pytest.raises(TypeError):
        g.integers(1.5, 5)


@pytest.mark.parametrize("dtype, table", [(np.uint32, "CROSS_FILL_U32"), (np.uint64, "CROSS_FILL_U64")])
def test_integers_match_cross_fixtures(dtype, table):
    for start, n, want, end in cross_cases("cross_fill_below.h", table):
        # The range picks the width: the 64-bit table only applies above 2**32.
        if (n > 2**32) != (table == "CROSS_FILL_U64"):
            continue
        g, _ = pair(42, start)
        got = g.integers(0, n, len(want), np.int64 if n <= 2**32 else dtype)
        assert got.tolist() == want and g.bit_generator.position == end


def test_generator_normal_matches_cross_fixtures():
    want = cross_floats("cross_normal.h", "CROSS_NORMAL")
    g = TandemGenerator(Tandem(42))
    g.bit_generator.position = 1
    assert np.allclose(g.standard_normal(want.size), want, rtol=1e-12, atol=0)
    assert g.bit_generator.position == cross_position("cross_normal.h", "CROSS_NORMAL_END_POS")


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
                                   np.int64, np.uint64])
def test_integers_cut_equals_whole(dtype, endpoint):
    # The widest range of each dtype that is not a full word, so the 32-bit and 64-bit
    # paths both reject draws (about a quarter for the two widest ranges), at a start that
    # is not word aligned.
    info = np.iinfo(dtype)
    low = info.min
    high = low + (info.max - low) // 4 * 3 if info.bits >= 32 else info.max
    cuts = (37, 1, 100, 62)
    whole, _ = pair(5, 1)
    want = whole.integers(low, high, sum(cuts), dtype, endpoint=endpoint)
    g, _ = pair(5, 1)
    parts = [g.integers(low, high, c, dtype, endpoint=endpoint) for c in cuts]
    assert want.dtype == dtype and np.array_equal(np.concatenate(parts), want)
    assert g.bit_generator.position == whole.bit_generator.position


def test_standard_normal_bits_match_tandem_c():
    # The bytes tandem-c's tools/dump_normals.c writes: f64 then f32 fills of 2e6 - 1 values
    # from five positions. Their FNV-1a hash is 0x9414e1315e2653be, recorded in
    # tandem-c's tests/test_normal_bits.c. SHA-256 over the same bytes is checked here
    # because a byte loop in Python is slow.
    h = hashlib.sha256()
    for start in (0, 1, 77, 12345, 1 << 30):
        g = TandemGenerator(Tandem(2026 + (7 << 64)))
        g.bit_generator.position = start
        h.update(g.standard_normal(2_000_000 - 1).tobytes())
        h.update(g.standard_normal(2_000_000 - 1, np.float32).tobytes())
    assert h.hexdigest() == "cfae418807a7d5f91ecd3e42c33a00943690c6e4b888ee39206738783efe9ded"


def test_generator_out_and_dtype_paths():
    g, ref = pair(8, 9)
    buf = np.empty((4, 5))
    assert g.random(out=buf) is buf and np.array_equal(buf.ravel(), ref.random(20))
    strided = np.empty(40, np.float32)[::2]
    g.standard_normal(dtype=np.float32, out=strided)
    assert np.array_equal(strided, ref.normal(20, np.float32))
    assert g.random((2, 3), np.float32).shape == (2, 3)
    assert g.standard_normal(size=(2, 3)).shape == (2, 3)
    with pytest.raises(TypeError):
        g.random(out=np.empty(3, np.float32))
    with pytest.raises(ValueError):
        g.random(5, out=np.empty(6))
    with pytest.raises(TypeError):
        g.random(2, np.int32)


def test_generator_falls_through_and_differs_from_numpy_algorithms():
    g = TandemGenerator(7)
    plain = Generator(Tandem(7))
    assert np.array_equal(g.permutation(20), plain.permutation(20))
    assert np.array_equal(g.exponential(size=5), plain.exponential(size=5))
    assert g.choice(10, 3).shape == (3,)
    a, b = TandemGenerator(7), Generator(Tandem(7))
    assert np.array_equal(a.random(50), b.random(50))
    assert not np.array_equal(a.standard_normal(50), b.standard_normal(50))
    assert not np.array_equal(a.integers(0, 0xC0000001, 200, np.uint32), b.integers(0, 0xC0000001, 200, np.uint32))


def test_generator_spawn_and_pickle():
    g = TandemGenerator(5)
    kids = g.spawn(2)
    assert all(type(k) is TandemGenerator for k in kids)
    assert kids[1].bit_generator.key == Tandem(5).split(1).key
    g.random(10)
    copy = pickle.loads(pickle.dumps(g))
    assert type(copy) is TandemGenerator and copy.bit_generator.state == g.bit_generator.state
    assert np.array_equal(copy.standard_normal(10), g.standard_normal(10))


def test_integers_width_follows_range_not_dtype():
    wide, narrow = pair(6, 17)[0], pair(6, 17)[0]
    a = wide.integers(0, 1000, 500, np.int64)
    assert np.array_equal(a, narrow.integers(0, 1000, 500, np.int32))
    assert a.dtype == np.int64
    ref = at_position(17, 6)
    assert np.array_equal(a, ref.below(1000, 500, np.uint32))
    assert wide.bit_generator.position == ref.position
    # Above 2**32 the draws are 64-bit.
    g, ref = pair(6, 17)
    assert np.array_equal(g.integers(0, 2**40, 50), ref.below(2**40, 50))


def test_integers_bool_and_array_bounds():
    g, ref = pair(2, 3)
    b = g.integers(0, 2, 100, dtype=np.bool_)
    assert b.dtype == np.bool_ and np.array_equal(b, ref.below(2, 100, np.uint32).astype(bool))
    assert g.integers(0, 1, dtype=np.bool_, endpoint=True) in (False, True)
    # Array bounds are NumPy's own algorithm.
    arr = TandemGenerator(9).integers([0, 10], [5, 20])
    assert arr.shape == (2,) and (arr >= [0, 10]).all() and (arr < [5, 20]).all()
