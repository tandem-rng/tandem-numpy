"""Agreement with the specification vectors and with dumps written by TandemRNG.jl."""

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
    """(n, want, end_pos) per range of one integer table in a tandem-c cross header."""
    text = (C_TESTS / header).read_text()
    table = text.split(name + "[] = {")[1].split("\n};")[0]
    return [
        (int(n), [int(x) for x in re.findall(r"(\d+)u", want)], int(end))
        for n, want, end in re.findall(r"\{(\d+)u(?:ll)?,\s*\{([^}]*)\},\s*(\d+)u\}", table)
    ]


def cross_floats(header, name):
    text = (C_TESTS / header).read_text()
    body = text.split(name + "[CROSS_NORMAL_COUNT] = {")[1].split("};")[0]
    return np.array([float(x) for x in re.findall(r"[-+]?\d[\d.]*(?:e[-+]?\d+)?", body)])


def cross_position(header, name):
    return int(re.search(name + r" = (\d+)u", (C_TESTS / header).read_text()).group(1))


def unaligned(seed=42):
    rng = Tandem(seed)
    rng.fill(np.empty(1, np.bool_))
    return rng


@pytest.mark.parametrize("dtype, table", [(np.uint32, "CROSS_U32"), (np.uint64, "CROSS_U64")])
def test_below_scalar_matches_tandem_cuda(dtype, table):
    cases = cross_cases("cross_below.h", table)
    assert len(cases) >= 5
    for n, want, end in cases:
        rng = unaligned()
        got = [rng.below(n, dtype=dtype) for _ in range(len(want))]
        assert got == want and rng.position == end


@pytest.mark.parametrize("dtype, table", [(np.uint32, "CROSS_FILL_U32"), (np.uint64, "CROSS_FILL_U64")])
def test_below_fill_matches_tandem_cuda(dtype, table):
    cases = cross_cases("cross_fill_below.h", table)
    assert len(cases) >= 5
    for n, want, end in cases:
        rng = unaligned()
        got = rng.below(n, len(want), dtype)
        assert got.dtype == dtype and got.tolist() == want and rng.position == end
        buf = np.empty(len(want), dtype)
        assert unaligned().below(n, out=buf, dtype=dtype) is buf and buf.tolist() == want


def test_below_edges():
    rng = Tandem(1)
    assert rng.below(0, dtype=np.uint32) == 0 and rng.position == 32
    assert rng.below(0) == 0 and rng.position == 128
    with pytest.raises(ValueError):
        rng.below(2**32, dtype=np.uint32)
    with pytest.raises(TypeError):
        rng.below(5, dtype=np.int64)


def test_normal_matches_tandem_cuda():
    want = cross_floats("cross_normal.h", "CROSS_NORMAL")
    rng = unaligned()
    got = np.array([rng.normal() for _ in want])
    assert np.allclose(got, want, rtol=1e-12, atol=0)
    assert rng.position == cross_position("cross_normal.h", "CROSS_NORMAL_END_POS")
    assert np.allclose(unaligned().normal(want.size), want, rtol=1e-12, atol=0)

    want = cross_floats("cross_normal.h", "CROSS_NORMALF")
    rng = unaligned()
    got = np.array([rng.normal(dtype=np.float32) for _ in want])
    # Float libm differs between platforms: 8 ulps and a floor near the zeros of cos.
    assert np.all(np.abs(got - want) <= 8 * 2.0**-23 * np.abs(want) + 1e-6)
    assert rng.position == cross_position("cross_normal.h", "CROSS_NORMALF_END_POS")


def test_normal_fill_equals_scalars_and_out():
    for dtype in (np.float64, np.float32):
        a, b = unaligned(7), unaligned(7)
        got = a.normal(1000, dtype)
        assert got.dtype == dtype
        assert np.array_equal(got, [b.normal(dtype=dtype) for _ in range(1000)])
        assert a.position == b.position
        buf = np.empty(10, dtype)
        assert Tandem(2).normal(out=buf, dtype=dtype) is buf
    z = Tandem(5).normal(200_000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02
