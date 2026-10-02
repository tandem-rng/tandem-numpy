"""Agreement with the specification vectors and with dumps written by TandemRNG.jl."""

import json
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
