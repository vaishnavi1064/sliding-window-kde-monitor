import numpy as np

from sketch.angular_hash import AngularHash, AngularHashBank


def test_bank_codes_match_scalar_reference():
    # The bank is an optimization of the scalar per-bit path (one matmul instead
    # of rows*k tiny np.dot calls). Pin them together so the fast path cannot
    # silently drift from the readable reference implementation.
    rng = np.random.default_rng(0)
    rows, k, dim = 12, 6, 9

    bank = AngularHashBank(rows, k, dim, np.random.default_rng(7))

    # Rebuild the equivalent scalar hashes from the bank's own hyperplanes so
    # the two differ only in how the bits are combined, not in the planes used.
    scalar = []
    for row in range(rows):
        bits = []
        for j in range(k):
            h = AngularHash(dim)
            h.w = bank.planes[row * k + j]
            bits.append(h)
        scalar.append(bits)

    for x in rng.normal(size=(40, dim)):
        expected = []
        for row in range(rows):
            code = 0
            for h in scalar[row]:
                code = code * 2 + (1 if h.eval(x) == 1 else 0)
            expected.append(code)
        assert bank.codes(x).tolist() == expected


def test_codes_stay_in_range():
    rng = np.random.default_rng(1)
    rows, k, dim = 8, 5, 6
    bank = AngularHashBank(rows, k, dim, np.random.default_rng(2))
    for x in rng.normal(size=(30, dim)):
        codes = bank.codes(x)
        assert codes.shape == (rows,)
        assert codes.min() >= 0
        assert codes.max() < 2**k


def test_distinct_bit_patterns_get_distinct_codes():
    # Guards Finding E: the reference's `code = bit*2 + code` collapsed to
    # 2*(number of set bits), so patterns with equal Hamming weight collided.
    # Proper concatenation must keep them apart.
    rows, k, dim = 1, 4, 3
    bank = AngularHashBank(rows, k, dim, np.random.default_rng(0))
    # Axis-aligned planes make the bit pattern directly controllable.
    bank.planes = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    # Both inputs set exactly two of the four bits, but different ones.
    a = np.array([1.0, -1.0, 0.0])
    b = np.array([-1.0, 1.0, 0.0])
    assert bank.codes(a)[0] != bank.codes(b)[0]
