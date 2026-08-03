import numpy as np


class AngularHash:
    """A single random-hyperplane (SimHash) bit.

    Kept as the readable scalar reference for one LSH bit; the sketches use
    AngularHashBank, which is equivalent but evaluates every bit at once.
    tests/test_hash_bank.py pins the two together.
    """

    def __init__(self, dim: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.w = rng.normal(0, 1, size=dim)

    def eval(self, x) -> int:
        return 1 if np.dot(self.w, x) >= 0 else -1


class AngularHashBank:
    """`rows` independent LSH functions, each concatenating `k` SimHash bits.

    Profiling the scalar version showed per-bit hashing at ~59% of update time:
    it issued rows*k separate tiny np.dot calls per element (2000 at rows=400,
    k=5), where call overhead dwarfs the arithmetic. Here all rows*k hyperplanes
    live in one matrix so a single matmul produces every bit, and the k bits of
    each row are packed into a cell code by a second small matmul.
    """

    def __init__(self, rows: int, k: int, dim: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.rows = rows
        self.k = k
        self.planes = rng.normal(0, 1, size=(rows * k, dim))
        # First bit is most significant, matching code = code*2 + bit.
        self.bit_values = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)

    def codes(self, x) -> np.ndarray:
        """Cell code per row, each in [0, 2^k)."""
        bits = (self.planes @ x >= 0).reshape(self.rows, self.k)
        return bits.astype(np.int64) @ self.bit_values
