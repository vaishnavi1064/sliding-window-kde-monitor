import math

import numpy as np


def l2_lsh_collision_probability(distance: float, width: float) -> float:
    """Probability two points at Euclidean `distance` collide under one p-stable
    hash of bucket width `width` (Datar et al. 2004).

    norm.cdf is expanded via erf so this needs no scipy.
    """
    if distance == 0:
        return 1.0
    normal_cdf = 0.5 * (1.0 + math.erf((-width / distance) / math.sqrt(2.0)))
    term1 = 1.0 - 2.0 * normal_cdf
    term2 = (2.0 * distance / (math.sqrt(2.0 * math.pi) * width)) * (
        1.0 - math.exp(-(width**2) / (2.0 * distance**2))
    )
    return term1 - term2


class PStableHashBank:
    """`rows` independent Euclidean LSH functions, each concatenating `k` p-stable
    hashes, folded into a bounded cell code.

    Vectorized in the same way as AngularHashBank: all rows*k projections come
    from one matmul.

    On folding: raw p-stable hashes are unbounded integers, so the paper bounds
    the range by rehashing. The reference does that by hashing each of the k
    values and *summing* them (`L2_hash_AKDE.py` lines 30-33), which is
    order-invariant and maps R^k distinct hash tuples onto at most k*R sums --
    concentrated near the mean, so far fewer in practice. That is the Euclidean
    analogue of Finding E and it destroys the same joint information. We instead
    fold with a polynomial hash, which preserves which value appeared in which
    position before reducing modulo the range.
    """

    _MULTIPLIER = 1_000_003

    def __init__(
        self,
        rows: int,
        k: int,
        dim: int,
        width: float,
        hash_range: int = 1 << 20,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        self.rows = rows
        self.k = k
        self.width = width
        self.hash_range = hash_range
        # Gaussian projections => 2-stable => Euclidean distance.
        self.projections = rng.normal(0, 1, size=(rows * k, dim))
        self.offsets = rng.uniform(0, width, size=rows * k)

    def codes(self, x) -> np.ndarray:
        """Cell code per row, each in [0, hash_range)."""
        raw = np.floor((self.projections @ x + self.offsets) / self.width).astype(np.int64)
        raw = raw.reshape(self.rows, self.k)
        code = np.zeros(self.rows, dtype=np.int64)
        for j in range(self.k):
            code = (code * self._MULTIPLIER + raw[:, j]) % self.hash_range
        return code
