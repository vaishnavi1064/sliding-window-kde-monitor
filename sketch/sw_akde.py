import numpy as np

from sketch.angular_hash import AngularHashBank
from sketch.exponential_histogram import ExponentialHistogram
from sketch.p_stable import PStableHashBank


class SlidingWindowKDE:
    """SW-AKDE sketch: a RACE array whose counters are exponential histograms,
    giving it sliding-window semantics (Danait, Das & Bhore, arXiv:2510.23039).

    Kernel-agnostic -- takes any hash bank exposing `rows` and `codes(x)`. Use
    SlidingWindowAngularKDE or SlidingWindowEuclideanKDE unless you are supplying
    your own bank.
    """

    def __init__(self, hashes, window_size: int, eh_relative_error: float = 0.1):
        self.hashes = hashes
        self.rows = hashes.rows
        self.window_size = window_size
        self.eh_relative_error = eh_relative_error
        self.cells: dict[tuple[int, int], ExponentialHistogram] = {}

    def update(self, x, t: int) -> None:
        """Record element `x` at logical time `t`.

        `t` must be a monotonic per-event counter, not a wall clock (Finding B).
        """
        codes = self.hashes.codes(x)
        for row in range(self.rows):
            key = (row, int(codes[row]))
            eh = self.cells.get(key)
            if eh is None:
                eh = ExponentialHistogram(self.window_size, self.eh_relative_error)
                self.cells[key] = eh
            # Unconditional add on both the create and existing-cell paths --
            # the reference only adds on the existing-cell branch, silently
            # dropping every cell's first arrival (Finding A).
            eh.add(t)

    def query(self, x, t: int) -> float:
        """Estimated kernel density around `x` over the last `window_size` elements.

        Mean over rows, per the paper's section 4.1 (median-of-means is plain
        RACE's estimator, not SW-AKDE's). `t` is the current logical clock, so
        cells that have gone cold expire on read instead of reporting a frozen
        count (Finding F).
        """
        codes = self.hashes.codes(x)
        total = 0.0
        for row in range(self.rows):
            eh = self.cells.get((row, int(codes[row])))
            if eh is not None:
                total += eh.count_estimate(t)
        return total / self.rows


class SlidingWindowAngularKDE(SlidingWindowKDE):
    """SW-AKDE with the angular (cosine) kernel via SimHash.

    Scale-invariant: two readings pointing the same direction are identical to
    this kernel regardless of magnitude.
    """

    def __init__(
        self,
        rows: int,
        k: int,
        dim: int,
        window_size: int,
        eh_relative_error: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            AngularHashBank(rows, k, dim, rng), window_size, eh_relative_error
        )
        self.k = k


class SlidingWindowEuclideanKDE(SlidingWindowKDE):
    """SW-AKDE with the Euclidean (L2) kernel via p-stable LSH.

    Magnitude-sensitive, unlike the angular kernel -- likely the better fit for
    sensor readings where the absolute level is itself the signal. `width` is the
    p-stable bucket width and sets the kernel bandwidth.
    """

    def __init__(
        self,
        rows: int,
        k: int,
        dim: int,
        width: float,
        window_size: int,
        eh_relative_error: float = 0.1,
        hash_range: int = 1 << 20,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            PStableHashBank(rows, k, dim, width, hash_range, rng),
            window_size,
            eh_relative_error,
        )
        self.k = k
        self.width = width
