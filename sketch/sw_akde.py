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

    def __init__(
        self,
        hashes,
        window_size: int,
        eh_relative_error: float = 0.1,
        compact_every: int | None = None,
    ):
        self.hashes = hashes
        self.rows = hashes.rows
        self.window_size = window_size
        self.eh_relative_error = eh_relative_error
        self.cells: dict[tuple[int, int], ExponentialHistogram] = {}
        # Reclaim dead cells once per window by default: any cell not touched
        # for a whole window is certainly empty, so this is the natural period.
        self.compact_every = window_size if compact_every is None else compact_every
        self._next_compaction = self.compact_every
        self.compactions = 0
        self.cells_reclaimed = 0

    def compact(self, t: int) -> int:
        """Drop cells whose contents have all expired. Returns how many went.

        Expiry is lazy and per-cell: a histogram only prunes itself when it is
        touched. A cell that goes cold is therefore never revisited and holds
        its buckets forever. Finding F covers the *correctness* half of this (a
        stale cell reports a frozen count, so density never decays); this is the
        *memory* half, which the paper's model hides.

        The published space bound O(RW/eps * log^2 N) counts a dense R x W
        array, where "unused cell" costs nothing extra. Any real implementation
        stores cells sparsely -- most are empty -- and then nothing bounds the
        dictionary: it accumulates one entry per distinct cell ever visited.
        Measured on MetroPT-3 before this existed, a rows=400 sketch reached
        2.7 GB part way through 1.5M readings while throughput collapsed from
        ~810 to ~300 updates/s under GC pressure.

        Dropping a fully expired histogram is semantically free: it contributes
        exactly zero to any query. Running once per window makes the sweep
        O(1) amortised per update.
        """
        dead = [key for key, eh in self.cells.items() if eh.is_expired(t)]
        for key in dead:
            del self.cells[key]
        self.compactions += 1
        self.cells_reclaimed += len(dead)
        return len(dead)

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

        if t >= self._next_compaction:
            self.compact(t)
            self._next_compaction = t + self.compact_every

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
        compact_every: int | None = None,
    ):
        super().__init__(
            AngularHashBank(rows, k, dim, rng),
            window_size,
            eh_relative_error,
            compact_every,
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
        compact_every: int | None = None,
    ):
        super().__init__(
            PStableHashBank(rows, k, dim, width, hash_range, rng),
            window_size,
            eh_relative_error,
            compact_every,
        )
        self.k = k
        self.width = width
