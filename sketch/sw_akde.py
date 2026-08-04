import numpy as np

from sketch.angular_hash import AngularHashBank
from sketch.cell_store import make_cell_store, resolve_backend
from sketch.p_stable import PStableHashBank


class SlidingWindowKDE:
    """SW-AKDE sketch: a RACE array whose counters are exponential histograms,
    giving it sliding-window semantics (Danait, Das & Bhore, arXiv:2510.23039).

    Kernel-agnostic -- takes any hash bank exposing `rows` and `codes(x)`. Use
    SlidingWindowAngularKDE or SlidingWindowEuclideanKDE unless you are supplying
    your own bank.

    This class owns the hashing and delegates the cell array to a store, which is
    either the pure-Python one or the C++17 core (see sketch/cell_store.py). The
    default is `"python"`: it is the implementation validated against brute-force
    ground truth, and results should not depend on whether the machine that ran
    them happened to have a compiler. Pass `backend="auto"` to prefer the native
    core when it is built, or `"native"` to require it.
    """

    def __init__(
        self,
        hashes,
        window_size: int,
        eh_relative_error: float = 0.1,
        compact_every: int | None = None,
        backend: str = "python",
    ):
        self.hashes = hashes
        self.rows = hashes.rows
        self.window_size = window_size
        self.eh_relative_error = eh_relative_error
        # Reclaim dead cells once per window by default: any cell not touched
        # for a whole window is certainly empty, so this is the natural period.
        self.compact_every = window_size if compact_every is None else compact_every
        # Resolved rather than as requested, so `backend` afterwards names the
        # core actually in use and a benchmark cannot mislabel which one ran.
        self.backend = resolve_backend(backend)
        self._store = make_cell_store(
            self.backend,
            self.rows,
            self.window_size,
            self.eh_relative_error,
            self.compact_every,
        )

    @property
    def cells(self):
        """The live cell dictionary. Python backend only.

        The native core keeps its cells in C++ hash maps with no Python objects
        to hand out; `cell_count` and `memory_bytes()` work on both.
        """
        cells = getattr(self._store, "cells", None)
        if cells is None:
            raise AttributeError(
                "the native backend holds no Python cell dictionary; "
                "use cell_count or memory_bytes() instead"
            )
        return cells

    @property
    def cell_count(self) -> int:
        return self._store.cell_count

    @property
    def compactions(self) -> int:
        return self._store.compactions

    @property
    def cells_reclaimed(self) -> int:
        return self._store.cells_reclaimed

    def memory_bytes(self) -> int:
        """Approximate bytes held by the cell array (see the store's docstring)."""
        return self._store.memory_bytes()

    def cell_state(self, row: int, code: int):
        """`(total, last, [(timestamp, size), ...])` for one cell, or None.

        Works on either backend, which `cells` cannot.
        """
        return self._store.cell_state(row, code)

    def compact(self, t: int) -> int:
        """Drop cells whose contents have all expired. Returns how many went."""
        return self._store.compact(t)

    def update(self, x, t: int) -> None:
        """Record element `x` at logical time `t`.

        `t` must be a monotonic per-event counter, not a wall clock (Finding B).
        """
        self._store.update(self.hashes.codes(x), t)

    def query(self, x, t: int) -> float:
        """Estimated kernel density around `x` over the last `window_size` elements.

        Mean over rows, per the paper's section 4.1 (median-of-means is plain
        RACE's estimator, not SW-AKDE's). `t` is the current logical clock, so
        cells that have gone cold expire on read instead of reporting a frozen
        count (Finding F).
        """
        return self._store.query(self.hashes.codes(x), t)


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
        backend: str = "python",
    ):
        super().__init__(
            AngularHashBank(rows, k, dim, rng),
            window_size,
            eh_relative_error,
            compact_every,
            backend,
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
        backend: str = "python",
    ):
        super().__init__(
            PStableHashBank(rows, k, dim, width, hash_range, rng),
            window_size,
            eh_relative_error,
            compact_every,
            backend,
        )
        self.k = k
        self.width = width
