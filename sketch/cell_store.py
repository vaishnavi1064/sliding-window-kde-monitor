"""The cell array behind the sketch, in two interchangeable implementations.

`SlidingWindowKDE` owns the hashing and delegates everything else -- the per-row
loop, the cell lookup, the exponential histograms, the compaction schedule -- to
a cell store. Two exist:

  * `PythonCellStore`, below. The Phase 1 port, validated against brute-force
    ground truth per CLAUDE.md section 9. It stays the oracle even now that the
    native core exists (section 13), and remains the default.
  * `sketch._native.CellArray`, the C++17 core from Phase 5, which implements the
    same four methods and the same attributes. It needs no Python adapter -- the
    binding exposes this interface directly.

`tests/test_native_parity.py` pins them together.
"""

import sys

from sketch.exponential_histogram import ExponentialHistogram

PYTHON = "python"
NATIVE = "native"
AUTO = "auto"


class PythonCellStore:
    """A RACE array whose counters are exponential histograms."""

    def __init__(
        self,
        rows: int,
        window_size: int,
        eh_relative_error: float,
        compact_every: int,
    ):
        self.rows = rows
        self.window_size = window_size
        self.eh_relative_error = eh_relative_error
        self.compact_every = compact_every
        self.cells: dict[tuple[int, int], ExponentialHistogram] = {}
        self._next_compaction = compact_every
        self.compactions = 0
        self.cells_reclaimed = 0

    @property
    def cell_count(self) -> int:
        return len(self.cells)

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

    def update(self, codes, t: int) -> None:
        """Record an element with per-row cell `codes` at logical time `t`.

        `t` must be a monotonic per-event counter, not a wall clock (Finding B).
        """
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

    def query(self, codes, t: int) -> float:
        """Mean cell count across rows, per the paper's section 4.1.

        Median-of-means is plain RACE's estimator, not SW-AKDE's. `t` is the
        current logical clock, so cells that have gone cold expire on read
        instead of reporting a frozen count (Finding F).
        """
        total = 0.0
        for row in range(self.rows):
            eh = self.cells.get((row, int(codes[row])))
            if eh is not None:
                total += eh.count_estimate(t)
        return total / self.rows

    def cell_state(self, row: int, code: int):
        """`(total, last, [(timestamp, size), ...])` for one cell, or None.

        Backend-agnostic introspection: the native core cannot hand out its
        histograms as Python objects, so this is the shape both cores report and
        what the parity tests compare cell by cell.
        """
        eh = self.cells.get((row, code))
        if eh is None:
            return None
        return (eh.total, eh.last, [(b.timestamp, b.size) for b in eh.buckets])

    def memory_bytes(self) -> int:
        """Approximate bytes held by the cell dictionary and its histograms.

        The native core reports the same quantity for its own layout, so the
        Phase 5 benchmark can compare like with like. Both are estimates -- here
        because sys.getsizeof cannot see shared or interned storage, there
        because the standard library does not expose per-node overhead -- so the
        benchmark quotes process RSS alongside them.
        """
        total = sys.getsizeof(self.cells)
        for key, eh in self.cells.items():
            total += sys.getsizeof(key) + sys.getsizeof(eh)
            total += sys.getsizeof(eh.buckets)
            total += sum(sys.getsizeof(b) for b in eh.buckets)
        return total


def native_available() -> bool:
    """True when the C++ core was compiled and can be imported."""
    from sketch import native

    return native.AVAILABLE


def resolve_backend(backend: str) -> str:
    """Turn a requested backend into the one that will actually be used.

    `auto` prefers the native core and silently falls back; `native` is an
    explicit demand and raises if it is unavailable, so a benchmark or a
    deployment cannot quietly measure the wrong core.
    """
    if backend == AUTO:
        return NATIVE if native_available() else PYTHON
    if backend == NATIVE:
        from sketch import native

        if not native.AVAILABLE:
            raise RuntimeError(
                "the native core is not built: "
                f"{native.IMPORT_ERROR}. Build it with `make native`, or pass "
                'backend="auto" to fall back to the Python core.'
            )
        return NATIVE
    if backend == PYTHON:
        return PYTHON
    raise ValueError(f"unknown backend: {backend!r} (expected 'python', 'native' or 'auto')")


def make_cell_store(
    backend: str,
    rows: int,
    window_size: int,
    eh_relative_error: float,
    compact_every: int,
):
    """Build the cell store for `backend`, which must already be resolved."""
    if backend == NATIVE:
        from sketch import native

        return native.module().CellArray(
            rows=rows,
            window_size=window_size,
            eh_relative_error=eh_relative_error,
            compact_every=compact_every,
        )
    return PythonCellStore(
        rows=rows,
        window_size=window_size,
        eh_relative_error=eh_relative_error,
        compact_every=compact_every,
    )
