"""The C++17 core against the pure-Python oracle (Phase 5).

CLAUDE.md section 13: the Python port stays the oracle after the native core
exists, and CI compares them. This file is that comparison.

The bar here is deliberately higher than "agrees within tolerance". Both cores
accumulate integer bucket counts and divide by two, then sum in row order, so
every query is a sum of exactly representable doubles added in the same
sequence. Their results should therefore be *bit-for-bit* equal, and these tests
assert `==` rather than `approx`. Anything looser would hide exactly the kind of
drift a reimplementation introduces -- an off-by-one in the merge cascade, say,
which perturbs a count by one element and would sail through a 1% tolerance.

Layered so a failure localises:
  1. the exponential histogram alone, bucket for bucket
  2. whole-sketch query output, both kernels
  3. per-cell internal state after a realistic run
  4. the native core against brute-force truth, i.e. the Phase 5 accuracy gate
"""

import random

import numpy as np
import pytest

from sketch import native
from sketch.brute_force import compute_true_kde_angular
from sketch.cell_store import resolve_backend
from sketch.exponential_histogram import ExponentialHistogram
from sketch.sw_akde import SlidingWindowAngularKDE, SlidingWindowEuclideanKDE

pytestmark = pytest.mark.skipif(
    not native.AVAILABLE,
    reason=f"native core not built ({native.IMPORT_ERROR}); build it with `make native`",
)


def python_buckets(eh: ExponentialHistogram) -> list[tuple[int, int]]:
    """The Python histogram's buckets in the tuple shape the native one returns."""
    return [(b.timestamp, b.size) for b in eh.buckets]


def assert_histograms_agree(python: ExponentialHistogram, cpp, t: int) -> None:
    assert python_buckets(python) == cpp.buckets()
    assert python.total == cpp.total
    assert python.last == cpp.last
    assert python.count_estimate(t) == cpp.count_estimate(t)


# ---------------------------------------------------------------- tier 1: the histogram


@pytest.mark.parametrize("relative_error", [0.5, 0.2, 0.1, 0.05, 0.01])
@pytest.mark.parametrize("window_size", [1, 7, 50, 256])
def test_histogram_matches_on_dense_stream(window_size: int, relative_error: float):
    """Every tick a hit: drives the merge cascade to its deepest, and evicts on
    almost every add. Also pins the derived parameters -- k = ceil(1/eps) and
    merge_cap = ceil(k/2)+1 are computed independently in each language, so a
    disagreement here would show up as divergent merging."""
    python = ExponentialHistogram(window_size=window_size, relative_error=relative_error)
    cpp = native.module().ExponentialHistogram(
        window_size=window_size, relative_error=relative_error
    )

    assert python.k == cpp.k

    for t in range(1, 601):
        python.add(t)
        cpp.add(t)
        assert_histograms_agree(python, cpp, t)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_histogram_matches_across_large_gaps(seed: int):
    """Gaps far wider than the window, so several buckets expire at once.

    This is the multi-bucket eviction path (Finding B) -- the `while` loop that
    replaced the reference's single `if`. A ring buffer evicts by bumping a head
    index instead of shifting a list, so this is where an indexing mistake would
    surface.
    """
    rng = random.Random(seed)
    window_size = 200
    python = ExponentialHistogram(window_size=window_size, relative_error=0.1)
    cpp = native.module().ExponentialHistogram(window_size=window_size, relative_error=0.1)

    t = 0
    for _ in range(400):
        t += rng.randint(1, 3 * window_size)
        python.add(t)
        cpp.add(t)
        assert_histograms_agree(python, cpp, t)


def test_histogram_matches_when_expiring_on_read():
    """Reading long after the last write must decay identically (Finding F).

    Queried at times the histogram never saw an add for, so all the expiry
    happens inside `count_estimate` rather than inside `add`.
    """
    python = ExponentialHistogram(window_size=64, relative_error=0.1)
    cpp = native.module().ExponentialHistogram(window_size=64, relative_error=0.1)

    for t in range(1, 101):
        python.add(t)
        cpp.add(t)

    # Walk the clock forward without adding anything: the count must fall to
    # zero in lockstep, and both must agree on when nothing is left.
    for t in range(100, 400, 7):
        assert python.count_estimate(t) == cpp.count_estimate(t)
        assert python.is_expired(t) == cpp.is_expired(t)
    assert python.is_expired(400) and cpp.is_expired(400)


def test_histogram_rejects_invalid_parameters():
    module = native.module()
    with pytest.raises(ValueError):
        module.ExponentialHistogram(window_size=0, relative_error=0.1)
    with pytest.raises(ValueError):
        module.ExponentialHistogram(window_size=10, relative_error=0.0)


# ------------------------------------------------------------------ tier 2: the sketch


def _gaussian_mixture(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    centers = rng.normal(scale=3.0, size=(3, dim))
    labels = rng.integers(0, 3, size=n)
    return centers[labels] + rng.normal(scale=1.0, size=(n, dim))


def _paired_angular(**kwargs) -> tuple[SlidingWindowAngularKDE, SlidingWindowAngularKDE]:
    """Two sketches differing only in backend.

    Seeding both hash banks identically is what makes bitwise comparison
    meaningful: same hyperplanes, same cell codes, so any difference in the
    output comes from the cell array rather than from the hashing.
    """
    return (
        SlidingWindowAngularKDE(rng=np.random.default_rng(1), backend="python", **kwargs),
        SlidingWindowAngularKDE(rng=np.random.default_rng(1), backend="native", **kwargs),
    )


@pytest.mark.parametrize("window_size", [64, 256])
@pytest.mark.parametrize("rows", [16, 200])
def test_angular_sketch_queries_are_bitwise_identical(rows: int, window_size: int):
    rng = np.random.default_rng(0)
    dim, n, k = 8, 900, 5
    stream = _gaussian_mixture(rng, n, dim)
    queries = _gaussian_mixture(rng, 15, dim)

    python, cpp = _paired_angular(rows=rows, k=k, dim=dim, window_size=window_size)

    for t, x in enumerate(stream, start=1):
        python.update(x, t)
        cpp.update(x, t)
        # Interleave queries with updates so expiry-on-read happens at the same
        # clock in both, rather than only comparing the end state.
        if t % 100 == 0:
            for q in queries[:3]:
                assert python.query(q, t) == cpp.query(q, t)

    for q in queries:
        assert python.query(q, n) == cpp.query(q, n)

    assert python.cell_count == cpp.cell_count
    assert python.compactions == cpp.compactions
    assert python.cells_reclaimed == cpp.cells_reclaimed


def test_euclidean_sketch_queries_are_bitwise_identical():
    """The Euclidean kernel folds its k hashes into a bounded range, so its cell
    codes are large and sparse rather than a dense [0, 2^k). Exercises the native
    hash map on keys the angular case never produces."""
    rng = np.random.default_rng(2)
    dim, n = 7, 800
    stream = _gaussian_mixture(rng, n, dim)
    queries = _gaussian_mixture(rng, 10, dim)

    shared = dict(rows=120, k=3, dim=dim, width=2.0, window_size=200)
    python = SlidingWindowEuclideanKDE(rng=np.random.default_rng(5), backend="python", **shared)
    cpp = SlidingWindowEuclideanKDE(rng=np.random.default_rng(5), backend="native", **shared)

    for t, x in enumerate(stream, start=1):
        python.update(x, t)
        cpp.update(x, t)

    for q in queries:
        assert python.query(q, n) == cpp.query(q, n)
    assert python.cell_count == cpp.cell_count


def test_compaction_schedule_matches():
    """Compaction is driven by the logical clock inside the cell array, so both
    cores must sweep at the same ticks and reclaim the same cells. Drift here
    would not change query results -- reclaiming is semantically free -- but it
    would mean the two cores hold different amounts of memory."""
    rng = np.random.default_rng(3)
    dim = 4
    python, cpp = _paired_angular(rows=8, k=6, dim=dim, window_size=100)

    for step in range(1500):
        t = step + 1
        # Drift steadily so cells are continuously abandoned and reclaimed.
        x = rng.normal(size=dim) + step * 0.5
        python.update(x, t)
        cpp.update(x, t)
        assert python.compactions == cpp.compactions
        assert python.cells_reclaimed == cpp.cells_reclaimed
        assert python.cell_count == cpp.cell_count


# -------------------------------------------------------- tier 3: per-cell internal state


def test_every_live_cell_holds_identical_state():
    """Aggregate agreement could in principle hide compensating differences
    between cells. This compares the two cores cell by cell: same set of live
    cells, and within each, the same buckets, total and last."""
    rng = np.random.default_rng(4)
    dim, n = 6, 700
    stream = _gaussian_mixture(rng, n, dim)

    python, cpp = _paired_angular(rows=32, k=4, dim=dim, window_size=150)
    for t, x in enumerate(stream, start=1):
        python.update(x, t)
        cpp.update(x, t)

    assert python.cell_count == cpp.cell_count
    assert python.cell_count > 0

    for row, code in python.cells:
        native_state = cpp.cell_state(row, code)
        assert native_state is not None, f"native core is missing cell ({row}, {code})"
        assert python.cell_state(row, code) == native_state, f"cell ({row}, {code})"

    # A cell neither core has ever seen reports absent, not empty.
    assert cpp.cell_state(0, 10**9) is None
    assert python.cell_state(0, 10**9) is None


# ------------------------------------------------- tier 4: accuracy gate and plumbing


def test_native_core_matches_brute_force_truth():
    """Phase 5's accuracy requirement, stated directly rather than inherited.

    Bitwise parity with the Python core already implies this, since that core is
    validated against brute-force truth in tests/test_sw_akde_windowed.py. Kept
    separate because "accuracy must still match Phase 1" is the phase's
    definition of done, and it should be checked against the ground truth itself
    rather than only via a chain of reasoning.
    """
    eh_relative_error = 0.1
    # Lemma 4.3: the KDE-level error composes from the EH's as 2*eps + eps^2.
    bound = 2 * eh_relative_error + eh_relative_error**2

    rng = np.random.default_rng(0)
    dim, n, k, window_size = 8, 1200, 5, 256
    stream = _gaussian_mixture(rng, n, dim)
    queries = _gaussian_mixture(rng, 12, dim)

    sketch = SlidingWindowAngularKDE(
        rows=400,
        k=k,
        dim=dim,
        window_size=window_size,
        eh_relative_error=eh_relative_error,
        rng=np.random.default_rng(1),
        backend="native",
    )
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)

    true_kde = compute_true_kde_angular(stream[-window_size:], queries, k)
    estimates = np.array([sketch.query(q, t=n) for q in queries])
    relative_error = np.abs(estimates - true_kde) / np.maximum(true_kde, 1e-9)

    assert relative_error.mean() <= bound


def test_backend_resolution():
    assert resolve_backend("python") == "python"
    assert resolve_backend("native") == "native"
    # With the core built, `auto` must choose it -- otherwise the deployed
    # pipeline would silently run the slow path.
    assert resolve_backend("auto") == "native"
    with pytest.raises(ValueError):
        resolve_backend("cpp")


def test_native_sketch_reports_its_backend_and_memory():
    sketch = SlidingWindowAngularKDE(
        rows=16, k=3, dim=4, window_size=50, rng=np.random.default_rng(0), backend="native"
    )
    assert sketch.backend == "native"
    for t in range(1, 101):
        sketch.update(np.random.default_rng(t).normal(size=4), t)

    assert sketch.memory_bytes() > 0
    # The native core has no Python cell dict to expose, and must say so clearly
    # rather than raising something obscure from inside the store.
    with pytest.raises(AttributeError, match="cell_count"):
        _ = sketch.cells


def test_native_core_rejects_a_mismatched_code_array():
    """The binding reads the code array through a raw pointer, so a wrong length
    would otherwise read past the end."""
    array = native.module().CellArray(
        rows=8, window_size=50, eh_relative_error=0.1, compact_every=50
    )
    with pytest.raises(ValueError, match="rows"):
        array.update(np.zeros(7, dtype=np.int64), 1)
    with pytest.raises(ValueError, match="1-D"):
        array.update(np.zeros((8, 2), dtype=np.int64), 1)


def test_native_cell_array_rejects_invalid_parameters():
    module = native.module()
    for kwargs in (
        dict(rows=0, window_size=50, eh_relative_error=0.1, compact_every=50),
        dict(rows=8, window_size=0, eh_relative_error=0.1, compact_every=50),
        dict(rows=8, window_size=50, eh_relative_error=0.0, compact_every=50),
        dict(rows=8, window_size=50, eh_relative_error=0.1, compact_every=0),
    ):
        with pytest.raises(ValueError):
            module.CellArray(**kwargs)
