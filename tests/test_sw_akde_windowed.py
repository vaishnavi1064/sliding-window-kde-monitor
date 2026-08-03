import numpy as np
import pytest

from sketch.brute_force import compute_true_kde_angular
from sketch.sw_akde import SlidingWindowAngularKDE

EH_RELATIVE_ERROR = 0.1
# Lemma 4.3: the KDE-level relative error eps composes from the EH's eps' as
# eps = 2*eps' + eps'^2. With eps' = 0.1 this is 0.21, the same figure the
# paper quotes for its own experiments (section 5.2).
THEORETICAL_BOUND = 2 * EH_RELATIVE_ERROR + EH_RELATIVE_ERROR**2


def _synthetic_gaussian_mixture(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    centers = rng.normal(scale=3.0, size=(3, dim))
    labels = rng.integers(0, 3, size=n)
    return centers[labels] + rng.normal(scale=1.0, size=(n, dim))


@pytest.mark.parametrize("window_size", [128, 256, 512])
@pytest.mark.parametrize("rows", [200, 400])
def test_windowed_kde_within_theoretical_bound(window_size: int, rows: int):
    # Tier 3: the windowed sketch vs sketch-independent brute-force truth over
    # the last N elements. Windows/rows are a subset of the paper's own sweeps
    # (windows 64..2048, rows 100..3200) sized to keep pytest fast; the full
    # sweep belongs in the Phase 3 evaluation notebook per CLAUDE.md section 9.
    #
    # Asserted on the mean relative error across queries, not per-query: the
    # paper's guarantee is probabilistic (holds with probability 1-delta), so
    # individual queries are expected to exceed the bound occasionally.
    #
    # Windows below ~128 are deliberately excluded. There the true KDE is small
    # enough that DGIM's `total - last/2` estimator has a systematic downward
    # bias of ~0.5 per row that does not shrink with more rows -- an inherent
    # small-count property of the estimator, not a defect in this port (adding
    # rows at window=64 plateaus at ~0.18-0.20 rather than converging).
    rng = np.random.default_rng(0)
    dim = 8
    n = 1200
    k = 5

    stream = _synthetic_gaussian_mixture(rng, n, dim)
    queries = _synthetic_gaussian_mixture(rng, 12, dim)

    sketch = SlidingWindowAngularKDE(
        rows=rows,
        k=k,
        dim=dim,
        window_size=window_size,
        eh_relative_error=EH_RELATIVE_ERROR,
        rng=np.random.default_rng(1),
    )
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)

    true_kde = compute_true_kde_angular(stream[-window_size:], queries, k)
    estimates = np.array([sketch.query(q, t=n) for q in queries])
    relative_error = np.abs(estimates - true_kde) / np.maximum(true_kde, 1e-9)

    assert relative_error.mean() <= THEORETICAL_BOUND


def test_every_cells_first_arrival_is_counted():
    """Finding A, at the sketch level.

    The paper's Algorithm 2 (§4.1) preprocessing reads:

        if A[i, j] is empty then
            Create an Exponential Histogram at A[i, j] with timestamp t
        else
            Add a 1 to the Exponential Histogram at A[i, j] with timestamp t

    The create branch has no corresponding "Add a 1", so every cell drops the
    very element that created it. The reference implementation
    (`Ang_hash_AKDE.py` lines 25-28) has the same asymmetry.

    A single element is the sharpest possible probe: correctly counted it gives
    density 1.0, and under the published branch structure it gives exactly 0.0.
    The tier-2 convergence test cannot see this -- a one-per-cell undercount
    stays inside its tolerance (measured: 0.074 vs 0.063 mean relative error,
    both well under its 0.30 assert), which is why this test exists separately.
    """
    rng = np.random.default_rng(0)
    sketch = SlidingWindowAngularKDE(
        rows=200,
        k=4,
        dim=6,
        window_size=1000,
        eh_relative_error=EH_RELATIVE_ERROR,
        rng=np.random.default_rng(1),
    )

    x = rng.normal(size=6)
    sketch.update(x, 1)

    # Every row placed this element in some cell, so the mean over rows is 1.
    assert sketch.query(x, t=1) == 1.0

    # And it keeps counting as more arrive in the same region.
    for t in range(2, 11):
        sketch.update(x, t)
    assert sketch.query(x, t=10) >= 9.0


def test_window_actually_expires_old_data():
    # Finding F, at the sketch level: cold cells must expire on read.
    #
    # The paper's Algorithm 2 (§4.1) query procedure reads the cell with no time
    # argument at all --
    #     c <- estimate of count in the Exponential Histogram at A[i, h_i(q)]
    # -- so it cannot expire before reading, and the reference's `count_est()`
    # likewise takes no timestamp. A region that goes quiet therefore reports a
    # frozen count forever. Reproducing that behaviour here leaves the density
    # unchanged (ratio 1.00) and fails the assert below, which passes at 0.00
    # once expiry-on-read is threaded through.
    #
    # Guards the sliding-window semantics themselves: a query matching only
    # data that has since fallen out of the window must decay toward zero,
    # while one matching recent data must not. Without correct expiry this
    # sketch would silently behave like un-windowed RACE.
    rng = np.random.default_rng(0)
    dim = 8
    window_size = 100
    k = 4

    old_region = rng.normal(size=(1, dim))
    new_region = -old_region

    sketch = SlidingWindowAngularKDE(
        rows=400,
        k=k,
        dim=dim,
        window_size=window_size,
        eh_relative_error=EH_RELATIVE_ERROR,
        rng=np.random.default_rng(1),
    )

    t = 0
    for _ in range(300):
        t += 1
        sketch.update(old_region[0] + rng.normal(scale=0.1, size=dim), t)
    density_when_fresh = sketch.query(old_region[0], t=t)

    for _ in range(300):
        t += 1
        sketch.update(new_region[0] + rng.normal(scale=0.1, size=dim), t)
    density_after_expiry = sketch.query(old_region[0], t=t)

    assert density_after_expiry < 0.25 * density_when_fresh
    assert sketch.query(new_region[0], t=t) > density_after_expiry
