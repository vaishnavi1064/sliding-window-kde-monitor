import numpy as np
import pytest

from sketch.brute_force import compute_true_kde_l2
from sketch.p_stable import PStableHashBank, l2_lsh_collision_probability
from sketch.sw_akde import SlidingWindowEuclideanKDE

EH_RELATIVE_ERROR = 0.1
THEORETICAL_BOUND = 2 * EH_RELATIVE_ERROR + EH_RELATIVE_ERROR**2
WIDTH = 4.0


def _synthetic_gaussian_mixture(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    centers = rng.normal(scale=3.0, size=(3, dim))
    labels = rng.integers(0, 3, size=n)
    return centers[labels] + rng.normal(scale=1.0, size=(n, dim))


def test_collision_probability_is_monotone_and_bounded():
    assert l2_lsh_collision_probability(0.0, WIDTH) == 1.0
    probabilities = [l2_lsh_collision_probability(d, WIDTH) for d in (0.5, 1, 2, 4, 8, 16)]
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    # Closer points must collide at least as often as distant ones.
    assert probabilities == sorted(probabilities, reverse=True)


def test_codes_respect_hash_range():
    rng = np.random.default_rng(0)
    rows, k, dim, hash_range = 16, 4, 7, 1024
    bank = PStableHashBank(
        rows, k, dim, WIDTH, hash_range=hash_range, rng=np.random.default_rng(1)
    )
    for x in rng.normal(size=(30, dim)) * 5:
        codes = bank.codes(x)
        assert codes.shape == (rows,)
        assert codes.min() >= 0
        assert codes.max() < hash_range


def test_fold_preserves_position_unlike_summing():
    # Guards Finding G: the reference folds the k hashes with `sum(...)`, which
    # is order-invariant, so permuting which value came from which hash function
    # yields the same cell. A polynomial fold must not.
    rows, k, dim = 1, 3, 2
    bank = PStableHashBank(rows, k, dim, width=1.0, hash_range=1 << 20)
    bank.projections = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    bank.offsets = np.zeros(3)

    a = bank.codes(np.array([5.0, 1.0]))[0]
    b = bank.codes(np.array([1.0, 5.0]))[0]
    # Same multiset of raw hash values, different positions -> different cells.
    assert a != b


def test_unwindowed_euclidean_converges_to_brute_force():
    # Tier 2: window >= stream length so nothing expires.
    rng = np.random.default_rng(0)
    dim, n, rows, k = 6, 400, 800, 2

    stream = _synthetic_gaussian_mixture(rng, n, dim)
    queries = _synthetic_gaussian_mixture(rng, 15, dim)

    sketch = SlidingWindowEuclideanKDE(
        rows=rows,
        k=k,
        dim=dim,
        width=WIDTH,
        window_size=n + 1,
        eh_relative_error=EH_RELATIVE_ERROR,
        rng=np.random.default_rng(1),
    )
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)

    true_kde = compute_true_kde_l2(stream, queries, k, WIDTH)
    estimates = np.array([sketch.query(q, t=n) for q in queries])
    relative_error = np.abs(estimates - true_kde) / np.maximum(true_kde, 1e-9)

    assert relative_error.mean() < 0.3


@pytest.mark.parametrize("window_size", [128, 256])
@pytest.mark.parametrize("rows", [200, 400])
def test_windowed_euclidean_within_theoretical_bound(window_size: int, rows: int):
    # Tier 3, Euclidean counterpart. Same caveats as the angular version: the
    # bound is probabilistic so we assert on the mean across queries, and very
    # small windows are excluded because of DGIM's small-count bias floor.
    rng = np.random.default_rng(0)
    dim, n, k = 6, 1200, 2

    stream = _synthetic_gaussian_mixture(rng, n, dim)
    queries = _synthetic_gaussian_mixture(rng, 12, dim)

    sketch = SlidingWindowEuclideanKDE(
        rows=rows,
        k=k,
        dim=dim,
        width=WIDTH,
        window_size=window_size,
        eh_relative_error=EH_RELATIVE_ERROR,
        rng=np.random.default_rng(1),
    )
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)

    true_kde = compute_true_kde_l2(stream[-window_size:], queries, k, WIDTH)
    estimates = np.array([sketch.query(q, t=n) for q in queries])
    relative_error = np.abs(estimates - true_kde) / np.maximum(true_kde, 1e-9)

    assert relative_error.mean() <= THEORETICAL_BOUND


def test_euclidean_kernel_sees_magnitude_that_angular_cannot():
    # The reason we ported this kernel at all: angular LSH is scale-invariant,
    # so a sensor drifting in magnitude along a fixed direction is invisible to
    # it. The Euclidean kernel must register that drift as a density change.
    rng = np.random.default_rng(0)
    dim, window_size, k = 5, 200, 2

    direction = np.ones(dim) / np.sqrt(dim)
    sketch = SlidingWindowEuclideanKDE(
        rows=400,
        k=k,
        dim=dim,
        width=WIDTH,
        window_size=window_size,
        eh_relative_error=EH_RELATIVE_ERROR,
        rng=np.random.default_rng(1),
    )

    baseline = direction * 5.0
    t = 0
    for _ in range(400):
        t += 1
        sketch.update(baseline + rng.normal(scale=0.2, size=dim), t)

    at_baseline = sketch.query(baseline, t=t)
    # Same direction, much larger magnitude -- angular LSH could not tell these
    # apart at all; the Euclidean kernel should see far lower density here.
    scaled_up = direction * 25.0
    assert sketch.query(scaled_up, t=t) < 0.2 * at_baseline
