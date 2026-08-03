import numpy as np

from sketch.brute_force import compute_true_kde_angular
from sketch.race import RACE
from sketch.sw_akde import SlidingWindowAngularKDE


def _synthetic_gaussian_mixture(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    centers = rng.normal(scale=3.0, size=(3, dim))
    labels = rng.integers(0, 3, size=n)
    return centers[labels] + rng.normal(scale=1.0, size=(n, dim))


def test_unwindowed_sketch_and_race_both_converge_to_brute_force():
    # Window >= stream length so nothing ever expires -- the sketch should
    # then behave like plain RACE, and both estimators (our mean-of-rows,
    # RACE's median-of-chunk-means) should independently converge to the same
    # sketch-independent brute-force target with enough rows.
    rng = np.random.default_rng(0)
    dim = 8
    n = 500
    rows = 800
    k = 5

    stream = _synthetic_gaussian_mixture(rng, n, dim)
    queries = _synthetic_gaussian_mixture(rng, 20, dim)

    true_kde = compute_true_kde_angular(stream, queries, k)

    sketch = SlidingWindowAngularKDE(
        rows=rows, k=k, dim=dim, window_size=n + 1, rng=np.random.default_rng(1)
    )
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)

    race = RACE(rows=rows, k=k, dim=dim, rng=np.random.default_rng(2))
    for x in stream:
        race.update(x)

    sketch_estimates = np.array([sketch.query(q, t=n) for q in queries])
    race_estimates = np.array([race.query(q) for q in queries])

    sketch_relative_error = np.abs(sketch_estimates - true_kde) / np.maximum(true_kde, 1e-9)
    race_relative_error = np.abs(race_estimates - true_kde) / np.maximum(true_kde, 1e-9)

    assert sketch_relative_error.mean() < 0.3
    assert race_relative_error.mean() < 0.3
