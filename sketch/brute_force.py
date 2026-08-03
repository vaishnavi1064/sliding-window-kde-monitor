"""Sketch-independent KDE ground truth.

These compute the true windowed density directly from the data, with no LSH or
sketch involved anywhere, and are the only thing we validate against -- never
the reference implementation's own outputs (CLAUDE.md Finding D).

Both return an unnormalized sum over the window, matching the scale of the
sketch estimators (Theorem 2.3: E[A[h(q)]] = sum of k^p(x,q) over the data).
Callers pass data already restricted to the window; no windowing happens here.
"""

import math

import numpy as np

from sketch.p_stable import l2_lsh_collision_probability


def compute_true_kde_angular(window: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Angular-kernel truth: sum over the window of (1 - angle/pi)^k."""
    window_norm = np.linalg.norm(window, axis=1, keepdims=True).clip(min=1e-12)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
    cosines = (query @ window.T) / (query_norm * window_norm.T)
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    return np.sum((1.0 - angles / math.pi) ** k, axis=1)


def compute_true_kde_l2(
    window: np.ndarray, query: np.ndarray, k: int, width: float
) -> np.ndarray:
    """Euclidean-kernel truth: sum over the window of p_collision(distance)^k.

    Note the `** k`. The reference's compute_true_kde_l2 omits it (its angular
    counterpart does not), so its L2 ground truth is only correct at k=1 -- which
    is what every driver in that repo passes, so its published numbers stand.
    """
    query_sq = np.sum(query**2, axis=1, keepdims=True)
    window_sq = np.sum(window**2, axis=1, keepdims=True).T
    squared = np.clip(query_sq + window_sq - 2.0 * (query @ window.T), a_min=0.0, a_max=None)
    distances = np.sqrt(squared)

    probabilities = np.vectorize(
        lambda d: l2_lsh_collision_probability(float(d), width)
    )(distances)
    return np.sum(probabilities**k, axis=1)
