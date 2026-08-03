import math

import numpy as np


def compute_true_kde_angular(window: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Sketch-independent angular-kernel KDE ground truth: sum, over `window`, of the
    per-row collision probability (1 - angle/pi)^k -- the same unnormalized scale as
    SlidingWindowAngularKDE.query() / RACE.query() (Theorem 2.3: E[A[h(q)]] = sum k^p(x,q)).
    Caller passes the already-windowed data (last N rows); no windowing done here.
    """
    window_norm = np.linalg.norm(window, axis=1, keepdims=True).clip(min=1e-12)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
    cosines = (query @ window.T) / (query_norm * window_norm.T)
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    return np.sum((1.0 - angles / math.pi) ** k, axis=1)
