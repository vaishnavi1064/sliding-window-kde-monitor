"""Build a sketch from settings, so producer/consumer/evaluation agree on config."""

from sketch.sw_akde import (
    SlidingWindowAngularKDE,
    SlidingWindowEuclideanKDE,
    SlidingWindowKDE,
)
from streaming.config import SETTINGS


def build_sketch(kernel: str, dim: int, seed: int | None = 0) -> SlidingWindowKDE:
    import numpy as np

    rng = np.random.default_rng(seed)

    if kernel == "euclidean":
        # Default for sensor data: the angular kernel is scale-invariant, so a
        # channel drifting in magnitude along a fixed direction is invisible to
        # it -- and on MetroPT the absolute level is the fault signal.
        return SlidingWindowEuclideanKDE(
            rows=SETTINGS.rows,
            k=SETTINGS.k,
            dim=dim,
            width=SETTINGS.lsh_width,
            window_size=SETTINGS.window_size,
            eh_relative_error=SETTINGS.eh_relative_error,
            rng=rng,
            backend=SETTINGS.sketch_backend,
        )
    if kernel == "angular":
        return SlidingWindowAngularKDE(
            rows=SETTINGS.rows,
            k=SETTINGS.k,
            dim=dim,
            window_size=SETTINGS.window_size,
            eh_relative_error=SETTINGS.eh_relative_error,
            rng=rng,
            backend=SETTINGS.sketch_backend,
        )
    raise ValueError(f"unknown kernel: {kernel}")
