import numpy as np


class AngularHash:
    def __init__(self, dim: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.w = rng.normal(0, 1, size=dim)

    def eval(self, x) -> int:
        return 1 if np.dot(self.w, x) >= 0 else -1
