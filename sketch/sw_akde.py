import numpy as np

from sketch.angular_hash import AngularHashBank
from sketch.exponential_histogram import ExponentialHistogram


class SlidingWindowAngularKDE:
    def __init__(
        self,
        rows: int,
        k: int,
        dim: int,
        window_size: int,
        eh_relative_error: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        self.rows = rows
        self.k = k
        self.window_size = window_size
        self.eh_relative_error = eh_relative_error
        self.hashes = AngularHashBank(rows, k, dim, rng)
        self.cells: dict[tuple[int, int], ExponentialHistogram] = {}

    def update(self, x, t: int) -> None:
        codes = self.hashes.codes(x)
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

    def query(self, x, t: int) -> float:
        # `t` is the current logical clock, required so cells that have gone
        # cold expire on read rather than reporting a frozen count (Finding F).
        codes = self.hashes.codes(x)
        total = 0.0
        for row in range(self.rows):
            eh = self.cells.get((row, int(codes[row])))
            if eh is not None:
                total += eh.count_estimate(t)
        return total / self.rows
