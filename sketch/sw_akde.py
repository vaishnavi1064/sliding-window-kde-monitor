import numpy as np

from sketch.angular_hash import AngularHash
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
        rng = rng or np.random.default_rng()
        self.rows = rows
        self.k = k
        self.window_size = window_size
        self.eh_relative_error = eh_relative_error
        self.hash_functions = [
            [AngularHash(dim, rng) for _ in range(k)] for _ in range(rows)
        ]
        self.cells: dict[tuple[int, int], ExponentialHistogram] = {}

    def _cell_code(self, row: int, x) -> int:
        code = 0
        for h in self.hash_functions[row]:
            bit = 1 if h.eval(x) == 1 else 0
            code = code * 2 + bit
        return code

    def update(self, x, t: int) -> None:
        for row in range(self.rows):
            key = (row, self._cell_code(row, x))
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
        total = 0.0
        for row in range(self.rows):
            eh = self.cells.get((row, self._cell_code(row, x)))
            if eh is not None:
                total += eh.count_estimate(t)
        return total / self.rows
