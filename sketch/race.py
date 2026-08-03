import numpy as np

from sketch.angular_hash import AngularHash


class RACE:
    """Plain un-windowed RACE (Coleman & Shrivastava, 2020) -- angular kernel.

    Used only as the tier-2 un-windowed oracle for validating SlidingWindowAngularKDE;
    not part of the production sketch.
    """

    def __init__(self, rows: int, k: int, dim: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.rows = rows
        self.hash_functions = [
            [AngularHash(dim, rng) for _ in range(k)] for _ in range(rows)
        ]
        self.counts: dict[tuple[int, int], int] = {}

    def _cell_code(self, row: int, x) -> int:
        code = 0
        for h in self.hash_functions[row]:
            bit = 1 if h.eval(x) == 1 else 0
            code = code * 2 + bit
        return code

    def update(self, x) -> None:
        for row in range(self.rows):
            key = (row, self._cell_code(row, x))
            self.counts[key] = self.counts.get(key, 0) + 1

    def query(self, x, chunk_size: int = 5) -> float:
        values = np.array(
            [self.counts.get((row, self._cell_code(row, x)), 0) for row in range(self.rows)],
            dtype=float,
        )
        num_chunks = self.rows // chunk_size
        chunks = [values[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
        if self.rows % chunk_size != 0:
            chunks.append(values[num_chunks * chunk_size :])
        means = [chunk.mean() for chunk in chunks]
        return float(np.median(means))
