import numpy as np

from sketch.angular_hash import AngularHashBank


class RACE:
    """Plain un-windowed RACE (Coleman & Shrivastava, 2020) -- angular kernel.

    Used only as the tier-2 un-windowed oracle for validating SlidingWindowAngularKDE;
    not part of the production sketch.
    """

    def __init__(self, rows: int, k: int, dim: int, rng: np.random.Generator | None = None):
        self.rows = rows
        self.hashes = AngularHashBank(rows, k, dim, rng)
        self.counts: dict[tuple[int, int], int] = {}

    def update(self, x) -> None:
        codes = self.hashes.codes(x)
        for row in range(self.rows):
            key = (row, int(codes[row]))
            self.counts[key] = self.counts.get(key, 0) + 1

    def query(self, x, chunk_size: int = 5) -> float:
        codes = self.hashes.codes(x)
        values = np.array(
            [self.counts.get((row, int(codes[row])), 0) for row in range(self.rows)],
            dtype=float,
        )
        num_chunks = self.rows // chunk_size
        chunks = [values[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
        if self.rows % chunk_size != 0:
            chunks.append(values[num_chunks * chunk_size :])
        means = [chunk.mean() for chunk in chunks]
        return float(np.median(means))
