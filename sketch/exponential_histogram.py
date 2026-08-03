import math

from sketch.buckets import Bucket


class ExponentialHistogram:
    def __init__(self, window_size: int, relative_error: float):
        self.window_size = window_size
        self.k = math.ceil(1 / relative_error)
        self.buckets: list[Bucket] = []
        self.last = 0
        self.total = 0

    def _evict_oldest(self) -> None:
        expired = self.buckets.pop(0)
        self.total -= expired.size
        self.last = self.buckets[0].size if self.buckets else 0

    def _expire(self, t: int) -> None:
        # A single cell's EH only gets touched when its own bucket is hit,
        # which can skip arbitrarily many logical-clock ticks (every other
        # element landed in a different cell/row). A gap that long can leave
        # more than one bucket expired at once, so eviction must be a loop --
        # a single check (as in the reference) under-evicts stale buckets
        # for any cell that goes cold for a while. See CLAUDE.md Finding B.
        #
        # Boundary is `<=`, not `<` (as in the reference): the window is the
        # last `window_size` elements, i.e. timestamps in (t - window_size, t]
        # per the paper's own T_t = {t-N+1, ..., t} (Problem 1.2) -- `<` keeps
        # one extra stale element at the edge.
        while self.buckets and self.buckets[0].timestamp <= t - self.window_size:
            self._evict_oldest()

    def add(self, t: int) -> None:
        self._expire(t)
        self.buckets.append(Bucket(t))
        self.total += 1
        cap = math.ceil(self.k / 2) + 1

        i = len(self.buckets) - 1
        while i > 0:
            j = i - 1
            run = 1
            size = self.buckets[i].size
            while j >= 0 and self.buckets[j].size == size:
                run += 1
                j -= 1
            if run <= cap:
                break
            del self.buckets[j + 1]
            self.buckets[j + 1].size = 2 * size
            if j + 1 == 0:
                self.last = 2 * size
            i = j + 1

    def count_estimate(self, t: int) -> float:
        # Expiry is lazy, so a cell that stopped receiving elements still holds
        # its old buckets -- reading without expiring first returns a frozen,
        # arbitrarily stale count. The reference's count_est() takes no time
        # argument at all and so cannot expire on read (nor can the paper's
        # Algorithm 2 query procedure, which reads the cell directly). That
        # breaks sliding-window semantics exactly where it matters most for
        # anomaly detection: a region going quiet is precisely the signal, and
        # without this the density never decays. See CLAUDE.md Finding F.
        self._expire(t)
        return self.total - self.last / 2.0
