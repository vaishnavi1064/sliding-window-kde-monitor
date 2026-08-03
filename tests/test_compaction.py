"""Cell reclamation: the memory half of Finding F.

Expiry is lazy and per-cell, so a cell that goes cold is never touched again and
holds its buckets forever. The paper's space bound assumes a dense R x W array
where that costs nothing; a sparse dictionary -- which any real implementation
uses, since most cells are empty -- grows one entry per distinct cell ever seen.
"""

import numpy as np

from sketch.sw_akde import SlidingWindowAngularKDE


def _sketch(window_size: int = 50, rows: int = 8, k: int = 3, dim: int = 4, **kwargs):
    return SlidingWindowAngularKDE(
        rows=rows,
        k=k,
        dim=dim,
        window_size=window_size,
        rng=np.random.default_rng(0),
        **kwargs,
    )


def test_cells_are_reclaimed_once_their_contents_expire():
    sketch = _sketch(window_size=50)
    rng = np.random.default_rng(1)

    # Fill with one region, then move permanently to a distant one.
    t = 0
    for _ in range(200):
        t += 1
        sketch.update(rng.normal(size=4) + 50.0, t)
    peak = len(sketch.cells)

    for _ in range(400):
        t += 1
        sketch.update(rng.normal(size=4) - 50.0, t)

    # The first region's cells cannot still be live: nothing has landed in them
    # for far longer than the window.
    assert sketch.cells_reclaimed > 0
    assert len(sketch.cells) < peak * 1.5


def test_bounded_memory_under_continuously_novel_input():
    # The pathological case for a sparse dictionary: every reading lands in a
    # region never seen before, so without reclamation the dict grows without
    # bound. With it, the live set stays proportional to the window.
    sketch = _sketch(window_size=100, rows=8, k=6, dim=4)
    rng = np.random.default_rng(2)

    sizes = []
    t = 0
    for step in range(3000):
        t += 1
        # Drift steadily so cells are continuously abandoned.
        sketch.update(rng.normal(size=4) + step * 0.5, t)
        if step % 500 == 499:
            sizes.append(len(sketch.cells))

    # Growth must flatten rather than track the number of readings.
    assert sizes[-1] < 4 * sizes[0], sizes
    assert sketch.cells_reclaimed > 0


def test_compaction_does_not_change_density_estimates():
    # Dropping a fully expired histogram must be semantically free: it
    # contributes exactly zero to any query.
    rng = np.random.default_rng(3)
    stream = rng.normal(size=(600, 4))
    queries = rng.normal(size=(10, 4))

    compacting = _sketch(window_size=80)
    never = _sketch(window_size=80, compact_every=10**9)

    for t, x in enumerate(stream, start=1):
        compacting.update(x, t)
        never.update(x, t)

    t = len(stream)
    for q in queries:
        assert compacting.query(q, t) == never.query(q, t)


def test_compaction_runs_on_schedule():
    sketch = _sketch(window_size=25)
    rng = np.random.default_rng(4)
    for t in range(1, 201):
        sketch.update(rng.normal(size=4), t)
    # 200 updates, one sweep per 25-tick window.
    assert sketch.compactions >= 200 // 25 - 1
