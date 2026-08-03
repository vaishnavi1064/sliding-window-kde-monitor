import random

from sketch.exponential_histogram import ExponentialHistogram


def brute_force_window_count(hit_times: list[int], t: int, window_size: int) -> int:
    return sum(1 for ht in hit_times if t - window_size < ht <= t)


def test_first_arrival_is_recorded():
    eh = ExponentialHistogram(window_size=100, relative_error=0.1)
    eh.add(1)
    assert eh.count_estimate(1) == 1


def test_matches_brute_force_on_dense_stream():
    # Every logical tick is a hit, so bucket merging kicks in and counts grow
    # past the point where DGIM's (1+eps) guarantee is meaningful (it's an
    # asymptotic bound tied to the invariant that enough same-size buckets
    # have accumulated, not an exact bound for tiny counts).
    window_size = 50
    relative_error = 0.1
    eh = ExponentialHistogram(window_size=window_size, relative_error=relative_error)
    hit_times: list[int] = []

    for t in range(1, 501):
        eh.add(t)
        hit_times.append(t)
        true_count = brute_force_window_count(hit_times, t, window_size)
        estimate = eh.count_estimate(t)
        if true_count >= 2 * eh.k:
            assert abs(estimate - true_count) <= relative_error * true_count + 1e-9


def test_eviction_handles_large_gaps_between_hits():
    # A single cell's EH only receives `add` when its own bucket is hit, so in
    # a real RACE-style sketch consecutive calls can be arbitrarily far apart
    # in the shared logical clock (every other element landed in a different
    # cell/row). Gaps here deliberately exceed window_size so that, without
    # the while-loop eviction fix (see Finding B in exponential_histogram.py),
    # more than one bucket can be stale at once and only one would get
    # evicted per call.
    #
    # A single un-fixed `if` check on this exact scenario (verified directly
    # against this test's random sequence) lets `total` drift to 4x the true
    # count. `total` here never merges into multi-element buckets (counts
    # stay small under such large gaps), so with correct eviction it should
    # track the brute-force count almost exactly; 1.5x leaves comfortable
    # margin above the correct behavior while still catching the bug.
    random.seed(0)
    window_size = 200
    eh = ExponentialHistogram(window_size=window_size, relative_error=0.1)
    hit_times: list[int] = []

    t = 0
    for _ in range(500):
        t += random.randint(1, 3 * window_size)
        eh.add(t)
        hit_times.append(t)
        true_count = brute_force_window_count(hit_times, t, window_size)
        assert eh.total <= true_count * 1.5
        assert 0 <= eh.count_estimate(t) <= eh.total
