// DGIM exponential histogram -- the native counterpart of
// sketch/exponential_histogram.py.
//
// The Python implementation stays the oracle (CLAUDE.md section 13): every
// difference here is representational, never behavioural, and
// tests/test_native_parity.py pins the two together bucket for bucket.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace swakde {

// Matches the Python Bucket (size + timestamp), at 16 bytes instead of a
// 56-byte PyObject plus an 8-byte list slot. There is one of these per element
// currently inside a cell's window, so this is where the memory goes.
struct Bucket {
    std::int64_t timestamp;
    std::int64_t size;
};

// k = ceil(1 / eps), matching Python's math.ceil(1 / relative_error).
inline std::int64_t histogram_k(double relative_error) {
    return static_cast<std::int64_t>(std::ceil(1.0 / relative_error));
}

// merge_cap = ceil(k / 2) + 1: how many buckets may share a size before two of
// them are merged. Python computes math.ceil(self.k / 2) + 1 on a float
// division, so the division is done in double here too.
inline std::int64_t histogram_merge_cap(std::int64_t k) {
    return static_cast<std::int64_t>(std::ceil(static_cast<double>(k) / 2.0)) + 1;
}

/// A sliding-window counter in O(log N / eps) space (Datar et al. 2002).
///
/// Two deliberate departures from the Python shape, both to serve the ~10^5
/// live cells a real run holds:
///
///  1. Buckets live in a power-of-two ring buffer rather than a list, so
///     evicting the oldest is a head bump instead of the O(bucket count) shift
///     that `list.pop(0)` performs. In a dense stream almost every `add`
///     evicts, so that shift was on the hot path.
///  2. `window_size` and `merge_cap` are passed per call instead of stored.
///     They are identical for every cell in a sketch, and storing them would
///     cost 16 bytes per cell to hold two numbers the owning CellArray already
///     knows.
class ExponentialHistogram {
public:
    void add(std::int64_t t, std::int64_t window_size, std::int64_t merge_cap) {
        expire(t, window_size);
        push_back(Bucket{t, 1});
        total_ += 1;

        // Cascading merge, step for step with the Python: walk back from the
        // newest bucket; wherever more than merge_cap buckets share a size,
        // drop the oldest of that run and double the next one, then continue
        // from there. `i` and `j` are signed because `j` walks off the front of
        // the buffer to -1.
        std::int64_t i = static_cast<std::int64_t>(count_) - 1;
        while (i > 0) {
            std::int64_t j = i - 1;
            std::int64_t run = 1;
            const std::int64_t size = at(i).size;
            while (j >= 0 && at(j).size == size) {
                ++run;
                --j;
            }
            if (run <= merge_cap) break;
            // Delete the run's oldest bucket, then double what has shifted
            // into its place -- the element formerly at j + 2.
            erase_at(j + 1);
            at(j + 1).size = 2 * size;
            if (j + 1 == 0) last_ = 2 * size;
            i = j + 1;
        }
    }

    /// DGIM's estimate: everything counted, less half the oldest bucket, whose
    /// contents straddle the window boundary by an unknown amount.
    double count_estimate(std::int64_t t, std::int64_t window_size) {
        // Expiry before reading, not just on update: a cell that stopped being
        // hit would otherwise report a frozen count forever, and a region going
        // quiet is exactly our anomaly signal (Finding F).
        expire(t, window_size);
        return static_cast<double>(total_) - static_cast<double>(last_) / 2.0;
    }

    /// True when nothing here is still inside the window, so the owning cell
    /// contributes exactly zero to any query and can be dropped outright.
    bool is_expired(std::int64_t t, std::int64_t window_size) {
        expire(t, window_size);
        return count_ == 0;
    }

    std::int64_t total() const { return total_; }
    std::int64_t last() const { return last_; }
    std::uint32_t bucket_count() const { return count_; }
    std::size_t capacity() const { return buf_.size(); }
    const Bucket& bucket(std::int64_t i) const { return peek(i); }

private:
    static constexpr std::size_t kInitialCapacity = 4;

    void expire(std::int64_t t, std::int64_t window_size) {
        // A `while`, not the reference's single `if`. One cell's histogram is
        // only touched when that cell is hit, which can skip arbitrarily many
        // logical ticks, leaving several buckets expired at once (Finding B).
        //
        // The boundary is `<=`: the window is the last `window_size` elements,
        // i.e. timestamps in (t - window_size, t], per the paper's own
        // T_t = {t-N+1, ..., t}.
        const std::int64_t cutoff = t - window_size;
        while (count_ > 0 && peek(0).timestamp <= cutoff) evict_oldest();
    }

    void evict_oldest() {
        total_ -= peek(0).size;
        head_ = (head_ + 1) & mask_;
        --count_;
        last_ = count_ > 0 ? peek(0).size : 0;
    }

    Bucket& at(std::int64_t i) {
        return buf_[(head_ + static_cast<std::size_t>(i)) & mask_];
    }

    const Bucket& peek(std::int64_t i) const {
        return buf_[(head_ + static_cast<std::size_t>(i)) & mask_];
    }

    void push_back(const Bucket& b) {
        if (static_cast<std::size_t>(count_) == buf_.size()) grow();
        buf_[(head_ + static_cast<std::size_t>(count_)) & mask_] = b;
        ++count_;
    }

    // Logical erase: shift the newer side down over the hole. Bounded by the
    // number of buckets newer than `p`, which the merge cascade keeps to
    // O(merge_cap) -- the run being merged sits near the tail.
    void erase_at(std::int64_t p) {
        for (std::int64_t q = p; q + 1 < static_cast<std::int64_t>(count_); ++q) {
            at(q) = at(q + 1);
        }
        --count_;
    }

    void grow() {
        // Capacity stays a power of two so indexing is a mask rather than a
        // modulo. Growth unrolls the ring back to head_ = 0.
        const std::size_t next_capacity = buf_.empty() ? kInitialCapacity : buf_.size() * 2;
        std::vector<Bucket> next(next_capacity);
        for (std::uint32_t q = 0; q < count_; ++q) next[q] = peek(q);
        buf_.swap(next);
        head_ = 0;
        mask_ = static_cast<std::uint32_t>(next_capacity - 1);
    }

    std::vector<Bucket> buf_;
    std::uint32_t head_ = 0;
    std::uint32_t count_ = 0;
    std::uint32_t mask_ = 0;  // capacity - 1; valid whenever count_ > 0
    std::int64_t total_ = 0;
    std::int64_t last_ = 0;
};

}  // namespace swakde
