// The RACE cell array whose counters are exponential histograms -- the native
// counterpart of the cell store in sketch/cell_store.py.
//
// Hashing deliberately stays in Python. It is already a single matmul over all
// rows*k hyperplanes (see AngularHashBank), and after that change it fell out
// of the profile entirely; docs/PERFORMANCE.md settles by measurement that the
// remaining cost is the scalar, branchy, allocation-heavy part ported here: the
// per-row loop, the cell lookup, and the histogram update.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exponential_histogram.hpp"

namespace swakde {

/// splitmix64's finaliser. Cell codes are dense small integers -- [0, 2^k) for
/// the angular kernel, a polynomial fold mod hash_range for the Euclidean one --
/// and the standard library hashes integers with the identity, which leaves that
/// structure intact. One multiply-xor round costs less than the probe chains it
/// avoids.
struct MixHash {
    std::size_t operator()(std::int64_t key) const noexcept {
        std::uint64_t x = static_cast<std::uint64_t>(key) + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<std::size_t>(x ^ (x >> 31));
    }
};

class CellArray {
public:
    using Row = std::unordered_map<std::int64_t, ExponentialHistogram, MixHash>;

    CellArray(std::int64_t rows, std::int64_t window_size, double eh_relative_error,
              std::int64_t compact_every)
        // Validation has to happen before any member is initialised: a negative
        // `rows` would otherwise size the vector from a huge size_t, and a
        // non-positive relative error would send histogram_k through a cast of
        // infinity, both before a constructor body could reject them.
        : rows_(checked_parameters(rows, window_size, eh_relative_error, compact_every)),
          window_size_(window_size),
          eh_relative_error_(eh_relative_error),
          k_(histogram_k(eh_relative_error)),
          merge_cap_(histogram_merge_cap(histogram_k(eh_relative_error))),
          compact_every_(compact_every),
          next_compaction_(compact_every) {}

    /// Record one element, given its per-row cell codes, at logical time `t`.
    /// `t` must be a monotonic per-event counter, not a wall clock (Finding B).
    void update(const std::int64_t* codes, std::int64_t t) {
        for (std::size_t row = 0; row < rows_.size(); ++row) {
            // Unconditional add on both the found and created paths. The
            // reference adds only when the cell already exists, so every cell
            // silently drops the element that created it (Finding A).
            rows_[row][codes[row]].add(t, window_size_, merge_cap_);
        }

        if (t >= next_compaction_) {
            compact(t);
            next_compaction_ = t + compact_every_;
        }
    }

    /// Mean cell count across rows, per the paper's section 4.1 (median-of-means
    /// is plain RACE's estimator, not SW-AKDE's). Accumulated in row order so
    /// the floating-point sum is bit-for-bit what the Python produces.
    double query(const std::int64_t* codes, std::int64_t t) {
        double total = 0.0;
        for (std::size_t row = 0; row < rows_.size(); ++row) {
            Row& cells = rows_[row];
            auto it = cells.find(codes[row]);
            if (it != cells.end()) total += it->second.count_estimate(t, window_size_);
        }
        return total / static_cast<double>(rows_.size());
    }

    /// Drop cells whose contents have all expired; returns how many went.
    ///
    /// The memory half of Finding F. Expiry is lazy and per-cell, so a cell that
    /// goes cold is never revisited and holds its buckets forever. The published
    /// space bound counts a dense array where an unused cell is free; any sparse
    /// implementation instead gains an entry per distinct cell ever visited, so
    /// memory tracks elements seen rather than window size. Dropping a fully
    /// expired histogram is semantically free -- it contributes exactly zero to
    /// any query.
    std::int64_t compact(std::int64_t t) {
        std::int64_t dead = 0;
        for (Row& cells : rows_) {
            for (auto it = cells.begin(); it != cells.end();) {
                if (it->second.is_expired(t, window_size_)) {
                    it = cells.erase(it);
                    ++dead;
                } else {
                    ++it;
                }
            }
        }
        ++compactions_;
        cells_reclaimed_ += dead;
        return dead;
    }

    std::int64_t rows() const { return static_cast<std::int64_t>(rows_.size()); }
    std::int64_t window_size() const { return window_size_; }
    double eh_relative_error() const { return eh_relative_error_; }
    std::int64_t k() const { return k_; }
    std::int64_t merge_cap() const { return merge_cap_; }
    std::int64_t compact_every() const { return compact_every_; }
    std::int64_t compactions() const { return compactions_; }
    std::int64_t cells_reclaimed() const { return cells_reclaimed_; }

    std::int64_t cell_count() const {
        std::int64_t n = 0;
        for (const Row& cells : rows_) n += static_cast<std::int64_t>(cells.size());
        return n;
    }

    std::int64_t bucket_count() const {
        std::int64_t n = 0;
        for (const Row& cells : rows_) {
            for (const auto& cell : cells) n += cell.second.bucket_count();
        }
        return n;
    }

    /// Structural cost of the cell array: the row maps, one node per live cell,
    /// and each histogram's bucket buffer.
    ///
    /// Node and slot overhead is inferred, not reported -- the standard library
    /// exposes neither -- so this is an estimate of the same kind as the
    /// sys.getsizeof walk in scripts/memory_check.py. The Phase 5 benchmark
    /// quotes process RSS alongside it so this figure is never the only
    /// evidence for a memory claim.
    std::size_t memory_bytes() const {
        std::size_t bytes = sizeof(*this) + rows_.capacity() * sizeof(Row);
        for (const Row& cells : rows_) {
            // bucket_count() here is the hash table's slot array, unrelated to
            // the histogram buckets counted below.
            bytes += cells.bucket_count() * sizeof(void*);
            for (const auto& cell : cells) {
                bytes += sizeof(std::pair<const std::int64_t, ExponentialHistogram>)
                         + sizeof(void*);  // node's next pointer
                bytes += cell.second.capacity() * sizeof(Bucket);
            }
        }
        return bytes;
    }

    /// One cell's histogram, or nullptr when that cell is not live. Lets the
    /// parity tests compare bucket-level state, not just query output.
    const ExponentialHistogram* cell(std::int64_t row, std::int64_t code) const {
        if (row < 0 || static_cast<std::size_t>(row) >= rows_.size()) return nullptr;
        const Row& cells = rows_[static_cast<std::size_t>(row)];
        auto it = cells.find(code);
        return it == cells.end() ? nullptr : &it->second;
    }

private:
    /// Rejects every invalid argument and returns the row count, so the checks
    /// run ahead of the member initialisations that would misbehave on them.
    static std::size_t checked_parameters(std::int64_t rows, std::int64_t window_size,
                                         double eh_relative_error,
                                         std::int64_t compact_every) {
        if (rows <= 0) throw std::invalid_argument("rows must be positive");
        if (window_size <= 0) throw std::invalid_argument("window_size must be positive");
        // Written as !(x > 0) so NaN is rejected too.
        if (!(eh_relative_error > 0.0)) {
            throw std::invalid_argument("eh_relative_error must be positive");
        }
        if (compact_every <= 0) throw std::invalid_argument("compact_every must be positive");
        return static_cast<std::size_t>(rows);
    }

    std::vector<Row> rows_;
    std::int64_t window_size_;
    double eh_relative_error_;
    std::int64_t k_;
    std::int64_t merge_cap_;
    std::int64_t compact_every_;
    std::int64_t next_compaction_;
    std::int64_t compactions_ = 0;
    std::int64_t cells_reclaimed_ = 0;
};

}  // namespace swakde
