// pybind11 bindings for the native core (Phase 5).
//
// Exposes exactly two things: the cell array the sketch runs on, and a
// standalone exponential histogram used only so the parity tests can compare
// bucket-level state against the Python oracle rather than just query output.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cell_array.hpp"
#include "exponential_histogram.hpp"

namespace py = pybind11;

namespace {

/// A single histogram with its parameters attached, mirroring the constructor
/// signature of sketch.exponential_histogram.ExponentialHistogram.
///
/// The core class takes window_size and merge_cap per call so that ~10^5 live
/// cells need not each store them; this facade holds them for the one case
/// where a histogram is used on its own.
class Histogram {
public:
    Histogram(std::int64_t window_size, double relative_error)
        : window_size_(window_size),
          k_(swakde::histogram_k(relative_error)),
          merge_cap_(swakde::histogram_merge_cap(swakde::histogram_k(relative_error))) {
        if (window_size <= 0) throw std::invalid_argument("window_size must be positive");
        if (!(relative_error > 0.0)) throw std::invalid_argument("relative_error must be positive");
    }

    void add(std::int64_t t) { eh_.add(t, window_size_, merge_cap_); }
    double count_estimate(std::int64_t t) { return eh_.count_estimate(t, window_size_); }
    bool is_expired(std::int64_t t) { return eh_.is_expired(t, window_size_); }

    std::int64_t window_size() const { return window_size_; }
    std::int64_t k() const { return k_; }
    std::int64_t merge_cap() const { return merge_cap_; }
    std::int64_t total() const { return eh_.total(); }
    std::int64_t last() const { return eh_.last(); }

    /// (timestamp, size) oldest first -- the same order as the Python list.
    std::vector<std::pair<std::int64_t, std::int64_t>> buckets() const {
        std::vector<std::pair<std::int64_t, std::int64_t>> out;
        out.reserve(eh_.bucket_count());
        for (std::uint32_t i = 0; i < eh_.bucket_count(); ++i) {
            const swakde::Bucket& b = eh_.bucket(i);
            out.emplace_back(b.timestamp, b.size);
        }
        return out;
    }

private:
    swakde::ExponentialHistogram eh_;
    std::int64_t window_size_;
    std::int64_t k_;
    std::int64_t merge_cap_;
};

// forcecast so an int32 or non-contiguous code array is converted rather than
// rejected; the hash banks already produce contiguous int64, so this is a
// safety net that normally does nothing.
using CodeArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

const std::int64_t* codes_pointer(const CodeArray& codes, std::int64_t rows) {
    if (codes.ndim() != 1) {
        throw std::invalid_argument("codes must be a 1-D array, one cell code per row");
    }
    if (codes.shape(0) != rows) {
        throw std::invalid_argument("codes has " + std::to_string(codes.shape(0))
                                    + " entries but the sketch has " + std::to_string(rows)
                                    + " rows");
    }
    return codes.data();
}

std::string compiler_description() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "unknown";
#endif
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "C++17 core for the SW-AKDE sketch (Phase 5). Semantics are pinned to "
              "the pure-Python oracle by tests/test_native_parity.py.";
    m.attr("__compiler__") = compiler_description();

    py::class_<Histogram>(m, "ExponentialHistogram")
        .def(py::init<std::int64_t, double>(), py::arg("window_size"), py::arg("relative_error"))
        .def("add", &Histogram::add, py::arg("t"))
        .def("count_estimate", &Histogram::count_estimate, py::arg("t"))
        .def("is_expired", &Histogram::is_expired, py::arg("t"))
        .def("buckets", &Histogram::buckets)
        .def_property_readonly("window_size", &Histogram::window_size)
        .def_property_readonly("k", &Histogram::k)
        .def_property_readonly("merge_cap", &Histogram::merge_cap)
        .def_property_readonly("total", &Histogram::total)
        .def_property_readonly("last", &Histogram::last);

    py::class_<swakde::CellArray>(m, "CellArray")
        .def(py::init<std::int64_t, std::int64_t, double, std::int64_t>(), py::arg("rows"),
             py::arg("window_size"), py::arg("eh_relative_error"), py::arg("compact_every"))
        .def(
            "update",
            [](swakde::CellArray& self, const CodeArray& codes, std::int64_t t) {
                const std::int64_t* pointer = codes_pointer(codes, self.rows());
                // Nothing below touches a Python object, and `codes` keeps the
                // buffer alive for the duration, so the row loop can run without
                // the GIL -- which also lets the consumer thread overlap it.
                py::gil_scoped_release unlock;
                self.update(pointer, t);
            },
            py::arg("codes"), py::arg("t"))
        .def(
            "query",
            [](swakde::CellArray& self, const CodeArray& codes, std::int64_t t) {
                const std::int64_t* pointer = codes_pointer(codes, self.rows());
                py::gil_scoped_release unlock;
                return self.query(pointer, t);
            },
            py::arg("codes"), py::arg("t"))
        .def("compact", &swakde::CellArray::compact, py::arg("t"),
             py::call_guard<py::gil_scoped_release>())
        .def("memory_bytes", &swakde::CellArray::memory_bytes)
        .def(
            "cell_state",
            [](const swakde::CellArray& self, std::int64_t row, std::int64_t code) -> py::object {
                const swakde::ExponentialHistogram* eh = self.cell(row, code);
                if (eh == nullptr) return py::none();
                std::vector<std::pair<std::int64_t, std::int64_t>> buckets;
                buckets.reserve(eh->bucket_count());
                for (std::uint32_t i = 0; i < eh->bucket_count(); ++i) {
                    buckets.emplace_back(eh->bucket(i).timestamp, eh->bucket(i).size);
                }
                return py::make_tuple(eh->total(), eh->last(), py::cast(buckets));
            },
            py::arg("row"), py::arg("code"),
            "(total, last, [(timestamp, size), ...]) for one cell, or None if it is not live.")
        .def_property_readonly("rows", &swakde::CellArray::rows)
        .def_property_readonly("window_size", &swakde::CellArray::window_size)
        .def_property_readonly("eh_relative_error", &swakde::CellArray::eh_relative_error)
        .def_property_readonly("k", &swakde::CellArray::k)
        .def_property_readonly("merge_cap", &swakde::CellArray::merge_cap)
        .def_property_readonly("compact_every", &swakde::CellArray::compact_every)
        .def_property_readonly("compactions", &swakde::CellArray::compactions)
        .def_property_readonly("cells_reclaimed", &swakde::CellArray::cells_reclaimed)
        .def_property_readonly("cell_count", &swakde::CellArray::cell_count)
        .def_property_readonly("bucket_count", &swakde::CellArray::bucket_count);
}
