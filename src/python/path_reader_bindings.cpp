// Python access to the paths of a container without decompressing it:
// gfa_compression.PathReader. Returns numpy arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "compute/indexed_paths.hpp"
#include "core/codec/serialization.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

template <typename T> py::array_t<T> to_numpy(const std::vector<T> &values) {
  return py::array_t<T>(values.size(), values.data());
}

class PathReader {
public:
  PathReader(const std::string &path, const std::string &index_path,
             uint32_t block_steps, int num_threads) {
    py::gil_scoped_release release;
    const gfaz::CompressedData data = gfaz::deserialize_compressed_data(path);
    paths_ = std::make_unique<gfaz::IndexedPaths>(
        data, gfaz::IndexedPathsOptions{index_path, block_steps, num_threads});
  }

  size_t num_haplotypes() const { return paths_->catalog.size(); }
  uint32_t num_nodes() const { return paths_->num_nodes(); }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    for (size_t h = 0; h < num_haplotypes(); ++h)
      out.push_back(paths_->catalog[h].name);
    return out;
  }

  std::vector<std::string> contigs() const {
    std::vector<std::string> out;
    for (size_t h = 0; h < num_haplotypes(); ++h)
      out.push_back(paths_->catalog[h].contig);
    return out;
  }

  std::vector<uint64_t> contig_offsets() const {
    std::vector<uint64_t> out;
    for (size_t h = 0; h < num_haplotypes(); ++h)
      out.push_back(paths_->catalog[h].contig_offset);
    return out;
  }

  std::vector<uint32_t> haplotypes_on(const std::string &contig) const {
    return paths_->catalog.on_contig(contig);
  }

  uint32_t step_count(size_t hap) const {
    return paths_->index.step_count(checked(hap));
  }
  uint64_t length(size_t hap) const {
    return paths_->index.bp_length(checked(hap));
  }

  py::array_t<uint32_t> node_lengths() const {
    return to_numpy(paths_->segment_lengths);
  }

  py::array_t<int32_t> steps(size_t hap, uint32_t start,
                             py::object stop) const {
    const uint32_t count = step_count(hap);
    const uint32_t end =
        stop.is_none() ? count : std::min(count, stop.cast<uint32_t>());
    std::vector<int32_t> nodes;
    if (start < end) {
      py::gil_scoped_release release;
      nodes.reserve(end - start);
      paths_->access.for_each_step(
          hap, start, end,
          [&](const gfaz::PathStep &step) { nodes.push_back(step.node); });
    }
    return to_numpy(nodes);
  }

  py::tuple steps_in_bases(size_t hap, uint64_t bp_begin,
                           uint64_t bp_end) const {
    checked(hap);
    std::vector<int32_t> nodes;
    std::vector<uint64_t> starts;
    uint32_t first_step = 0;
    {
      py::gil_scoped_release release;
      paths_->access.for_each_step_in_bases(
          hap, bp_begin, bp_end, [&](const gfaz::PathStep &step) {
            if (nodes.empty())
              first_step = step.step;
            nodes.push_back(step.node);
            starts.push_back(step.bp_begin);
          });
    }
    return py::make_tuple(first_step, to_numpy(nodes), to_numpy(starts));
  }

  py::dict locate(uint32_t node) const {
    if (node == 0 || node > num_nodes())
      throw std::out_of_range("node id outside 1..num_nodes");
    std::vector<uint32_t> haps, steps;
    std::vector<uint64_t> starts;
    std::vector<uint8_t> reverse;
    {
      py::gil_scoped_release release;
      paths_->access.for_each_step_on_nodes(
          node, node, [&](size_t hap, const gfaz::PathStep &step) {
            haps.push_back(static_cast<uint32_t>(hap));
            steps.push_back(step.step);
            starts.push_back(step.bp_begin);
            reverse.push_back(step.node < 0);
          });
    }
    py::dict out;
    out["haplotype"] = to_numpy(haps);
    out["step"] = to_numpy(steps);
    out["start"] = to_numpy(starts);
    out["reverse"] = to_numpy(reverse).cast<py::array_t<bool>>();
    return out;
  }

private:
  size_t checked(size_t hap) const {
    if (hap >= num_haplotypes())
      throw std::out_of_range("haplotype index out of range");
    return hap;
  }

  std::unique_ptr<gfaz::IndexedPaths> paths_;
};

} // namespace

void bind_path_reader(py::module_ &m) {
  py::class_<PathReader>(
      m, "PathReader",
      "Random access to the paths and walks of a .gfaz without decompressing "
      "it. Haplotypes are numbered [paths..., walks...]; node ids are the "
      "container's 1-based ids, negative for reverse orientation.")
      .def(py::init<const std::string &, const std::string &, uint32_t, int>(),
           py::arg("path"), py::arg("index_path") = "",
           py::arg("block_steps") = 4096, py::arg("num_threads") = 0,
           "Open a container. index_path names a saved `gfaz index` file; "
           "when empty the index is built here.")
      .def_property_readonly("num_haplotypes", &PathReader::num_haplotypes)
      .def_property_readonly("num_nodes", &PathReader::num_nodes)
      .def_property_readonly("names", &PathReader::names,
                             "Display name of every haplotype.")
      .def_property_readonly("contigs", &PathReader::contigs,
                             "P-line name or sample#hap#seq of every haplotype.")
      .def_property_readonly("contig_offsets", &PathReader::contig_offsets,
                             "Contig coordinate of each haplotype's first base.")
      .def("haplotypes_on", &PathReader::haplotypes_on, py::arg("contig"),
           "Indices of the haplotypes lying on a contig.")
      .def("step_count", &PathReader::step_count, py::arg("haplotype"))
      .def("length", &PathReader::length, py::arg("haplotype"),
           "Length of a haplotype in bases.")
      .def("node_lengths", &PathReader::node_lengths,
           "uint32 array; entry n - 1 is the length of node n.")
      .def("steps", &PathReader::steps, py::arg("haplotype"),
           py::arg("start") = 0, py::arg("stop") = py::none(),
           "Signed node ids of steps [start, stop) as an int32 array.")
      .def("steps_in_bases", &PathReader::steps_in_bases, py::arg("haplotype"),
           py::arg("bp_begin"), py::arg("bp_end"),
           "Steps overlapping bases [bp_begin, bp_end) of a haplotype: "
           "(index of the first step, int32 node ids, uint64 start offsets).")
      .def("locate", &PathReader::locate, py::arg("node"),
           "Every visit to a node: dict of arrays haplotype, step, start "
           "(base offset in the haplotype) and reverse.");
}
