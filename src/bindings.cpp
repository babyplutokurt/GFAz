#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "add_haplotypes_workflow.hpp"
#include "compression_workflow.hpp"
#include "debug_log.hpp"
#include "decompression_workflow.hpp"
#include "extraction_workflow.hpp"
#include "gfa_parser.hpp"
#include "gfa_writer.hpp"
#include "gpu/gfa_graph_gpu.hpp"
#include "serialization.hpp"
#include <iostream>
#include <stack>

#ifdef ENABLE_CUDA
#include "gpu/codec_gpu.cuh"
#include "gpu/codec_gpu_nvcomp.cuh"
#include "gpu/compression_workflow_gpu.hpp"
#include "gpu/decompression_workflow_gpu.hpp"
#include "gpu/serialization_gpu.hpp"
#endif

namespace py = pybind11;

namespace {
constexpr int kDefaultRounds = 8;
constexpr int kDefaultFreqThreshold = 2;
constexpr int kDefaultDeltaRound = 1;
constexpr int kDefaultNumThreads = 0;
} // namespace

GfaGraph parse_gfa_file(const std::string &file_path) {
  GfaParser parser;
  return parser.parse(file_path);
}

std::vector<int32_t>
reconstruct_path_cpu(const std::vector<int32_t> &compressed_path,
                     const std::map<uint32_t, uint64_t> &rulebook,
                     uint32_t min_rule_id) {
  std::vector<int32_t> result;
  result.reserve(compressed_path.size() * 2);

  for (int32_t root_node : compressed_path) {
    std::stack<int32_t> stack;
    stack.push(root_node);

    while (!stack.empty()) {
      int32_t node = stack.top();
      stack.pop();
      int32_t abs_node = std::abs(node);

      if ((uint32_t)abs_node >= min_rule_id) {
        auto it = rulebook.find(abs_node);
        if (it != rulebook.end()) {
          uint64_t kmer = it->second;
          int32_t first = (int32_t)(kmer >> 32);
          int32_t second = (int32_t)(kmer & 0xFFFFFFFF);

          if (node < 0) {
            stack.push(-first);
            stack.push(-second);
          } else {
            stack.push(second);
            stack.push(first);
          }
        } else {
          result.push_back(node);
        }
      } else {
        result.push_back(node);
      }
    }
  }
  return result;
}

PYBIND11_MODULE(gfa_compression, m) {
  // Module metadata and shared defaults.
  m.doc() = "Python bindings for the GFA compression library.\n"
            "Stable APIs are exposed at the module root. Experimental GPU\n"
            "helpers are available under gfa_compression.experimental.gpu.";

  m.attr("DEFAULT_ROUNDS") = kDefaultRounds;
  m.attr("DEFAULT_FREQ_THRESHOLD") = kDefaultFreqThreshold;
  m.attr("DEFAULT_DELTA_ROUND") = kDefaultDeltaRound;
  m.attr("DEFAULT_NUM_THREADS") = kDefaultNumThreads;

  m.def("has_gpu_backend", []() {
#ifdef ENABLE_CUDA
    return true;
#else
    return false;
#endif
  });

  // Core data types shared by CPU and GPU workflows.
  py::class_<WalkData>(m, "WalkData")
      .def(py::init<>())
      .def_readwrite("walks", &WalkData::walks)
      .def_readwrite("sample_ids", &WalkData::sample_ids)
      .def_readwrite("hap_indices", &WalkData::hap_indices)
      .def_readwrite("seq_ids", &WalkData::seq_ids)
      .def_readwrite("seq_starts", &WalkData::seq_starts)
      .def_readwrite("seq_ends", &WalkData::seq_ends);

  py::class_<WalkLookupKey>(m, "WalkLookupKey")
      .def(py::init<>())
      .def_readwrite("sample_id", &WalkLookupKey::sample_id)
      .def_readwrite("hap_index", &WalkLookupKey::hap_index)
      .def_readwrite("seq_id", &WalkLookupKey::seq_id)
      .def_readwrite("seq_start", &WalkLookupKey::seq_start)
      .def_readwrite("seq_end", &WalkLookupKey::seq_end);

  py::class_<GfaGraph>(m, "GfaGraph")
      .def(py::init<>())
      .def_readonly("node_name_to_id", &GfaGraph::node_name_to_id)
      .def_readonly("node_id_to_name", &GfaGraph::node_id_to_name)
      .def_readwrite("path_names", &GfaGraph::path_names)
      .def_readwrite("paths", &GfaGraph::paths)
      .def_readwrite("path_overlaps", &GfaGraph::path_overlaps)
      .def_readwrite("node_sequences", &GfaGraph::node_sequences)
      .def_readwrite("walks", &GfaGraph::walks);

  py::class_<LayerRuleRange>(m, "LayerRuleRange")
      .def(py::init<>())
      .def_readonly("k", &LayerRuleRange::k)
      .def_readonly("start_id", &LayerRuleRange::start_id)
      .def_readonly("end_id", &LayerRuleRange::end_id)
      .def_readonly("flattened_offset", &LayerRuleRange::flattened_offset)
      .def_readonly("element_count", &LayerRuleRange::element_count);

  py::class_<ZstdCompressedBlock>(m, "ZstdCompressedBlock")
      .def(py::init<>())
      .def_readonly("payload", &ZstdCompressedBlock::payload)
      .def_readonly("original_size", &ZstdCompressedBlock::original_size);

#ifdef ENABLE_CUDA
  py::class_<gpu_codec::NvcompCompressedBlock>(m, "NvcompCompressedBlock")
      .def(py::init<>())
      .def_readonly("payload", &gpu_codec::NvcompCompressedBlock::payload)
      .def_readonly("original_size",
                    &gpu_codec::NvcompCompressedBlock::original_size);
#endif

  py::class_<CompressedOptionalFieldColumn>(m, "CompressedOptionalFieldColumn")
      .def(py::init<>())
      .def_readonly("tag", &CompressedOptionalFieldColumn::tag)
      .def_readonly("type", &CompressedOptionalFieldColumn::type)
      .def_readonly("num_elements",
                    &CompressedOptionalFieldColumn::num_elements)
      .def_readonly("int_values_zstd",
                    &CompressedOptionalFieldColumn::int_values_zstd)
      .def_readonly("float_values_zstd",
                    &CompressedOptionalFieldColumn::float_values_zstd)
      .def_readonly("char_values_zstd",
                    &CompressedOptionalFieldColumn::char_values_zstd)
      .def_readonly("strings_zstd",
                    &CompressedOptionalFieldColumn::strings_zstd)
      .def_readonly("string_lengths_zstd",
                    &CompressedOptionalFieldColumn::string_lengths_zstd);

  py::class_<CompressedData>(m, "CompressedData")
      .def(py::init<>())
      .def_readonly("layer_rule_ranges", &CompressedData::layer_rule_ranges)
      .def_readonly("sequence_lengths", &CompressedData::sequence_lengths)
      .def_readonly("rules_first_zstd", &CompressedData::rules_first_zstd)
      .def_readonly("rules_second_zstd", &CompressedData::rules_second_zstd)
      .def_readonly("paths_zstd", &CompressedData::paths_zstd)
      .def_readonly("delta_round", &CompressedData::delta_round)
      .def_readonly("names_zstd", &CompressedData::names_zstd)
      .def_readonly("name_lengths_zstd", &CompressedData::name_lengths_zstd)
      .def_readonly("overlaps_zstd", &CompressedData::overlaps_zstd)
      .def_readonly("overlap_lengths_zstd",
                    &CompressedData::overlap_lengths_zstd)
      .def_readonly("segment_sequences_zstd",
                    &CompressedData::segment_sequences_zstd)
      .def_readonly("segment_seq_lengths_zstd",
                    &CompressedData::segment_seq_lengths_zstd)
      .def_readonly("segment_optional_fields_zstd",
                    &CompressedData::segment_optional_fields_zstd)
      .def_readonly("num_links", &CompressedData::num_links)
      .def_readonly("link_from_ids_zstd", &CompressedData::link_from_ids_zstd)
      .def_readonly("link_to_ids_zstd", &CompressedData::link_to_ids_zstd)
      .def_readonly("link_from_orients_zstd",
                    &CompressedData::link_from_orients_zstd)
      .def_readonly("link_to_orients_zstd",
                    &CompressedData::link_to_orients_zstd)
      .def_readonly("link_overlap_nums_zstd",
                    &CompressedData::link_overlap_nums_zstd)
      .def_readonly("link_overlap_ops_zstd",
                    &CompressedData::link_overlap_ops_zstd)
      .def_readonly("walk_lengths", &CompressedData::walk_lengths)
      .def_readonly("walks_zstd", &CompressedData::walks_zstd)
      .def_readonly("walk_sample_ids_zstd",
                    &CompressedData::walk_sample_ids_zstd)
      .def_readonly("walk_sample_id_lengths_zstd",
                    &CompressedData::walk_sample_id_lengths_zstd)
      .def_readonly("walk_hap_indices_zstd",
                    &CompressedData::walk_hap_indices_zstd)
      .def_readonly("walk_seq_ids_zstd", &CompressedData::walk_seq_ids_zstd)
      .def_readonly("walk_seq_id_lengths_zstd",
                    &CompressedData::walk_seq_id_lengths_zstd)
      .def_readonly("walk_seq_starts_zstd",
                    &CompressedData::walk_seq_starts_zstd)
      .def_readonly("walk_seq_ends_zstd", &CompressedData::walk_seq_ends_zstd);

  // Stable CPU-oriented APIs at module root.
  m.def("compress", &compress_gfa,
        "Compress a GFA file (returns CompressedData)",
        py::arg("gfa_file_path"), py::arg("num_rounds") = kDefaultRounds,
        py::arg("freq_threshold") = kDefaultFreqThreshold,
        py::arg("delta_round") = kDefaultDeltaRound,
        py::arg("num_threads") = kDefaultNumThreads);

  m.def(
      "decompress",
      [](const CompressedData &data, int num_threads) {
        GfaGraph graph;
        decompress_gfa(data, graph, num_threads);
        return graph;
      },
      "Decompress CompressedData to GfaGraph", py::arg("data"),
      py::arg("num_threads") = kDefaultNumThreads);

  m.def(
      "verify_round_trip",
      [](const GfaGraph &original, const GfaGraph &decompressed) {
        // Verify header
        if (original.header_line != decompressed.header_line) {
          std::cerr << "Verification Failed: Header mismatch. Original: '"
                    << original.header_line << "', Decompressed: '"
                    << decompressed.header_line << "'" << std::endl;
          return false;
        }

        if (original.paths.size() != decompressed.paths.size()) {
          std::cerr << "Verification Failed: Path count mismatch. Original: "
                    << original.paths.size()
                    << ", Decompressed: " << decompressed.paths.size()
                    << std::endl;
          return false;
        }

        // Bounds check for path_names and path_overlaps
        if (original.path_names.size() != original.paths.size() ||
            decompressed.path_names.size() != decompressed.paths.size()) {
          std::cerr
              << "Verification Failed: path_names size mismatch with paths."
              << std::endl;
          return false;
        }
        if (original.path_overlaps.size() != original.paths.size() ||
            decompressed.path_overlaps.size() != decompressed.paths.size()) {
          std::cerr
              << "Verification Failed: path_overlaps size mismatch with paths."
              << std::endl;
          return false;
        }

        for (size_t i = 0; i < original.paths.size(); ++i) {
          const std::vector<NodeId> &original_path = original.paths[i];
          const std::vector<NodeId> &decompressed_path = decompressed.paths[i];

          if (original_path.size() != decompressed_path.size()) {
            std::cerr << "Verification Failed: Path index " << i
                      << " length mismatch. Original: " << original_path.size()
                      << ", Decomp: " << decompressed_path.size() << std::endl;
            return false;
          }

          for (size_t j = 0; j < original_path.size(); ++j) {
            if (original_path[j] != decompressed_path[j]) {
              std::cerr << "Verification Failed: Path index " << i
                        << " element " << j << " mismatch." << std::endl;
              return false;
            }
          }

          // Check Overlap
          if (original.path_overlaps[i] != decompressed.path_overlaps[i]) {
            std::cerr << "Verification Failed: Path index " << i
                      << " overlap mismatch." << std::endl;
            return false;
          }

          // Check Name
          if (original.path_names[i] != decompressed.path_names[i]) {
            std::cerr << "Verification Failed: Path index " << i
                      << " name mismatch." << std::endl;
            return false;
          }
        }

        // Verify Segments (Sequences)
        // IDs are 1-based, index 0 is placeholder
        if (original.node_sequences.size() !=
            decompressed.node_sequences.size()) {
          std::cerr
              << "Verification Failed: Node sequence count mismatch. Original: "
              << original.node_sequences.size()
              << ", Decompressed: " << decompressed.node_sequences.size()
              << std::endl;
          return false;
        }

        for (size_t i = 1; i < original.node_sequences.size(); ++i) {
          const std::string &orig_seq = original.node_sequences[i];
          const std::string &dec_seq = decompressed.node_sequences[i];

          if (orig_seq != dec_seq) {
            // Try to report ID. Since we assume ID = index, we can say Node ID
            // i.
            std::cerr << "Verification Failed: Sequence mismatch for Node ID "
                      << i << ". Original: "
                      << orig_seq.substr(0,
                                         std::min<size_t>(20, orig_seq.size()))
                      << "..."
                      << ", Decompressed: "
                      << dec_seq.substr(0, std::min<size_t>(20, dec_seq.size()))
                      << "..." << std::endl;
            return false;
          }
        }

        // Verify Links
        if (original.links.from_ids.size() !=
            decompressed.links.from_ids.size()) {
          std::cerr << "Verification Failed: Link count mismatch. Original: "
                    << original.links.from_ids.size()
                    << ", Decompressed: " << decompressed.links.from_ids.size()
                    << std::endl;
          return false;
        }

        for (size_t i = 0; i < original.links.from_ids.size(); ++i) {
          if (original.links.from_ids[i] != decompressed.links.from_ids[i] ||
              original.links.to_ids[i] != decompressed.links.to_ids[i] ||
              original.links.from_orients[i] !=
                  decompressed.links.from_orients[i] ||
              original.links.to_orients[i] !=
                  decompressed.links.to_orients[i] ||
              original.links.overlap_nums[i] !=
                  decompressed.links.overlap_nums[i] ||
              original.links.overlap_ops[i] !=
                  decompressed.links.overlap_ops[i]) {
            std::cerr << "Verification Failed: Link index " << i << " mismatch."
                      << std::endl;
            return false;
          }
        }

        // Verify Segment Optional Fields
        if (original.segment_optional_fields.size() !=
            decompressed.segment_optional_fields.size()) {
          std::cerr << "Verification Failed: Segment optional field column "
                       "count mismatch. Original: "
                    << original.segment_optional_fields.size()
                    << ", Decompressed: "
                    << decompressed.segment_optional_fields.size() << std::endl;
          return false;
        }

        for (size_t col = 0; col < original.segment_optional_fields.size();
             ++col) {
          const auto &orig_col = original.segment_optional_fields[col];
          const auto &dec_col = decompressed.segment_optional_fields[col];

          if (orig_col.tag != dec_col.tag || orig_col.type != dec_col.type) {
            std::cerr << "Verification Failed: Segment optional field column "
                      << col << " tag/type mismatch." << std::endl;
            return false;
          }

          // Check values based on type
          if (orig_col.int_values != dec_col.int_values ||
              orig_col.float_values != dec_col.float_values ||
              orig_col.char_values != dec_col.char_values ||
              orig_col.concatenated_strings != dec_col.concatenated_strings ||
              orig_col.string_lengths != dec_col.string_lengths ||
              orig_col.b_subtypes != dec_col.b_subtypes ||
              orig_col.b_lengths != dec_col.b_lengths ||
              orig_col.b_concat_bytes != dec_col.b_concat_bytes) {
            std::cerr << "Verification Failed: Segment optional field column "
                      << col << " (tag=" << orig_col.tag << ") values mismatch."
                      << std::endl;
            return false;
          }
        }

        // Verify Link Optional Fields
        if (original.link_optional_fields.size() !=
            decompressed.link_optional_fields.size()) {
          std::cerr << "Verification Failed: Link optional field column count "
                       "mismatch. Original: "
                    << original.link_optional_fields.size()
                    << ", Decompressed: "
                    << decompressed.link_optional_fields.size() << std::endl;
          return false;
        }

        for (size_t col = 0; col < original.link_optional_fields.size();
             ++col) {
          const auto &orig_col = original.link_optional_fields[col];
          const auto &dec_col = decompressed.link_optional_fields[col];

          if (orig_col.tag != dec_col.tag || orig_col.type != dec_col.type) {
            std::cerr << "Verification Failed: Link optional field column "
                      << col << " tag/type mismatch." << std::endl;
            return false;
          }

          if (orig_col.int_values != dec_col.int_values ||
              orig_col.float_values != dec_col.float_values ||
              orig_col.char_values != dec_col.char_values ||
              orig_col.concatenated_strings != dec_col.concatenated_strings ||
              orig_col.string_lengths != dec_col.string_lengths ||
              orig_col.b_subtypes != dec_col.b_subtypes ||
              orig_col.b_lengths != dec_col.b_lengths ||
              orig_col.b_concat_bytes != dec_col.b_concat_bytes) {
            std::cerr << "Verification Failed: Link optional field column "
                      << col << " (tag=" << orig_col.tag << ") values mismatch."
                      << std::endl;
            return false;
          }
        }

        // Verify J-lines (jumps)
        if (original.jumps.size() != decompressed.jumps.size()) {
          std::cerr << "Verification Failed: J-lines count mismatch. Original: "
                    << original.jumps.size()
                    << ", Decompressed: " << decompressed.jumps.size()
                    << std::endl;
          return false;
        }
        if (original.jumps.from_ids != decompressed.jumps.from_ids ||
            original.jumps.to_ids != decompressed.jumps.to_ids ||
            original.jumps.from_orients != decompressed.jumps.from_orients ||
            original.jumps.to_orients != decompressed.jumps.to_orients ||
            original.jumps.distances != decompressed.jumps.distances) {
          std::cerr << "Verification Failed: J-lines content mismatch."
                    << std::endl;
          return false;
        }

        // Verify C-lines (containments)
        if (original.containments.size() != decompressed.containments.size()) {
          std::cerr << "Verification Failed: C-lines count mismatch. Original: "
                    << original.containments.size()
                    << ", Decompressed: " << decompressed.containments.size()
                    << std::endl;
          return false;
        }
        if (original.containments.container_ids !=
                decompressed.containments.container_ids ||
            original.containments.contained_ids !=
                decompressed.containments.contained_ids ||
            original.containments.container_orients !=
                decompressed.containments.container_orients ||
            original.containments.contained_orients !=
                decompressed.containments.contained_orients ||
            original.containments.positions !=
                decompressed.containments.positions ||
            original.containments.overlaps !=
                decompressed.containments.overlaps) {
          std::cerr << "Verification Failed: C-lines content mismatch."
                    << std::endl;
          return false;
        }

        // Verify W-lines (Walks)
        if (original.walks.walks.size() != decompressed.walks.walks.size()) {
          std::cerr << "Verification Failed: Walk count mismatch. Original: "
                    << original.walks.walks.size()
                    << ", Decompressed: " << decompressed.walks.walks.size()
                    << std::endl;
          return false;
        }

        for (size_t w = 0; w < original.walks.walks.size(); ++w) {
          // Check walk sequence
          if (original.walks.walks[w] != decompressed.walks.walks[w]) {
            std::cerr << "Verification Failed: Walk " << w
                      << " sequence mismatch. "
                      << "Original size: " << original.walks.walks[w].size()
                      << ", Decompressed size: "
                      << decompressed.walks.walks[w].size() << std::endl;
            return false;
          }

          // Check walk metadata
          if (w < original.walks.sample_ids.size() &&
              w < decompressed.walks.sample_ids.size() &&
              original.walks.sample_ids[w] !=
                  decompressed.walks.sample_ids[w]) {
            std::cerr << "Verification Failed: Walk " << w
                      << " sample_id mismatch." << std::endl;
            return false;
          }

          if (w < original.walks.hap_indices.size() &&
              w < decompressed.walks.hap_indices.size() &&
              original.walks.hap_indices[w] !=
                  decompressed.walks.hap_indices[w]) {
            std::cerr << "Verification Failed: Walk " << w
                      << " hap_index mismatch." << std::endl;
            return false;
          }

          if (w < original.walks.seq_ids.size() &&
              w < decompressed.walks.seq_ids.size() &&
              original.walks.seq_ids[w] != decompressed.walks.seq_ids[w]) {
            std::cerr << "Verification Failed: Walk " << w
                      << " seq_id mismatch." << std::endl;
            return false;
          }

          if (w < original.walks.seq_starts.size() &&
              w < decompressed.walks.seq_starts.size() &&
              original.walks.seq_starts[w] !=
                  decompressed.walks.seq_starts[w]) {
            std::cerr << "Verification Failed: Walk " << w
                      << " seq_start mismatch." << std::endl;
            return false;
          }

          if (w < original.walks.seq_ends.size() &&
              w < decompressed.walks.seq_ends.size() &&
              original.walks.seq_ends[w] != decompressed.walks.seq_ends[w]) {
            std::cerr << "Verification Failed: Walk " << w
                      << " seq_end mismatch." << std::endl;
            return false;
          }
        }

        GFAZ_LOG("Verification Successful: All fields match.");
        return true;
      },
      "Verify round-trip correctness", py::arg("original"),
      py::arg("decompressed"));
  m.attr("verify_roundtrip") = m.attr("verify_round_trip");

  // Stable naming-normalized aliases.
  m.def("parse", &parse_gfa_file, "Parse a GFA file", py::arg("file_path"));
  m.def("parse_gfa", &parse_gfa_file, "Parse a GFA file", py::arg("file_path"));
  m.def("compress_file", &compress_gfa,
        "Alias for compress with normalized argument names",
        py::arg("gfa_file_path"), py::arg("rounds") = kDefaultRounds,
        py::arg("threshold") = kDefaultFreqThreshold,
        py::arg("delta_round") = kDefaultDeltaRound,
        py::arg("threads") = kDefaultNumThreads);
  m.def(
      "decompress_data",
      [](const CompressedData &data, int threads) {
        GfaGraph graph;
        decompress_gfa(data, graph, threads);
        return graph;
      },
      "Alias for decompress with normalized argument names", py::arg("data"),
      py::arg("threads") = kDefaultNumThreads);

  m.def("serialize", &serialize_compressed_data, "Serialize to binary file",
        py::arg("data"), py::arg("output_path"));
  m.def("serialize_file", &serialize_compressed_data,
        "Serialize to binary file", py::arg("data"), py::arg("output_path"));

  m.def("deserialize", &deserialize_compressed_data,
        "Deserialize from binary file", py::arg("input_path"));
  m.def("deserialize_file", &deserialize_compressed_data,
        "Deserialize from binary file", py::arg("input_path"));

#ifdef ENABLE_CUDA
  m.def("serialize_gpu", &serialize_compressed_data_gpu,
        "Serialize GPU compressed data to binary file", py::arg("data"),
        py::arg("output_path"));

  m.def("deserialize_gpu", &deserialize_compressed_data_gpu,
        "Deserialize GPU compressed data from binary file",
        py::arg("input_path"));
#endif

  m.def("write_gfa", &write_gfa, "Write GfaGraph to GFA file", py::arg("graph"),
        py::arg("output_path"));
  m.def("write_gfa_from_compressed_data", &write_gfa_from_compressed_data,
        "Write GFA directly from CompressedData without materializing full path/walk vectors",
        py::arg("data"), py::arg("output_path"),
        py::arg("num_threads") = kDefaultNumThreads);
  m.def("extract_path_line", &extract_path_line_by_name,
        "Extract a single P-line from CompressedData", py::arg("data"),
        py::arg("path_name"),
        py::arg("num_threads") = kDefaultNumThreads);
  m.def("extract_path_lines", &extract_path_lines_by_name,
        "Extract multiple P-lines from CompressedData", py::arg("data"),
        py::arg("path_names"),
        py::arg("num_threads") = kDefaultNumThreads);
  m.def("extract_walk_line", &extract_walk_line,
        "Extract a single W-line from CompressedData", py::arg("data"),
        py::arg("sample_id"), py::arg("hap_index"), py::arg("seq_id"),
        py::arg("seq_start"), py::arg("seq_end"),
        py::arg("num_threads") = kDefaultNumThreads);
  m.def("extract_walk_line_by_name", &extract_walk_line_by_name,
        "Extract a single W-line from CompressedData using walk name",
        py::arg("data"), py::arg("walk_name"),
        py::arg("num_threads") = kDefaultNumThreads);
  m.def("extract_walk_lines", &extract_walk_lines,
        "Extract multiple W-lines from CompressedData using full walk identifiers",
        py::arg("data"), py::arg("walk_keys"),
        py::arg("num_threads") = kDefaultNumThreads);
  m.def("extract_walk_lines_by_name", &extract_walk_lines_by_name,
        "Extract multiple W-lines from CompressedData using walk names",
        py::arg("data"), py::arg("walk_names"),
        py::arg("num_threads") = kDefaultNumThreads);
  m.def(
      "add_haplotypes",
      [](CompressedData data, const std::string &haplotypes_path, int threads) {
        add_haplotypes(data, haplotypes_path, threads);
        return data;
      },
      "Append path-only or walk-only haplotypes to CompressedData using the existing rulebook",
      py::arg("data"), py::arg("haplotypes_path"),
      py::arg("threads") = kDefaultNumThreads);

  // CPU helper utilities.
  m.def("cpu_reconstruct_path", &reconstruct_path_cpu,
        "Reconstruct path on CPU", py::arg("compressed_path"),
        py::arg("rulebook"), py::arg("min_rule_id"));

  py::class_<FlattenedStrings>(m, "FlattenedStrings")
      .def(py::init<>())
      .def_readonly("data", &FlattenedStrings::data)
      .def_readonly("lengths", &FlattenedStrings::lengths)
      .def("count", &FlattenedStrings::count)
      .def("total_chars", &FlattenedStrings::total_chars);

  py::class_<FlattenedPaths>(m, "FlattenedPaths")
      .def(py::init<>())
      .def_readonly("data", &FlattenedPaths::data)
      .def_readonly("lengths", &FlattenedPaths::lengths)
      .def("num_paths", &FlattenedPaths::num_paths)
      .def("total_nodes", &FlattenedPaths::total_nodes);

  // GPU-friendly graph/view data types available at module root.
  py::class_<GfaGraph_gpu>(m, "GfaGraph_gpu")
      .def(py::init<>())
      .def_readonly("header_line", &GfaGraph_gpu::header_line)
      .def_readonly("num_segments", &GfaGraph_gpu::num_segments)
      .def_readonly("num_paths", &GfaGraph_gpu::num_paths)
      .def_readonly("num_walks", &GfaGraph_gpu::num_walks)
      .def_readonly("num_links", &GfaGraph_gpu::num_links)
      .def_readonly("node_names", &GfaGraph_gpu::node_names)
      .def_readonly("node_sequences", &GfaGraph_gpu::node_sequences)
      .def_readonly("paths", &GfaGraph_gpu::paths)
      .def_readonly("path_names", &GfaGraph_gpu::path_names)
      .def_readonly("path_overlaps", &GfaGraph_gpu::path_overlaps)
      .def_readonly("walk_sample_ids", &GfaGraph_gpu::walk_sample_ids)
      .def_readonly("walk_hap_indices", &GfaGraph_gpu::walk_hap_indices)
      .def_readonly("walk_seq_ids", &GfaGraph_gpu::walk_seq_ids)
      .def_readonly("walk_seq_starts", &GfaGraph_gpu::walk_seq_starts)
      .def_readonly("walk_seq_ends", &GfaGraph_gpu::walk_seq_ends)
      .def_readonly("link_from_ids", &GfaGraph_gpu::link_from_ids)
      .def_readonly("link_to_ids", &GfaGraph_gpu::link_to_ids)
      .def_readonly("jump_from_ids", &GfaGraph_gpu::jump_from_ids)
      .def_readonly("jump_to_ids", &GfaGraph_gpu::jump_to_ids)
      .def_readonly("containment_container_ids",
                    &GfaGraph_gpu::containment_container_ids)
      .def_readonly("containment_contained_ids",
                    &GfaGraph_gpu::containment_contained_ids)
      .def("paths_total_elements", &GfaGraph_gpu::paths_total_elements)
      .def("segments_total_chars", &GfaGraph_gpu::segments_total_chars)
      .def("num_jumps", &GfaGraph_gpu::num_jumps)
      .def("num_containments", &GfaGraph_gpu::num_containments);

  // Stable high-level GPU layout conversion helpers.
  m.def("convert_to_gpu_layout", &convert_to_gpu_layout,
        "Convert GfaGraph to GPU-friendly layout", py::arg("graph"));

  m.def("convert_from_gpu_layout", &convert_from_gpu_layout,
        "Convert GfaGraph_gpu back to GfaGraph", py::arg("gpu_graph"));

  py::module_ experimental =
      m.def_submodule("experimental", "Experimental or low-level APIs.");

#ifdef ENABLE_CUDA
  py::module_ experimental_gpu = experimental.def_submodule(
      "gpu", "Experimental low-level GPU helper APIs.");

  // Experimental low-level GPU codec/rule-table helpers.
  experimental_gpu.def("delta_encode_paths", &gpu_codec::delta_encode_paths,
                       "Delta encode flattened paths on GPU using CUB "
                       "(in-place)",
                       py::arg("paths"));

  experimental_gpu.def("delta_decode_paths", &gpu_codec::delta_decode_paths,
                       "Delta decode flattened paths on GPU using CUB "
                       "(in-place)",
                       py::arg("paths"));

  experimental_gpu.def("find_max_abs_node", &gpu_codec::find_max_abs_node,
                       "Find maximum absolute node ID in flattened paths on "
                       "GPU",
                       py::arg("paths"));

  experimental_gpu.def("find_repeated_2mers", &gpu_codec::find_repeated_2mers,
                       "Find 2-mers that appear >= 2 times in the paths using "
                       "GPU",
                       py::arg("paths"));

  // Expose opaque GPU rule-table pointer as an integer handle.
  experimental_gpu.def(
      "create_rule_table",
      [](const std::vector<uint64_t> &rules, uint32_t start_id) -> size_t {
        void *ptr = gpu_codec::create_rule_table_gpu(rules, start_id);
        return reinterpret_cast<size_t>(ptr);
      },
      "Create GPU Rule Hash Table");

  experimental_gpu.def(
      "apply_2mer_rules",
      [](GfaGraph_gpu &graph, size_t table_ptr, uint32_t num_rules,
         uint32_t start_id) {
        void *ptr = reinterpret_cast<void *>(table_ptr);
        std::vector<uint8_t> rules_used(num_rules, 0);
        gpu_codec::apply_2mer_rules_gpu(graph.paths, ptr, rules_used, start_id);
        return rules_used;
      },
      "Apply 2-mer rules on GPU (Modifies graph.paths). Returns rules_used "
      "vector.",
      py::arg("graph"), py::arg("table_ptr"), py::arg("num_rules"),
      py::arg("start_id"));

  experimental_gpu.def(
      "compact_rules_and_remap", &gpu_codec::compact_rules_and_remap_gpu,
      "Compact rules and remap path IDs on GPU", py::arg("paths"),
      py::arg("rules_used"), py::arg("current_rules"), py::arg("start_id"));

  experimental_gpu.def(
      "sort_rules_and_remap", &gpu_codec::sort_rules_and_remap_gpu,
      "Sort rules by value (for delta efficiency) and remap "
      "path IDs on GPU",
      py::arg("paths"), py::arg("current_rules"), py::arg("start_id"));

  experimental_gpu.def(
      "free_rule_table",
      [](size_t table_ptr) {
        void *ptr = reinterpret_cast<void *>(table_ptr);
        gpu_codec::free_rule_table_gpu(ptr);
      },
      "Free GPU Rule Hash Table");

  experimental_gpu.def(
      "run_compression_2mer",
      [](FlattenedPaths &paths, uint32_t next_starting_id, int num_rounds) {
        // We need to return the updated next_starting_id and the
        // master_rulebook
        std::map<uint32_t, uint64_t> master_rulebook;

        gpu_codec::run_compression_layer_2mer_gpu(paths, next_starting_id,
                                                  num_rounds, master_rulebook);

        return py::make_tuple(next_starting_id, master_rulebook);
      },
      "Run full 2-mer compression layer on GPU", py::arg("paths"),
      py::arg("next_starting_id"), py::arg("num_rounds") = kDefaultRounds);

  experimental_gpu.def(
      "run_compression_2mer_device",
      [](FlattenedPaths &paths, uint32_t next_starting_id, int num_rounds) {
        std::map<uint32_t, uint64_t> master_rulebook;

        gpu_codec::FlattenedPathsDevice device_paths =
            gpu_codec::copy_paths_to_device(paths);
        gpu_codec::run_compression_layer_2mer_gpu_device(
            device_paths, next_starting_id, num_rounds, master_rulebook);
        gpu_codec::copy_paths_to_host(device_paths, paths);

        return py::make_tuple(next_starting_id, master_rulebook);
      },
      "Run full 2-mer compression layer on GPU using device-side paths",
      py::arg("paths"), py::arg("next_starting_id"),
      py::arg("num_rounds") = kDefaultRounds);

  // GPU-resident compressed formats and path-compression helpers.

  py::class_<gpu_compression::GPURuleRange>(m, "GPURuleRange")
      .def(py::init<>())
      .def_readonly("start_id", &gpu_compression::GPURuleRange::start_id)
      .def_readonly("count", &gpu_compression::GPURuleRange::count);

  py::class_<gpu_compression::CompressedData_gpu>(m, "CompressedData_gpu")
      .def(py::init<>())
      .def_readonly(
          "encoded_path_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::encoded_path_zstd_nvcomp)
      .def_readonly(
          "rules_first_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::rules_first_zstd_nvcomp)
      .def_readonly(
          "rules_second_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::rules_second_zstd_nvcomp)
      .def_readonly("layer_ranges",
                    &gpu_compression::CompressedData_gpu::layer_ranges)
      .def_readonly(
          "path_lengths_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::path_lengths_zstd_nvcomp)
      .def_readonly("names_zstd_nvcomp",
                    &gpu_compression::CompressedData_gpu::names_zstd_nvcomp)
      .def_readonly(
          "name_lengths_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::name_lengths_zstd_nvcomp)
      .def_readonly("overlaps_zstd_nvcomp",
                    &gpu_compression::CompressedData_gpu::overlaps_zstd_nvcomp)
      .def_readonly(
          "overlap_lengths_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::overlap_lengths_zstd_nvcomp)
      .def_readonly(
          "segment_sequences_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::segment_sequences_zstd_nvcomp)
      .def_readonly(
          "segment_seq_lengths_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::segment_seq_lengths_zstd_nvcomp)
      .def_readonly("segment_optional_fields_zstd_nvcomp",
                    &gpu_compression::CompressedData_gpu::
                        segment_optional_fields_zstd_nvcomp)
      .def_readonly("header_line",
                    &gpu_compression::CompressedData_gpu::header_line)
      .def_readonly(
          "node_names_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::node_names_zstd_nvcomp)
      .def_readonly(
          "node_name_lengths_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::node_name_lengths_zstd_nvcomp)
      .def_readonly(
          "link_from_ids_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::link_from_ids_zstd_nvcomp)
      .def_readonly(
          "link_to_ids_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::link_to_ids_zstd_nvcomp)
      .def_readonly(
          "link_from_orients_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::link_from_orients_zstd_nvcomp)
      .def_readonly(
          "link_to_orients_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::link_to_orients_zstd_nvcomp)
      .def_readonly(
          "link_overlap_nums_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::link_overlap_nums_zstd_nvcomp)
      .def_readonly(
          "link_overlap_ops_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::link_overlap_ops_zstd_nvcomp)
      .def_readonly("num_links",
                    &gpu_compression::CompressedData_gpu::num_links)
      .def_readonly("link_optional_fields_zstd_nvcomp",
                    &gpu_compression::CompressedData_gpu::
                        link_optional_fields_zstd_nvcomp)
      .def_readonly("num_paths",
                    &gpu_compression::CompressedData_gpu::num_paths)
      .def_readonly("num_walks",
                    &gpu_compression::CompressedData_gpu::num_walks)
      .def_readonly(
          "walk_sample_ids_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::walk_sample_ids_zstd_nvcomp)
      .def_readonly(
          "walk_hap_indices_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::walk_hap_indices_zstd_nvcomp)
      .def_readonly(
          "walk_seq_ids_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::walk_seq_ids_zstd_nvcomp)
      .def_readonly(
          "walk_seq_starts_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::walk_seq_starts_zstd_nvcomp)
      .def_readonly(
          "walk_seq_ends_zstd_nvcomp",
          &gpu_compression::CompressedData_gpu::walk_seq_ends_zstd_nvcomp)
      .def_readonly("num_jumps_stored",
                    &gpu_compression::CompressedData_gpu::num_jumps_stored)
      .def_readonly(
          "num_containments_stored",
          &gpu_compression::CompressedData_gpu::num_containments_stored)
      .def("total_rules", &gpu_compression::CompressedData_gpu::total_rules)
      .def("num_rounds", &gpu_compression::CompressedData_gpu::num_rounds)
      .def("min_rule_id", &gpu_compression::CompressedData_gpu::min_rule_id);

  py::class_<gpu_compression::CompressedOptionalFieldColumn_gpu>(
      m, "CompressedOptionalFieldColumnGpu")
      .def(py::init<>())
      .def_readonly("tag",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::tag)
      .def_readonly("type",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::type)
      .def_readonly(
          "num_elements",
          &gpu_compression::CompressedOptionalFieldColumn_gpu::num_elements)
      .def_readonly("int_values_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        int_values_zstd_nvcomp)
      .def_readonly("float_values_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        float_values_zstd_nvcomp)
      .def_readonly("char_values_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        char_values_zstd_nvcomp)
      .def_readonly("strings_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        strings_zstd_nvcomp)
      .def_readonly("string_lengths_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        string_lengths_zstd_nvcomp)
      .def_readonly("b_subtypes_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        b_subtypes_zstd_nvcomp)
      .def_readonly("b_lengths_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        b_lengths_zstd_nvcomp)
      .def_readonly("b_concat_bytes_zstd_nvcomp",
                    &gpu_compression::CompressedOptionalFieldColumn_gpu::
                        b_concat_bytes_zstd_nvcomp);

  experimental_gpu.def(
      "run_path_compression",
      [](const FlattenedPaths &paths, int num_rounds) {
        return gpu_compression::run_path_compression_gpu(paths, num_rounds);
      },
      "GPU-resident path compression with minimal host-device transfers.\n"
      "Returns CompressedData_gpu containing encoded_path, compressed rules, "
      "layer_ranges, path_lengths.",
      py::arg("paths"), py::arg("num_rounds") = kDefaultRounds);

  // Stable high-level GPU compression/decompression APIs at module root.
  m.def(
      "compress_gfa_gpu",
      [](const std::string &gfa_file_path, int num_rounds) {
        return gpu_compression::compress_gfa_gpu(gfa_file_path, num_rounds);
      },
      "Parse GFA and run GPU path compression with path metadata compression.\n"
      "Rules are stored as delta-encoded + nvComp ZSTD compressed first/second "
      "element arrays.",
      py::arg("gfa_file_path"), py::arg("num_rounds") = kDefaultRounds);

  m.def(
      "compress_gpu_graph",
      [](const GfaGraph_gpu &gpu_graph, int num_rounds) {
        return gpu_compression::compress_gpu_graph(gpu_graph, num_rounds);
      },
      "GPU compression from pre-converted GfaGraph_gpu.\n"
      "Use this for accurate timing of compression-only (no parsing).",
      py::arg("gpu_graph"), py::arg("num_rounds") = kDefaultRounds);

  // Experimental convenience helpers built on compressed GPU payloads.
  experimental_gpu.def(
      "build_rulebook",
      [](const gpu_compression::CompressedData_gpu &data) {
        return gpu_compression::build_rulebook(data);
      },
      "Build rulebook map from CompressedData_gpu.\n"
      "Decompresses and inverse delta-encodes the rules.\n"
      "Returns dict mapping rule_id -> packed_2mer (uint64).",
      py::arg("data"));

  experimental_gpu.def(
      "get_min_rule_id",
      [](const std::vector<gpu_compression::GPURuleRange> &layer_ranges) {
        if (layer_ranges.empty()) {
          return (uint32_t)0;
        }
        return layer_ranges[0].start_id;
      },
      "Get minimum rule ID from layer ranges (for reconstruction).",
      py::arg("layer_ranges"));

  experimental_gpu.def(
      "decompress_encoded_path",
      [](const gpu_codec::NvcompCompressedBlock &block) {
        return gpu_codec::nvcomp_zstd_decompress_int32(block);
      },
      "Decompress encoded_path from NvcompCompressedBlock to vector<int32>.",
      py::arg("block"));

  experimental_gpu.def(
      "decompress_path_lengths",
      [](const gpu_codec::NvcompCompressedBlock &block) {
        return gpu_codec::nvcomp_zstd_decompress_uint32(block);
      },
      "Decompress path_lengths from NvcompCompressedBlock to vector<uint32>.",
      py::arg("block"));

  experimental_gpu.def(
      "decompress_paths_gpu",
      [](const gpu_compression::CompressedData_gpu &data) {
        return gpu_decompression::decompress_paths_gpu(data);
      },
      "GPU-accelerated path decompression.\n"
      "Decompresses nvComp blocks, inverse delta-encodes rules, and expands "
      "path on GPU.\n"
      "Returns FlattenedPaths with decompressed data and lengths.",
      py::arg("data"));

  // Backward-compatible aliases for legacy flat module names.
  m.attr("gpu_delta_encode_paths") =
      experimental_gpu.attr("delta_encode_paths");
  m.attr("gpu_delta_decode_paths") =
      experimental_gpu.attr("delta_decode_paths");
  m.attr("gpu_find_max_abs_node") = experimental_gpu.attr("find_max_abs_node");
  m.attr("gpu_find_repeated_2mers") =
      experimental_gpu.attr("find_repeated_2mers");
  m.attr("gpu_create_rule_table") = experimental_gpu.attr("create_rule_table");
  m.attr("gpu_apply_2mer_rules") = experimental_gpu.attr("apply_2mer_rules");
  m.attr("gpu_compact_rules_and_remap") =
      experimental_gpu.attr("compact_rules_and_remap");
  m.attr("gpu_sort_rules_and_remap") =
      experimental_gpu.attr("sort_rules_and_remap");
  m.attr("gpu_free_rule_table") = experimental_gpu.attr("free_rule_table");
  m.attr("gpu_run_compression_2mer") =
      experimental_gpu.attr("run_compression_2mer");
  m.attr("gpu_run_compression_2mer_device") =
      experimental_gpu.attr("run_compression_2mer_device");
  m.attr("gpu_run_path_compression") =
      experimental_gpu.attr("run_path_compression");
  m.attr("build_rulebook") = experimental_gpu.attr("build_rulebook");
  m.attr("get_min_rule_id") = experimental_gpu.attr("get_min_rule_id");
  m.attr("decompress_encoded_path") =
      experimental_gpu.attr("decompress_encoded_path");
  m.attr("decompress_path_lengths") =
      experimental_gpu.attr("decompress_path_lengths");
  m.attr("decompress_paths_gpu") =
      experimental_gpu.attr("decompress_paths_gpu");

  m.def(
      "decompress_to_gpu_layout",
      [](const gpu_compression::CompressedData_gpu &data) {
        return gpu_decompression::decompress_to_gpu_layout(data);
      },
      "Full GPU decompression: CompressedData_gpu -> GfaGraph_gpu.\n"
      "Decompresses paths, names, overlaps, and segment sequences.",
      py::arg("data"));

  m.def(
      "verify_gpu_round_trip",
      [](const GfaGraph_gpu &original,
         const GfaGraph_gpu &decompressed) -> bool {
        bool success = true;

        // Compare metadata
        if (original.num_segments != decompressed.num_segments) {
          std::cerr
              << "GPU Verification Failed: num_segments mismatch. Original: "
              << original.num_segments
              << ", Decompressed: " << decompressed.num_segments << std::endl;
          success = false;
        }
        if (original.num_paths != decompressed.num_paths) {
          std::cerr << "GPU Verification Failed: num_paths mismatch. Original: "
                    << original.num_paths
                    << ", Decompressed: " << decompressed.num_paths
                    << std::endl;
          success = false;
        }
        if (original.num_walks != decompressed.num_walks) {
          std::cerr << "GPU Verification Failed: num_walks mismatch. Original: "
                    << original.num_walks
                    << ", Decompressed: " << decompressed.num_walks
                    << std::endl;
          success = false;
        }
        if (original.num_links != decompressed.num_links) {
          std::cerr << "GPU Verification Failed: num_links mismatch. Original: "
                    << original.num_links
                    << ", Decompressed: " << decompressed.num_links
                    << std::endl;
          success = false;
        }

        // Compare paths (most critical!)
        if (original.paths.data != decompressed.paths.data) {
          std::cerr
              << "GPU Verification Failed: paths.data mismatch. Original size: "
              << original.paths.data.size()
              << ", Decompressed size: " << decompressed.paths.data.size()
              << std::endl;
          // Find first mismatch
          for (size_t i = 0; i < std::min(original.paths.data.size(),
                                          decompressed.paths.data.size());
               ++i) {
            if (original.paths.data[i] != decompressed.paths.data[i]) {
              std::cerr << "  First mismatch at index " << i << ": "
                        << original.paths.data[i] << " vs "
                        << decompressed.paths.data[i] << std::endl;
              break;
            }
          }
          success = false;
        }
        if (original.paths.lengths != decompressed.paths.lengths) {
          std::cerr << "GPU Verification Failed: paths.lengths mismatch."
                    << std::endl;
          success = false;
        }

        // Compare header line
        if (original.header_line != decompressed.header_line) {
          std::cerr << "GPU Verification Failed: header_line mismatch."
                    << std::endl;
          success = false;
        }

        // Compare node names
        if (original.node_names.data != decompressed.node_names.data) {
          std::cerr << "GPU Verification Failed: node_names.data mismatch."
                    << std::endl;
          success = false;
        }
        if (original.node_names.lengths != decompressed.node_names.lengths) {
          std::cerr << "GPU Verification Failed: node_names.lengths mismatch."
                    << std::endl;
          success = false;
        }

        // Compare path names
        if (original.path_names.data != decompressed.path_names.data) {
          std::cerr << "GPU Verification Failed: path_names.data mismatch."
                    << std::endl;
          success = false;
        }
        if (original.path_names.lengths != decompressed.path_names.lengths) {
          std::cerr << "GPU Verification Failed: path_names.lengths mismatch."
                    << std::endl;
          success = false;
        }

        // Compare path overlaps
        if (original.path_overlaps.data != decompressed.path_overlaps.data) {
          std::cerr << "GPU Verification Failed: path_overlaps.data mismatch."
                    << std::endl;
          success = false;
        }
        if (original.path_overlaps.lengths !=
            decompressed.path_overlaps.lengths) {
          std::cerr
              << "GPU Verification Failed: path_overlaps.lengths mismatch."
              << std::endl;
          success = false;
        }

        // Compare walk metadata
        if (original.walk_sample_ids.data !=
                decompressed.walk_sample_ids.data ||
            original.walk_sample_ids.lengths !=
                decompressed.walk_sample_ids.lengths) {
          std::cerr << "GPU Verification Failed: walk_sample_ids mismatch."
                    << std::endl;
          success = false;
        }
        if (original.walk_hap_indices != decompressed.walk_hap_indices) {
          std::cerr << "GPU Verification Failed: walk_hap_indices mismatch."
                    << std::endl;
          success = false;
        }
        if (original.walk_seq_ids.data != decompressed.walk_seq_ids.data ||
            original.walk_seq_ids.lengths !=
                decompressed.walk_seq_ids.lengths) {
          std::cerr << "GPU Verification Failed: walk_seq_ids mismatch."
                    << std::endl;
          success = false;
        }
        if (original.walk_seq_starts != decompressed.walk_seq_starts) {
          std::cerr << "GPU Verification Failed: walk_seq_starts mismatch."
                    << std::endl;
          success = false;
        }
        if (original.walk_seq_ends != decompressed.walk_seq_ends) {
          std::cerr << "GPU Verification Failed: walk_seq_ends mismatch."
                    << std::endl;
          success = false;
        }

        // Compare node sequences
        if (original.node_sequences.data != decompressed.node_sequences.data) {
          std::cerr << "GPU Verification Failed: node_sequences.data mismatch. "
                       "Sizes: "
                    << original.node_sequences.data.size() << " vs "
                    << decompressed.node_sequences.data.size() << std::endl;
          success = false;
        }
        if (original.node_sequences.lengths !=
            decompressed.node_sequences.lengths) {
          std::cerr
              << "GPU Verification Failed: node_sequences.lengths mismatch."
              << std::endl;
          success = false;
        }

        // Compare links
        if (original.link_from_ids != decompressed.link_from_ids) {
          std::cerr
              << "GPU Verification Failed: link_from_ids mismatch. Sizes: "
              << original.link_from_ids.size() << " vs "
              << decompressed.link_from_ids.size() << std::endl;
          success = false;
        }
        if (original.link_to_ids != decompressed.link_to_ids) {
          std::cerr << "GPU Verification Failed: link_to_ids mismatch."
                    << std::endl;
          success = false;
        }
        if (original.link_from_orients != decompressed.link_from_orients) {
          std::cerr << "GPU Verification Failed: link_from_orients mismatch."
                    << std::endl;
          success = false;
        }
        if (original.link_to_orients != decompressed.link_to_orients) {
          std::cerr << "GPU Verification Failed: link_to_orients mismatch."
                    << std::endl;
          success = false;
        }

        // Compare segment optional fields
        if (original.segment_optional_fields.size() !=
            decompressed.segment_optional_fields.size()) {
          std::cerr << "GPU Verification Failed: segment_optional_fields count "
                       "mismatch. "
                    << original.segment_optional_fields.size() << " vs "
                    << decompressed.segment_optional_fields.size() << std::endl;
          success = false;
        } else {
          for (size_t i = 0; i < original.segment_optional_fields.size(); ++i) {
            const auto &orig_col = original.segment_optional_fields[i];
            const auto &decomp_col = decompressed.segment_optional_fields[i];
            if (orig_col.tag != decomp_col.tag ||
                orig_col.type != decomp_col.type) {
              std::cerr << "GPU Verification Failed: segment_optional_fields["
                        << i << "] tag/type mismatch." << std::endl;
              success = false;
            }
            // Check type-specific data
            if (orig_col.int_values != decomp_col.int_values) {
              std::cerr << "GPU Verification Failed: segment_optional_fields["
                        << i << "] (" << orig_col.tag
                        << ") int_values mismatch." << std::endl;
              success = false;
            }
          }
        }

        // Compare J-lines (structured columnar)
        if (original.jump_from_ids != decompressed.jump_from_ids) {
          std::cerr << "GPU Verification Failed: jump_from_ids mismatch."
                    << std::endl;
          success = false;
        }
        if (original.jump_to_ids != decompressed.jump_to_ids) {
          std::cerr << "GPU Verification Failed: jump_to_ids mismatch."
                    << std::endl;
          success = false;
        }
        if (original.jump_from_orients != decompressed.jump_from_orients) {
          std::cerr << "GPU Verification Failed: jump_from_orients mismatch."
                    << std::endl;
          success = false;
        }
        if (original.jump_to_orients != decompressed.jump_to_orients) {
          std::cerr << "GPU Verification Failed: jump_to_orients mismatch."
                    << std::endl;
          success = false;
        }
        if (original.jump_distances.data != decompressed.jump_distances.data ||
            original.jump_distances.lengths !=
                decompressed.jump_distances.lengths) {
          std::cerr << "GPU Verification Failed: jump_distances mismatch."
                    << std::endl;
          success = false;
        }
        if (original.jump_rest_fields.data !=
                decompressed.jump_rest_fields.data ||
            original.jump_rest_fields.lengths !=
                decompressed.jump_rest_fields.lengths) {
          std::cerr << "GPU Verification Failed: jump_rest_fields mismatch."
                    << std::endl;
          success = false;
        }

        // Compare C-lines (structured columnar)
        if (original.containment_container_ids !=
            decompressed.containment_container_ids) {
          std::cerr
              << "GPU Verification Failed: containment_container_ids mismatch."
              << std::endl;
          success = false;
        }
        if (original.containment_contained_ids !=
            decompressed.containment_contained_ids) {
          std::cerr
              << "GPU Verification Failed: containment_contained_ids mismatch."
              << std::endl;
          success = false;
        }
        if (original.containment_container_orients !=
            decompressed.containment_container_orients) {
          std::cerr << "GPU Verification Failed: containment_container_orients "
                       "mismatch."
                    << std::endl;
          success = false;
        }
        if (original.containment_contained_orients !=
            decompressed.containment_contained_orients) {
          std::cerr << "GPU Verification Failed: containment_contained_orients "
                       "mismatch."
                    << std::endl;
          success = false;
        }
        if (original.containment_positions !=
            decompressed.containment_positions) {
          std::cerr
              << "GPU Verification Failed: containment_positions mismatch."
              << std::endl;
          success = false;
        }
        if (original.containment_overlaps.data !=
                decompressed.containment_overlaps.data ||
            original.containment_overlaps.lengths !=
                decompressed.containment_overlaps.lengths) {
          std::cerr << "GPU Verification Failed: containment_overlaps mismatch."
                    << std::endl;
          success = false;
        }
        if (original.containment_rest_fields.data !=
                decompressed.containment_rest_fields.data ||
            original.containment_rest_fields.lengths !=
                decompressed.containment_rest_fields.lengths) {
          std::cerr
              << "GPU Verification Failed: containment_rest_fields mismatch."
              << std::endl;
          success = false;
        }

        if (success) {
          GFAZ_LOG("GPU Verification PASSED: All fields match exactly.");
        }
        return success;
      },
      "Verify that two GfaGraph_gpu objects are identical.\n"
      "Returns True if all fields match, False otherwise with error details.",
      py::arg("original"), py::arg("decompressed"));
#endif
}
