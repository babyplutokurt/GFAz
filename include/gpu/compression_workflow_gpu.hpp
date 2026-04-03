#ifndef COMPRESSION_WORKFLOW_GPU_HPP
#define COMPRESSION_WORKFLOW_GPU_HPP

#include "compression_workflow.hpp"
#include "gpu/codec_gpu_nvcomp.cuh"
#include "gpu/gfa_graph_gpu.hpp"
#include <cstdint>
#include <map>
#include <vector>

namespace gpu_compression {

// Simplified rule range for GPU output (no k or flattened_offset needed)
struct GPURuleRange {
  uint32_t start_id;
  uint32_t count;
};

// Segment Optional Fields (each column is nvComp ZSTD compressed)
struct CompressedOptionalFieldColumn_gpu {
  std::string tag; // Two-character tag, e.g., "LN", "RC"
  char type;       // GFA type: 'A', 'i', 'f', 'Z', 'J', 'H', 'B'
  size_t num_elements = 0;

  gpu_codec::NvcompCompressedBlock int_values_zstd_nvcomp;   // type 'i'
  gpu_codec::NvcompCompressedBlock float_values_zstd_nvcomp; // type 'f'
  gpu_codec::NvcompCompressedBlock char_values_zstd_nvcomp;  // type 'A'
  gpu_codec::NvcompCompressedBlock strings_zstd_nvcomp; // types 'Z', 'J', 'H'
  gpu_codec::NvcompCompressedBlock
      string_lengths_zstd_nvcomp; // lengths for string types

  gpu_codec::NvcompCompressedBlock b_subtypes_zstd_nvcomp; // type 'B'
  gpu_codec::NvcompCompressedBlock b_lengths_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock b_concat_bytes_zstd_nvcomp;
};

// GPU-backend compressed data structure (analogous to CPU's CompressedData)
// Contains all data required for full GFA round-trip
struct CompressedData_gpu {
  // ====== ENCODED PATH DATA ======
  // Delta-encoded + rule-replaced, then nvComp ZSTD compressed
  gpu_codec::NvcompCompressedBlock encoded_path_zstd_nvcomp;

  // ====== RULES ======
  // Rules are split into first/second elements, delta-encoded, then nvComp ZSTD
  // compressed
  gpu_codec::NvcompCompressedBlock rules_first_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock rules_second_zstd_nvcomp;

  // Layer ranges: start_id and count per compression round
  std::vector<GPURuleRange> layer_ranges;

  // ====== PATH/WALK SPLIT INFO ======
  uint32_t num_paths = 0; // Number of P-line paths in the combined path array
  uint32_t num_walks = 0; // Number of W-line walks in the combined path array

  // Original path+walk lengths (for splitting encoded_path back into individual
  // paths) nvComp ZSTD compressed
  gpu_codec::NvcompCompressedBlock path_lengths_zstd_nvcomp;

  // ====== PATH METADATA (P-lines only) ======
  gpu_codec::NvcompCompressedBlock names_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock name_lengths_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock overlaps_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock overlap_lengths_zstd_nvcomp;

  // ====== WALK METADATA (W-lines only) ======
  gpu_codec::NvcompCompressedBlock walk_sample_ids_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock walk_sample_id_lengths_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock walk_hap_indices_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock walk_seq_ids_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock walk_seq_id_lengths_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock walk_seq_starts_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock walk_seq_ends_zstd_nvcomp;

  // ====== HEADER ======
  std::string header_line; // Small, stored as-is (not compressed)

  // ====== SEGMENT DATA ======
  gpu_codec::NvcompCompressedBlock segment_sequences_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock segment_seq_lengths_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock node_names_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock node_name_lengths_zstd_nvcomp;

  // Segment Optional Fields (each column is nvComp ZSTD compressed)
  std::vector<CompressedOptionalFieldColumn_gpu>
      segment_optional_fields_zstd_nvcomp;

  // ====== LINK DATA ======
  gpu_codec::NvcompCompressedBlock
      link_from_ids_zstd_nvcomp; // delta + zigzag + varint, then nvComp
  gpu_codec::NvcompCompressedBlock
      link_to_ids_zstd_nvcomp; // delta + zigzag + varint, then nvComp
  gpu_codec::NvcompCompressedBlock
      link_from_orients_zstd_nvcomp; // bit-packed, then nvComp
  gpu_codec::NvcompCompressedBlock
      link_to_orients_zstd_nvcomp; // bit-packed, then nvComp
  gpu_codec::NvcompCompressedBlock
      link_overlap_nums_zstd_nvcomp; // direct nvComp
  gpu_codec::NvcompCompressedBlock
      link_overlap_ops_zstd_nvcomp; // direct nvComp
  size_t num_links = 0;

  // Link Optional Fields (each column is nvComp ZSTD compressed)
  std::vector<CompressedOptionalFieldColumn_gpu>
      link_optional_fields_zstd_nvcomp;

  // ====== JUMP DATA (J-lines) - structured columnar, matching CPU backend
  // ======
  size_t num_jumps_stored = 0;
  gpu_codec::NvcompCompressedBlock jump_from_ids_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock jump_to_ids_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock jump_from_orients_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock jump_to_orients_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock jump_distances_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock jump_distance_lengths_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock jump_rest_fields_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock jump_rest_lengths_zstd_nvcomp;

  // ====== CONTAINMENT DATA (C-lines) - structured columnar, matching CPU
  // backend ======
  size_t num_containments_stored = 0;
  gpu_codec::NvcompCompressedBlock containment_container_ids_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_contained_ids_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_container_orients_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_contained_orients_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_positions_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_overlaps_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_overlap_lengths_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_rest_fields_zstd_nvcomp;
  gpu_codec::NvcompCompressedBlock containment_rest_lengths_zstd_nvcomp;

  // Convenience methods
  size_t total_rules() const {
    size_t total = 0;
    for (const auto &range : layer_ranges) {
      total += range.count;
    }
    return total;
  }
  size_t num_rounds() const { return layer_ranges.size(); }
  uint32_t min_rule_id() const {
    return layer_ranges.empty() ? 0 : layer_ranges[0].start_id;
  }
};

/**
 * GPU-resident path compression with minimal host-device transfers.
 *
 * This function:
 * 1. Copies input paths to GPU once
 * 2. Computes start_id on GPU (max abs value + 1)
 * 3. Performs delta encoding on GPU
 * 4. For each round, finds 2-mers, applies rules, compacts, and sorts - ALL ON
 * GPU
 * 5. Accumulates rules in a GPU device vector (no per-round copies to host)
 * 6. Only at the end, copies results back to host
 *
 * @param paths Input flattened paths (host memory)
 * @param num_rounds Maximum number of compression rounds
 * @return CompressedData_gpu containing encoded path, all rules, and layer
 * ranges
 */
CompressedData_gpu run_path_compression_gpu(const FlattenedPaths &paths,
                                            int num_rounds);

/**
 * High-level GPU compression entry point (parse + compress).
 * Mirrors CPU compress_gfa but uses GPU path compression and nvComp Zstd for
 * path metadata.
 */
CompressedData_gpu compress_gfa_gpu(const std::string &gfa_file_path,
                                    int num_rounds);

/**
 * GPU compression from GfaGraph_gpu (no parsing).
 * Use this for accurate timing of compression-only.
 *
 * @param gpu_graph Pre-converted GPU graph
 * @param num_rounds Number of compression rounds
 * @return CompressedData_gpu containing all compressed fields
 */
CompressedData_gpu compress_gpu_graph(const GfaGraph_gpu &gpu_graph,
                                      int num_rounds);

void set_gpu_compression_debug(bool enabled);

/**
 * Build a rulebook map from the flat rules vector and layer ranges.
 * Useful for round-trip verification with CPU reconstruction.
 *
 * @param data CompressedData_gpu containing all_rules and layer_ranges
 * @return Map from rule_id -> packed_2mer
 */
std::map<uint32_t, uint64_t> build_rulebook(const CompressedData_gpu &data);

} // namespace gpu_compression

#endif // COMPRESSION_WORKFLOW_GPU_HPP
