#ifndef CODEC_GPU_CUH
#define CODEC_GPU_CUH

#include <cstdint>
#include <vector>
#include <map>
#include <cstddef>
#include <thrust/device_vector.h>
#include "gpu/gfa_graph_gpu.hpp"

namespace gpu_codec {

// Device-side flattened paths for GPU-only workflows.
struct FlattenedPathsDevice {
    thrust::device_vector<int32_t> data;
    thrust::device_vector<uint32_t> lengths;

    size_t num_paths() const { return lengths.size(); }
    size_t total_nodes() const { return data.size(); }
};

// Host <-> Device helpers for flattened paths
FlattenedPathsDevice copy_paths_to_device(const FlattenedPaths& paths);
void copy_paths_to_host(const FlattenedPathsDevice& device_paths, FlattenedPaths& paths);

/**
 * Delta encode flattened paths on GPU using CUB DeviceAdjacentDifference
 *
 * Transforms path data in-place: data[i] = data[i] - data[i-1]
 * First element remains unchanged (delta from 0)
 *
 * @param paths Input/output flattened paths (modified in-place)
 */
void delta_encode_paths(FlattenedPaths& paths);
void delta_encode_paths_device(FlattenedPathsDevice& paths);

/**
 * Delta decode flattened paths on GPU using CUB DeviceScan::InclusiveSum
 *
 * Restores original path data from deltas: data[i] = sum(delta[0..i])
 *
 * @param paths Input/output delta-encoded paths (restored in-place)
 */
void delta_decode_paths(FlattenedPaths& paths);

/**
 * Find the maximum absolute node ID in the flattened paths on GPU.
 *
 * Uses CUB DeviceReduce::Max with a custom transform iterator to handle
 * absolute value conversion on the fly.
 *
 * @param paths Flattened paths
 * @return Maximum absolute value found (0 if empty)
 */
uint32_t find_max_abs_node(const FlattenedPaths& paths);

/**
 * Find repeated 2-mers (frequency >= 2) in the paths.
 * 
 * 1. Generates all 2-mers (canonicalized) from paths.
 * 2. Sorts them on GPU.
 * 3. Filters to keep only those that appear 2+ times.
 * 4. Returns unique set of repeated 2-mers (Packed2mer/uint64_t).
 * 
 * @param paths Flattened paths
 * @return Vector of repeated 2-mers (on host)
 */
std::vector<uint64_t> find_repeated_2mers(const FlattenedPaths& paths);
std::vector<uint64_t> find_repeated_2mers_device(const FlattenedPathsDevice& paths);

// Create hash table mapping 2-mer -> RuleID on GPU
struct GPUHashTable; 
void* create_rule_table_gpu(const std::vector<uint64_t>& rules, uint32_t start_id);

// Apply 2-mer rules
void apply_2mer_rules_gpu(FlattenedPaths& paths, void* d_table_handle, std::vector<uint8_t>& rules_used, uint32_t start_id);
void apply_2mer_rules_gpu_device(FlattenedPathsDevice& paths, void* d_table_handle, std::vector<uint8_t>& rules_used, uint32_t start_id);

/**
 * Compact rules and remap path nodes in one go.
 */
void compact_rules_and_remap_gpu(
    FlattenedPaths& paths,
    const std::vector<uint8_t>& rules_used,
    std::vector<uint64_t>& current_rules,
    uint32_t start_id);
void compact_rules_and_remap_gpu_device(
    FlattenedPathsDevice& paths,
    const std::vector<uint8_t>& rules_used,
    std::vector<uint64_t>& current_rules,
    uint32_t start_id);

/**
 * Sort rules by their 2-mer value (for better delta compression) and remap path IDs.
 */
void sort_rules_and_remap_gpu(
    FlattenedPaths& paths,
    std::vector<uint64_t>& current_rules,
    uint32_t start_id);
void sort_rules_and_remap_gpu_device(
    FlattenedPathsDevice& paths,
    std::vector<uint64_t>& current_rules,
    uint32_t start_id);

/**
 * Legacy: Orchestrates the full 2-mer compression layer on GPU for multiple rounds.
 * Uses master_rulebook map (less efficient). 
 * Prefer run_path_compression_gpu from compression_workflow_gpu.hpp instead.
 */
void run_compression_layer_2mer_gpu(
    FlattenedPaths& paths, 
    uint32_t& next_starting_id, 
    int num_rounds, 
    std::map<uint32_t, uint64_t>& master_rulebook);
void run_compression_layer_2mer_gpu_device(
    FlattenedPathsDevice& paths,
    uint32_t& next_starting_id,
    int num_rounds,
    std::map<uint32_t, uint64_t>& master_rulebook);

// ============================================================================
// Device-vector helpers for GPU-resident workflow
// ============================================================================

uint32_t find_max_abs_device(const thrust::device_vector<int32_t>& d_data);

void delta_encode_device_vec(thrust::device_vector<int32_t>& d_data);

thrust::device_vector<uint64_t> find_repeated_2mers_device_vec(
    const thrust::device_vector<int32_t>& d_data);

void* create_rule_table_gpu_from_device(
    const thrust::device_vector<uint64_t>& rules,
    uint32_t start_id);

void apply_2mer_rules_device_vec(
    thrust::device_vector<int32_t>& d_data,
    void* d_table_handle,
    thrust::device_vector<uint8_t>& rules_used,
    uint32_t start_id);

void compact_rules_and_remap_device_vec(
    thrust::device_vector<int32_t>& d_data,
    const thrust::device_vector<uint8_t>& rules_used,
    thrust::device_vector<uint64_t>& rules,
    uint32_t start_id);

void sort_rules_and_remap_device_vec(
    thrust::device_vector<int32_t>& d_data,
    thrust::device_vector<uint64_t>& rules,
    uint32_t start_id);

/**
 * Split packed 2-mer rules into first/second element vectors and delta-encode them.
 * 
 * Input:  rules[i] = (first << 32) | (second & 0xFFFFFFFF)
 * Output: d_first[i] = first element, d_second[i] = second element (both delta-encoded)
 * 
 * @param d_rules Device vector of packed 2-mers
 * @param d_first Output: delta-encoded first elements
 * @param d_second Output: delta-encoded second elements
 */
void split_and_delta_encode_rules_device_vec(
    const thrust::device_vector<uint64_t>& d_rules,
    thrust::device_vector<int32_t>& d_first,
    thrust::device_vector<int32_t>& d_second);

// ============================================================================
// Segmented (boundary-aware) device-vector helpers
// ============================================================================

/**
 * Compute boundary masks from segment offsets.
 *
 * @param d_offsets Exclusive prefix sum of per-segment lengths
 * @param num_segments Number of segments (paths + walks)
 * @param total_nodes Total number of nodes in the flat array
 * @param d_is_first Output: d_is_first[idx] = 1 if idx is the first node of a segment
 * @param d_is_last  Output: d_is_last[idx] = 1 if idx is the last node of a segment
 */
void compute_boundary_masks(
    const thrust::device_vector<uint64_t>& d_offsets,
    uint32_t num_segments,
    size_t total_nodes,
    thrust::device_vector<uint8_t>& d_is_first,
    thrust::device_vector<uint8_t>& d_is_last);

/**
 * Segmented delta encode: per-traversal adjacent difference.
 * First element of each segment is unchanged (matches CPU semantics).
 *
 * @param d_data Input/output: modified in-place
 * @param d_is_first Boundary mask: 1 at the first element of each segment
 */
void segmented_delta_encode_device_vec(
    thrust::device_vector<int32_t>& d_data,
    const thrust::device_vector<uint8_t>& d_is_first);

/**
 * Boundary-aware 2-mer discovery: skips pairs that cross segment boundaries.
 *
 * @param d_data Flat path data
 * @param d_is_last Boundary mask: 1 at the last element of each segment
 * @return Device vector of unique repeated 2-mers (canonical form)
 */
thrust::device_vector<uint64_t> find_repeated_2mers_segmented_device_vec(
    const thrust::device_vector<int32_t>& d_data,
    const thrust::device_vector<uint8_t>& d_is_last);

/**
 * Boundary-aware rule application: skips replacements that cross segment boundaries.
 * Also computes new per-segment lengths after compaction.
 *
 * @param d_data Input/output: encoded path data (shrinks after rule application)
 * @param d_table_handle GPU hash table mapping 2-mer -> rule ID
 * @param rules_used Output: marks which rules were actually applied
 * @param start_id First rule ID in the current round
 * @param d_offsets Exclusive prefix sum of per-segment lengths
 * @param num_segments Number of segments
 * @return New per-segment lengths after compaction
 */
thrust::device_vector<uint32_t> apply_2mer_rules_segmented_device_vec(
    thrust::device_vector<int32_t>& d_data,
    void* d_table_handle,
    thrust::device_vector<uint8_t>& rules_used,
    uint32_t start_id,
    const thrust::device_vector<uint64_t>& d_offsets,
    uint32_t num_segments);

// ============================================================================
// GPU Path Expansion (Decompression)
// ============================================================================

/**
 * Expand rules in a path using single-pass algorithm with pre-expanded rules.
 *
 * Algorithm (optimized):
 * Phase 1 - Pre-expand rules (done once, O(num_rules)):
 *   1. Compute final expansion size of each rule
 *   2. Expand each rule to its final form (all raw nodes) in parallel
 *
 * Phase 2 - Expand path (single pass, O(path_size)):
 *   1. Compute output offsets using pre-computed rule sizes
 *   2. Copy pre-expanded rules to output in single pass
 *
 * Complexity: O(num_rules) + O(path_size)
 * (vs O(num_compression_rounds * path_size) for iterative version)
 *
 * @param d_encoded_path Input: compressed path with rule IDs
 * @param d_rules_first First elements of rules (NOT delta-encoded)
 * @param d_rules_second Second elements of rules (NOT delta-encoded)
 * @param min_rule_id Minimum rule ID (values < this are raw nodes)
 * @param num_rules Total number of rules
 * @return Expanded path with all raw node IDs
 */
thrust::device_vector<int32_t> expand_path_device_vec(
    const thrust::device_vector<int32_t>& d_encoded_path,
    const thrust::device_vector<int32_t>& d_rules_first,
    const thrust::device_vector<int32_t>& d_rules_second,
    uint32_t min_rule_id,
    size_t num_rules);

/**
 * @deprecated Use expand_path_device_vec instead.
 *
 * Iterative multi-pass expansion. Each pass expands one level of rules.
 * Kept for performance comparison with the new single-pass algorithm.
 *
 * Complexity: O(num_compression_rounds * path_size)
 */
thrust::device_vector<int32_t> expand_path_device_vec_iterative(
    const thrust::device_vector<int32_t>& d_encoded_path,
    const thrust::device_vector<int32_t>& d_rules_first,
    const thrust::device_vector<int32_t>& d_rules_second,
    uint32_t min_rule_id,
    size_t num_rules);

/**
 * Inverse delta-encode a device vector (prefix sum).
 *
 * @param d_delta_encoded Input: delta-encoded values
 * @return Original values before delta encoding
 */
thrust::device_vector<int32_t> inverse_delta_decode_device_vec(
    const thrust::device_vector<int32_t>& d_delta_encoded);

/**
 * Segmented inverse delta-decode a device vector (per-segment prefix sum).
 * Each segment is independently prefix-summed, matching segmented delta encode.
 *
 * @param d_delta_encoded Input: delta-encoded values (segmented)
 * @param d_offsets Exclusive prefix sum of per-segment lengths
 * @param num_segments Number of segments
 * @param total_nodes Total number of elements
 * @return Original values before segmented delta encoding
 */
thrust::device_vector<int32_t> segmented_inverse_delta_decode_device_vec(
    const thrust::device_vector<int32_t>& d_delta_encoded,
    const thrust::device_vector<uint64_t>& d_offsets,
    uint32_t num_segments,
    size_t total_nodes);

// Cleanup
void free_rule_table_gpu(void* d_table_handle);

// ============================================================================
// GPU Bit-Packing for Orientations
// ============================================================================

/**
 * Pack orientation chars into bytes on GPU.
 * Each byte stores 8 orientations, with '-' = bit 1, '+' = bit 0.
 * 
 * @param orients Input: vector of '+' or '-' chars
 * @return Packed bytes ((size + 7) / 8 bytes)
 */
std::vector<uint8_t> pack_orientations_gpu(const std::vector<char>& orients);

/**
 * Unpack bytes into orientation chars on GPU.
 * Bit 1 maps to '-', bit 0 maps to '+'.
 * 
 * @param packed Input: packed bytes
 * @param num_orients Number of orientations to unpack
 * @return Unpacked orientation chars
 */
std::vector<char> unpack_orientations_gpu(const std::vector<uint8_t>& packed, size_t num_orients);

} // namespace gpu_codec

#endif // CODEC_GPU_CUH

