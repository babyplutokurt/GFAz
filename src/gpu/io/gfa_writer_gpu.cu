#include "gpu/io/gfa_writer_gpu.hpp"
#include "gpu/decompression/decompression_primitives_gpu.hpp"
#include "gpu/decompression/decompression_workflow_gpu_internal.hpp"
#include "gpu/io/gfa_writer_gpu_direct.hpp"
#include "io/gfa_write_utils.hpp"
#include "io/gfa_writer.hpp"
#include "utils/runtime_utils.hpp"
#include "workflows/decompression_debug.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thrust/device_vector.h>

#ifdef ENABLE_CUDA

namespace {

using Clock = std::chrono::high_resolution_clock;
using gfz::runtime_utils::elapsed_ms;
using namespace gfz::decompression_debug;

} // namespace

void write_gfa_from_compressed_data_gpu(
    const CompressedData &data, const std::string &output_path,
    gpu_decompression::GpuDecompressionOptions options) {
  if (options.use_legacy_full_decompression) {
    GfaGraph graph = gpu_decompression::decompress_to_host_graph(data, options);
    write_gfa(graph, output_path);
    return;
  }

  std::vector<TimedDebugStage> debug_stages;
  const auto writer_total_start = Clock::now();
  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("GFA writer error: failed to open output file: " +
                             output_path);
  }

  const gpu_decompression::GpuTraversalRulebook rulebook =
      gpu_decompression::prepare_gpu_traversal_rulebook(data);
  debug_stages.push_back({"decode rules", rulebook.host_decode_ms});

  const gpu_decompression::GpuDirectWriterStaticFields static_fields =
      gpu_decompression::decode_gpu_direct_writer_static_fields(data);
  debug_stages.push_back({"decode static graph fields",
                          static_fields.decode_ms});

  auto t0 = Clock::now();
  gpu_decompression::write_gpu_direct_writer_static_fields(out, data,
                                                           static_fields);
  auto t1 = Clock::now();
  debug_stages.push_back({"write static graph fields", elapsed_ms(t0, t1)});

  if (!data.paths_zstd.payload.empty() && !data.original_path_lengths.empty()) {
    const gpu_decompression::GpuTraversalPayload path_payload =
        gpu_decompression::prepare_gpu_traversal_payload(
            data.paths_zstd, data.original_path_lengths);
    const gpu_decompression::GpuPathWriterMetadata path_metadata =
        gpu_decompression::decode_gpu_path_writer_metadata(data);
    debug_stages.push_back({"decode path payload", path_payload.host_decode_ms});
    debug_stages.push_back({"decode path metadata", path_metadata.decode_ms});
    t0 = Clock::now();
    gpu_decompression::stream_gpu_traversal_to_host(
        path_payload, rulebook, options,
        [&](const gpu_decompression::RollingPathPinnedHostBuffer &buffer) {
          gpu_decompression::write_gpu_path_chunk_lines(out, buffer,
                                                        path_metadata);
        },
        {.num_host_buffers = 2,
         .max_expanded_chunk_bytes =
             std::max<size_t>(1, options.max_expanded_chunk_bytes)});
    t1 = Clock::now();
    debug_stages.push_back({"expand+write paths (GPU rolling)",
                            elapsed_ms(t0, t1)});
  }

  if (!data.walks_zstd.payload.empty() && !data.original_walk_lengths.empty()) {
    const gpu_decompression::GpuTraversalPayload walk_payload =
        gpu_decompression::prepare_gpu_traversal_payload(
            data.walks_zstd, data.original_walk_lengths);
    const gpu_decompression::GpuWalkWriterMetadata walk_metadata =
        gpu_decompression::decode_gpu_walk_writer_metadata(data);
    debug_stages.push_back({"decode walk payload", walk_payload.host_decode_ms});
    debug_stages.push_back({"decode walk metadata", walk_metadata.decode_ms});
    t0 = Clock::now();
    gpu_decompression::stream_gpu_traversal_to_host(
        walk_payload, rulebook, options,
        [&](const gpu_decompression::RollingPathPinnedHostBuffer &buffer) {
          gpu_decompression::write_gpu_walk_chunk_lines(out, buffer,
                                                        walk_metadata);
        },
        {.num_host_buffers = 2,
         .max_expanded_chunk_bytes =
             std::max<size_t>(1, options.max_expanded_chunk_bytes)});
    t1 = Clock::now();
    debug_stages.push_back({"expand+write walks (GPU rolling)",
                            elapsed_ms(t0, t1)});
  }

  out.close();

  if (gpu_decompression::decompression_debug_enabled()) {
    print_cpu_decompression_summary("GPU Direct Writer", data);
    print_cpu_decompression_timing(
        {"GPU Direct Writer", "rolling direct-writer path", debug_stages,
         elapsed_ms(writer_total_start, Clock::now())});
  }
}

#endif
