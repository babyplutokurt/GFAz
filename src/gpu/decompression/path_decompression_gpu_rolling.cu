#include "gpu/decompression/decompression_workflow_gpu_internal.hpp"
#include "gpu/core/codec_gpu.cuh"
#include "gpu/decompression/path_decompression_gpu_rolling.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <thrust/copy.h>

namespace gpu_decompression {

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(const Clock::time_point &start,
                  const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void check_cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

struct ScopedCudaStream {
  cudaStream_t stream = nullptr;

  explicit ScopedCudaStream(unsigned int flags = cudaStreamDefault) {
    check_cuda(cudaStreamCreateWithFlags(&stream, flags),
               "cudaStreamCreateWithFlags(rolling copy stream)");
  }

  ~ScopedCudaStream() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  ScopedCudaStream(const ScopedCudaStream &) = delete;
  ScopedCudaStream &operator=(const ScopedCudaStream &) = delete;
};

struct ScopedCudaEvent {
  cudaEvent_t event = nullptr;

  explicit ScopedCudaEvent(unsigned int flags = cudaEventDisableTiming) {
    check_cuda(cudaEventCreateWithFlags(&event, flags),
               "cudaEventCreateWithFlags(rolling decode ready)");
  }

  ~ScopedCudaEvent() {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }

  ScopedCudaEvent(const ScopedCudaEvent &) = delete;
  ScopedCudaEvent &operator=(const ScopedCudaEvent &) = delete;
};

} // namespace

void decode_rolling_path_chunk_to_device(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const RollingPathDecodePlan &plan,
    thrust::device_vector<int32_t> &d_chunk_workspace, size_t chunk_index) {
  const auto &chunk = plan.schedule.chunks.at(chunk_index);
  if (plan.d_rule_sizes.empty()) {
    d_chunk_workspace.resize(chunk.expanded_count());
    thrust::copy(
        d_encoded_path.begin() + static_cast<std::ptrdiff_t>(chunk.encoded_begin),
        d_encoded_path.begin() + static_cast<std::ptrdiff_t>(chunk.encoded_end),
        d_chunk_workspace.begin());
    return;
  }
  gpu_codec::expand_and_inverse_decode_chunk_device(
      d_encoded_path, plan.d_output_offsets, plan.d_expanded_rules,
      plan.d_rule_offsets, plan.d_rule_sizes, d_chunk_workspace,
      plan.d_offs_final, chunk.encoded_begin, chunk.encoded_end,
      chunk.expanded_begin, chunk.expanded_end, chunk.segment_begin,
      chunk.segment_end, plan.min_rule_id, plan.max_rule_id,
      static_cast<size_t>(plan.schedule.output_size));
}

void prepare_rolling_path_host_buffer(const RollingPathDecodePlan &plan,
                                      size_t chunk_index,
                                      RollingPathHostBuffer &host_buffer) {
  const RollingPathChunkMetadata metadata =
      describe_rolling_path_chunk(plan, chunk_index);
  host_buffer.segment_begin = metadata.segment_begin;
  host_buffer.segment_end = metadata.segment_end;
  host_buffer.expanded_begin = metadata.expanded_begin;
  host_buffer.expanded_end = metadata.expanded_end;
  host_buffer.node_count = metadata.node_count;
  host_buffer.lengths = metadata.lengths;

  if (host_buffer.node_capacity < host_buffer.node_count) {
    throw std::runtime_error(
        "Rolling host buffer capacity is smaller than decoded chunk size");
  }
  if (host_buffer.node_count > 0 && host_buffer.host_nodes == nullptr) {
    throw std::runtime_error(
        "Rolling host buffer is missing host storage for decoded chunk");
  }
}

void copy_rolling_path_chunk_to_host_buffer(
    const RollingPathDecodePlan &plan,
    const thrust::device_vector<int32_t> &d_chunk_workspace, size_t chunk_index,
    RollingPathHostBuffer &host_buffer) {
  prepare_rolling_path_host_buffer(plan, chunk_index, host_buffer);
  thrust::copy(d_chunk_workspace.begin(),
               d_chunk_workspace.begin() +
                   static_cast<std::ptrdiff_t>(host_buffer.node_count),
               host_buffer.host_nodes);
}

void ensure_rolling_path_pinned_host_buffer_capacity(
    RollingPathPinnedHostBuffer &host_buffer, size_t required_capacity) {
  if (host_buffer.node_capacity >= required_capacity &&
      host_buffer.host_nodes != nullptr) {
    return;
  }

  if (host_buffer.host_nodes != nullptr) {
    check_cuda(cudaFreeHost(host_buffer.host_nodes),
               "cudaFreeHost(rolling pinned buffer)");
    host_buffer.host_nodes = nullptr;
    host_buffer.node_capacity = 0;
  }

  if (required_capacity == 0) {
    return;
  }

  check_cuda(cudaMallocHost(reinterpret_cast<void **>(&host_buffer.host_nodes),
                            required_capacity * sizeof(int32_t)),
             "cudaMallocHost(rolling pinned buffer)");
  host_buffer.node_capacity = required_capacity;
}

void release_rolling_path_pinned_host_buffer(
    RollingPathPinnedHostBuffer &host_buffer) {
  if (host_buffer.ready != nullptr) {
    check_cuda(cudaEventDestroy(host_buffer.ready),
               "cudaEventDestroy(rolling pinned buffer ready)");
    host_buffer.ready = nullptr;
  }
  if (host_buffer.host_nodes != nullptr) {
    check_cuda(cudaFreeHost(host_buffer.host_nodes),
               "cudaFreeHost(rolling pinned buffer)");
    host_buffer.host_nodes = nullptr;
  }
  host_buffer.node_capacity = 0;
  host_buffer.node_count = 0;
  host_buffer.segment_begin = 0;
  host_buffer.segment_end = 0;
  host_buffer.expanded_begin = 0;
  host_buffer.expanded_end = 0;
  host_buffer.lengths.clear();
}

void copy_rolling_path_chunk_to_pinned_host_async(
    const RollingPathDecodePlan &plan,
    const thrust::device_vector<int32_t> &d_chunk_workspace, size_t chunk_index,
    RollingPathPinnedHostBuffer &host_buffer, cudaStream_t copy_stream) {
  const RollingPathChunkMetadata metadata =
      describe_rolling_path_chunk(plan, chunk_index);
  ensure_rolling_path_pinned_host_buffer_capacity(host_buffer,
                                                  metadata.node_count);
  host_buffer.node_count = metadata.node_count;
  host_buffer.segment_begin = metadata.segment_begin;
  host_buffer.segment_end = metadata.segment_end;
  host_buffer.expanded_begin = metadata.expanded_begin;
  host_buffer.expanded_end = metadata.expanded_end;
  host_buffer.lengths = metadata.lengths;

  if (host_buffer.ready == nullptr) {
    check_cuda(cudaEventCreateWithFlags(&host_buffer.ready,
                                        cudaEventDisableTiming),
               "cudaEventCreateWithFlags(rolling pinned buffer ready)");
  }

  if (host_buffer.node_count > 0) {
    check_cuda(cudaMemcpyAsync(
                   host_buffer.host_nodes,
                   thrust::raw_pointer_cast(d_chunk_workspace.data()),
                   host_buffer.node_count * sizeof(int32_t),
                   cudaMemcpyDeviceToHost, copy_stream),
               "cudaMemcpyAsync(rolling chunk to pinned host)");
  }
  check_cuda(cudaEventRecord(host_buffer.ready, copy_stream),
             "cudaEventRecord(rolling pinned buffer ready)");
}

void wait_for_rolling_path_pinned_host_buffer(
    const RollingPathPinnedHostBuffer &host_buffer) {
  if (host_buffer.ready == nullptr) {
    return;
  }
  check_cuda(cudaEventSynchronize(host_buffer.ready),
             "cudaEventSynchronize(rolling pinned buffer ready)");
}

void stream_decompress_paths_gpu_rolling(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, RollingPathChunkConsumer consumer,
    RollingPathStreamOptions stream_options) {
  if (!consumer) {
    throw std::runtime_error(
        "Rolling stream decompression requires a chunk consumer");
  }

  const uint32_t resolved_traversals_per_chunk =
      std::max<uint32_t>(1, traversals_per_chunk);
  const size_t num_host_buffers = std::max<size_t>(1, stream_options.num_host_buffers);

  if (decompression_debug_enabled()) {
    std::cout << "[GPU Decompress] Streaming rolling path chunks with "
              << num_host_buffers << " pinned host buffers ("
              << resolved_traversals_per_chunk << " traversals per chunk), "
              << "rolling output chunk="
              << (stream_options.rolling_output_chunk_bytes /
                  (1024.0 * 1024.0))
              << " MiB, min_rule_id=" << min_rule_id << std::endl;
  }

  RollingPathDecodePlan plan = prepare_rolling_path_decode_plan(
      d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules,
      d_lens_final, resolved_traversals_per_chunk,
      stream_options.rolling_output_chunk_bytes);
  thrust::device_vector<int32_t> d_chunk_workspace;

  std::vector<RollingPathPinnedHostBuffer> host_buffers(num_host_buffers);
  ScopedCudaStream copy_stream(cudaStreamNonBlocking);
  ScopedCudaEvent decode_ready;

  std::mutex mutex;
  std::condition_variable cv;
  std::deque<size_t> free_indices;
  std::deque<size_t> ready_indices;
  std::exception_ptr worker_error;
  bool producer_done = false;

  for (size_t i = 0; i < host_buffers.size(); ++i) {
    free_indices.push_back(i);
  }

  auto writer_thread = std::thread([&]() {
    try {
      while (true) {
        size_t buffer_index = 0;
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&]() {
            return worker_error != nullptr || !ready_indices.empty() ||
                   producer_done;
          });
          if (worker_error != nullptr) {
            return;
          }
          if (ready_indices.empty()) {
            if (producer_done) {
              return;
            }
            continue;
          }
          buffer_index = ready_indices.front();
          ready_indices.pop_front();
        }

        wait_for_rolling_path_pinned_host_buffer(host_buffers[buffer_index]);
        consumer(host_buffers[buffer_index]);

        {
          std::lock_guard<std::mutex> lock(mutex);
          free_indices.push_back(buffer_index);
        }
        cv.notify_all();
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        worker_error = std::current_exception();
      }
      cv.notify_all();
    }
  });

  try {
    for (size_t chunk_index = 0; chunk_index < plan.schedule.chunks.size();
         ++chunk_index) {
      size_t buffer_index = 0;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() {
          return worker_error != nullptr || !free_indices.empty();
        });
        if (worker_error != nullptr) {
          std::rethrow_exception(worker_error);
        }
        buffer_index = free_indices.front();
        free_indices.pop_front();
      }

      decode_rolling_path_chunk_to_device(d_encoded_path, plan,
                                          d_chunk_workspace, chunk_index);
      check_cuda(cudaEventRecord(decode_ready.event, 0),
                 "cudaEventRecord(rolling decode ready)");
      check_cuda(cudaStreamWaitEvent(copy_stream.stream, decode_ready.event, 0),
                 "cudaStreamWaitEvent(rolling copy waits for decode)");
      copy_rolling_path_chunk_to_pinned_host_async(
          plan, d_chunk_workspace, chunk_index, host_buffers[buffer_index],
          copy_stream.stream);
      // The rolling streaming path currently reuses a single device workspace
      // for all chunks. Do not launch the next decode until the D2H copy from
      // that workspace has completed, or the next chunk will overwrite the
      // source buffer before the async transfer finishes.
      check_cuda(cudaStreamSynchronize(copy_stream.stream),
                 "cudaStreamSynchronize(rolling copy before workspace reuse)");

      {
        std::lock_guard<std::mutex> lock(mutex);
        ready_indices.push_back(buffer_index);
      }
      cv.notify_all();
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      producer_done = true;
    }
    cv.notify_all();
    writer_thread.join();
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (worker_error == nullptr) {
        worker_error = std::current_exception();
      }
      producer_done = true;
    }
    cv.notify_all();
    writer_thread.join();
  }

  check_cuda(cudaStreamSynchronize(copy_stream.stream),
             "cudaStreamSynchronize(rolling copy stream)");
  for (auto &host_buffer : host_buffers) {
    release_rolling_path_pinned_host_buffer(host_buffer);
  }

  if (worker_error != nullptr) {
    std::rethrow_exception(worker_error);
  }
}

void decompress_paths_gpu_rolling(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, size_t rolling_output_chunk_bytes,
    std::vector<int32_t> &out_data) {
  const uint32_t resolved_traversals_per_chunk =
      std::max<uint32_t>(1, traversals_per_chunk);

  if (decompression_debug_enabled()) {
    std::cout << "[GPU Decompress] Expanding path with rolling chunk "
                 "scheduler ("
              << resolved_traversals_per_chunk
              << " traversals per chunk, rolling output chunk "
              << (rolling_output_chunk_bytes / (1024.0 * 1024.0))
              << " MiB), min_rule_id=" << min_rule_id
              << std::endl;
  }

  const auto prepare_start = Clock::now();
  RollingPathDecodePlan plan = prepare_rolling_path_decode_plan(
      d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules,
      d_lens_final, resolved_traversals_per_chunk,
      std::max<size_t>(1, rolling_output_chunk_bytes));
  const auto prepare_end = Clock::now();
  out_data.resize(static_cast<size_t>(plan.schedule.output_size));
  thrust::device_vector<int32_t> d_chunk_workspace;

  double decode_ms = 0.0;
  double copy_ms = 0.0;
  for (size_t chunk_index = 0; chunk_index < plan.schedule.chunks.size();
       ++chunk_index) {
    const auto decode_start = Clock::now();
    decode_rolling_path_chunk_to_device(d_encoded_path, plan, d_chunk_workspace,
                                        chunk_index);
    const auto decode_end = Clock::now();
    RollingPathHostBuffer host_buffer;
    host_buffer.host_nodes =
        out_data.data() + plan.schedule.chunks[chunk_index].expanded_begin;
    host_buffer.node_capacity =
        out_data.size() -
        static_cast<size_t>(plan.schedule.chunks[chunk_index].expanded_begin);
    const auto copy_start = Clock::now();
    copy_rolling_path_chunk_to_host_buffer(plan, d_chunk_workspace, chunk_index,
                                           host_buffer);
    const auto copy_end = Clock::now();

    decode_ms += elapsed_ms(decode_start, decode_end);
    copy_ms += elapsed_ms(copy_start, copy_end);
  }

  if (decompression_debug_enabled()) {
    std::cout << "[GPU Decompress][Rolling] prepare context + schedule: "
              << std::fixed << std::setprecision(2)
              << elapsed_ms(prepare_start, prepare_end) << " ms" << std::endl;
    std::cout << "[GPU Decompress][Rolling] chunk decode kernels: "
              << std::fixed << std::setprecision(2) << decode_ms << " ms"
              << std::endl;
    std::cout << "[GPU Decompress][Rolling] chunk device->host copies: "
              << std::fixed << std::setprecision(2) << copy_ms << " ms"
              << std::endl;
    std::cout << "[GPU Decompress][Rolling] chunks: "
              << plan.schedule.chunks.size() << ", output nodes: "
              << out_data.size() << std::endl;
  }
}

} // namespace gpu_decompression
