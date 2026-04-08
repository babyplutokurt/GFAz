#include "gpu/decompression_workflow_gpu_internal.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/path_decompression_gpu_rolling.hpp"

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
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

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

struct FinalExpansionSizeOp {
  const int64_t *d_rule_sizes;
  uint32_t min_rule_id;
  uint32_t max_rule_id;

  __host__ __device__ FinalExpansionSizeOp(const int64_t *sizes,
                                           uint32_t min_id, uint32_t max_id)
      : d_rule_sizes(sizes), min_rule_id(min_id), max_rule_id(max_id) {}

  __host__ __device__ __forceinline__ int64_t operator()(int32_t val) const {
    uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);
    if (abs_val >= min_rule_id && abs_val < max_rule_id) {
      return d_rule_sizes[abs_val - min_rule_id];
    }
    return 1;
  }
};

gpu_codec::RollingDecodeSchedule build_passthrough_schedule(
    const thrust::device_vector<uint32_t> &d_lens_final, size_t encoded_size,
    uint32_t traversals_per_chunk) {
  gpu_codec::RollingDecodeSchedule schedule;
  const uint32_t num_segs_final = static_cast<uint32_t>(d_lens_final.size());
  if (num_segs_final == 0) {
    schedule.output_size = static_cast<int64_t>(encoded_size);
    return schedule;
  }

  thrust::device_vector<uint64_t> d_offs_final(num_segs_final);
  thrust::exclusive_scan(d_lens_final.begin(), d_lens_final.end(),
                         d_offs_final.begin(), uint64_t(0));

  schedule.expanded_offsets.resize(num_segs_final);
  thrust::copy(d_offs_final.begin(), d_offs_final.end(),
               schedule.expanded_offsets.begin());
  schedule.encoded_offsets.assign(schedule.expanded_offsets.begin(),
                                  schedule.expanded_offsets.end());
  schedule.output_size = static_cast<int64_t>(encoded_size);

  uint32_t current_seg_begin = 0;
  for (uint32_t i = 0; i < num_segs_final; ++i) {
    const int64_t seg_expanded_start =
        static_cast<int64_t>(schedule.expanded_offsets[current_seg_begin]);
    const int64_t next_expanded_start =
        (i + 1 < num_segs_final)
            ? static_cast<int64_t>(schedule.expanded_offsets[i + 1])
            : static_cast<int64_t>(encoded_size);
    const int64_t size_so_far = next_expanded_start - seg_expanded_start;

    bool split = false;
    if (i - current_seg_begin >= traversals_per_chunk) {
      split = true;
    }
    if (size_so_far >= 32 * 1024 * 1024 && i > current_seg_begin) {
      split = true;
    }
    if (i + 1 == num_segs_final) {
      split = true;
    }

    if (!split) {
      continue;
    }

    const uint32_t chunk_seg_end = i + 1;
    const int64_t chunk_begin =
        static_cast<int64_t>(schedule.expanded_offsets[current_seg_begin]);
    schedule.chunks.push_back({current_seg_begin, chunk_seg_end, chunk_begin,
                               next_expanded_start, chunk_begin,
                               next_expanded_start});
    current_seg_begin = chunk_seg_end;
  }

  return schedule;
}

void populate_rolling_path_host_buffer_metadata(
    const RollingPathDecodeContext &context, size_t chunk_index,
    uint32_t &segment_begin, uint32_t &segment_end, int64_t &expanded_begin,
    int64_t &expanded_end, size_t &node_count, std::vector<uint32_t> &lengths) {
  const auto &chunk = context.schedule.chunks.at(chunk_index);
  node_count = static_cast<size_t>(chunk.expanded_count());
  segment_begin = chunk.segment_begin;
  segment_end = chunk.segment_end;
  expanded_begin = chunk.expanded_begin;
  expanded_end = chunk.expanded_end;
  lengths.assign(context.lengths.begin() + chunk.segment_begin,
                 context.lengths.begin() + chunk.segment_end);
}

} // namespace

RollingPathDecodeContext prepare_rolling_path_decode(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk) {
  RollingPathDecodeContext context;
  context.min_rule_id = min_rule_id;
  context.max_rule_id = min_rule_id + static_cast<uint32_t>(num_rules);
  context.lengths.resize(d_lens_final.size());
  thrust::copy(d_lens_final.begin(), d_lens_final.end(), context.lengths.begin());

  if (num_rules == 0 || d_encoded_path.empty()) {
    context.schedule = build_passthrough_schedule(
        d_lens_final, d_encoded_path.size(), traversals_per_chunk);
    context.d_offs_final = thrust::device_vector<uint64_t>(
        context.schedule.expanded_offsets.begin(),
        context.schedule.expanded_offsets.end());
    return context;
  }

  context.d_rule_sizes.resize(num_rules);
  gpu_codec::compute_rule_final_sizes_device_vec(
      d_rules_first, d_rules_second, context.d_rule_sizes, min_rule_id);

  context.d_rule_offsets.resize(num_rules);
  thrust::exclusive_scan(context.d_rule_sizes.begin(), context.d_rule_sizes.end(),
                         context.d_rule_offsets.begin());

  const int64_t total_expanded_size =
      context.d_rule_offsets.back() + context.d_rule_sizes.back();
  context.d_expanded_rules.resize(total_expanded_size);
  gpu_codec::expand_rules_to_buffer_device_vec(
      d_rules_first, d_rules_second, context.d_rule_offsets,
      context.d_expanded_rules, min_rule_id);

  context.d_output_offsets.resize(d_encoded_path.size());
  FinalExpansionSizeOp size_op(
      thrust::raw_pointer_cast(context.d_rule_sizes.data()), min_rule_id,
      context.max_rule_id);
  auto size_iter =
      thrust::make_transform_iterator(d_encoded_path.begin(), size_op);
  const int64_t output_size = thrust::transform_reduce(
      d_encoded_path.begin(), d_encoded_path.end(), size_op, int64_t(0),
      thrust::plus<int64_t>());
  thrust::exclusive_scan(size_iter, size_iter + d_encoded_path.size(),
                         context.d_output_offsets.begin());

  context.schedule = gpu_codec::build_rolling_decode_schedule(
      context.d_output_offsets, d_lens_final, d_encoded_path.size(),
      output_size, traversals_per_chunk);
  context.d_offs_final = thrust::device_vector<uint64_t>(
      context.schedule.expanded_offsets.begin(),
      context.schedule.expanded_offsets.end());

  return context;
}

void decode_rolling_path_chunk_to_device(
    const thrust::device_vector<int32_t> &d_encoded_path,
    RollingPathDecodeContext &context, size_t chunk_index) {
  const auto &chunk = context.schedule.chunks.at(chunk_index);
  if (context.d_rule_sizes.empty()) {
    context.d_chunk_workspace.resize(chunk.expanded_count());
    thrust::copy(
        d_encoded_path.begin() + static_cast<std::ptrdiff_t>(chunk.encoded_begin),
        d_encoded_path.begin() + static_cast<std::ptrdiff_t>(chunk.encoded_end),
        context.d_chunk_workspace.begin());
    return;
  }
  gpu_codec::expand_and_inverse_decode_chunk_device(
      d_encoded_path, context.d_output_offsets, context.d_expanded_rules,
      context.d_rule_offsets, context.d_rule_sizes, context.d_chunk_workspace,
      context.d_offs_final, chunk.encoded_begin, chunk.encoded_end,
      chunk.expanded_begin, chunk.expanded_end, chunk.segment_begin,
      chunk.segment_end, context.min_rule_id, context.max_rule_id,
      static_cast<size_t>(context.schedule.output_size));
}

void prepare_rolling_path_host_buffer(const RollingPathDecodeContext &context,
                                      size_t chunk_index,
                                      RollingPathHostBuffer &host_buffer) {
  populate_rolling_path_host_buffer_metadata(
      context, chunk_index, host_buffer.segment_begin, host_buffer.segment_end,
      host_buffer.expanded_begin, host_buffer.expanded_end,
      host_buffer.node_count, host_buffer.lengths);

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
    const RollingPathDecodeContext &context, size_t chunk_index,
    RollingPathHostBuffer &host_buffer) {
  prepare_rolling_path_host_buffer(context, chunk_index, host_buffer);
  thrust::copy(context.d_chunk_workspace.begin(),
               context.d_chunk_workspace.begin() +
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
    const RollingPathDecodeContext &context, size_t chunk_index,
    RollingPathPinnedHostBuffer &host_buffer, cudaStream_t copy_stream) {
  size_t node_count = 0;
  uint32_t segment_begin = 0;
  uint32_t segment_end = 0;
  int64_t expanded_begin = 0;
  int64_t expanded_end = 0;
  std::vector<uint32_t> lengths;
  populate_rolling_path_host_buffer_metadata(
      context, chunk_index, segment_begin, segment_end, expanded_begin,
      expanded_end, node_count, lengths);
  ensure_rolling_path_pinned_host_buffer_capacity(host_buffer,
                                                  node_count);
  host_buffer.node_count = node_count;
  host_buffer.segment_begin = segment_begin;
  host_buffer.segment_end = segment_end;
  host_buffer.expanded_begin = expanded_begin;
  host_buffer.expanded_end = expanded_end;
  host_buffer.lengths = std::move(lengths);

  if (host_buffer.ready == nullptr) {
    check_cuda(cudaEventCreateWithFlags(&host_buffer.ready,
                                        cudaEventDisableTiming),
               "cudaEventCreateWithFlags(rolling pinned buffer ready)");
  }

  if (host_buffer.node_count > 0) {
    check_cuda(cudaMemcpyAsync(
                   host_buffer.host_nodes,
                   thrust::raw_pointer_cast(context.d_chunk_workspace.data()),
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
              << "min_rule_id=" << min_rule_id << std::endl;
  }

  RollingPathDecodeContext context = prepare_rolling_path_decode(
      d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules,
      d_lens_final, resolved_traversals_per_chunk);

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
    for (size_t chunk_index = 0; chunk_index < context.schedule.chunks.size();
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

      decode_rolling_path_chunk_to_device(d_encoded_path, context, chunk_index);
      check_cuda(cudaEventRecord(decode_ready.event, 0),
                 "cudaEventRecord(rolling decode ready)");
      check_cuda(cudaStreamWaitEvent(copy_stream.stream, decode_ready.event, 0),
                 "cudaStreamWaitEvent(rolling copy waits for decode)");
      copy_rolling_path_chunk_to_pinned_host_async(
          context, chunk_index, host_buffers[buffer_index], copy_stream.stream);

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
    uint32_t traversals_per_chunk, std::vector<int32_t> &out_data) {
  const uint32_t resolved_traversals_per_chunk =
      std::max<uint32_t>(1, traversals_per_chunk);

  if (decompression_debug_enabled()) {
    std::cout << "[GPU Decompress] Expanding path with rolling chunk "
                 "scheduler ("
              << resolved_traversals_per_chunk
              << " traversals per chunk), min_rule_id=" << min_rule_id
              << std::endl;
  }

  const auto prepare_start = Clock::now();
  RollingPathDecodeContext context = prepare_rolling_path_decode(
      d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules,
      d_lens_final, resolved_traversals_per_chunk);
  const auto prepare_end = Clock::now();
  out_data.resize(static_cast<size_t>(context.schedule.output_size));

  double decode_ms = 0.0;
  double copy_ms = 0.0;
  for (size_t chunk_index = 0; chunk_index < context.schedule.chunks.size();
       ++chunk_index) {
    const auto decode_start = Clock::now();
    decode_rolling_path_chunk_to_device(d_encoded_path, context, chunk_index);
    const auto decode_end = Clock::now();
    RollingPathHostBuffer host_buffer;
    host_buffer.host_nodes =
        out_data.data() + context.schedule.chunks[chunk_index].expanded_begin;
    host_buffer.node_capacity =
        out_data.size() -
        static_cast<size_t>(context.schedule.chunks[chunk_index].expanded_begin);
    const auto copy_start = Clock::now();
    copy_rolling_path_chunk_to_host_buffer(context, chunk_index, host_buffer);
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
              << context.schedule.chunks.size() << ", output nodes: "
              << out_data.size() << std::endl;
  }
}

} // namespace gpu_decompression
