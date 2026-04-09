#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace gfz::runtime_utils {

struct ProcessMemorySnapshot {
  size_t vm_rss_kb = 0;
  size_t vm_hwm_kb = 0;
  size_t rss_anon_kb = 0;
};

template <typename ClockT>
double elapsed_ms(const std::chrono::time_point<ClockT> &start,
                  const std::chrono::time_point<ClockT> &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double gbps_from_mb(double size_mb, double time_ms);

std::string format_size(size_t bytes);

ProcessMemorySnapshot read_process_memory_snapshot();

std::string format_memory_snapshot(const ProcessMemorySnapshot &snapshot);

} // namespace gfz::runtime_utils
