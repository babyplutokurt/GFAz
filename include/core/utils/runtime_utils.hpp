#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace gfaz::runtime_utils {

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

std::string format_size(size_t bytes);

ProcessMemorySnapshot read_process_memory_snapshot();

std::string format_memory_snapshot(const ProcessMemorySnapshot &snapshot);

// Asks the kernel to back a large, randomly accessed buffer with huge pages
// where transparent huge pages are available on request. Only a hint: it does
// nothing for small buffers, on other platforms, or when the kernel declines.
// GFAZ_NO_HUGE_PAGES=1 turns it off.
void advise_huge_pages(void *begin, size_t bytes);

} // namespace gfaz::runtime_utils
