#include "core/utils/runtime_utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace gfaz::runtime_utils {

std::string format_size(size_t bytes) {
  std::ostringstream oss;
  if (bytes >= 1024 * 1024)
    oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0))
        << " MB";
  else if (bytes >= 1024)
    oss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
  else
    oss << bytes << " Bytes";
  return oss.str();
}

void advise_huge_pages(void *begin, size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  constexpr size_t kHugePage = size_t{2} << 20;
  constexpr size_t kMinBytes = size_t{8} << 20;
  static const bool disabled = std::getenv("GFAZ_NO_HUGE_PAGES") != nullptr;
  if (disabled || begin == nullptr || bytes < kMinBytes)
    return;
  const uintptr_t first =
      (reinterpret_cast<uintptr_t>(begin) + kHugePage - 1) / kHugePage *
      kHugePage;
  const uintptr_t last =
      (reinterpret_cast<uintptr_t>(begin) + bytes) / kHugePage * kHugePage;
  if (last > first)
    madvise(reinterpret_cast<void *>(first), last - first, MADV_HUGEPAGE);
#else
  (void)begin;
  (void)bytes;
#endif
}

ProcessMemorySnapshot read_process_memory_snapshot() {
  ProcessMemorySnapshot snapshot;

  std::ifstream status("/proc/self/status");
  if (!status)
    return snapshot;

  std::string line;
  while (std::getline(status, line)) {
    auto parse_kb_field = [&](const char *prefix, size_t &out_value) {
      const std::string_view view(line);
      const std::string_view key(prefix);
      if (view.size() < key.size() || view.substr(0, key.size()) != key)
        return false;

      std::istringstream iss(std::string(view.substr(key.size())));
      size_t value = 0;
      std::string unit;
      if (iss >> value >> unit) {
        out_value = value;
        return true;
      }
      return false;
    };

    if (parse_kb_field("VmRSS:", snapshot.vm_rss_kb))
      continue;
    if (parse_kb_field("RssAnon:", snapshot.rss_anon_kb))
      continue;
    parse_kb_field("VmHWM:", snapshot.vm_hwm_kb);
  }

  return snapshot;
}

std::string format_memory_snapshot(const ProcessMemorySnapshot &snapshot) {
  std::ostringstream oss;
  oss << "RssAnon=" << format_size(snapshot.rss_anon_kb * 1024)
      << " | VmRSS=" << format_size(snapshot.vm_rss_kb * 1024)
      << " | VmHWM=" << format_size(snapshot.vm_hwm_kb * 1024);
  return oss.str();
}

} // namespace gfaz::runtime_utils
