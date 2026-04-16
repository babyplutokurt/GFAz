#ifndef THREADING_UTILS_HPP
#define THREADING_UTILS_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>

#endif

inline int resolve_omp_thread_count(int num_threads) {
#ifdef _OPENMP
  if (num_threads > 0) {
    return num_threads;
  }
  if (num_threads < 0) {
    return std::max(1, omp_get_max_threads());
  }

  // User override for this project.
  if (const char *env = std::getenv("GFAZ_NUM_THREADS")) {
    char *end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && parsed > 0) {
      return static_cast<int>(parsed);
    }
  }

  // Respect explicit OpenMP runtime override.
  if (const char *env = std::getenv("OMP_NUM_THREADS")) {
    char *end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && parsed > 0) {
      return static_cast<int>(parsed);
    }
  }

  // Conservative auto policy to avoid oversubscription on large HPC nodes:
  // half of logical CPUs, capped at 8 threads.
  const int logical_cpus = omp_get_num_procs();
  const int half = std::max(1, logical_cpus / 2);
  return std::min(half, 8);
#else
  (void)num_threads;
  return 1;
#endif
}

inline bool gfaz_threading_debug_enabled() {
  if (const char *env = std::getenv("GFAZ_THREADING_DEBUG")) {
    return env[0] != '\0' && env[0] != '0';
  }
  return false;
}

// RAII wrapper to set/restore OpenMP thread count.
class ScopedOMPThreads {
public:
  explicit ScopedOMPThreads(int num_threads) : set_(false), resolved_threads_(1) {
#ifdef _OPENMP
    original_threads_ = omp_get_max_threads();
    const char *mode = "auto";
    if (num_threads < 0) {
      resolved_threads_ = std::max(1, original_threads_);
      mode = "inherit";
    } else {
      resolved_threads_ = resolve_omp_thread_count(num_threads);
      mode = num_threads > 0 ? "explicit" : "auto";
      if (resolved_threads_ > 0 && original_threads_ != resolved_threads_) {
        omp_set_num_threads(resolved_threads_);
        set_ = true;
      }
    }

    if (gfaz_threading_debug_enabled()) {
      std::cerr << "[GFAZ Threading] requested=" << num_threads
                << " mode=" << mode
                << " original_omp_max=" << original_threads_
                << " effective_threads=" << resolved_threads_
                << " changed_runtime=" << (set_ ? "yes" : "no") << std::endl;
    }
#else
    if (gfaz_threading_debug_enabled()) {
      std::cerr << "[GFAZ Threading] requested=" << num_threads
                << " mode=serial"
                << " effective_threads=1"
                << " changed_runtime=no" << std::endl;
    }
#endif
    (void)num_threads;
  }

  ~ScopedOMPThreads() {
#ifdef _OPENMP
    if (set_)
      omp_set_num_threads(original_threads_);
#endif
  }

  int effective_threads() const { return resolved_threads_; }

private:
  int original_threads_ = 0;
  bool set_;
  int resolved_threads_;
};

#endif
