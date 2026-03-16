#pragma once
#include <cstdlib>
#include <iostream>

// Singleton-style debug flag, checked once from GFA_COMPRESSION_DEBUG env var.
// Usage:  if (gfaz_debug_enabled()) { std::cerr << "info" << std::endl; }
//     or: GFAZ_LOG("message" << value << " more");
inline bool gfaz_debug_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("GFA_COMPRESSION_DEBUG");
    return env && (std::string(env) == "1" || std::string(env) == "true");
  }();
  return enabled;
}

// Convenience macro: GFAZ_LOG(expr)
// Outputs to std::cerr only when GFA_COMPRESSION_DEBUG=1 or true.
#define GFAZ_LOG(expr)                                                         \
  do {                                                                         \
    if (gfaz_debug_enabled()) {                                                \
      std::cerr << expr << std::endl;                                          \
    }                                                                          \
  } while (0)
