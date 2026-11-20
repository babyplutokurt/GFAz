#ifndef COMPUTE_PACKED_UINTS_HPP
#define COMPUTE_PACKED_UINTS_HPP

#include <cstdint>
#include <cstring>
#include <vector>

namespace gfaz {

// A fixed-length array of unsigned values kept in 1, 2 or 4 bytes each: the
// narrowest width that holds max_value. Different elements may be written
// from different threads.
class PackedUints {
public:
  PackedUints(uint64_t size, uint32_t max_value)
      : width_(max_value <= 0xFFu ? 1 : max_value <= 0xFFFFu ? 2 : 4),
        bytes_(size * width_) {}

  unsigned width() const { return width_; }

  uint32_t operator[](uint64_t i) const {
    const uint8_t *p = bytes_.data() + i * width_;
    if (width_ == 1)
      return *p;
    if (width_ == 2) {
      uint16_t v;
      std::memcpy(&v, p, sizeof(v));
      return v;
    }
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
  }

  void set(uint64_t i, uint32_t value) {
    uint8_t *p = bytes_.data() + i * width_;
    if (width_ == 1) {
      *p = static_cast<uint8_t>(value);
    } else if (width_ == 2) {
      const uint16_t v = static_cast<uint16_t>(value);
      std::memcpy(p, &v, sizeof(v));
    } else {
      std::memcpy(p, &value, sizeof(value));
    }
  }

private:
  unsigned width_;
  std::vector<uint8_t> bytes_;
};

} // namespace gfaz

#endif // COMPUTE_PACKED_UINTS_HPP
