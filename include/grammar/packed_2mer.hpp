#ifndef PACKED_2MER_HPP
#define PACKED_2MER_HPP

#include <cstdint>
#include <algorithm>


// Packed2mer: Two int32_t NodeIds packed into a single int64_t
// Allows faster hashing (native int64 hash), better cache locality,
// and simpler sorting for rule reordering.
using Packed2mer = int64_t;

// Pack two NodeIds into a Packed2mer
// High 32 bits: first node, Low 32 bits: second node
inline Packed2mer pack_2mer(int32_t first, int32_t second) {
    return (static_cast<int64_t>(first) << 32) | static_cast<uint32_t>(second);
}

// Unpack the first gfaz::NodeId (high 32 bits)
inline int32_t unpack_first(Packed2mer packed) {
    return static_cast<int32_t>(packed >> 32);
}

// Unpack the second gfaz::NodeId (low 32 bits)
inline int32_t unpack_second(Packed2mer packed) {
    return static_cast<int32_t>(packed);  // Truncates to low 32 bits
}

// Unpack both NodeIds at once
inline void unpack_2mer(Packed2mer packed, int32_t& first, int32_t& second) {
    first = unpack_first(packed);
    second = unpack_second(packed);
}

// Reverse a 2-mer: (-second, -first)
// GFA convention: reversing path orientation negates all node IDs and reverses order
inline Packed2mer reverse_2mer(Packed2mer packed) {
    int32_t first = unpack_first(packed);
    int32_t second = unpack_second(packed);
    return pack_2mer(-second, -first);
}

// Get the canonical form of a 2-mer (smaller of forward and reverse)
// Uses int64 comparison for consistency
inline Packed2mer canonical_2mer(Packed2mer packed) {
    Packed2mer rev = reverse_2mer(packed);
    return std::min(packed, rev);
}

#endif // PACKED_2MER_HPP

