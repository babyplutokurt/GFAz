#ifndef COMPUTE_HAPLOTYPE_CATALOG_HPP
#define COMPUTE_HAPLOTYPE_CATALOG_HPP

// Names and coordinates of every haplotype (P-line or W-line) in a container,
// numbered [paths... walks...] like TraversalSet.

#include "core/model/compressed_data.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gfaz {

struct Haplotype {
  // Unique display name: the P-line name, or sample#hap#seq[:start-end] for a
  // W-line (tquery::walk_reference_name).
  std::string name;
  // The sequence the haplotype lies on, without coordinates: the P-line name,
  // or sample#hap#seq. Several W-lines can share a contig.
  std::string contig;
  // Contig coordinate of the haplotype's first base (the W-line start, or 0).
  uint64_t contig_offset = 0;

  bool is_walk = false;
  // W-line fields; unused for a P-line.
  std::string sample;
  uint32_t hap_index = 0;
  std::string seq_id;
  int64_t seq_start = -1;
  int64_t seq_end = -1;
};

class HaplotypeCatalog {
public:
  explicit HaplotypeCatalog(const CompressedData &data);

  size_t size() const { return haplotypes_.size(); }
  const Haplotype &operator[](size_t hap) const { return haplotypes_[hap]; }

  // Haplotypes lying on a contig, in container order; empty if there is none.
  const std::vector<uint32_t> &on_contig(const std::string &contig) const;

private:
  std::vector<Haplotype> haplotypes_;
  std::unordered_map<std::string, std::vector<uint32_t>> by_contig_;
};

} // namespace gfaz

#endif // COMPUTE_HAPLOTYPE_CATALOG_HPP
