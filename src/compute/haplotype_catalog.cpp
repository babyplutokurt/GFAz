#include "compute/haplotype_catalog.hpp"

#include "compute/traversal_query.hpp"

namespace gfaz {

HaplotypeCatalog::HaplotypeCatalog(const CompressedData &data) {
  const size_t num_paths = data.sequence_lengths.size();
  const size_t num_walks = data.walk_lengths.size();
  haplotypes_.reserve(num_paths + num_walks);

  if (num_paths > 0) {
    std::vector<std::string> names =
        tquery::load_path_names(data, num_paths, "haplotype catalog");
    for (std::string &name : names) {
      Haplotype hap;
      hap.contig = name;
      hap.name = std::move(name);
      haplotypes_.push_back(std::move(hap));
    }
  }

  if (num_walks > 0) {
    tquery::WalkIdentityColumns walks =
        tquery::load_walk_identity(data, num_walks, "haplotype catalog");
    for (size_t i = 0; i < num_walks; ++i) {
      Haplotype hap;
      hap.is_walk = true;
      hap.sample = std::move(walks.samples[i]);
      hap.hap_index = walks.haps[i];
      hap.seq_id = std::move(walks.seqs[i]);
      hap.seq_start = walks.starts[i];
      hap.seq_end = walks.ends[i];
      hap.name = tquery::walk_reference_name(hap.sample, hap.hap_index,
                                             hap.seq_id, hap.seq_start,
                                             hap.seq_end);
      hap.contig = tquery::walk_reference_name(hap.sample, hap.hap_index,
                                               hap.seq_id, -1, -1);
      hap.contig_offset =
          hap.seq_start > 0 ? static_cast<uint64_t>(hap.seq_start) : 0;
      haplotypes_.push_back(std::move(hap));
    }
  }

  for (size_t h = 0; h < haplotypes_.size(); ++h)
    by_contig_[haplotypes_[h].contig].push_back(static_cast<uint32_t>(h));
}

const std::vector<uint32_t> &
HaplotypeCatalog::on_contig(const std::string &contig) const {
  static const std::vector<uint32_t> none;
  const auto it = by_contig_.find(contig);
  return it == by_contig_.end() ? none : it->second;
}

} // namespace gfaz
