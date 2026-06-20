#include "compute/stats_workflow.hpp"

#include "core/codec/codec.hpp"

#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace gfaz {

void graph_stats_to_tsv(const CompressedData &data, const StatsOptions &options,
                        std::ostream &out) {
  if (options.base_content) {
    // A/C/G/T tally over the concatenated segment sequences (uppercased, as odgi
    // does). Only the four canonical bases are reported, matching `odgi stats -b`.
    const std::string concat =
        Codec::zstd_decompress_string(data.segment_sequences_zstd);
    uint64_t counts[4] = {0, 0, 0, 0}; // A, C, G, T
    for (char ch : concat) {
      switch (std::toupper(static_cast<unsigned char>(ch))) {
      case 'A': ++counts[0]; break;
      case 'C': ++counts[1]; break;
      case 'G': ++counts[2]; break;
      case 'T': ++counts[3]; break;
      default: break;
      }
    }
    out << "A\t" << counts[0] << '\n'
        << "C\t" << counts[1] << '\n'
        << "G\t" << counts[2] << '\n'
        << "T\t" << counts[3] << '\n';
    return;
  }

  const std::vector<uint32_t> segment_lengths =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  uint64_t total_length = 0;
  for (uint32_t len : segment_lengths)
    total_length += len;

  const uint64_t num_nodes = segment_lengths.size();
  const uint64_t num_edges = data.num_links;
  // odgi treats W-lines as paths, so both P- and W-lines count here.
  const uint64_t num_paths =
      data.sequence_lengths.size() + data.walk_lengths.size();

  // steps = total node visits = pre-grammar (original) traversal lengths summed.
  uint64_t num_steps = 0;
  for (uint32_t s : data.original_path_lengths)
    num_steps += s;
  for (uint32_t s : data.original_walk_lengths)
    num_steps += s;

  out << "#length\tnodes\tedges\tpaths\tsteps\n";
  out << total_length << '\t' << num_nodes << '\t' << num_edges << '\t'
      << num_paths << '\t' << num_steps << '\n';
}

} // namespace gfaz
