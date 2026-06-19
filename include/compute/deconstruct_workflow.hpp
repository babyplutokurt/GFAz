#ifndef DECONSTRUCT_WORKFLOW_HPP
#define DECONSTRUCT_WORKFLOW_HPP

#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"
#include "compute/grouping_mode.hpp"

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace gfaz {

struct DeconstructOptions {
  // Reference path/walk names to deconstruct against (each becomes a VCF
  // CHROM). Required unless reference_prefixes is given.
  std::vector<std::string> reference_names;
  // Reference path/walk name *prefixes* (vg `-P` parity): every stored
  // path/walk name beginning with one of these is used as a reference contig.
  // E.g. "CHM13" selects all CHM13#0#chr* chromosomes. Combined with
  // reference_names; at least one of the two must be non-empty.
  std::vector<std::string> reference_prefixes;
  // How non-reference traversals collapse into VCF sample columns.
  GroupingMode grouping = GroupingMode::Sample;
  int num_threads = kDefaultNumThreads;
  // Emit per-sample GT columns (false -> site + INFO only).
  bool emit_gt = true;
  // Emit graph-level annotations for closer `vg deconstruct` parity (off by
  // default to keep the output lean): the AT (allele traversal) INFO field, the
  // snarl boundary id in the ID column, and the full PanSN contig name
  // (sample#hap#seq) in CHROM instead of the bare sequence name. Node ids in AT
  // / ID are gfaz's own 1-based numbering (the .gfaz node space; original GFA
  // segment names are not retained), so they are not byte-identical to vg's.
  // Only affects the snarl writer for AT/ID; CHROM naming applies to both.
  bool emit_at = false;
  // Sites whose reference span exceeds this (bp) are emitted as a single
  // <CPX>-flagged record instead of enumerating every allele. 0 disables.
  uint64_t max_site_length = 0;
  // Use topology-based snarl (superbubble) enumeration built from the stored
  // L-line links, instead of the linear reference-unique-anchor heuristic.
  // Site boundaries then come from graph structure rather than the reference
  // projection, and alleles are observed by streaming each path once. On by
  // default; the linear heuristic is a legacy mode (set false via --linear).
  bool use_snarls = true;
  // Emit one record per *top-level* snarl, matching `vg deconstruct`'s default
  // granularity. Sites come from the global biconnected decomposition of the
  // node-end graph (see snarl_finder.cpp): each non-trivial biconnected block
  // the reference threads through is one snarl, and blocks with a
  // cyclic/ambiguous reference traversal are dropped (as vg does). This
  // collapses the leaf-bubble chains and nested bubbles the leaf-superbubble
  // superset reports separately, and suppresses the tangled satellite/palindrome
  // regions vg also skips. This is the default, since producing output identical
  // to `vg deconstruct` is the goal. Set false (via the legacy --snarl flag) to
  // get the leaf-superbubble superset instead.
  bool vg_compat = true;
};

// Derive a VCF (header + records) from the compressed traversals and write it
// to `out`. Operates directly on the .gfaz container without materializing the
// original GFA. See docs/workflows/DECONSTRUCT_WORKFLOW.md for the algorithm.
void deconstruct_to_vcf(const CompressedData &data,
                        const DeconstructOptions &options, std::ostream &out);

} // namespace gfaz

#endif // DECONSTRUCT_WORKFLOW_HPP
