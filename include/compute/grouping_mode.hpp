#ifndef GFAZ_COMPUTE_GROUPING_MODE_HPP
#define GFAZ_COMPUTE_GROUPING_MODE_HPP

namespace gfaz {

// How to map P/W-lines to "haplotype" identity before coverage counting.
// Shared across the compute engine (growth, pav, deconstruct, and future
// path-iterative workflows) so none of them has to depend on another
// workflow's header just to name a grouping.
//   PerPathWalk:  each P-line and each W-line is its own haplotype (GFAz
//                 default; simple, but inflates N when one haplotype is split
//                 into multiple walks/paths).
//   SampleHapSeq: group by (sample, hap, seqid) after stripping PanSN
//                 ":start-end". Matches Panacus default (id() + clear_coords).
//   SampleHap:    group by (sample, hap). Matches Panacus --groupby-haplotype
//                 (-H). Typical HPRC "per-haplotype" growth curve.
//   Sample:       group by sample. Matches Panacus --groupby-sample (-S).
enum class GroupingMode {
  PerPathWalk,
  SampleHapSeq,
  SampleHap,
  Sample,
};

} // namespace gfaz

#endif // GFAZ_COMPUTE_GROUPING_MODE_HPP
