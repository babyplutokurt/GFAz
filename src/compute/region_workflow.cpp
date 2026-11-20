#include "compute/region_workflow.hpp"

#include "compute/indexed_paths.hpp"
#include "compute/traversal_query.hpp"
#include "core/codec/codec.hpp"
#include "core/io/gfa_write_utils.hpp"
#include "core/utils/threading_utils.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {

namespace {

IndexedPathsOptions paths_options(const RegionOptions &options) {
  IndexedPathsOptions paths;
  paths.index_path = options.index_path;
  paths.block_steps = options.index_block_steps;
  paths.num_threads = options.num_threads;
  return paths;
}

struct ContigInterval {
  std::string contig;
  uint64_t begin = 0;
  uint64_t end = std::numeric_limits<uint64_t>::max();
};

bool parse_coordinate(const std::string &text, uint64_t &value) {
  if (text.empty() || text.size() > 18)
    return false;
  value = 0;
  for (char c : text) {
    if (!std::isdigit(static_cast<unsigned char>(c)))
      return false;
    value = value * 10 + static_cast<uint64_t>(c - '0');
  }
  return true;
}

// "CONTIG:START-END", or a bare contig name. A name that itself ends in
// ":digits-digits" is taken whole when it is a contig of the container.
ContigInterval parse_region(const std::string &text,
                            const HaplotypeCatalog &catalog) {
  ContigInterval region;
  region.contig = text;
  if (!catalog.on_contig(text).empty())
    return region;

  const size_t colon = text.rfind(':');
  const size_t dash =
      colon == std::string::npos ? std::string::npos : text.find('-', colon);
  uint64_t begin = 0, end = 0;
  if (dash == std::string::npos ||
      !parse_coordinate(text.substr(colon + 1, dash - colon - 1), begin) ||
      !parse_coordinate(text.substr(dash + 1), end))
    throw std::runtime_error("region: no path or walk named '" + text + "'");
  if (begin >= end)
    throw std::runtime_error("region: empty interval in '" + text + "'");

  region.contig = text.substr(0, colon);
  region.begin = begin;
  region.end = end;
  if (catalog.on_contig(region.contig).empty())
    throw std::runtime_error("region: no path or walk named '" +
                             region.contig + "'");
  return region;
}

// A run of steps of one haplotype; the unit both of "this haplotype touches the
// region here" and of the final subpaths.
struct StepRun {
  uint32_t hap = 0;
  uint32_t step_begin = 0;
  uint32_t step_end = 0;
  uint64_t bp_begin = 0;
  uint64_t bp_end = 0;
};

// Grows runs from steps arriving in path order: a step joins the open run if it
// starts within max_gap bases of the run's end, otherwise it starts a new run.
class RunBuilder {
public:
  RunBuilder(uint64_t max_gap, std::vector<StepRun> &out)
      : max_gap_(max_gap), out_(out) {}

  void add(uint32_t hap, uint32_t step_begin, uint32_t step_end,
           uint64_t bp_begin, uint64_t bp_end) {
    if (open_ && run_.hap == hap && bp_begin <= run_.bp_end + max_gap_) {
      run_.step_end = std::max(run_.step_end, step_end);
      run_.bp_end = std::max(run_.bp_end, bp_end);
      return;
    }
    flush();
    run_ = StepRun{hap, step_begin, step_end, bp_begin, bp_end};
    open_ = true;
  }

  void flush() {
    if (open_)
      out_.push_back(run_);
    open_ = false;
  }

private:
  uint64_t max_gap_;
  std::vector<StepRun> &out_;
  StepRun run_;
  bool open_ = false;
};

// Mark the nodes the reference haplotypes visit inside the regions. Returns the
// smallest and largest marked node id (first > second if nothing was marked).
std::pair<uint32_t, uint32_t>
mark_reference_nodes(const IndexedPaths &ctx, const RegionOptions &options,
                     std::vector<uint8_t> &in_region) {
  uint32_t min_node = std::numeric_limits<uint32_t>::max();
  uint32_t max_node = 0;
  for (const std::string &text : options.regions) {
    const ContigInterval region = parse_region(text, ctx.catalog);
    bool overlaps = false;
    for (uint32_t hap : ctx.catalog.on_contig(region.contig)) {
      // Contig coordinates -> coordinates of this haplotype.
      const uint64_t offset = ctx.catalog[hap].contig_offset;
      const uint64_t length = ctx.index.bp_length(hap);
      if (region.end <= offset || region.begin >= offset + length)
        continue;
      const uint64_t begin = region.begin > offset ? region.begin - offset : 0;
      const uint64_t end = std::min(region.end - offset, length);
      ctx.access.for_each_step_in_bases(hap, begin, end,
                                        [&](const PathStep &step) {
        const uint32_t node = tquery::abs_node_id(step.node);
        if (node == 0 || node >= in_region.size())
          return;
        in_region[node] = 1;
        min_node = std::min(min_node, node);
        max_node = std::max(max_node, node);
        overlaps = true;
      });
    }
    if (!overlaps)
      throw std::runtime_error("region: '" + text +
                               "' does not overlap its path or walk");
  }
  return {min_node, max_node};
}

// Every stretch of every haplotype that visits a marked node, joined across
// gaps of at most max_gap_bp bases.
std::vector<StepRun> find_runs(const IndexedPaths &ctx,
                               const RegionOptions &options,
                               const std::vector<uint8_t> &in_region,
                               uint32_t min_node, uint32_t max_node) {
  // Only blocks whose node range reaches the marked nodes can visit one.
  std::vector<std::pair<uint32_t, uint32_t>> candidates;
  ctx.index.for_each_block_overlapping(
      min_node, max_node, [&](size_t hap, size_t block) {
        candidates.emplace_back(static_cast<uint32_t>(hap),
                                static_cast<uint32_t>(block));
      });
  std::sort(candidates.begin(), candidates.end());

  ScopedOMPThreads omp_scope(options.num_threads);
  const int num_threads = std::max(1, omp_scope.effective_threads());
  std::vector<std::vector<StepRun>> per_thread(num_threads);
#pragma omp parallel num_threads(num_threads)
  {
#ifdef _OPENMP
    std::vector<StepRun> &hits = per_thread[omp_get_thread_num()];
#else
    std::vector<StepRun> &hits = per_thread[0];
#endif
#pragma omp for schedule(dynamic, 16)
    for (long long i = 0; i < static_cast<long long>(candidates.size()); ++i) {
      const uint32_t hap = candidates[i].first;
      const uint32_t block = candidates[i].second;
      RunBuilder runs(options.max_gap_bp, hits);
      ctx.access.for_each_step_in_blocks(
          hap, block, block + 1, [&](const PathStep &step) {
            const uint32_t node = tquery::abs_node_id(step.node);
            if (node < in_region.size() && in_region[node])
              runs.add(hap, step.step, step.step + 1, step.bp_begin,
                       step.bp_end());
          });
      runs.flush();
    }
  }

  // Join the per-block runs across block boundaries. Sorting first makes the
  // result independent of how the blocks were scheduled.
  std::vector<StepRun> hits;
  for (std::vector<StepRun> &part : per_thread) {
    hits.insert(hits.end(), part.begin(), part.end());
    std::vector<StepRun>().swap(part);
  }
  std::sort(hits.begin(), hits.end(), [](const StepRun &a, const StepRun &b) {
    return a.hap != b.hap ? a.hap < b.hap : a.step_begin < b.step_begin;
  });
  std::vector<StepRun> merged;
  RunBuilder runs(options.max_gap_bp, merged);
  for (const StepRun &hit : hits)
    runs.add(hit.hap, hit.step_begin, hit.step_end, hit.bp_begin, hit.bp_end);
  runs.flush();
  return merged;
}

RegionSubgraph select_regions(const IndexedPaths &ctx,
                              const RegionOptions &options) {
  if (options.regions.empty())
    throw std::runtime_error("region: no region given");

  RegionSubgraph result;
  result.node_selected.assign(ctx.segment_lengths.size() + 1, 0);

  std::vector<uint8_t> in_region(ctx.segment_lengths.size() + 1, 0);
  const auto [min_node, max_node] =
      mark_reference_nodes(ctx, options, in_region);
  const std::vector<StepRun> runs =
      find_runs(ctx, options, in_region, min_node, max_node);

  // Decode each run in full: the steps between two visits to the region are
  // part of the subgraph too.
  result.subpaths.resize(runs.size());
  uint8_t *selected = result.node_selected.data();
  const size_t num_selectable = result.node_selected.size();
  ScopedOMPThreads omp_scope(options.num_threads);
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < static_cast<long long>(runs.size()); ++i) {
    const StepRun &run = runs[i];
    Subpath &subpath = result.subpaths[i];
    subpath.haplotype = run.hap;
    subpath.step_begin = run.step_begin;
    subpath.step_end = run.step_end;
    subpath.bp_begin = run.bp_begin;
    subpath.bp_end = run.bp_end;
    subpath.nodes.reserve(run.step_end - run.step_begin);
    ctx.access.for_each_step(run.hap, run.step_begin, run.step_end,
                             [&](const PathStep &step) {
      subpath.nodes.push_back(step.node);
      const uint32_t node = tquery::abs_node_id(step.node);
      if (node < num_selectable) {
#pragma omp atomic write
        selected[node] = 1;
      }
    });
  }
  return result;
}

void write_segments(const CompressedData &data,
                    const std::vector<uint32_t> &segment_lengths,
                    const std::vector<uint8_t> &node_selected,
                    const SegmentNames &names, std::ostream &out) {
  namespace wu = gfa_write_utils;
  const std::string sequences =
      Codec::zstd_decompress_string(data.segment_sequences_zstd);
  std::vector<OptionalFieldColumn> fields;
  fields.reserve(data.segment_optional_fields_zstd.size());
  for (const CompressedOptionalFieldColumn &column :
       data.segment_optional_fields_zstd)
    fields.push_back(wu::decompress_optional_column(column));
  const wu::FieldOffsets field_offsets = wu::build_field_offsets(fields);

  std::string line;
  size_t offset = 0;
  for (size_t i = 0; i < segment_lengths.size(); ++i) {
    const uint32_t length = segment_lengths[i];
    if (offset + length > sequences.size())
      throw std::runtime_error("region: segment sequence column is truncated");
    if (node_selected[i + 1]) {
      line.clear();
      wu::append_segment_line(line, static_cast<uint32_t>(i + 1),
                              sequences.data() + offset, length, fields,
                              field_offsets, names);
      out.write(line.data(), static_cast<std::streamsize>(line.size()));
    }
    offset += length;
  }
}

void write_links(const CompressedData &data,
                 const std::vector<uint8_t> &node_selected,
                 const SegmentNames &names, std::ostream &out) {
  namespace wu = gfa_write_utils;
  const LinkData links = wu::decompress_links(data);
  std::vector<OptionalFieldColumn> fields;
  fields.reserve(data.link_optional_fields_zstd.size());
  for (const CompressedOptionalFieldColumn &column :
       data.link_optional_fields_zstd)
    fields.push_back(wu::decompress_optional_column(column));
  const wu::FieldOffsets field_offsets = wu::build_field_offsets(fields);

  auto selected = [&](uint32_t node) {
    return node < node_selected.size() && node_selected[node];
  };
  std::string line;
  for (size_t i = 0; i < links.from_ids.size(); ++i) {
    if (!selected(links.from_ids[i]) || !selected(links.to_ids[i]))
      continue;
    line.clear();
    wu::append_link_line(line, links, i, fields, field_offsets, names);
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

void write_subpaths(const IndexedPaths &ctx, const RegionSubgraph &subgraph,
                    std::ostream &out) {
  namespace wu = gfa_write_utils;
  for (const Subpath &subpath : subgraph.subpaths) {
    const Haplotype &hap = ctx.catalog[subpath.haplotype];
    const bool whole = subpath.bp_begin == 0 &&
                       subpath.bp_end == ctx.index.bp_length(subpath.haplotype);
    std::string line;
    if (hap.is_walk) {
      // A W-line carries its own coordinates; shift them to the stretch kept.
      // Unknown coordinates ('*') stay unknown when the walk is kept whole.
      const bool keep = whole && (hap.seq_start < 0 || hap.seq_end < 0);
      const int64_t start =
          keep ? hap.seq_start
               : static_cast<int64_t>(hap.contig_offset + subpath.bp_begin);
      const int64_t end =
          keep ? hap.seq_end
               : static_cast<int64_t>(hap.contig_offset + subpath.bp_end);
      line = wu::format_walk_line(hap.sample, hap.hap_index, hap.seq_id, start,
                                  end, subpath.nodes.data(),
                                  subpath.nodes.size(), ctx.names);
    } else {
      // A partial P-line is named NAME:START-END, the PanSN subrange form.
      std::string name = hap.name;
      if (!whole)
        name += ":" + std::to_string(subpath.bp_begin) + "-" +
                std::to_string(subpath.bp_end);
      line = wu::format_path_line(name, subpath.nodes.data(),
                                  subpath.nodes.size(), "*", ctx.names);
    }
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

} // namespace

RegionSubgraph select_regions(const CompressedData &data,
                              const RegionOptions &options) {
  const IndexedPaths ctx(data, paths_options(options));
  return select_regions(ctx, options);
}

void extract_regions_to_gfa(const CompressedData &data,
                            const RegionOptions &options, std::ostream &out) {
  const IndexedPaths ctx(data, paths_options(options));
  const RegionSubgraph subgraph = select_regions(ctx, options);

  if (!data.header_line.empty())
    out << data.header_line << '\n';
  write_segments(data, ctx.segment_lengths, subgraph.node_selected, ctx.names,
                 out);
  write_links(data, subgraph.node_selected, ctx.names, out);
  write_subpaths(ctx, subgraph, out);
}

} // namespace gfaz
