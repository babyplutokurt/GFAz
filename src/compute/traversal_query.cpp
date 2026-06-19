#include "compute/traversal_query.hpp"

#include "core/codec/codec.hpp"

#include <cctype>
#include <stdexcept>

namespace gfaz {
namespace tquery {

namespace {

bool build_rule_recursive(uint32_t rule_id, RuleLeafCache &cache,
                          const std::vector<int32_t> &first,
                          const std::vector<int32_t> &second) {
  const uint32_t idx = rule_id - cache.min_rule_id;
  if (cache.ready[idx])
    return true;

  const int32_t a = first[idx];
  const int32_t b = second[idx];

  std::vector<int32_t> leaves;

  auto append_child = [&](int32_t child) -> bool {
    const uint32_t abs_c = static_cast<uint32_t>(std::abs(child));
    if (abs_c >= cache.min_rule_id && abs_c < cache.max_rule_id) {
      if (!build_rule_recursive(abs_c, cache, first, second))
        return false;
      const uint32_t cidx = abs_c - cache.min_rule_id;
      const std::vector<int32_t> &cl = cache.forward[cidx];
      const size_t projected = leaves.size() + cl.size();
      if (projected * sizeof(int32_t) > cache.budget_bytes)
        return false;
      if (child >= 0) {
        leaves.insert(leaves.end(), cl.begin(), cl.end());
      } else {
        leaves.reserve(projected);
        for (auto it = cl.rbegin(); it != cl.rend(); ++it)
          leaves.push_back(-*it);
      }
    } else {
      leaves.push_back(child);
    }
    return true;
  };

  if (!append_child(a))
    return false;
  if (!append_child(b))
    return false;

  const size_t needed = leaves.size() * sizeof(int32_t);
  if (cache.bytes_used + needed > cache.budget_bytes)
    return false;

  cache.bytes_used += needed;
  cache.forward[idx] = std::move(leaves);
  cache.ready[idx] = 1;
  return true;
}

// Resolve a rule-cache byte budget from an environment variable, falling back
// to default_bytes when unset/invalid.
size_t resolve_rule_cache_budget(const char *env_name, size_t default_bytes) {
  if (const char *env = std::getenv(env_name)) {
    char *end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (end != env && *end == '\0' && parsed >= 0)
      return static_cast<size_t>(parsed);
  }
  return default_bytes;
}

// Populate cache.forward/ready for every rule that fits within the budget.
void build_rule_cache(RuleLeafCache &cache, const std::vector<int32_t> &first,
                      const std::vector<int32_t> &second) {
  if (cache.budget_bytes == 0)
    return;
  for (uint32_t rid = cache.min_rule_id; rid < cache.max_rule_id; ++rid) {
    build_rule_recursive(rid, cache, first, second);
  }
}

} // namespace

RuleLeafCache make_rule_cache(uint32_t min_rule_id,
                              const std::vector<int32_t> &rules_first,
                              const std::vector<int32_t> &rules_second,
                              const char *budget_env, size_t default_budget) {
  RuleLeafCache cache;
  cache.min_rule_id = min_rule_id;
  cache.max_rule_id =
      min_rule_id + static_cast<uint32_t>(rules_first.size());
  cache.budget_bytes = resolve_rule_cache_budget(budget_env, default_budget);
  cache.forward.assign(rules_first.size(), {});
  cache.ready.assign(rules_first.size(), 0);
  build_rule_cache(cache, rules_first, rules_second);
  return cache;
}

Rulebook load_rulebook(const CompressedData &data) {
  Rulebook rb;
  rb.rules_first = Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  rb.rules_second = Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
  Codec::delta_decode_int32(rb.rules_first);
  Codec::delta_decode_int32(rb.rules_second);
  rb.min_rule_id = data.min_rule_id();
  rb.max_rule_id =
      rb.min_rule_id + static_cast<uint32_t>(rb.rules_first.size());
  return rb;
}

LoadedTraversals load_traversals(const CompressedData &data) {
  LoadedTraversals t;
  t.paths_flat = Codec::zstd_decompress_int32_vector(data.paths_zstd);
  t.walks_flat = Codec::zstd_decompress_int32_vector(data.walks_zstd);
  t.slices.reserve(data.sequence_lengths.size() + data.walk_lengths.size());
  build_slices(t.paths_flat, data.sequence_lengths, data.original_path_lengths,
               t.slices);
  build_slices(t.walks_flat, data.walk_lengths, data.original_walk_lengths,
               t.slices);
  return t;
}

void build_slices(const std::vector<int32_t> &flat,
                  const std::vector<uint32_t> &lengths,
                  const std::vector<uint32_t> &original_lengths,
                  std::vector<HapSlice> &out) {
  size_t offset = 0;
  for (size_t i = 0; i < lengths.size(); ++i) {
    const uint32_t enc_len = lengths[i];
    const uint32_t orig_len =
        (i < original_lengths.size()) ? original_lengths[i] : enc_len;
    if (offset + enc_len > flat.size())
      throw std::runtime_error("traversal_query: encoded traversal block is "
                               "truncated");
    out.push_back(HapSlice{flat.data() + offset, enc_len, orig_len});
    offset += enc_len;
  }
}

void decode_one_haplotype_general(const int32_t *encoded, size_t encoded_len,
                                  uint32_t original_len, int delta_round,
                                  uint32_t min_rule_id, uint32_t max_rule_id,
                                  const std::vector<int32_t> &rules_first,
                                  const std::vector<int32_t> &rules_second,
                                  const RuleLeafCache &cache,
                                  std::vector<NodeId> &decoded) {
  decoded.clear();
  decoded.reserve(original_len);
  auto push = [&](int32_t v) { decoded.push_back(v); };
  stream_hap_leaves(encoded, encoded_len, min_rule_id, max_rule_id, rules_first,
                    rules_second, cache, push);
  for (int r = 0; r < delta_round; ++r) {
    for (size_t i = 1; i < decoded.size(); ++i)
      decoded[i] = decoded[i] + decoded[i - 1];
  }
}

namespace {

void strip_pansn_coords_inplace(std::string &s) {
  const size_t colon = s.rfind(':');
  if (colon == std::string::npos || colon == 0)
    return;
  const size_t dash = s.find('-', colon + 1);
  if (dash == std::string::npos || colon + 1 == dash || dash + 1 == s.size())
    return;
  for (size_t i = colon + 1; i < dash; ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return;
  for (size_t i = dash + 1; i < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return;
  s.erase(colon);
}

} // namespace

// PanSN parse mirroring Panacus's from_str + clear_coords: regex
// ^([^#]+)(#[^#]+)?(#[^#].*)?$, with ":start-end" stripped from the last
// populated field. `has_hap`/`has_seq` mirror PathSegment's Option fields, so a
// malformed name (e.g. "sample#", "sample##seq") falls back to fewer fields.
PansnParts parse_pansn_path_name(const std::string &name) {
  PansnParts p;
  const size_t h1 = name.find('#');
  if (h1 == std::string::npos) {
    p.sample = name;
    strip_pansn_coords_inplace(p.sample);
    return p;
  }
  p.sample = name.substr(0, h1);
  const size_t h2 = name.find('#', h1 + 1);
  auto take_two_field = [&](const std::string &hap_raw) {
    if (hap_raw.empty())
      return; // Invalid PanSN ("sample#"): fall back to single-field key.
    p.hap = hap_raw;
    p.has_hap = true;
    strip_pansn_coords_inplace(p.hap);
  };
  if (h2 == std::string::npos) {
    take_two_field(name.substr(h1 + 1));
    return p;
  }
  if (h2 == h1 + 1) {
    // "sample##...": doesn't match PanSN at all; keep sample only.
    return p;
  }
  const std::string hap_raw = name.substr(h1 + 1, h2 - h1 - 1);
  std::string seq_raw = name.substr(h2 + 1);
  if (seq_raw.empty() || seq_raw[0] == '#') {
    // Group-3 regex fails; Panacus backtracks to the 2-field match.
    take_two_field(hap_raw);
    return p;
  }
  p.hap = hap_raw;
  p.has_hap = true;
  p.seq = std::move(seq_raw);
  p.has_seq = true;
  strip_pansn_coords_inplace(p.seq);
  return p;
}

std::string path_group_key(const std::string &name, GroupingMode mode) {
  if (mode == GroupingMode::PerPathWalk)
    return name;
  const PansnParts p = parse_pansn_path_name(name);
  switch (mode) {
  case GroupingMode::Sample:
    return p.sample;
  case GroupingMode::SampleHap:
    // Panacus: format!("{}#{}", sample, hap.unwrap_or("")).
    return p.sample + "#" + (p.has_hap ? p.hap : std::string());
  case GroupingMode::SampleHapSeq:
  default:
    if (p.has_hap)
      return p.has_seq ? (p.sample + "#" + p.hap + "#" + p.seq)
                       : (p.sample + "#" + p.hap);
    if (p.has_seq)
      return p.sample + "#*#" + p.seq;
    return p.sample;
  }
}

std::string walk_group_key(const std::string &sample, uint32_t hap,
                           const std::string &seq, GroupingMode mode) {
  switch (mode) {
  case GroupingMode::Sample:
    return sample;
  case GroupingMode::SampleHap:
    return sample + "#" + std::to_string(hap);
  case GroupingMode::SampleHapSeq:
    return sample + "#" + std::to_string(hap) + "#" + seq;
  case GroupingMode::PerPathWalk:
  default:
    return sample + "#" + std::to_string(hap) + "#" + seq;
  }
}

std::string walk_reference_name(const std::string &sample, uint32_t hap,
                                const std::string &seq, int64_t start,
                                int64_t end) {
  std::string name = sample + "#" + std::to_string(hap) + "#" + seq;
  if (start != -1 || end != -1)
    name += ":" + std::to_string(start) + "-" + std::to_string(end);
  return name;
}

void reconstruct_strings(const std::string &concat,
                         const std::vector<uint32_t> &lengths,
                         std::vector<std::string> &out, const char *label) {
  out.clear();
  out.reserve(lengths.size());
  size_t off = 0;
  for (uint32_t len : lengths) {
    if (off + len > concat.size())
      throw std::runtime_error(std::string("traversal_query: truncated ") +
                               label + " string column");
    out.push_back(concat.substr(off, len));
    off += len;
  }
}

std::vector<std::string> decompress_strings(const ZstdCompressedBlock &strings,
                                            const ZstdCompressedBlock &lengths,
                                            const char *label) {
  std::vector<std::string> out;
  reconstruct_strings(Codec::zstd_decompress_string(strings),
                      Codec::zstd_decompress_uint32_vector(lengths), out,
                      label);
  return out;
}

std::vector<std::string> load_path_names(const CompressedData &data,
                                         size_t num_paths, const char *label) {
  std::vector<std::string> names =
      decompress_strings(data.names_zstd, data.name_lengths_zstd, "path name");
  if (names.size() != num_paths)
    throw std::runtime_error(std::string(label) + ": path name count mismatch");
  return names;
}

WalkIdentityColumns load_walk_identity(const CompressedData &data,
                                       size_t num_walks, const char *label) {
  WalkIdentityColumns w;
  w.samples = decompress_strings(data.walk_sample_ids_zstd,
                                 data.walk_sample_id_lengths_zstd, "walk sample");
  w.seqs = decompress_strings(data.walk_seq_ids_zstd,
                              data.walk_seq_id_lengths_zstd, "walk seq");
  w.haps = Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  w.starts =
      Codec::decompress_varint_int64(data.walk_seq_starts_zstd, num_walks);
  w.ends = Codec::decompress_varint_int64(data.walk_seq_ends_zstd, num_walks);
  if (w.samples.size() != num_walks || w.seqs.size() != num_walks ||
      w.haps.size() != num_walks || w.starts.size() != num_walks ||
      w.ends.size() != num_walks)
    throw std::runtime_error(std::string(label) +
                             ": walk metadata count mismatch");
  return w;
}

} // namespace tquery
} // namespace gfaz
