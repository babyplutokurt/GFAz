#pragma once

#include "core/model/compressed_data.hpp"
#include "core/model/gfa_graph.hpp"

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace gfaz {

using GfaTagList = std::vector<std::string>;

/**
 * Receives decoded GFA records from a GFAz container.
 *
 * Every reference passed to a callback is valid only for the duration of that
 * callback. Callbacks are invoked synchronously in type-grouped canonical
 * order: H, S, L, J, C, P, then W. Records retain their source order within
 * each group, but the original ordering between different record types is not
 * represented by the GFAz container.
 *
 * Segments are numbered 1..N in S-line order, and links, jumps, containments,
 * paths and walks refer to them by that number. on_named_segment also gives
 * the name the segment has in the GFA, which may be any string; a consumer that
 * needs original names records them there and maps the numbers itself.
 *
 * Orientation booleans are false for forward ('+') and true for reverse ('-').
 * Oriented NodeId values use the same convention through their sign: positive
 * is forward and negative is reverse.
 *
 * The default implementations ignore records so consumers only need to
 * override the record types they use.
 */
class GfaRecordVisitor {
public:
  virtual ~GfaRecordVisitor() = default;

  virtual void on_header(const std::string &) {}

  // Called once per segment: number, GFA name, sequence, tags. The default
  // drops the name and calls on_segment, so a visitor overrides one of the two.
  virtual void on_named_segment(uint32_t id, const std::string &,
                                const std::string &sequence,
                                const GfaTagList &tags) {
    on_segment(id, sequence, tags);
  }

  virtual void on_segment(uint32_t, const std::string &, const GfaTagList &) {}

  virtual void on_link(uint32_t, bool, uint32_t, bool, const std::string &,
                       const GfaTagList &) {}

  virtual void on_path(const std::string &, const std::vector<NodeId> &,
                       const std::string &, const GfaTagList &) {}

  virtual void on_walk(const std::string &, uint32_t, const std::string &,
                       int64_t, int64_t, const std::vector<NodeId> &) {}

  virtual void on_jump(uint32_t, bool, uint32_t, bool, const std::string &,
                       const std::string &) {}

  virtual void on_containment(uint32_t, bool, uint32_t, bool, uint32_t,
                              const std::string &, const std::string &) {}
};

/** Decode an already-deserialized GFAz container into semantic GFA records. */
void decode_gfa_records(const CompressedData &data, GfaRecordVisitor &visitor,
                        int num_threads = 0);

/** Deserialize and decode a GFAz container from the stream's current position.
 */
void decode_gfa_records(std::istream &input, GfaRecordVisitor &visitor,
                        int num_threads = 0);

/** Deserialize and decode a GFAz file. */
void decode_gfa_records(const std::string &input_path,
                        GfaRecordVisitor &visitor, int num_threads = 0);

} // namespace gfaz
