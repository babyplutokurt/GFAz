#include "compress/compression_workflow.hpp"
#include "compress/io/gfa_decoder.hpp"
#include "core/codec/serialization.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

namespace {

std::string join(const std::vector<std::string> &values) {
  std::string result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      result += ',';
    result += values[i];
  }
  return result;
}

std::string visits_text(const std::vector<gfaz::NodeId> &visits) {
  std::string result;
  for (size_t i = 0; i < visits.size(); ++i) {
    if (i > 0)
      result += ',';
    result += std::to_string(visits[i]);
  }
  return result;
}

const char *orientation(bool is_reverse) {
  return is_reverse ? "-" : "+";
}

struct RecordingVisitor final : gfaz::GfaRecordVisitor {
  std::vector<std::string> events;

  void on_header(const std::string &header_line) override {
    events.push_back(header_line);
  }

  void on_segment(uint32_t id, const std::string &sequence,
                  const gfaz::GfaTagList &tags) override {
    events.push_back("S|" + std::to_string(id) + "|" + sequence + "|" +
                     join(tags));
  }

  void on_link(uint32_t from_id, bool from_is_reverse, uint32_t to_id,
               bool to_is_reverse, const std::string &overlap,
               const gfaz::GfaTagList &tags) override {
    events.push_back("L|" + std::to_string(from_id) + "|" +
                     orientation(from_is_reverse) + "|" +
                     std::to_string(to_id) + "|" +
                     orientation(to_is_reverse) + "|" + overlap + "|" +
                     join(tags));
  }

  void on_path(const std::string &name,
               const std::vector<gfaz::NodeId> &visits,
               const std::string &overlap,
               const gfaz::GfaTagList &tags) override {
    events.push_back("P|" + name + "|" + visits_text(visits) + "|" + overlap +
                     "|" + join(tags));
  }

  void on_walk(const std::string &sample_id, uint32_t hap_index,
               const std::string &sequence_id, int64_t sequence_start,
               int64_t sequence_end,
               const std::vector<gfaz::NodeId> &visits) override {
    events.push_back("W|" + sample_id + "|" + std::to_string(hap_index) + "|" +
                     sequence_id + "|" + std::to_string(sequence_start) + "|" +
                     std::to_string(sequence_end) + "|" + visits_text(visits));
  }

  void on_jump(uint32_t from_id, bool from_is_reverse, uint32_t to_id,
               bool to_is_reverse, const std::string &distance,
               const std::string &rest_fields) override {
    events.push_back("J|" + std::to_string(from_id) + "|" +
                     orientation(from_is_reverse) + "|" +
                     std::to_string(to_id) + "|" +
                     orientation(to_is_reverse) + "|" + distance + "|" +
                     rest_fields);
  }

  void on_containment(uint32_t container_id, bool container_is_reverse,
                      uint32_t contained_id, bool contained_is_reverse,
                      uint32_t position, const std::string &overlap,
                      const std::string &rest_fields) override {
    events.push_back("C|" + std::to_string(container_id) + "|" +
                     orientation(container_is_reverse) + "|" +
                     std::to_string(contained_id) + "|" +
                     orientation(contained_is_reverse) + "|" +
                     std::to_string(position) + "|" + overlap + "|" +
                     rest_fields);
  }
};

class UnseekableMemoryBuffer final : public std::streambuf {
public:
  explicit UnseekableMemoryBuffer(std::string &bytes) {
    char *begin = bytes.data();
    setg(begin, begin, begin + bytes.size());
  }

protected:
  pos_type seekoff(off_type, std::ios_base::seekdir,
                   std::ios_base::openmode) override {
    return pos_type(off_type(-1));
  }

  pos_type seekpos(pos_type, std::ios_base::openmode) override {
    return pos_type(off_type(-1));
  }
};

void require_events(const std::vector<std::string> &actual,
                    const std::vector<std::string> &expected,
                    const std::string &source) {
  if (actual == expected)
    return;

  std::ostringstream message;
  message << source << " callbacks differ";
  const size_t count = std::max(actual.size(), expected.size());
  for (size_t i = 0; i < count; ++i) {
    const std::string got = i < actual.size() ? actual[i] : "<missing>";
    const std::string want = i < expected.size() ? expected[i] : "<missing>";
    if (got != want)
      message << "\n  [" << i << "] expected: " << want << "\n      actual: "
              << got;
  }
  throw std::runtime_error(message.str());
}

const std::vector<std::string> expected_events = {
    "H\tVN:Z:1.1\tTS:Z:front door",
    "S|1|A|LN:i:1,SZ:Z:segment_one",
    "S|2|CC|LN:i:2",
    "S|3|GGG|LN:i:3",
    "S|4|TTTT|LN:i:4",
    "L|1|+|2|-|1M|ID:Z:forward_reverse",
    "L|2|-|3|+|2M|ID:Z:reverse_forward",
    "J|2|-|3|+|-4|SC:i:1\tJZ:Z:jump_tag",
    "C|1|-|3|+|2|1M|ID:Z:containment_tag",
    "P|p1|1,-2,3|1M,2M|PT:Z:path tag,PI:i:7",
    "P|p2|-3,2|*|",
    "W|sample|1|chr1|0|6|1,-2,3",
    "W|sample|2|chr2|-1|-1|-3,2",
};

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " FIXTURE TEMP_GFAZ\n";
    return 2;
  }

  const std::string fixture = argv[1];
  const std::string temp_gfaz = argv[2];

  try {
    const gfaz::CompressedData data =
        compress_gfa(fixture, 4, 2, 1, 2, false);

    RecordingVisitor data_visitor;
    gfaz::decode_gfa_records(data, data_visitor, 2);
    require_events(data_visitor.events, expected_events, "CompressedData");

    gfaz::serialize_compressed_data(data, temp_gfaz);

    RecordingVisitor file_visitor;
    gfaz::decode_gfa_records(temp_gfaz, file_visitor, 2);
    require_events(file_visitor.events, expected_events, "file");

    std::ifstream input(temp_gfaz, std::ios::binary);
    if (!input)
      throw std::runtime_error("failed to reopen temporary GFAz");
    std::string bytes((std::istreambuf_iterator<char>(input)),
                      std::istreambuf_iterator<char>());
    UnseekableMemoryBuffer buffer(bytes);
    std::istream unseekable(&buffer);

    RecordingVisitor stream_visitor;
    gfaz::decode_gfa_records(unseekable, stream_visitor, 2);
    require_events(stream_visitor.events, expected_events,
                   "unseekable stream");

    std::remove(temp_gfaz.c_str());
    std::cout << "PASS front-door GFA decoder\n";
    return 0;
  } catch (const std::exception &error) {
    std::remove(temp_gfaz.c_str());
    std::cerr << "FAIL front-door GFA decoder: " << error.what() << '\n';
    return 1;
  }
}
