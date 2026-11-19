// Cross-checks GfazHandleGraph against a plain reading of the decompressed GFA.
//
//   gfaz_handle_graph_test <graph.gfaz> <graph.gfa> [block_steps] [cached_blocks]
//
// The GFA must be the `gfaz decompress` output of the container, so node ids
// agree. Every interface method is compared with a brute-force answer.

#include "handlegraph_adapter/gfaz_handle_graph.hpp"

#include <handlegraph/util.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <string>
#include <vector>

using namespace handlegraph;

namespace {

struct ExpectedPath {
  std::vector<std::pair<nid_t, bool>> steps;
};

struct ExpectedGraph {
  std::map<nid_t, std::string> sequences;
  std::set<std::pair<std::pair<nid_t, bool>, std::pair<nid_t, bool>>> edges;
  std::vector<ExpectedPath> paths; // P-lines first, then W-lines
};

[[noreturn]] void fail(const std::string &what) {
  std::cerr << "FAIL: " << what << std::endl;
  std::exit(1);
}

void expect(bool condition, const std::string &what) {
  if (!condition)
    fail(what);
}

std::vector<std::string> split(const std::string &line, char sep) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, sep))
    fields.push_back(field);
  return fields;
}

char complement(char c) {
  switch (c) {
  case 'A': return 'T';
  case 'C': return 'G';
  case 'G': return 'C';
  case 'T': return 'A';
  default: return c;
  }
}

std::string reverse_complement(const std::string &s) {
  std::string out(s.rbegin(), s.rend());
  for (char &c : out)
    c = complement(c);
  return out;
}

ExpectedGraph read_gfa(const std::string &path) {
  ExpectedGraph graph;
  std::vector<ExpectedPath> walks;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    const std::vector<std::string> f = split(line, '\t');
    if (f[0] == "S") {
      graph.sequences[std::stoll(f[1])] = f[2];
    } else if (f[0] == "L") {
      const std::pair<nid_t, bool> a{std::stoll(f[1]), f[2] == "-"};
      const std::pair<nid_t, bool> b{std::stoll(f[3]), f[4] == "-"};
      // Store an edge in both directions so lookups need no canonical form.
      graph.edges.insert({a, b});
      graph.edges.insert({{b.first, !b.second}, {a.first, !a.second}});
    } else if (f[0] == "P") {
      ExpectedPath p;
      for (const std::string &step : split(f[2], ','))
        p.steps.push_back({std::stoll(step.substr(0, step.size() - 1)),
                           step.back() == '-'});
      graph.paths.push_back(std::move(p));
    } else if (f[0] == "W") {
      ExpectedPath p;
      const std::string &walk = f[6];
      for (size_t i = 0; i < walk.size();) {
        size_t j = i + 1;
        while (j < walk.size() && walk[j] != '<' && walk[j] != '>')
          ++j;
        p.steps.push_back({std::stoll(walk.substr(i + 1, j - i - 1)),
                           walk[i] == '<'});
        i = j;
      }
      walks.push_back(std::move(p));
    }
  }
  graph.paths.insert(graph.paths.end(), walks.begin(), walks.end());
  return graph;
}

void check_nodes(const gfaz::GfazHandleGraph &graph, const ExpectedGraph &want) {
  expect(graph.get_node_count() == want.sequences.size(), "node count");
  expect(graph.min_node_id() == 1 &&
             graph.max_node_id() == static_cast<nid_t>(want.sequences.size()),
         "node id range");
  size_t total = 0, seen = 0;
  graph.for_each_handle([&](const handle_t &handle) {
    const nid_t id = graph.get_id(handle);
    const std::string &sequence = want.sequences.at(id);
    expect(graph.has_node(id) && !graph.get_is_reverse(handle), "handle");
    expect(graph.get_sequence(handle) == sequence, "forward sequence");
    expect(graph.get_sequence(graph.flip(handle)) == reverse_complement(sequence),
           "reverse sequence");
    expect(graph.get_length(handle) == sequence.size(), "length");
    expect(graph.get_base(graph.flip(handle), 0) ==
               reverse_complement(sequence)[0],
           "reverse base");
    expect(graph.get_subsequence(graph.flip(handle), 1, 2) ==
               reverse_complement(sequence).substr(1, 2),
           "reverse subsequence");
    total += sequence.size();
    ++seen;
  });
  expect(seen == want.sequences.size(), "for_each_handle visits every node");
  expect(graph.get_total_length() == total, "total length");
  expect(!graph.has_node(0) && !graph.has_node(want.sequences.size() + 1),
         "has_node outside range");
}

void check_edges(const gfaz::GfazHandleGraph &graph, const ExpectedGraph &want) {
  std::set<std::pair<std::pair<nid_t, bool>, std::pair<nid_t, bool>>> got;
  graph.for_each_handle([&](const handle_t &forward) {
    for (const handle_t &handle : {forward, graph.flip(forward)}) {
      const std::pair<nid_t, bool> here{graph.get_id(handle),
                                        graph.get_is_reverse(handle)};
      size_t right = 0, left = 0;
      graph.follow_edges(handle, false, [&](const handle_t &next) {
        got.insert({here, {graph.get_id(next), graph.get_is_reverse(next)}});
        ++right;
      });
      graph.follow_edges(handle, true, [&](const handle_t &prev) {
        expect(want.edges.count({{graph.get_id(prev), graph.get_is_reverse(prev)},
                                 here}) == 1,
               "left edge exists");
        ++left;
      });
      expect(graph.get_degree(handle, false) == right, "right degree");
      expect(graph.get_degree(handle, true) == left, "left degree");
    }
  });
  expect(got == want.edges, "edge set");
  size_t edge_count = 0;
  graph.for_each_edge([&](const edge_t &) { ++edge_count; });
  expect(edge_count == graph.get_edge_count(), "edge count");
  for (const auto &edge : want.edges)
    expect(graph.has_edge(graph.get_handle(edge.first.first, edge.first.second),
                          graph.get_handle(edge.second.first,
                                           edge.second.second)),
           "has_edge");
}

void check_paths(const gfaz::GfazHandleGraph &graph, const ExpectedGraph &want) {
  expect(graph.get_path_count() == want.paths.size(), "path count");
  std::vector<path_handle_t> handles;
  std::set<std::string> names;
  graph.for_each_path_handle([&](const path_handle_t &path) {
    handles.push_back(path);
    names.insert(graph.get_path_name(path));
    expect(graph.get_path_handle(graph.get_path_name(path)) == path,
           "path name round trip");
  });
  expect(names.size() == want.paths.size(), "path names are unique");

  // Paths are checked from several threads to exercise the per-thread cache.
#pragma omp parallel for schedule(dynamic, 1)
  for (long long i = 0; i < static_cast<long long>(handles.size()); ++i) {
    const path_handle_t path = handles[i];
    const ExpectedPath &expected = want.paths[i];
    expect(graph.get_step_count(path) == expected.steps.size(), "step count");

    size_t position = 0, index = 0;
    for (step_handle_t step = graph.path_begin(path);
         step != graph.path_end(path); step = graph.get_next_step(step)) {
      expect(index < expected.steps.size(), "path is not too long");
      const handle_t handle = graph.get_handle_of_step(step);
      expect(graph.get_id(handle) == expected.steps[index].first &&
                 graph.get_is_reverse(handle) == expected.steps[index].second,
             "step handle");
      expect(graph.get_path_handle_of_step(step) == path, "step path");
      expect(graph.get_position_of_step(step) == position, "step position");
      expect(graph.has_next_step(step) == (index + 1 < expected.steps.size()),
             "has_next_step");
      expect(graph.has_previous_step(step) == (index > 0), "has_previous_step");
      const size_t length = graph.get_length(handle);
      if (length > 0) {
        expect(graph.get_step_at_position(path, position) == step,
               "step at its first base");
        expect(graph.get_step_at_position(path, position + length - 1) == step,
               "step at its last base");
      }
      position += length;
      ++index;
    }
    expect(index == expected.steps.size(), "path is not too short");
    expect(graph.get_path_length(path) == position, "path length");
    expect(graph.get_step_at_position(path, position) == graph.path_end(path),
           "position past the end");

    // And backwards.
    index = expected.steps.size();
    for (step_handle_t step = graph.path_back(path);
         step != graph.path_front_end(path);
         step = graph.get_previous_step(step)) {
      --index;
      expect(graph.get_id(graph.get_handle_of_step(step)) ==
                 expected.steps[index].first,
             "reverse iteration");
    }
    expect(index == 0, "reverse iteration reaches the front");
  }
}

void check_steps_on_nodes(const gfaz::GfazHandleGraph &graph,
                          const ExpectedGraph &want) {
  // (path, step index, reversed) of every visit, by node.
  std::map<nid_t, std::set<std::tuple<size_t, size_t, bool>>> visits;
  for (size_t p = 0; p < want.paths.size(); ++p)
    for (size_t i = 0; i < want.paths[p].steps.size(); ++i)
      visits[want.paths[p].steps[i].first].insert(
          {p, i, want.paths[p].steps[i].second});

  const size_t stride = std::max<size_t>(1, want.sequences.size() / 400);
  for (nid_t id = 1; id <= static_cast<nid_t>(want.sequences.size());
       id += stride) {
    std::set<std::tuple<size_t, size_t, bool>> got;
    const handle_t handle = graph.get_handle(id, true);
    graph.for_each_step_position_on_handle(
        handle, [&](const step_handle_t &step, const bool &reversed,
                    const size_t &position) {
          expect(graph.get_position_of_step(step) == position,
                 "position reported with a step on a node");
          const bool step_reversed =
              graph.get_is_reverse(graph.get_handle_of_step(step));
          // The handle asked about is the reverse strand.
          expect(reversed == !step_reversed, "relative orientation");
          got.insert({as_integer(graph.get_path_handle_of_step(step)),
                      static_cast<size_t>(as_integers(step)[1]), step_reversed});
          return true;
        });
    expect(got == visits[id], "steps on node " + std::to_string(id));
    expect(graph.get_step_count(handle) == visits[id].size(),
           "step count on node");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <graph.gfaz> <graph.gfa> [block_steps] [cached_blocks]\n";
    return 2;
  }
  gfaz::GfazHandleGraphOptions options;
  if (argc > 3)
    options.block_steps = static_cast<uint32_t>(std::stoul(argv[3]));
  if (argc > 4)
    options.cached_blocks = std::stoul(argv[4]);
  const std::unique_ptr<gfaz::GfazHandleGraph> graph =
      gfaz::GfazHandleGraph::load(argv[1], options);
  const ExpectedGraph want = read_gfa(argv[2]);

  check_nodes(*graph, want);
  check_edges(*graph, want);
  check_paths(*graph, want);
  check_steps_on_nodes(*graph, want);
  std::cout << "handle graph adapter checks passed (" << graph->get_node_count()
            << " nodes, " << graph->get_edge_count() << " edges, "
            << graph->get_path_count() << " paths)\n";
  return 0;
}
