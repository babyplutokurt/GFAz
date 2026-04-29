# GFAz Growth Workflow

This document describes what the `growth` analysis computes, how Panacus
computes it from a GFA, and how `gfaz growth` computes the same node-growth
curve directly from compressed `.gfaz` data.

## What Growth Computes

Pangenome growth estimates the expected number of distinct countables observed
as more path/walk groups are sampled. In the current GFAz implementation, the
countable is a **node**. A node counts at most once per path/walk group,
regardless of how many times that group traverses it.

Let:

- `N` be the number of path/walk groups.
- `c(v)` be the number of groups that contain node `v`.
- `hist[c]` be the number of nodes with group coverage exactly `c`.
- `k` be the number of groups sampled.

For union growth (`coverage >= 1`, `quorum >= 0`), node `v` is present in a
random subset of `k` groups unless all selected groups come from the `N - c(v)`
groups that do not contain it. Therefore:

```
P(v is present in a random k-group subset)
    = 1 - C(N - c(v), k) / C(N, k)
```

The expected growth value at `k` is:

```
growth[k] = sum over coverage c:
              hist[c] * (1 - C(N - c, k) / C(N, k))
```

GFAz computes this probability using a stable product form instead of directly
forming binomial coefficients:

```
C(N - c, k) / C(N, k)
    = product over i = 0..c-1 of (N - k - i) / (N - i)
```

This avoids integer overflow on large pangenomes.

## How Panacus Does It

Panacus `growth` is implemented as a graph analysis that first builds the
coverage histogram and then computes growth from that histogram.

At a high level:

1. The CLI creates an `AnalysisRun` with `AnalysisParameter::Growth`.
2. The analysis requests `InputRequirement::Hist`.
3. `GraphBroker` loads the raw GFA into `GraphStorage`.
4. `GraphStorage::from_gfa()` scans the GFA text and builds:
   - a node-name to integer-ID map,
   - node lengths,
   - path/walk metadata.
5. `GraphBroker::set_abaci_by_total()` reopens the GFA and parses `P` and `W`
   lines to build `ItemTable` data: for each path/walk, the list of node IDs it
   traverses.
6. `AbacusByTotal::item_table_to_abacus()` walks the path order and counts, for
   each node, how many groups contain it. A `last` array prevents duplicate
   counting of the same node inside one group.
7. `Hist::from_abacus()` converts the per-node coverage vector into
   `hist[c] = number of nodes with coverage c`.
8. `Growth::set_inner()` calls `Hist::calc_all_growths()` to compute growth
   curves from the histogram.

The important cost is that Panacus starts from a **text GFA** at query time.
Even if parts of path-sequence parsing are internally chunked, the workflow
still needs to scan and parse the GFA, build node ID mappings, tokenize path
and walk records, and materialize generic item tables before growth can be
computed.

This design is flexible: the same machinery supports multiple count types,
subsets, excludes, custom grouping, and other analyses. But for a simple node
growth query it does significant generic preprocessing work.

## How GFAz Does It

`gfaz growth` starts from `CompressedData`, the in-memory representation of a
`.gfaz` file. The compression step has already converted text GFA records into
typed compressed columns and integer path/walk streams, so growth does not need
to parse the original GFA.

The public entry point is:

```
GrowthResult compute_growth(const CompressedData &data,
                            int num_threads,
                            GroupingMode mode);
```

The CLI in `src/cli/growth_command.cpp` only parses arguments, deserializes the
`.gfaz`, calls `compute_growth()`, and prints the result.

### Step 1: Load the Node and Path/Walk Metadata

GFAz infers the number of nodes from `segment_seq_lengths_zstd`. It does not
need the segment sequences themselves for node growth; it only needs the number
of segment IDs that can appear in paths/walks.

It then reads:

- `data.sequence_lengths` and `data.walk_lengths`, which give the encoded
  length of each path/walk slice,
- `data.original_path_lengths` and `data.original_walk_lengths`, used if full
  materialization is needed,
- compressed path and walk streams,
- grammar rule arrays.

### Step 2: Decompress Encoded Integer Streams

`compute_growth()` decompresses:

```
rules_first_zstd
rules_second_zstd
paths_zstd
walks_zstd
```

The rules are delta-decoded. The path and walk streams are already integer
streams over node IDs and grammar rule IDs. This is the main difference from
Panacus: the hot path starts from integer arrays, not GFA text.

### Step 3: Build Slices

GFAz stores all encoded paths in one flat vector and all encoded walks in
another. `build_slices()` turns those flat vectors plus their length arrays
into `HapSlice` records:

```
struct HapSlice {
  const int32_t *encoded;
  uint32_t enc_len;
  uint32_t orig_len;
};
```

Each slice is one path or one walk. It points into the decompressed integer
stream; it does not copy the path/walk data.

### Step 4: Build Groups

Growth counts node presence per group. In `PerPathWalk` mode, every path/walk
is its own group.

For Panacus-style grouping, GFAz builds a key per slice:

- `SampleHapSeq`: `sample#hap#seq`, matching Panacus default path identity.
- `SampleHap`: `sample#hap`.
- `Sample`: `sample`.

For `P` lines, path names are decompressed from `names_zstd` and parsed as PanSN
names. For `W` lines, sample, haplotype, and sequence ID are already stored as
separate compressed columns.

The result is:

```
groups[gid] = list of slice IDs belonging to that group
```

All slices in one group share a single deduplication stamp during coverage
counting, so a node is counted once for the group even if it appears in
multiple slices in that group.

### Step 5: Stream Nodes and Count Coverage

GFAz allocates:

```
cov[node_id] = number of groups containing node_id
```

Then it processes groups in parallel with OpenMP.

Each thread owns:

```
last_seen[node_id]
stamp
```

For each group:

1. Increment the thread-local `stamp`.
2. Stream every slice in the group.
3. Expand grammar rules as needed.
4. For each node ID encountered:
   - convert signed orientation to absolute node ID,
   - ignore invalid ID `0` or IDs above `num_nodes`,
   - if `last_seen[node_id] != stamp`, set it to `stamp` and increment
     `cov[node_id]`.

The update is:

```
if node has not been seen in this group:
    cov[node] += 1
```

So yes: GFAz checks, for each segment/node, how many groups contain it. It does
not count raw occurrences. A node appearing three times in the same group still
adds only `1` to that node's coverage.

### Step 6: Expand Grammar Rules Lazily

Encoded path/walk streams may contain grammar rule IDs instead of raw node IDs.
When a rule ID is encountered, GFAz recursively expands the rule and streams
the leaf node IDs to the same coverage-update callback.

For common `delta_round` values:

- `delta_round == 0`: stream leaves directly.
- `delta_round == 1`: maintain a running prefix sum while streaming leaves.
- `delta_round >= 2`: materialize the decoded haplotype into a temporary buffer
  and apply multiple inverse-delta passes.

The important property is that, for `delta_round` 0 or 1, GFAz does not need to
materialize a full decompressed GFA path. It streams the decoded node IDs
directly into the coverage counter.

### Step 7: Build the Coverage Histogram

After all groups are processed, GFAz converts the coverage vector into a
histogram:

```
hist[c] = number of nodes where cov[node] == c
```

This pass is also parallelized. Each thread builds a local histogram, then the
local histograms are reduced into `result.hist`.

### Step 8: Compute the Growth Curve

Finally, GFAz computes the closed-form expected growth curve:

```
growth[k] = sum over c:
              hist[c] * (1 - C(N - c, k) / C(N, k))
```

This pass is parallelized over `k`. The current implementation corresponds to
Panacus union-growth semantics: node count, `coverage >= 1`, `quorum >= 0`.

## Why This Is Faster

The runtime win comes from where the work happens.

Panacus performs GFA parsing, node-name lookup, path/walk tokenization, item
table construction, abacus construction, histogram construction, and growth
calculation during the `growth` command.

GFAz performs text parsing and path encoding during compression. At `growth`
time, it reads typed compressed blocks, decompresses compact integer arrays,
streams grammar-expanded node IDs, and updates a coverage vector directly.

The GFAz hot path therefore avoids:

- scanning raw GFA text,
- parsing `S`, `P`, and `W` records,
- mapping segment names to IDs during growth,
- materializing generic Panacus item tables,
- expanding paths into textual or full GFA form.

For repeated downstream analyses, the `.gfaz` file acts as a compressed
preprocessed representation. The compression/indexing cost is paid once, while
growth queries reuse the compact encoded structure.

## Reference Benchmark

Captured on HPRC `hprc-v1.1-mc-chm13` with 16 threads. The raw GFA used by
Panacus is 50,763,189,595 bytes.

Commands:

```bash
/usr/bin/time -v ./build/bin/gfaz growth \
    -j 16 --group-by sample \
    /data1/work/kurty/gfaz/hprc-v1.1-mc-chm13.gfaz \
    > output_hprc.1.1.txt

/usr/bin/time -v ./target/release/panacus growth \
    /home/kurty/data/gfa/hprc-v1.1-mc-chm13.gfa \
    -t 16 \
    > hprc_1.1.tsv
```

| Tool | Input | Wall time | Max RSS | CPU |
|:---|:---|---:|---:|---:|
| GFAz growth | `.gfaz` | 0:06.33 | 6,102,928 KB | 908% |
| Panacus growth | `.gfa` | 23:48.12 | 44,690,224 KB | 99% |

For these runs, GFAz is about **226x faster** by wall time and uses about
**7.3x less peak memory**. Panacus spent most of the elapsed time loading and
parsing the raw GFA path/walk sequences; its log reported 1,354.63 s in
`parsing path + walk sequences` before abacus and histogram construction.

Note: the exact commands above use `--group-by sample` for GFAz, while the
Panacus command uses its default path identity grouping (`sample#hap#seq` for
PanSN-style names). The comparison is therefore an end-to-end performance
snapshot for the shown commands, not a strict semantic-equivalence benchmark.

## Current Scope

The current `gfaz growth` implementation is intentionally narrower than the
full Panacus command:

- count type: node,
- coverage threshold: `>= 1`,
- quorum threshold: `>= 0`,
- output: growth curve and internal node coverage histogram,
- grouping: per path/walk, sample-hap-seq, sample-hap, or sample.

Future extensions can reuse the same basic structure, but bp growth, edge
growth, custom subset/exclude coordinates, and non-zero quorum thresholds need
additional data structures or formulas.
