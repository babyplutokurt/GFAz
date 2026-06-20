# Extending the GFAz compute engine — how to add a new app

This is the developer guide for adding a new **compute app** (a subcommand that
analyzes a `.gfaz` container directly, like `growth` / `pav` / `deconstruct`).
It documents the shared "extension surface" you build on, the canonical app
skeleton, every wiring touch point, and the conventions a new app must follow.

If you just want to *use* an existing app, read its workflow spec under
[`workflows/`](workflows/) instead. For the strategic picture (which apps to
build next, and why), see
[`design/COMPUTE_ENGINE_DIRECTION.md`](design/COMPUTE_ENGINE_DIRECTION.md) and
[`design/DOWNSTREAM_APPLICATIONS.md`](design/DOWNSTREAM_APPLICATIONS.md).

## 1. Architecture in one paragraph

The build is three static libraries: **`gfaz_core`** (model, codec, utils) ←
**`gfaz_compress`** (GFA→`.gfaz`) and **`gfaz_compute`** (analytics on `.gfaz`).
A compute app lives in `gfaz_compute` and depends only on `gfaz_core` — **never
on `gfaz_compress`**. It reads a deserialized `CompressedData`, computes a
statistic by streaming traversals, and writes its result (VCF / TSV / FASTA /
…) to **stdout** (there is no `-o` flag; users redirect with `>`). All three
shipped apps follow one pattern; a fourth should too.

## 2. The canonical app skeleton

Every shipped app does the same five things. The decode, identity/grouping,
threading, and topology plumbing is already factored out — a new app is mostly
"pick an accumulator and a visitor."

```text
1. deserialize_compressed_data(path)        -> CompressedData     (in the CLI command)
2. load_rulebook(data)                       -> Rulebook          (if you decode node ids)
   load_traversals(data)                     -> LoadedTraversals  (slices over flat arrays)
   make_rule_cache(..., "GFAZ_<APP>_RULE_CACHE_BYTES", 0)         (rule-leaf cache; off by default)
3. build identity/grouping from PanSN names  -> per-slice keys     (parse_pansn_path_name, *_group_key)
4. ScopedOMPThreads + per-thread:                                  (parallel over slices/snarls)
     stream_decoded_nodes(slice, ..., visit) -> visitor updates a thread-local accumulator
5. merge per-thread accumulators (stable), normalize, emit to std::ostream
```

Minimal annotated template for `src/compute/<app>_workflow.cpp`:

```cpp
#include "compute/<app>_workflow.hpp"
#include "compute/traversal_query.hpp"          // load_traversals, stream_decoded_nodes, ...
#include "core/utils/threading_utils.hpp"        // ScopedOMPThreads
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {
namespace { using namespace gfaz::tquery; }      // tquery::  decode/identity layer

void <app>_run(const CompressedData &data, const <App>Options &options,
               std::ostream &out) {
  Rulebook rb = load_rulebook(data);             // rules_first/second + id range
  LoadedTraversals loaded = load_traversals(data);
  const std::vector<HapSlice> &slices = loaded.slices;   // [paths... walks...]
  RuleLeafCache cache = make_rule_cache(
      rb.min_rule_id, rb.rules_first, rb.rules_second,
      "GFAZ_<APP>_RULE_CACHE_BYTES", /*default_budget=*/0);
  const int delta_round = data.delta_round;

  // ... build per-slice grouping keys with parse_pansn_path_name / *_group_key ...

  ScopedOMPThreads omp(options.num_threads);
  const int T = std::max(1, omp.effective_threads());
  std::vector<Accum> per_thread(T);
#pragma omp parallel num_threads(T)
  {
    const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
    Accum &acc = per_thread[tid];
    std::vector<NodeId> scratch;                  // reused decode buffer
#pragma omp for schedule(dynamic, 16)
    for (long long s = 0; s < (long long)slices.size(); ++s) {
      auto visit = [&](NodeId node) { /* update acc with abs_node_id(node), sign, etc. */ };
      stream_decoded_nodes(slices[s], delta_round, rb.min_rule_id, rb.max_rule_id,
                           rb.rules_first, rb.rules_second, cache, scratch, visit);
    }
  }
  // merge per_thread (stable order!), normalize, write to `out`.
}
} // namespace gfaz
```

Worked examples, simplest to richest:

- **`src/compute/growth_workflow.cpp`** — pure path-iterative; node-coverage
  bitset accumulator. The cleanest illustration of the skeleton.
- **`src/compute/pav_workflow.cpp`** — reference projection + BED windows +
  grouping into a matrix.
- **`src/compute/deconstruct_workflow.cpp`** — the richest: topology (snarls),
  multi-pass, per-thread record buffers, the determinism + flat-pool patterns
  below.

## 3. Stable extension surface (the API to depend on)

These declarations are the contract a compute app builds on. Treat them as the
stable surface; if you need something not here, prefer adding it to these shared
headers over duplicating logic in your workflow.

### `include/compute/traversal_query.hpp` (namespace `gfaz::tquery`)

| Symbol | Line | Contract |
|---|---|---|
| `load_traversals(data) -> LoadedTraversals` | 171 | Decompress P/W node arrays + build the combined `slices` (`[paths… walks…]`). Owns the flat arrays — must outlive slice use. |
| `struct HapSlice {encoded, enc_len, orig_len}` | 149 | One haplotype's encoded node span. |
| `build_slices(flat, lengths, original_lengths, out)` | 155 | Build slices over your own flat arrays (rarely needed directly). |
| `load_rulebook(data) -> Rulebook` | 68 | Decompress + delta-decode both rule arrays; compute `[min_rule_id, max_rule_id)`. |
| `make_rule_cache(min_id, first, second, env, default_budget)` | 52 | One-call rule-leaf cache setup; resolves byte budget from env. Use `default_budget=0` for single-stream apps. |
| `stream_decoded_nodes(slice, delta_round, min, max, first, second, cache, scratch, visit)` | 186 | **The workhorse.** Streams a slice's decoded signed node ids to `visit` without materializing a vector (delta 0/1 fast paths). |
| `stream_hap_leaves(...)`, `expand_rule_visit(...)` | 128, 75 | Lower-level grammar expansion; `stream_decoded_nodes` is the normal entry point. |
| `decode_one_haplotype_general(...)` | 176 | Full decode into a `vector<NodeId>` (general delta path). |
| `abs_node_id(node) -> uint32_t` | 70 | `|node|`; the sign of `NodeId` is orientation. |
| `parse_pansn_path_name(name) -> PansnParts` | 225 | Split `sample#hap#seq` (coords stripped). |
| `path_group_key(name, mode)` / `walk_group_key(sample, hap, seq, mode)` | 226 / 227 | Grouping key for a P-line / W-line under a `GroupingMode`. |
| `walk_reference_name(sample, hap, seq, start, end)` | 229 | Canonical W-line reference name (with `:start-end`). |
| `load_path_names(data, n, ctx)` | 249 | P-line names. |
| `load_walk_identity(data, n, ctx) -> WalkIdentityColumns` | 269 | W-line sample/hap/seq/start/end columns. |
| `reconstruct_strings(...)`, `decompress_strings(...)` | 237, 241 | String-payload helpers. |

### `include/compute/snarl_finder.hpp` (only if you need topology)

| Symbol | Line | Contract |
|---|---|---|
| `build_doubled_graph_from_links(data, num_nodes) -> DoubledGraph` | 62 | Bidirected node-side graph from stored L-lines. |
| `build_segment_graph_from_links(data, num_nodes) -> SegmentGraph` | 94 | Undirected segment graph (for the top-level/biconnected decomposition). |
| `find_reference_snarls(g, ...) -> vector<ReferenceSnarl>` | 79 | Leaf-superbubble snarls projected on the reference. |
| `find_reference_snarls_top_level(sg, ...)` | 112 | Top-level (biconnected) snarls — the `vg deconstruct`-matching granularity. |

### `include/compute/grouping_mode.hpp`

`enum class GroupingMode { PerPathWalk, SampleHapSeq, SampleHap, Sample }` — the
shared grouping vocabulary; map your `-S/-H/-p` flags to this.

### `gfaz_core` headers you will use

| Symbol | Header | Contract |
|---|---|---|
| `CompressedData` | `core/model/compressed_data.hpp` | The deserialized container (segments, rulebook, traversals, links, `delta_round`, …). |
| `deserialize_compressed_data(path)` | `core/codec/serialization.hpp` | Load a `.gfaz` into `CompressedData` (call in the CLI command). |
| `ScopedOMPThreads`, `.effective_threads()` | `core/utils/threading_utils.hpp` | RAII OpenMP thread-count scope; the standard parallel wrapper. |
| `complement_base`, `reverse_complement[_inplace]` | `core/utils/sequence_utils.hpp` | DNA helpers (orientation-resolved sequence). |
| `kDefaultNumThreads` | `core/defaults.hpp` | Default for an Options `num_threads`. |

## 4. Wiring checklist (registering the app)

All lists in the build/CLI are **explicit** (no globs), so a new app touches a
fixed set of files. Using `deconstruct` as the live reference:

1. `include/compute/<app>_workflow.hpp` — the `<App>Options` struct + the entry
   function (`<app>_run(const CompressedData&, const <App>Options&, std::ostream&)`).
2. `src/compute/<app>_workflow.cpp` — implementation (helpers in an anonymous
   namespace).
3. `src/cli/<app>_command.cpp` — `int do_<app>(int, char**)`: getopt parse →
   fill `<App>Options` → `deserialize_compressed_data` → call the workflow.
4. `include/cli/commands.hpp` — declare `int do_<app>(int, char*[]);`.
5. `src/cli/gfaz_cli.cpp` — add an `else if (command == "<app>")` dispatch arm.
6. `include/cli/common.hpp` + `src/cli/common.cpp` — declare and define
   `print_<app>_help()`, **and** add the app to `print_usage()`'s USAGE +
   SUBCOMMANDS lists.
7. `CMakeLists.txt` — add `src/compute/<app>_workflow.cpp` to
   **`GFAZ_COMPUTE_SOURCES`** (~line 153) **and** `src/cli/<app>_command.cpp` to
   the **`add_executable(gfaz …)`** source list (~line 265).
8. Tests + docs (see §6).

## 5. Conventions & contracts

- **Output to stdout.** No `-o`. Write to the `std::ostream&` the workflow is
  handed; the command passes `std::cout`.
- **CLI flags** (reuse the shipped vocabulary): `-i/--input`, `-t/-j` threads
  (`>0` explicit, `0` auto, `<0` inherit OpenMP), `-S/-H/-p` grouping →
  `GroupingMode`, and `-r/-P` reference / reference-prefix selection if the app
  is reference-based (see `deconstruct_command.cpp`).
- **Determinism — output must be thread-count invariant.** Accumulate into
  per-thread buffers, then merge and `stable_sort` by a *stable key* (e.g. source
  slice id), never relying on thread scheduling order. See the `obs_by_snarl`
  stable sort in `deconstruct_workflow.cpp` and the guard test
  `tests/regression/test_thread_determinism.py`.
- **Memory discipline at chromosome scale.** Prefer one flat pool + fixed-size
  records over a `vector` per item (per-item vectors pay a control block + malloc
  each — fatal at hundreds of millions of items). Guard 32-bit index widths with
  a clean `throw` rather than silent wrap. Free per-thread buffers as you merge so
  the per-thread and merged copies don't both reside at full size. See `ObsRec` /
  `interior_pool` in `deconstruct_workflow.cpp`.
- **Rule cache off by default.** Single-stream apps (each slice decoded once)
  don't benefit from eagerly expanding the rulebook to leaf arrays; pass
  `default_budget=0` to `make_rule_cache` and expose a per-app env override
  (`GFAZ_<APP>_RULE_CACHE_BYTES`).
- **Uppercase once, not per base.** If you spell sequence, uppercase the shared
  segment pool a single time (parallel) rather than per emitted base — see
  `load_segments` in `deconstruct_workflow.cpp`.

## 6. Tests & docs to add

- `tests/regression/test_<app>.py` driving the built CLI on small fixtures under
  `tests/fixtures/`; register it in `tests/run_all.py`.
- If a reference tool exists (vg / odgi / panacus), add
  `tests/concordance/test_<app>_vs_<tool>.py`.
- A workflow spec `docs/workflows/<APP>_WORKFLOW.md`, and a pointer from
  [`docs/README.md`](README.md).

## 7. Known follow-ups (not yet done)

These are shared-surface improvements flagged for when an interactive/per-query
app needs them — see
[`design/DOWNSTREAM_APPLICATIONS.md`](design/DOWNSTREAM_APPLICATIONS.md):

- Walk-by-name lookup is currently an O(N) linear scan
  (`src/compress/extraction_workflow.cpp`); fine for whole-pangenome batch apps,
  but fix before any per-query service.
- The repeated "decode reference profile + `offset_at` prefix sums + identity/
  grouping setup" boilerplate could be extracted into a shared helper once a 4th
  app needs the exact same prologue.
