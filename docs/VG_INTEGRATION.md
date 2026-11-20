# Running vg directly on a `.gfaz`

With the [libhandlegraph adapter](HANDLEGRAPH_ADAPTER.md), vg reads a `.gfaz`
in place for every command that only needs a read-only graph. Before, vg
converted the container to its own in-memory graph first, which is where most
of the time and memory went.

The vg side lives on the `gfaz_native_sync` branch of the vg fork: the Makefile
builds `gfaz_core`, `gfaz_compress`, `gfaz_compute` and `gfaz_handlegraph`, and
`src/io/register_loader_gfaz.cpp` registers `gfaz::GfazHandleGraph` for
`PathPositionHandleGraph`, `PathHandleGraph` and `HandleGraph`. Commands that
ask for a mutable graph still get the converted graph. `VG_GFAZ_CONVERT=1`
forces conversion everywhere, for comparison.

## Against XG and GBZ

`scripts/benchmark/vg_formats.sh` runs the same vg commands on the `.gfaz` read
in place and on vg's own indexed format built from the same GFA, next to the
`gfaz` command that answers the same question. 16 threads, files in the page
cache, wall time and peak memory of the whole command.

HPRC chr6 (4.1 GB GFA; `.gfaz` 0.11 GB with a 69 MB saved index, XG 6.5 GB),
two runs each, the slower shown:

| | vg on `.gfaz` | vg on XG | `gfaz` |
|---|---|---|---|
| `stats -z -l` | 0.33 s, 0.55 GB | 2.0 s, 6.4 GB | 0.05 s, 0.14 GB |
| `chunk -p` 100 kb, `-c 10` | 1.7 s, 0.55 GB | 3.1 s, 6.4 GB | 0.44 s, 0.61 GB (`extract-region`) |
| `paths -F -Q grch38#chr6` | 1.3 s, 1.2 GB | 4.5 s, 7.2 GB | |
| `snarls` | 48 s, 3.8 GB | 81 s, 9.8 GB | |
| `deconstruct -p grch38#chr6` | 103 s, 3.8 GB | 128 s, 9.8 GB | 7.0 s, 5.1 GB |

The deconstruct VCF and the path FASTA are byte-identical between the two vg
runs. With the saved index written by `gfaz index -b 1024`, deconstruct on the
`.gfaz` takes 85 s.

HPRC v1.1 (50.8 GB GFA, 92.9 M nodes, 30,640 walks, 4.6 B steps; `.gfaz`
2.27 GB built in 25 s, GBZ 3.28 GB built in 497 s), one run each, no saved
index:

| | vg on `.gfaz` | vg on GBZ | `gfaz` |
|---|---|---|---|
| `stats -z -l` | 9.0 s, 8.5 GB | 19.8 s, 11.0 GB | 1.1 s, 2.6 GB |
| `paths -L` | 8.8 s, 8.5 GB | 9.0 s, 11.0 GB | |
| `paths -F -Q CHM13#0#chr20` | 9.4 s, 8.5 GB | 9.7 s, 11.0 GB | |
| `chunk -p` 100 kb, `-c 10` | 10.0 s, 8.5 GB | 10.8 s, 14.9 GB | 11.3 s, 11.5 GB (`extract-region`) |
| `snarls` | 185 s, 49.7 GB | 204 s, 53.8 GB | |
| `deconstruct -p CHM13#0#chr20` | 330 s, 49.7 GB | 1,645 s, 55.7 GB | 21 s, 22.7 GB |

The two vg columns do not return the same bytes on HPRC v1.1: a GBZ names and
lists paths differently, `chunk` on it returns far fewer paths, and the
deconstruct VCFs differ by 0.01% in size. Those differences have not been
analysed. `gfaz deconstruct` uses its own snarl decomposition and conflict
policy, so its VCF is not the one vg writes (see
[workflows/DECONSTRUCT_WORKFLOW.md](workflows/DECONSTRUCT_WORKFLOW.md)).

Where the remaining time goes: on the snarl and deconstruct rows most of it is
vg's own work on the graph (snarl finding, and parsing path names once per
traversal), not the adapter. `gfaz extract-region` without a saved index
spends its time loading the container and building the index.

## Against conversion

Measured on an earlier build (branch `gfaz_native`), before the index and
memory changes above; the native column is slower there than it is now.

16 threads, files in the page cache, one run each. "Convert" is the previous
behaviour. Times are whole commands, including loading.

HPRC chr6 (4.1 GB GFA, 4.8 M nodes, 1,410 paths, 400 M steps):

| vg command | Native | Convert | Output |
|---|---|---|---|
| `stats -z -l` | 0.7 s, 0.8 GB | 105 s, 6.6 GB | identical |
| `paths -L` | 0.7 s, 0.8 GB | 105 s, 6.6 GB | identical |
| `paths -E` | 10.2 s, 0.8 GB | 115 s, 6.6 GB | identical |
| `paths -F -Q chm13#chr6` | 1.4 s, 1.5 GB | 105 s, 6.6 GB | identical |
| `find -p` 100 kb, `-c 10` | 2.6 s, 0.8 GB | 136 s, 7.8 GB | native has more paths (below) |
| `chunk -p` 100 kb, `-c 10` | 2.3 s, 0.8 GB | 106 s, 6.6 GB | native has more paths (below) |
| `snarls` | 55 s, 4.5 GB | 158 s, 9.1 GB | same snarls but one (below) |
| `deconstruct -p chm13#chr6` | 178 s, 4.5 GB | 190 s, 18.6 GB | identical VCF |
| `gbwt -E --gbz-format` (export GBZ) | 324 s, 3.5 GB | 450 s, 8.0 GB | same graph and paths (below) |

HPRC v1.1 (50.8 GB GFA, 92.9 M nodes, 30,640 walks, 4.6 B steps):

| vg command | Native | Convert |
|---|---|---|
| `stats -z -l` | 11 s, 14.2 GB | 1,257 s, 72.7 GB |
| `paths -F -Q CHM13#0#chr6` | 12 s, 14.2 GB | not run |
| `find -p` 100 kb, `-c 10` | 15 s, 14.2 GB | not run |
| `chunk -p` 100 kb, `-c 10` | 14 s, 14.2 GB | not run |

On HPRC v1.1 conversion alone takes 21 minutes, so only `stats` was run that
way; its output matches the native one.

What the numbers say:

- Loading is the difference. The native graph is ready in under a second on
  chr6 and in about 11 s on HPRC v1.1; conversion takes 105 s and 21 min.
- Commands that walk paths or read nodes and edges inherit that gain in full.
- `deconstruct` asks which paths cross each snarl boundary over the whole
  graph, the adapter's most expensive query. It ends up level with the
  converted graph in time and uses a quarter of the memory. Without the
  adapter's block cache it took 719 s.

## Differences in output

- **`find` and `chunk` return more.** vg adds a position overlay to graphs
  that lack path positions, and that overlay indexes reference paths only. The
  adapter already has positions for every path, so the sub-paths of all
  haplotypes come back, as they do for an XG graph.
- **`snarls`:** 1,509,320 of 1,509,321 snarl boundaries are the same on chr6.
  One differs, and some snarls are written from the other end or under a
  different parent. vg's snarl finder depends on the order in which edges are
  visited, and the adapter lists edges sorted by node id.
- **GBZ export:** the two GBZ files have the same nodes, edges, path names and
  path lengths but different sizes (156 MB native, 177 MB converted), because
  paths are inserted in a different order.

## Limits

- Read-only. `vg mod`, `vg clip`, `vg convert` and other commands that build
  or change a graph use the converted graph, as before.
- Node ids are assigned by the rule of vg's own GFA import (numeric segment
  names are the ids), and segment names are reported through
  `NamedNodeBackTranslation`; see the adapter doc. On graphs named by
  strings, by sparse numbers and by zero-padded numbers, `vg convert -f` gives
  the same GFA from the `.gfaz` (read in place or converted) as from the
  original GFA, up to which end an L-line is written from.
- Read mapping with Giraffe needs a GBZ and its indexes. The route is
  `vg gbwt -x graph.gfaz -E --gbz-format -g graph.gbz`, which now reads the
  container in place.
- Not compared here: vg on the plain GFA, XG at whole-genome scale, and the
  commands that have not been run on a graph read in place (`call`, `pack`,
  the distance index, `autoindex`, Giraffe).

## vg's own tests

The seven vg test files that cover `.gfaz` input (`03_vg_view`, `05_vg_find`,
`10_vg_stats`, `11_vg_paths`, `45_vg_sort`, `48_vg_convert`, `53_clip`) pass
with the native graph. `vg stats -F` now reports `GFAZ` for a container read in
place; the test was updated for that. The full suite (1,897 scripted tests and
1,069 unit cases) passes on `gfaz_native_sync`.

## Reproducing

```bash
# in the vg checkout, branch gfaz_native_sync
make -j16
bin/vg stats -z -l graph.gfaz                  # native
VG_GFAZ_CONVERT=1 bin/vg stats -z -l graph.gfaz  # previous behaviour
```
