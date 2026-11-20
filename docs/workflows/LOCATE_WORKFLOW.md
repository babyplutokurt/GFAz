# `gfaz locate` and `gfaz index`

`gfaz locate` answers "where does every haplotype visit this node?" directly
from a `.gfaz`. `gfaz index` saves the path index that `locate` and
`extract-region` use, so repeated queries skip rebuilding it.

```bash
gfaz index  -i graph.gfaz                 # writes graph.gfaz.pidx
gfaz locate -i graph.gfaz -n 2000000 -n 2000001
```

## Output

Tab-delimited, one line per visit, in the order the nodes were given and then
by haplotype and position:

```text
#node    path                          strand  step    start     end
2000000  HG00673#1#JAHBBZ010000003.1   -       557867  34430162  34430190
```

- `path`: the P-line name, or `sample#hap#seq` of a W-line.
- `step`: 0-based index of the visit in its P- or W-line.
- `[start, end)`: the bases the node covers on that sequence (for a W-line,
  including the walk's own start offset).

`-n` takes a segment's name as the GFA writes it (`2000000`, `s17`, `utg000042l`),
and the `#node` column repeats it.

## The path index

A container stores each haplotype as a compressed sequence that can only be
decoded from its start. The index cuts every haplotype into blocks of about
4096 steps and records, per block, where it starts (in encoded elements, steps
and bases), the decoder state at that point, and the smallest and largest node
id it visits. Two things follow:

- **Random access.** A step or base position is reached by decoding one block.
- **Node lookup.** Only blocks whose id range covers a node can visit it. Graph
  builders number nodes roughly along the genome, so that is typically one block
  per haplotype. On a graph with scattered ids more blocks qualify and lookups
  slow down, but stay correct.

The index is not part of the `.gfaz` format. `gfaz index` writes it to
`<input>.pidx`; `locate` and `extract-region` use that file when it exists
(`-x` names another), and build the index in memory otherwise. An index built
from a different container is rejected.

## Cost

16 threads. "Query" is the whole command, including loading the container.

| Graph | Index build | Index size | `locate`, saved index | `locate`, no index | `extract-region` 100 kb, saved index |
|---|---|---|---|---|---|
| HPRC chr6 | 0.4 s | 4 MB | 0.15 s | 0.5 s | 0.36 s |
| HPRC v1.1 | 5.9 s | 47 MB | 2.2 s | 5.7 s | 7.4 s |

With a saved index the remaining time is decoding whole columns of the
container (rulebook and traversals for both commands; segment sequences and
links for `extract-region`). The lookup itself takes milliseconds, which is
what a long-running process that loads the container once would pay per query.

## Validation

`tests/regression/test_locate.py` compares `locate` with a brute-force scan of
the decompressed GFA, with and without a saved index and across block sizes,
and checks that a stale index is refused.
