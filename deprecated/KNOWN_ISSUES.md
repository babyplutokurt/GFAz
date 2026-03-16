# Known Issues

## Issue 1: L-line Overlap CIGAR Truncation

**Status**: Known limitation (not fixed)

**Description**: The Link overlap field only stores a single number and operation, but GFA spec allows full CIGAR strings.

**Current storage**:
```cpp
struct LinkData {
  std::vector<uint32_t> overlap_nums;  // Only ONE number
  std::vector<char> overlap_ops;       // Only ONE operation
};
```

**GFA Spec**: Overlap can be `*` or full CIGAR like `([0-9]+[MIDNSHPX=])+`

**Example**:
- Input: `L 1 + 2 - 100M50I20D`
- Stored: `num=100, op='M'`
- Output: `L 1 + 2 - 100M`
- Lost: `50I20D`

**Impact**: Complex CIGAR strings are truncated to first operation only.

**Workaround**: Most GFA files use simple overlaps like `0M` or `*`. If your files have complex CIGARs, this needs to be fixed by storing overlap as a full string.

---

## Issue 2: P-line Separator Loss (GFA v1.2)

**Status**: Known limitation (not fixed)

**Description**: GFA v1.2 uses `;` separator for Jump connections and `,` for Link connections in paths. Current implementation only handles `,`.

**Current parsing** (gfa_parser.cpp:513):
```cpp
if (i == nodes_str.size() || nodes_str[i] == ',') {
```

**Current writing** (gfa_writer.cpp:163):
```cpp
if (n > 0)
    line += ',';
```

**GFA v1.2 Example**:
```
P    first    11+,12-    *       # uses Link connection
P    second   11+;12-    *       # uses Jump connection
```

**Impact**:
- Paths using `;` separators may be parsed incorrectly
- All separators become `,` on output, losing Link vs Jump distinction

**Workaround**: Only affects GFA v1.2 files using Jump connections in paths. GFA v1.0/v1.1 files are unaffected.

---