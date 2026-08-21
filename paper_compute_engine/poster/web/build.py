#!/usr/bin/env python3
"""Generate the GFAz poster (42 x 45 in) as a single self-contained HTML file.

All numbers come from README.md (post-'Refresh CPU compression benchmarks').
Charts are emitted as inline SVG so they stay vector-sharp in the printed PDF.

    python3 build.py            -> poster.html
    node ../render.js poster.html poster.pdf
"""
import math
from pathlib import Path

OUT = Path(__file__).parent / "poster.html"

INK    = "#15181E"
MUTED  = "#6B7280"
CHERRY = "#9D2235"
TEAL   = "#0F6E70"
AMBER  = "#B57200"
LINE   = "#D9DCE1"
PAPER  = "#FBFAF8"
WASH   = "#F2F1EE"

# --------------------------------------------------------------------- data --
# ratio: gzip, Zstd, GBZ, GFAz(CPU)
RATIO = [
    ("chr1",       5.59, 7.54,  9.52, 35.4),
    ("chr6",       5.04, 6.99, 19.2,  35.4),
    ("E. coli",    4.69, 5.67,  5.58, 18.4),
    ("HPRC v1.1",  4.02, 5.32, 14.0,  22.4),
    ("HPRC v2.0",  4.19, 6.49, 66.8,  83.8),
    ("HPRC v2.1",  4.19, 6.43, 64.2,  82.8),
]

# throughput MiB/s: (label, Zstd, GBZ, GFAz CPU, GFAz GPU)  None = n/a
COMP = [
    ("chr1",      2178, 12.1, 1320, 2754),
    ("chr6",      1712, 10.7, 1355, 3791),
    ("E. coli",   1356, 20.2,  226,  678),
    ("HPRC v1.1", 1657, 84.5,  291, 4843),
    ("HPRC v2.0", 1514, 130,   555, None),
    ("HPRC v2.1", 1540, 136,   538, None),
]
DECOMP = [
    ("chr1",      1618, 284, 2307, 8124),
    ("chr6",      1515, 281, 2943, 7230),
    ("E. coli",   1258, 197,  834, 1430),
    ("HPRC v1.1", 1234, 650, 2292, 9435),
    ("HPRC v2.0", 1240, 648, 5426, None),
    ("HPRC v2.1", 1241, 652, 5325, None),
]

# compute engine: (command, baseline tool, graph, baseline, gfaz, speedup, mem)
CE = [
    ("deconstruct", "vg deconstruct", "chr1",      "805 s",     "11.3 s",   71,  3.7),
    ("deconstruct", "vg deconstruct", "HGSVC3",    "132.9 min", "8.7 min",  15,  7.6),
    ("growth",      "Panacus",        "chr1",      "18.3 s",    "0.74 s",   25,  9.3),
    ("growth",      "Panacus",        "HPRC v2.0", "245 min",   "39.8 s",  369, 25.0),
    ("pav",         "odgi pav",       "chr1",      "3499 s",    "11.7 s",  299,  3.4),
    ("pav",         "odgi pav",       "chr6",      "4759 s",    "7.8 s",   613,  3.0),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------ chart helpers --
def dot_plot(rows, series, title_vals=True, log=False, width=1600, row_h=68,
             pad_l=230, pad_r=250, pad_t=64, pad_b=52, fmt="{:g}×", ticks=None):
    """Cleveland dot plot: one row per dataset, one dot per tool.

    Far more compact than grouped bars for 3-4 series, and the eye reads the
    GFAz gap directly as horizontal distance.
    """
    names = [x[0] for x in series]
    cols  = [x[1] for x in series]
    plot_w = width - pad_l - pad_r
    height = pad_t + len(rows) * row_h + pad_b
    vals = [v for r in rows for v in r[1:] if v]
    vmin, vmax = min(vals), max(vals)

    def x(v):
        if log:
            lo, hi = math.log10(vmin), math.log10(vmax)
            return pad_l + plot_w * (math.log10(v) - lo) / (hi - lo)
        return pad_l + plot_w * (v - 0) / vmax

    p = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']

    # legend
    lx = pad_l
    for name, col in zip(names, cols):
        p.append(f'<circle cx="{lx+9}" cy="{pad_t-40}" r="11" fill="{col}"/>')
        p.append(f'<text x="{lx+28}" y="{pad_t-31}" class="lg">{esc(name)}</text>')
        lx += 46 + len(name) * 15.5

    # gridlines
    if ticks:
        for t in ticks:
            gx = x(t)
            p.append(f'<line x1="{gx:.1f}" y1="{pad_t-8}" x2="{gx:.1f}" '
                     f'y2="{pad_t+len(rows)*row_h}" stroke="{LINE}" stroke-width="2"/>')
            p.append(f'<text x="{gx:.1f}" y="{pad_t+len(rows)*row_h+38}" class="gl">'
                     f'{fmt.format(t)}</text>')

    for i, row in enumerate(rows):
        cy = pad_t + i * row_h + row_h / 2
        p.append(f'<text x="{pad_l-26}" y="{cy+11}" class="ylab">{esc(row[0])}</text>')
        pts = [(v, c) for v, c in zip(row[1:], cols) if v]
        if pts:
            xs = [x(v) for v, _ in pts]
            p.append(f'<line x1="{min(xs):.1f}" y1="{cy:.1f}" x2="{max(xs):.1f}" '
                     f'y2="{cy:.1f}" stroke="{LINE}" stroke-width="4" stroke-linecap="round"/>')
        for j, v in enumerate(row[1:]):
            if not v:
                continue
            last = (j == len(cols) - 1)
            r = 17 if last else 12
            p.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="{r}" fill="{cols[j]}" '
                     f'{"stroke=\"#fff\" stroke-width=\"3\"" if last else ""}/>')
        # value label for the winning series, at a fixed right-hand column
        gv = next((v for v in reversed(row[1:]) if v), None)
        if title_vals and gv:
            p.append(f'<text x="{width-pad_r+34}" y="{cy+13}" class="val hi">'
                     f'{fmt.format(gv)}</text>')
    p.append("</svg>")
    return "".join(p)


def speedup_chart(width=1500, row_h=76, pad_l=330, pad_r=300, pad_t=22):
    """Log-scale speedup bars for the compute engine."""
    height = pad_t + len(CE) * row_h + 52
    plot_w = width - pad_l - pad_r
    hi, lo = math.log10(700), math.log10(10)

    def x(v):
        return plot_w * (math.log10(v) - lo) / (hi - lo)

    p = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for gv in (10, 100, 700):
        gx = pad_l + x(gv)
        p.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" '
                 f'y2="{pad_t+len(CE)*row_h}" stroke="{LINE}" stroke-width="2"/>')
        p.append(f'<text x="{gx:.1f}" y="{pad_t+len(CE)*row_h+38}" class="gl">{gv}×</text>')
    prev = None
    for i, (cmd, base, graph, bt, gt, sp, mem) in enumerate(CE):
        y = pad_t + i * row_h
        col = {"deconstruct": CHERRY, "growth": TEAL, "pav": AMBER}[cmd]
        if cmd != prev:
            p.append(f'<text x="0" y="{y+34}" class="cmd" fill="{col}">{esc(cmd)}</text>')
            p.append(f'<text x="0" y="{y+62}" class="vs">vs. {esc(base)}</text>')
            prev = cmd
        p.append(f'<text x="{pad_l-18}" y="{y+row_h/2+10}" class="glab">{esc(graph)}</text>')
        w = max(x(sp), 3)
        bh = row_h - 26
        p.append(f'<rect x="{pad_l}" y="{y+11}" width="{w:.1f}" height="{bh}" rx="4" '
                 f'fill="{col}" fill-opacity="0.9"/>')
        p.append(f'<text x="{pad_l+w+16}" y="{y+11+bh*0.5:.1f}" class="sp">{sp}×</text>')
        p.append(f'<text x="{pad_l+w+16}" y="{y+11+bh*0.95:.1f}" class="spsub">'
                 f'{bt} → {gt}</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- diagrams ---
def pipeline_svg():
    """Compression / decompression pipeline, redrawn clean (and typo-free)."""
    def stage(x, y, w, h, title, sub, fill, stroke, tcol):
        s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="2.5"/>')
        s += f'<text x="{x+w/2}" y="{y+(h/2 if not sub else h/2-9)}" class="st" fill="{tcol}">{esc(title)}</text>'
        if sub:
            s += f'<text x="{x+w/2}" y="{y+h/2+22}" class="stsub">{esc(sub)}</text>'
        return s

    def arrow(x1, y, x2):
        return (f'<line x1="{x1}" y1="{y}" x2="{x2-13}" y2="{y}" stroke="{INK}" '
                f'stroke-width="3"/><path d="M{x2} {y} L{x2-15} {y-8} L{x2-15} {y+8} Z" fill="{INK}"/>')

    W, H = 1000, 556
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">',
         '<g transform="translate(0,34)">']

    # --- compression row
    p.append(f'<text x="0" y="24" class="rowlab" fill="{CHERRY}">COMPRESS</text>')
    p.append(stage(0, 44, 132, 92, ".gfa", "369 GB", "#fff", LINE, INK))
    p.append(arrow(132, 90, 176))
    p.append(stage(176, 44, 168, 92, "parse", "GfaGraph", WASH, LINE, INK))
    p.append(arrow(344, 90, 388))
    # three parallel columns
    p.append(f'<rect x="388" y="18" width="452" height="150" rx="11" fill="none" '
             f'stroke="{LINE}" stroke-width="2.5" stroke-dasharray="8 7"/>')
    p.append(f'<text x="614" y="10" class="grp">columnar, type-specific</text>')
    p.append(stage(404, 34, 130, 52, "segments", "", "#fff", LINE, INK))
    p.append(stage(404, 96, 130, 52, "links / opt.", "", "#fff", LINE, INK))
    p.append(stage(552, 34, 272, 114, "traversals", "iterative 2-mer grammar",
                   "#FBEFF1", CHERRY, CHERRY))
    p.append(f'<text x="688" y="140" class="badge">&gt;90% of the file</text>')
    p.append(arrow(840, 90, 884))
    p.append(stage(884, 44, 116, 92, "Zstd", "", WASH, LINE, INK))

    # --- container
    cy = 218
    p.append(f'<rect x="330" y="{cy}" width="340" height="76" rx="10" fill="{INK}"/>')
    p.append(f'<text x="500" y="{cy+34}" class="ctr">.gfaz container</text>')
    p.append(f'<text x="500" y="{cy+60}" class="ctrsub">4.5 GB &#183; 84&#215; smaller &#183; lossless</text>')
    p.append(f'<line x1="500" y1="152" x2="500" y2="{cy-4}" stroke="{INK}" stroke-width="3"/>')
    p.append(f'<path d="M500 {cy} L492 {cy-16} L508 {cy-16} Z" fill="{INK}"/>')
    p.append(f'<line x1="500" y1="{cy+76}" x2="500" y2="342" stroke="{INK}" stroke-width="3"/>')
    p.append(f'<path d="M500 358 L492 342 L508 342 Z" fill="{INK}"/>')

    # --- decompression row
    dy = 388
    p.append(f'<text x="0" y="{dy-16}" class="rowlab" fill="{TEAL}">DECOMPRESS &#8212; streaming</text>')
    p.append(stage(0, dy, 152, 92, "de-Zstd", "", WASH, LINE, INK))
    p.append(arrow(152, dy + 46, 196))
    p.append(stage(196, dy, 244, 92, "expand rules", "O(1) array lookup",
                   "#E9F4F4", TEAL, TEAL))
    p.append(arrow(440, dy + 46, 484))
    p.append(stage(484, dy, 250, 92, "restore records", "", "#fff", LINE, INK))
    p.append(arrow(734, dy + 46, 778))
    p.append(stage(778, dy, 222, 92, ".gfa", "5.3 GB/s", "#fff", LINE, INK))
    p.append("</g></svg>")
    return "".join(p)


def stream_svg():
    """Materialize-then-fold vs. stream-the-compressed-columns."""
    W, H = 1130, 342
    def box(x, y, w, h, t, sub, fill, stroke, tcol=INK):
        s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="2.5"/>'
             f'<text x="{x+w/2}" y="{y+h/2-6}" class="st" fill="{tcol}">{esc(t)}</text>')
        if sub:
            s += f'<text x="{x+w/2}" y="{y+h/2+26}" class="stsub">{esc(sub)}</text>'
        return s

    def arrow(x1, y, x2):
        return (f'<line x1="{x1}" y1="{y}" x2="{x2-13}" y2="{y}" stroke="{INK}" '
                f'stroke-width="3"/><path d="M{x2} {y} L{x2-15} {y-8} L{x2-15} {y+8} Z" fill="{INK}"/>')

    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    p.append(f'<text x="0" y="20" class="rowlab" fill="{MUTED}">TODAY &#8212; materialize, then fold</text>')
    p.append(box(0, 38, 150, 92, "GFA", "369 GB", "#fff", LINE))
    p.append(arrow(150, 84, 194))
    p.append(box(194, 38, 470, 92, "whole graph + snarl / item index",
                 "330–400 GB resident", "#FBEFF1", CHERRY, CHERRY))
    p.append(arrow(664, 84, 708))
    p.append(box(708, 38, 150, 92, "statistic", "MB", "#E9F4F4", TEAL, TEAL))

    p.append(f'<text x="0" y="212" class="rowlab" fill="{CHERRY}">GFAz &#8212; stream the compressed columns</text>')
    p.append(box(0, 230, 150, 92, ".gfaz", "4.5 GB", "#FBEFF1", CHERRY, CHERRY))
    p.append(arrow(150, 276, 194))
    p.append(box(194, 230, 470, 92, "grammar-compressed traversal stream",
                 "+ compact accumulator", "#fff", LINE))
    p.append(arrow(664, 276, 708))
    p.append(box(708, 230, 150, 92, "statistic", "MB", "#E9F4F4", TEAL, TEAL))
    p.append(f'<text x="878" y="270" class="win">peak RSS</text>')
    p.append(f'<text x="878" y="298" class="win">below the</text>')
    p.append(f'<text x="878" y="326" class="win">original GFA</text>')
    p.append("</svg>")
    return "".join(p)


def snarl_svg():
    """deconstruct worked example: two chained snarls -> two VCF records."""
    W, H = 1120, 288
    def seg(x, y, name, dna, fill, stroke):
        return (f'<rect x="{x}" y="{y}" width="84" height="84" rx="10" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="3"/>'
                f'<text x="{x+42}" y="{y+34}" class="sg">{name}</text>'
                f'<text x="{x+42}" y="{y+66}" class="sgd">{dna}</text>')

    def edge(x1, y1, x2, y2, col, dash=""):
        d = f' stroke-dasharray="11 9"' if dash else ""
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
                f'stroke-width="4"{d} marker-end="url(#ah{"t" if dash else "c"})"/>')

    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">',
         '<defs>',
         f'<marker id="ahc" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
         f'<path d="M0 0 L9 4.5 L0 9 Z" fill="{CHERRY}"/></marker>',
         f'<marker id="aht" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
         f'<path d="M0 0 L9 4.5 L0 9 Z" fill="{TEAL}"/></marker>',
         '</defs>', '<g transform="translate(0,30)">']
    # snarl brackets
    p.append(f'<rect x="176" y="16" width="120" height="216" rx="12" fill="none" '
             f'stroke="{MUTED}" stroke-width="2.5" stroke-dasharray="6 8"/>')
    p.append(f'<text x="236" y="8" class="sn">snarl [A,D]</text>')
    p.append(f'<rect x="500" y="16" width="120" height="112" rx="12" fill="none" '
             f'stroke="{MUTED}" stroke-width="2.5" stroke-dasharray="6 8"/>')
    p.append(f'<text x="560" y="8" class="sn">snarl [D,G]</text>')
    # edges
    p.append(edge(96, 152, 176, 80, TEAL, True))     # A->B
    p.append(edge(96, 166, 176, 190, CHERRY))        # A->C
    p.append(edge(296, 80, 376, 152, TEAL, True))    # B->D
    p.append(edge(296, 190, 376, 166, CHERRY))       # C->D
    p.append(edge(464, 152, 500, 80, CHERRY))        # D->E
    p.append(edge(620, 80, 700, 152, CHERRY))        # E->G
    p.append(edge(464, 178, 700, 178, TEAL, True))   # D->G
    # nodes
    p.append(seg(12, 118, "A", "GAT", WASH, LINE))
    p.append(seg(196, 36, "B", "T", "#E9F4F4", TEAL))
    p.append(seg(196, 148, "C", "C", "#FBEFF1", CHERRY))
    p.append(seg(380, 118, "D", "GA", WASH, LINE))
    p.append(seg(520, 36, "E", "CC", "#FBEFF1", CHERRY))
    p.append(seg(704, 118, "G", "TT", WASH, LINE))
    # legend
    p.append(f'<line x1="800" y1="46" x2="840" y2="46" stroke="{CHERRY}" stroke-width="4"/>')
    p.append(f'<text x="850" y="53" class="lg2">reference / h1</text>')
    p.append(f'<line x1="800" y1="82" x2="840" y2="82" stroke="{TEAL}" stroke-width="4" stroke-dasharray="11 9"/>')
    p.append(f'<text x="850" y="89" class="lg2">h2</text>')
    p.append("</g></svg>")
    return "".join(p)


# -------------------------------------------------------------------- build --
ratio_chart = dot_plot(
    RATIO,
    [("gzip", "#C6CAD1"), ("Zstd", "#98A0AC"), ("GBZ", "#59636F"), ("GFAz", CHERRY)],
    fmt="{:g}×", ticks=[10, 25, 50, 84])

decomp_chart = dot_plot(
    [(r[0], r[1], r[2], r[3], r[4]) for r in DECOMP],
    [("Zstd", "#98A0AC"), ("GBZ", "#59636F"), ("GFAz CPU", CHERRY), ("GFAz GPU", TEAL)],
    fmt="{:g}", log=True, ticks=[200, 1000, 5000, 9435], pad_r=270)

ce_rows = "".join(
    f'<tr><td class="mono">{esc(c)}</td><td>{esc(g)}</td>'
    f'<td class="num dim">{esc(b)}</td><td class="num strong">{esc(gt)}</td>'
    f'<td class="num hi">{s}×</td><td class="num">{m:g}×</td></tr>'
    for c, _, g, b, gt, s, m in CE)

CMDS = [("deconstruct", "vg deconstruct", "per-site allele table"),
        ("growth", "Panacus", "coverage → histogram"),
        ("depth", "odgi depth", "per-node counters"),
        ("stats", "odgi stats", "metadata totals"),
        ("pav", "odgi pav", "node→groups CSR"),
        ("similarity", "odgi similarity", "node→groups CSR")]
cmd_rows = "".join(
    f'<tr><td class="mono strong">{esc(a)}</td><td class="mono dim">{esc(b)}</td>'
    f'<td class="dim">{esc(c)}</td></tr>' for a, b, c in CMDS)

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>GFAz Poster</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  @page {{ size: 42in 45in; margin: 0; }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink:{INK}; --muted:{MUTED}; --cherry:{CHERRY}; --teal:{TEAL};
    --line:{LINE}; --paper:{PAPER}; --wash:{WASH};
  }}
  html {{ font-size: 16px; }}
  body {{
    width: 42in; height: 45in; overflow: hidden;
    background: var(--paper); color: var(--ink);
    font-family: Inter, system-ui, sans-serif;
    font-size: 25pt; line-height: 1.42;
    -webkit-font-smoothing: antialiased;
    padding: 0.62in 0.72in 0.5in;
    display: flex; flex-direction: column;
  }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.92em; }}
  .dim {{ color: var(--muted); }}
  .strong {{ font-weight: 600; }}
  b, strong {{ font-weight: 650; }}

  /* ---------- header ---------- */
  header {{ border-bottom: 5px solid var(--ink); padding-bottom: 0.26in; }}
  h1 {{
    font-family: Archivo, sans-serif; font-weight: 800; font-size: 92pt;
    line-height: 1.02; letter-spacing: -0.022em; max-width: 30in;
  }}
  h1 em {{ font-style: normal; color: var(--cherry); }}
  .byline {{ display: flex; justify-content: space-between; align-items: flex-end;
             margin-top: 0.24in; gap: 1in; }}
  .authors {{ font-size: 30pt; font-weight: 550; letter-spacing: -0.01em; }}
  .affil {{ font-size: 24pt; color: var(--muted); margin-top: 0.06in; }}
  .venue {{ text-align: right; font-size: 23pt; color: var(--muted); line-height: 1.5;
            white-space: nowrap; }}
  .venue b {{ color: var(--ink); display: block; font-size: 26pt; }}

  /* ---------- stat band ---------- */
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr);
            border-bottom: 2px solid var(--line); }}
  .stat {{ padding: 0.34in 0 0.32in; text-align: center; border-left: 2px solid var(--line); }}
  .stat:first-child {{ border-left: 0; }}
  .stat .n {{ font-family: Archivo, sans-serif; font-weight: 700; font-size: 78pt;
              line-height: 1; letter-spacing: -0.035em; color: var(--cherry); }}
  .stat .k {{ font-size: 25pt; font-weight: 600; margin-top: 0.1in; }}
  .stat .s {{ font-size: 21pt; color: var(--muted); margin-top: 0.03in; }}

  /* ---------- columns ---------- */
  main {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.72in;
          flex: 1; min-height: 0; padding-top: 0.4in; }}
  .col {{ display: flex; flex-direction: column; gap: 0.3in; min-width: 0; }}
  .parthead {{ display: flex; align-items: baseline; gap: 0.2in;
               border-bottom: 3px solid var(--ink); padding-bottom: 0.12in; }}
  .parthead .no {{ font-family: Archivo, sans-serif; font-weight: 800; font-size: 40pt;
                   color: var(--cherry); letter-spacing: -0.02em; }}
  .parthead h2 {{ font-family: Archivo, sans-serif; font-weight: 700; font-size: 40pt;
                  letter-spacing: -0.015em; }}
  .parthead .tag {{ margin-left: auto; font-size: 21pt; color: var(--muted); }}

  .lbl {{ font-size: 19pt; font-weight: 650; letter-spacing: 0.1em;
          text-transform: uppercase; color: var(--muted); margin-bottom: 0.12in; }}
  .note {{ font-size: 21.5pt; color: var(--muted); line-height: 1.4; }}
  .note b {{ color: var(--ink); }}

  svg.chart {{ width: 100%; height: auto; display: block; }}
  svg text {{ font-family: Inter, sans-serif; }}
  .ylab {{ font-size: 25px; font-weight: 600; text-anchor: end; fill: {INK}; }}
  .val  {{ font-size: 23px; font-weight: 500; fill: {MUTED}; }}
  .val.hi {{ font-weight: 700; fill: {CHERRY}; font-size: 26px; }}
  .lg   {{ font-size: 24px; font-weight: 600; fill: {INK}; }}
  .lg2  {{ font-size: 25px; font-weight: 550; fill: {MUTED}; }}
  .na   {{ font-size: 21px; fill: #B6BCC5; font-style: italic; }}
  .gl   {{ font-size: 23px; fill: {MUTED}; text-anchor: middle; }}
  .cmd  {{ font-size: 32px; font-weight: 700; font-family: "JetBrains Mono", monospace; }}
  .vs   {{ font-size: 24px; fill: {MUTED}; }}
  .glab {{ font-size: 27px; font-weight: 600; text-anchor: end; fill: {INK}; }}
  .sp   {{ font-size: 40px; font-weight: 800; fill: {INK}; font-family: Archivo, sans-serif; }}
  .spsub{{ font-size: 23px; fill: {MUTED}; }}
  .st   {{ font-size: 27px; font-weight: 650; text-anchor: middle; }}
  .stsub{{ font-size: 22px; fill: {MUTED}; text-anchor: middle; }}
  .rowlab {{ font-size: 23px; font-weight: 700; letter-spacing: 0.12em; }}
  .grp  {{ font-size: 22px; fill: {MUTED}; text-anchor: middle; font-style: italic; }}
  .badge {{ font-size: 21px; fill: {CHERRY}; text-anchor: middle; font-weight: 600; }}
  .ctr  {{ font-size: 34px; font-weight: 700; fill: #fff; text-anchor: middle;
           font-family: Archivo, sans-serif; }}
  .ctrsub {{ font-size: 23px; fill: #C9CDD4; text-anchor: middle; }}
  .win  {{ font-size: 24px; fill: {TEAL}; font-weight: 650; }}
  .sg   {{ font-size: 30px; font-weight: 700; text-anchor: middle;
           font-family: "JetBrains Mono", monospace; }}
  .sgd  {{ font-size: 24px; fill: {MUTED}; text-anchor: middle;
           font-family: "JetBrains Mono", monospace; }}
  .sn   {{ font-size: 23px; fill: {MUTED}; text-anchor: middle; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 22.5pt; }}
  th {{ font-size: 19pt; font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--muted); text-align: left; padding: 0 0.1in 0.1in; }}
  td {{ padding: 0.105in 0.1in; border-top: 2px solid var(--line); }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.hi {{ color: var(--cherry); font-weight: 700; }}

  .callout {{ border-left: 7px solid var(--cherry); padding: 0.02in 0 0.02in 0.28in; }}
  .callout .big {{ font-family: Archivo, sans-serif; font-weight: 700; font-size: 33pt;
                   line-height: 1.2; letter-spacing: -0.012em; }}
  .callout p {{ font-size: 22.5pt; color: var(--muted); margin-top: 0.12in; line-height: 1.4; }}
  .spacer {{ flex: 1; }}

  footer {{ display: flex; align-items: center; gap: 0.5in;
            border-top: 5px solid var(--ink); padding-top: 0.24in; margin-top: 0.34in; }}
  footer .txt {{ flex: 1; font-size: 21.5pt; line-height: 1.45; color: var(--muted); }}
  footer .txt b {{ color: var(--ink); }}
  footer img {{ width: 1.45in; height: 1.45in; display: block; }}
  .qr {{ text-align: center; font-size: 17pt; color: var(--muted); }}
</style></head>
<body>

<header>
  <h1>GFAz: order-of-magnitude pangenome analytics by <em>computing over compression</em></h1>
  <div class="byline">
    <div>
      <div class="authors">Taolue Yang &nbsp;&#183;&nbsp; Youyuan Liu &nbsp;&#183;&nbsp; Bo Jiang &nbsp;&#183;&nbsp; Xinghua Shi &nbsp;&#183;&nbsp; Sian Jin</div>
      <div class="affil">Department of Computer &amp; Information Sciences, Temple University</div>
    </div>
    <div class="venue"><b>T2T Face-to-Face 2026</b>UC Santa Cruz &#183; to appear at ICS&#8202;&#39;26</div>
  </div>
</header>

<section class="stats">
  <div class="stat"><div class="n">84&#215;</div><div class="k">smaller</div><div class="s">369 GB &#8594; 4.5 GB</div></div>
  <div class="stat"><div class="n">5.3<span style="font-size:44pt"> GB/s</span></div><div class="k">decompression</div><div class="s">CPU, 32 threads</div></div>
  <div class="stat"><div class="n">369&#215;</div><div class="k">faster analytics</div><div class="s">245 min &#8594; 40 s</div></div>
  <div class="stat"><div class="n">99.99<span style="font-size:44pt">%</span></div><div class="k">VCF concordance</div><div class="s">vs. vg deconstruct</div></div>
</section>

<main>
  <!-- ============================ COLUMN 1 ============================ -->
  <div class="col">
    <div class="parthead"><span class="no">01</span><h2>Compression engine</h2>
      <span class="tag">lossless &#183; full GFA semantics &#183; CPU + GPU</span></div>

    <div>{pipeline_svg()}</div>

    <div>
      <div class="lbl">Compression ratio &#8212; higher is better</div>
      {ratio_chart}
    </div>

    <div>
      <div class="lbl">Decompression throughput, MiB/s &#8212; log scale</div>
      {decomp_chart}
      <div class="note" style="margin-top:0.14in">Compression reaches <b>1355 MiB/s</b> on
      CPU (32 threads, end-to-end CLI wall time) and <b>4843 MiB/s</b> on GPU.
      GFAz wins ratio on <b>every</b> dataset while decompressing faster than Zstd
      &#8212; and unlike SQZ and GBZ it preserves the complete GFA.</div>
    </div>

    <div>
      <div class="lbl">GFAz CPU, end to end</div>
      <table>
        <tr><th>Graph</th><th class="num">GFA</th><th class="num">Ratio</th>
            <th class="num">Compress</th><th class="num">Decompress</th></tr>
        <tr><td>chr1 (PGGB)</td><td class="num dim">7.1 GB</td><td class="num hi">35.4×</td>
            <td class="num">1320 MiB/s</td><td class="num">2307 MiB/s</td></tr>
        <tr><td>HPRC v1.1</td><td class="num dim">48 GB</td><td class="num hi">22.4×</td>
            <td class="num">291 MiB/s</td><td class="num">2292 MiB/s</td></tr>
        <tr><td>HPRC v2.1</td><td class="num dim">369 GB</td><td class="num hi">82.8×</td>
            <td class="num">538 MiB/s</td><td class="num">5325 MiB/s</td></tr>
      </table>
    </div>

    <div class="spacer"></div>

    <div class="callout">
      <div class="big">Best ratio on every graph tested &#8212; without giving up speed.</div>
      <p>Traversals are &gt;90% of a GFA and collapse under iterative 2-mer grammar
      compression; everything else gets columnar, type-specific encoding. Both
      directions are linear-time and fully parallel.</p>
    </div>
  </div>

  <!-- ============================ COLUMN 2 ============================ -->
  <div class="col">
    <div class="parthead"><span class="no">02</span><h2>Compute engine</h2>
      <span class="tag">run the analysis <em>on</em> the container</span></div>

    <div>{stream_svg()}</div>

    <div>
      <div class="lbl">Six analyses, one streaming decoder</div>
      <table>
        <tr><th>GFAz command</th><th>reproduces</th><th>accumulator</th></tr>
        {cmd_rows}
      </table>
    </div>

    <div>
      <div class="lbl">Speedup over the standard tool &#8212; log scale</div>
      {speedup_chart()}
    </div>

    <div>
      <div class="lbl">GFA &#8594; VCF with no graph in memory</div>
      {snarl_svg()}
      <table style="margin-top:0.12in">
        <tr><th class="mono">#CHROM</th><th class="mono num">POS</th><th class="mono">REF</th>
            <th class="mono">ALT</th><th class="mono">INFO</th><th class="mono num">h1 h2</th></tr>
        <tr><td class="mono">chr1</td><td class="mono num">4</td><td class="mono">C</td>
            <td class="mono">T</td><td class="mono dim">AC=1;AN=2;AF=0.5</td>
            <td class="mono num">0&nbsp;&nbsp;&nbsp;1</td></tr>
        <tr><td class="mono">chr1</td><td class="mono num">6</td><td class="mono">ACC</td>
            <td class="mono">A</td><td class="mono dim">AC=1;AN=2;AF=0.5</td>
            <td class="mono num">0&nbsp;&nbsp;&nbsp;1</td></tr>
      </table>
      <div class="note" style="margin-top:0.12in"><b>Phase 1</b> recovers snarls from
      topology alone (biconnected decomposition, O(V+E), decodes zero haplotypes).
      <b>Phase 2</b> streams each haplotype once through a boundary state machine;
      a reverse boundary pair normalizes inversions.</div>
    </div>

    <div class="spacer"></div>

    <div class="callout">
      <div class="big">A pangenome graph does not have to be unpacked to be queried.</div>
      <p>Outputs match the baselines: <b>growth</b> reproduces Panacus&#8217; curve exactly,
      <b>deconstruct</b> agrees with vg to within 0.2% of records, <b>pav</b> matrices are
      structurally identical to odgi&#8217;s. On the 358/369 GB HPRC graphs vg and odgi
      cannot load the input at all. Open: whole-genome <b>pav</b> still exceeds memory
      there &#8212; its node-set index is not yet compressed.</p>
    </div>
  </div>
</main>

<footer>
  <div class="qr"><img src="qr-code.png" alt="repository QR code"><div>code</div></div>
  <div class="txt">
    <b>github.com/babyplutokurt/GFAz</b> &nbsp;&#183;&nbsp; <b>taolue.yang@temple.edu</b><br>
    Yang, Liu, Jiang, Shi, Jin. <i>GFAz: Order-of-Magnitude Pangenome Analytics by
    Computing over Compression.</i> ICS&#8202;&#39;26. doi:10.1145/3797905.3807870<br>
    We thank the Human Pangenome Reference Consortium (BioProject PRJNA730823) and its
    funder, NHGRI. Supported in part by NIH award R01GM157443.
  </div>
  <div class="qr"><img src="qr-paper.png" alt="paper DOI QR code"><div>paper</div></div>
</footer>

</body></html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({len(HTML)/1024:.0f} KB)")
