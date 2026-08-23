#!/usr/bin/env python3
"""Generate the CPU-focused 42 x 45 inch GFAz conference poster."""

from pathlib import Path
import math


OUT = Path(__file__).parent / "poster.html"

DATA = [
  ("chr1",      (5.59, 7.54, 9.52, 35.4),  (46.2, 2178, 12.1, 1320), (359, 1618, 284, 2307)),
  ("chr6",      (5.04, 6.99, 19.2, 35.4),  (41.0, 1712, 10.7, 1355), (348, 1515, 281, 2943)),
  ("E. coli",   (4.69, 5.67, 5.58, 18.4),  (33.3, 1356, 20.2, 226),  (310, 1258, 197, 834)),
  ("HPRC v1.1", (4.02, 5.32, 14.0, 22.4),  (36.4, 1657, 84.5, 291),  (319, 1234, 650, 2292)),
  ("HPRC v2.0", (4.19, 6.49, 66.8, 83.8),  (49.1, 1514, 130, 555),   (342, 1240, 648, 5426)),
  ("HPRC v2.1", (4.19, 6.43, 64.2, 82.8),  (48.9, 1540, 136, 538),   (343, 1241, 652, 5325)),
]
TOOLS = ["gzip", "Zstd", "GBZ", "GFAz"]
COLORS = ["#c8cdd1", "#798892", "#15908c", "#a51c38"]


def dot_plot(metric, title, subtitle, log=False, ticks=()):
  """Compact horizontal Cleveland plot with GFAz labels at right."""
  width, left, right, top, row_h, bottom = 1180, 145, 160, 125, 118, 64
  height = top + len(DATA) * row_h + bottom
  values = [v for _, ratios, comp, decomp in DATA for v in (ratios, comp, decomp)[metric]]
  vmax = max(values)
  vmin = min(values)

  def xpos(v):
    if log:
      lo, hi = math.log10(vmin), math.log10(vmax)
      return left + (width - left - right) * (math.log10(v) - lo) / (hi - lo)
    return left + (width - left - right) * v / vmax

  parts = [f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">']
  parts.append(f'<text x="0" y="32" class="plot-title">{title}</text>')
  parts.append(f'<text x="0" y="66" class="plot-sub">{subtitle}</text>')
  lx = left
  for tool, color in zip(TOOLS, COLORS):
    parts.append(f'<circle cx="{lx}" cy="94" r="9" fill="{color}"/>')
    parts.append(f'<text x="{lx+16}" y="101" class="legend">{tool}</text>')
    lx += 135
  for tick in ticks:
    x = xpos(tick)
    parts.append(f'<line x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{top+len(DATA)*row_h}" class="grid"/>')
    label = f'{tick:g}×' if metric == 0 else f'{tick:g}'
    parts.append(f'<text x="{x:.1f}" y="{height-12}" class="tick">{label}</text>')
  for i, (name, ratios, comp, decomp) in enumerate(DATA):
    vals = (ratios, comp, decomp)[metric]
    cy = top + i * row_h + row_h / 2
    xs = [xpos(v) for v in vals]
    parts.append(f'<text x="{left-18}" y="{cy+7}" class="ylabel">{name}</text>')
    parts.append(f'<line x1="{min(xs):.1f}" y1="{cy}" x2="{max(xs):.1f}" y2="{cy}" class="range"/>')
    for j, (v, x) in enumerate(zip(vals, xs)):
      radius = 15 if j == 3 else 10
      parts.append(f'<circle cx="{x:.1f}" cy="{cy}" r="{radius}" fill="{COLORS[j]}" class="dot"/>')
    label = f'{vals[3]:g}×' if metric == 0 else f'{vals[3]:,.0f}'
    parts.append(f'<text x="{width-right+22}" y="{cy+8}" class="winner">{label}</text>')
  parts.append('</svg>')
  return "".join(parts)


ratio_plot = dot_plot(0, "Compression ratio", "higher is better", ticks=(10, 25, 50, 84))
comp_plot = dot_plot(1, "Compression throughput", "MiB/s · end-to-end CPU wall time · log scale", True, (10, 100, 1000, 2000))
decomp_plot = dot_plot(2, "Decompression throughput", "MiB/s · CPU streaming direct writer · log scale", True, (200, 500, 1000, 3000, 5000))


def analysis_plot():
  rows = [("deconstruct · chr1",71,"vg"),("deconstruct · HGSVC3",15,"vg"),("growth · chr1",25,"Panacus"),("growth · HPRC v2.0",369,"Panacus"),("pav · chr1",299,"odgi"),("pav · chr6",613,"odgi")]
  width, height, left, right, top, row_h = 1180, 900, 285, 135, 120, 112
  lo, hi = math.log10(10), math.log10(700)
  xpos = lambda v: left + (width-left-right)*(math.log10(v)-lo)/(hi-lo)
  p = [f'<svg class="plot" viewBox="0 0 {width} {height}">','<text x="0" y="32" class="plot-title">Direct-analysis speedup</text>','<text x="0" y="66" class="plot-sub">16 CPU threads · log scale · baseline = 1×</text>']
  for tick in (10,100,700):
    x=xpos(tick); p += [f'<line x1="{x:.1f}" y1="{top-15}" x2="{x:.1f}" y2="{top+len(rows)*row_h}" class="grid"/>',f'<text x="{x:.1f}" y="{height-12}" class="tick">{tick}×</text>']
  colors=["#a51c38","#a51c38","#087f7b","#087f7b","#cf9221","#cf9221"]
  for i,((label,value,baseline),color) in enumerate(zip(rows,colors)):
    cy=top+i*row_h+row_h/2; x=xpos(value)
    p += [f'<text x="{left-18}" y="{cy-4}" class="ylabel">{label}</text>',f'<text x="{left-18}" y="{cy+22}" class="baseline">vs. {baseline}</text>',f'<rect x="{left}" y="{cy-18}" width="{max(4,x-left):.1f}" height="36" rx="8" fill="{color}"/>',f'<text x="{x+16:.1f}" y="{cy+9}" class="winner">{value}×</text>']
  return "".join(p)+"</svg>"


speedup_plot = analysis_plot()

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GFAz — T2T 2026</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
@page { size: 42in 45in; margin: 0; }
* { box-sizing: border-box; }
:root { --ink:#17232b; --muted:#52636e; --red:#a51c38; --teal:#087f7b; --pale:#edf3f1; --line:#d4dcda; --paper:#fffefa; --navy:#263b50; --gold:#cf9221; }
html,body { margin:0; width:42in; height:45in; background:var(--paper); color:var(--ink); font-family:Inter,sans-serif; }
.poster { width:42in; height:45in; overflow:hidden; padding:.58in .68in .4in; display:flex; flex-direction:column; gap:.27in; }
h1,h2,h3,p { margin:0; }
.header { display:grid; grid-template-columns:1fr 6.2in; align-items:end; gap:.3in; padding-bottom:.22in; border-bottom:.055in solid var(--ink); }
.eyebrow { font:700 .27in Archivo; letter-spacing:.11em; text-transform:uppercase; color:var(--red); }
h1 { font:800 .86in/.98 Archivo; letter-spacing:-.035em; margin:.07in 0 .1in; }
h1 em { color:var(--red); font-style:normal; }
.dek { font:500 .3in/1.35 Inter; color:var(--muted); }
.authors { margin-top:.13in; font:600 .25in/1.3 Inter; }
.event { text-align:right; font:700 .3in/1.35 Archivo; }.event span { display:block; color:var(--muted); font:500 .21in Inter; }
.banner { display:grid; grid-template-columns:9.6in 1fr; gap:.22in; align-items:stretch; }
.big-result { background:var(--red); color:white; border-radius:.13in; padding:.2in .3in; display:flex; align-items:center; gap:.25in; }
.big-result strong { font:800 .68in Archivo; white-space:nowrap; }.big-result span { font:600 .21in/1.3 Inter; }
.claim { background:var(--pale); border-left:.09in solid var(--teal); padding:.2in .3in; display:flex; align-items:center; font:700 .31in/1.35 Archivo; }

.columns { flex:1; min-height:0; display:grid; grid-template-columns:repeat(3,1fr); gap:.3in; }
.col { min-height:0; display:flex; flex-direction:column; gap:.28in; }
.section { border-top:.055in solid var(--ink); padding-top:.17in; }
.section h2 { font:800 .36in/1.1 Archivo; margin-bottom:.14in; }.section h2 b { color:var(--red); margin-right:.1in; }
.section p { font:500 .22in/1.48 Inter; color:#34454f; }.section p+p { margin-top:.12in; }
.lead { font-size:.25in!important; font-weight:600!important; color:var(--ink)!important; }
.accent { color:var(--red); font-weight:700; }.teal { color:var(--teal); font-weight:700; }

.problem-visual { margin:.2in 0; display:grid; grid-template-columns:1fr .65in 1fr; align-items:center; gap:.08in; }
.graph { height:2.65in; position:relative; background:#f5f6f4; border-radius:.12in; overflow:hidden; }
.node { position:absolute; width:.42in; height:.42in; border:.035in solid var(--navy); border-radius:50%; background:white; display:grid; place-items:center; font:700 .16in Archivo; z-index:2; }
.edge { position:absolute; height:.035in; background:#99a5aa; transform-origin:left center; }
.walks { height:2.65in; display:flex; flex-direction:column; justify-content:center; gap:.11in; }
.walk { height:.28in; border-radius:1in; background:linear-gradient(90deg,var(--teal) 0 25%,var(--red) 25% 65%,var(--gold) 65%); opacity:.85; }
.plus { text-align:center; font:800 .4in Archivo; color:var(--red); }
.statline { display:grid; grid-template-columns:repeat(2,1fr); gap:.12in; margin-top:.2in; }
.statline div { background:#f3f5f3; padding:.16in; }.statline strong { display:block; font:800 .3in Archivo; color:var(--red); }.statline span { font:500 .17in/1.3 Inter; color:var(--muted); }

.pipeline { margin:.2in 0; }
.pipe-row { display:grid; grid-template-columns:1fr .18in 1fr .18in 1fr; gap:.05in; align-items:center; }
.box { min-height:.9in; padding:.12in; display:flex; flex-direction:column; justify-content:center; text-align:center; background:#f4f5f3; border:.02in solid var(--line); border-radius:.1in; }
.box strong { font:700 .2in Archivo; }.box span { font:500 .15in/1.25 Inter; color:var(--muted); margin-top:.03in; }.box.hot { background:#f8e9ed; border-color:#d899a7; }.arr { text-align:center; color:var(--red); font:800 .24in Archivo; }
.grammar { margin:.17in 0; padding:.2in; color:white; background:var(--navy); border-radius:.12in; font:600 .18in/1.55 'JetBrains Mono'; }
.grammar mark { background:var(--red); color:white; padding:.04in .08in; border-radius:.05in; }.grammar small { display:block; color:#cbd5dc; font:500 .15in/1.35 Inter; margin-top:.09in; }
.bullets { display:grid; gap:.1in; }.bullet { padding-left:.18in; border-left:.055in solid var(--teal); }.bullet strong { display:block; font:700 .2in Archivo; }.bullet span { font:500 .17in/1.35 Inter; color:var(--muted); }

.plot-wrap { margin-top:.1in; padding:.13in 0; }.plot { width:100%; display:block; overflow:visible; }
.plot-title { font:800 30px Archivo; fill:var(--ink); }.plot-sub { font:500 19px Inter; fill:var(--muted); }
.baseline { font:500 15px Inter; fill:#7b878e; text-anchor:end; }
.legend { font:600 18px Inter; fill:var(--muted); }.grid { stroke:#dce2e0; stroke-width:2; }.range { stroke:#ced6d4; stroke-width:5; stroke-linecap:round; }
.ylabel { font:700 20px Inter; fill:var(--ink); text-anchor:end; }.tick { font:500 16px Inter; fill:#728089; text-anchor:middle; }.winner { font:800 22px Archivo; fill:var(--red); }.dot { stroke:white; stroke-width:3; }
.protocol { margin-top:.1in; padding:.14in .17in; background:#f4f5f3; font:500 .16in/1.38 Inter; color:var(--muted); }

.center-figure { min-height:7.4in; padding:.22in; background:linear-gradient(145deg,#f1f7f5,#fff); border:.025in solid #b9d1cd; border-radius:.14in; display:flex; flex-direction:column; justify-content:center; }
.center-title { font:800 .31in Archivo; text-align:center; }.center-sub { font:500 .19in/1.4 Inter; color:var(--muted); text-align:center; margin:.06in auto .2in; max-width:10in; }
.traversal { display:flex; justify-content:center; gap:.06in; margin:.17in 0; }.step { width:.56in; height:.56in; border-radius:50%; display:grid; place-items:center; background:white; border:.035in solid var(--navy); font:700 .18in Archivo; }.step.shared { background:#f8e9ed; border-color:var(--red); color:var(--red); }
.down { text-align:center; font:800 .35in Archivo; color:var(--teal); margin:.05in 0; }
.compressed { width:72%; margin:auto; padding:.2in; color:white; background:var(--navy); border-radius:.12in; text-align:center; }.compressed strong { font:800 .28in Archivo; }.compressed span { display:block; margin-top:.04in; font:500 .17in Inter; color:#d6e0e5; }
.fork { display:grid; grid-template-columns:repeat(3,1fr); gap:.12in; margin-top:.25in; }.fork div { padding:.17in; background:white; border-top:.055in solid var(--teal); box-shadow:0 .02in .08in #dce6e3; text-align:center; }.fork strong { display:block; font:700 .21in Archivo; }.fork span { font:500 .16in/1.3 Inter; color:var(--muted); }

.compute-flow { margin:.17in 0; display:grid; grid-template-columns:1fr .18in 1.2fr .18in 1fr; align-items:center; }.compute-flow .box { min-height:1.15in; }.compute-flow .box.hot2 { background:#e5f3f1; border-color:#80bab6; }
.code { padding:.17in .2in; background:var(--navy); color:white; border-radius:.1in; font:600 .17in/1.45 'JetBrains Mono'; margin:.15in 0; }
.analyses { display:grid; gap:.12in; }.analysis { display:grid; grid-template-columns:1.8in 1fr 1.7in; gap:.12in; align-items:center; padding:.16in 0; border-bottom:.02in solid var(--line); }.analysis h3 { font:800 .22in Archivo; }.analysis p { font-size:.17in; line-height:1.35; }.analysis strong { font:800 .26in Archivo; color:var(--red); text-align:right; }

.result-cards { display:grid; grid-template-columns:1fr 1fr; gap:.12in; margin:.15in 0; }.result { padding:.17in; background:#f3f5f3; }.result strong { display:block; font:800 .3in Archivo; color:var(--teal); }.result span { font:500 .17in/1.35 Inter; color:var(--muted); }
.case { margin:.18in 0; background:var(--red); color:white; padding:.22in; border-radius:.12in; }.case h3 { font:800 .26in Archivo; }.case .numbers { display:grid; grid-template-columns:repeat(2,1fr); gap:.12in; margin-top:.14in; }.case strong { display:block; font:800 .36in Archivo; }.case span { font:500 .16in/1.3 Inter; color:#f4dbe1; }
.limit { padding:.18in; background:#fbf2dc; border-left:.06in solid var(--gold); font:500 .18in/1.4 Inter; }.limit strong { color:#875b07; }
.takeaway { min-height:6.5in; display:flex; flex-direction:column; justify-content:center; padding:.28in; background:var(--navy); color:white; border-radius:.12in; }.takeaway h2 { font:800 .42in/1.18 Archivo; }.takeaway h2 em { color:#f2b9c5; font-style:normal; }.takeaway p { margin-top:.17in; color:#dce5ea; }.takeaway ul { margin:.2in 0 0; padding-left:.28in; font:600 .19in/1.55 Inter; }

.footer { display:grid; grid-template-columns:.78in 1fr .78in; align-items:center; gap:.16in; border-top:.045in solid var(--ink); padding-top:.13in; }.footer img { width:.72in; height:.72in; }.foot { font:500 .16in/1.35 Inter; color:var(--muted); }.foot strong { color:var(--ink); }.right { text-align:right; }
</style></head><body><main class="poster">
<header class="header"><div><div class="eyebrow">Computing over compression</div><h1>GFAz: analyze pangenome graphs <em>without unpacking them</em></h1><p class="dek">A lossless, structure-aware GFA compressor whose traversal grammar doubles as a streaming compute substrate.</p><p class="authors">Taolue Yang · Youyuan Liu · Bo Jiang · Xinghua Shi · Sian Jin — Temple University</p></div><div class="event">T2T Face-to-Face 2026<span>UC Santa Cruz · CPU results shown throughout</span></div></header>
<section class="banner"><div class="big-result"><strong>369 GB → 4.5 GB</strong><span>HPRC v2.1 becomes 82.8× smaller while retaining full GFA semantics</span></div><div class="claim">Compress the repeated walks once. Then stream them directly into decompression or analysis.</div></section>
<section class="columns">
  <div class="col">
    <section class="section"><h2><b>01</b>The scaling problem is in the walks</h2><p class="lead">Pangenome topology grows slowly; traversal records grow with every haplotype.</p><div class="problem-visual"><div class="graph"><span class="node" style="left:8%;top:42%">1</span><span class="node" style="left:39%;top:18%">2</span><span class="node" style="left:39%;top:67%">3</span><span class="node" style="left:74%;top:42%">4</span><i class="edge" style="left:17%;top:48%;width:31%;transform:rotate(-18deg)"></i><i class="edge" style="left:17%;top:52%;width:31%;transform:rotate(20deg)"></i><i class="edge" style="left:48%;top:30%;width:34%;transform:rotate(18deg)"></i><i class="edge" style="left:48%;top:73%;width:34%;transform:rotate(-20deg)"></i></div><div class="plus">+</div><div class="walks"><div class="walk"></div><div class="walk" style="width:94%"></div><div class="walk" style="width:97%"></div><div class="walk" style="width:88%"></div><div class="walk" style="width:100%"></div></div></div><p>Across human graphs, P/W records dominate the bytes. Closely related haplotypes repeat long node-ID substrings, but gzip and Zstd see only short byte windows.</p><div class="statline"><div><strong>&gt;90%</strong><span>of large GFAs can be traversal records</span></div><div><strong>linear growth</strong><span>as samples and haplotypes are added</span></div></div></section>
    <section class="section"><h2><b>02</b>Compress the structure, not just the text</h2><div class="pipeline"><div class="pipe-row"><div class="box"><strong>GFA text</strong><span>S, L/J/C, P/W, optional tags</span></div><div class="arr">→</div><div class="box"><strong>typed columns</strong><span>integer IDs, packed orientations</span></div><div class="arr">→</div><div class="box hot"><strong>traversal grammar</strong><span>delta + iterative 2-mers</span></div></div></div><div class="grammar"><mark>+12 +13</mark> +21 <mark>+12 +13</mark> −8 → <mark>R₁</mark> +21 <mark>R₁</mark> −8<small>R₁ → (+12,+13) and −R₁ → (−13,−12): one rule represents both orientations. Later rounds combine nodes and rules into longer shared phrases.</small></div><div class="bullets"><div class="bullet"><strong>Columnar and lossless</strong><span>Every GFA record type and optional field is preserved with a type-specific encoding.</span></div><div class="bullet"><strong>Linear-time grammar</strong><span>Frequent adjacent pairs are canonicalized across forward and reverse-complement orientation.</span></div><div class="bullet"><strong>One portable container</strong><span>The CPU path serializes a shared <code>.gfaz</code> schema; queries can touch only the columns they need.</span></div></div></section>
    <section class="section"><h2><b>03</b>Best ratio across every graph</h2><div class="plot-wrap">__RATIO__</div><p class="protocol">Current repository README data. Ratio is original GFA size divided by compressed size. GFAz preserves full GFA semantics; GBZ preserves a graph/haplotype index rather than lossless GFA text.</p></section>
    <section class="section"><h2><b>04</b>CPU compression speed</h2><div class="plot-wrap">__COMP__</div><p class="protocol">32 threads. End-to-end CLI wall time, including input parsing, grammar construction, entropy encoding, and serialization. Inputs are staged from NVMe immediately before each run.</p></section>
  </div>
  <div class="col">
    <section class="section"><h2><b>05</b>One representation, two streaming paths</h2><div class="center-figure"><div class="center-title">Repeated haplotype walks become a reusable traversal store</div><div class="center-sub">Grammar rules stay compressed until a consumer asks for nodes. Expansion is linear in emitted traversal length.</div><div class="traversal"><span class="step">1</span><span class="step shared">2</span><span class="step shared">7</span><span class="step shared">9</span><span class="step">12</span><span class="step shared">2</span><span class="step shared">7</span><span class="step shared">9</span><span class="step">18</span></div><div class="down">↓</div><div class="compressed"><strong>.gfaz traversal grammar</strong><span>flat encoded P/W streams + rule arrays + typed metadata columns</span></div><div class="fork"><div><strong>restore GFA</strong><span>expand and write records in order</span></div><div><strong>select a walk</strong><span>decode one zero-copy slice</span></div><div><strong>fold an analysis</strong><span>visit nodes without graph materialization</span></div></div></div></section>
    <section class="section"><h2><b>06</b>Streaming decompression avoids reconstruction</h2><div class="compute-flow"><div class="box"><strong>de-Zstd columns</strong><span>load the encoded blocks</span></div><div class="arr">→</div><div class="box hot2"><strong>expand + inverse delta</strong><span>O(1) rule-array lookup; prefix sum fused into emission</span></div><div class="arr">→</div><div class="box"><strong>direct writer</strong><span>emit GFA without rebuilding GfaGraph</span></div></div><p>The default CPU decoder uses thread-local output buffers and ordered writes. Fixed-width fields are restored from columns while traversal nodes stream through the grammar expander.</p></section>
    <section class="section"><h2><b>07</b>CPU decompression speed</h2><div class="plot-wrap">__DECOMP__</div><p class="protocol">Inputs are preloaded into page cache; decompression writes to <code>/dev/null</code> to isolate compute throughput. GFAz is fastest on five of six datasets and reaches 5.4 GiB/s on HPRC v2.0.</p></section>
    <section class="section"><h2><b>08</b>The compressed file is also the compute engine</h2><p class="lead">Most path-centric analyses need a fold over haplotypes—not a fully materialized graph.</p><div class="compute-flow"><div class="box"><strong>.gfaz slice</strong><span>encoded traversal</span></div><div class="arr">→</div><div class="box hot2"><strong>StreamDecodedNodes</strong><span>visitor is inlined through grammar expansion</span></div><div class="arr">→</div><div class="box"><strong>compact state</strong><span>statistic or output record</span></div></div><div class="code">StreamDecodedNodes(haplotype, visit_node)</div><div class="analyses"><div class="analysis"><h3>deconstruct</h3><p>Find top-level snarls, then stream haplotypes through boundary states to emit VCF.</p><strong>15–71×</strong></div><div class="analysis"><h3>growth</h3><p>Fold node coverage by sample/haplotype group while traversals remain compressed.</p><strong>25–369×</strong></div><div class="analysis"><h3>pav</h3><p>Accumulate presence/absence ratios over reference windows from streamed walks.</p><strong>299–613×</strong></div></div><p style="margin-top:.12in">Speedups are against vg deconstruct, Panacus, and odgi pav at 16 threads.</p></section>
  </div>
  <div class="col">
    <section class="section"><h2><b>09</b>The answers agree</h2><div class="result-cards"><div class="result"><strong>~99.99%</strong><span>deconstruct position concordance with vg; record counts within ~0.2%</span></div><div class="result"><strong>identical</strong><span>growth curves at every Panacus point</span></div><div class="result"><strong>same shape</strong><span>PAV matrices reproduce odgi’s reference-window semantics</span></div><div class="result"><strong>thread stable</strong><span>byte-identical VCF output at 1 and 16 threads</span></div></div></section>
    <section class="section"><h2><b>10</b>Computing over compression pays off</h2><div class="plot-wrap">__SPEEDUP__</div><p class="protocol">Current README comparisons at 16 CPU threads. Baselines read uncompressed GFA; GFAz reads the <code>.gfaz</code> container directly.</p></section>
    <section class="section"><h2><b>11</b>Whole-genome scale changes what is possible</h2><div class="case"><h3>HPRC v2.0 growth</h3><div class="numbers"><div><strong>245 min</strong><span>Panacus · 327 GB RSS</span></div><div><strong>39.8 s</strong><span>GFAz · 12.9 GB RSS</span></div></div></div><p>GFAz reads a ~4.5 GB container and reproduces the exact growth curve. On the largest graphs, vg and odgi cannot load the uncompressed input within the 503 GB node.</p></section>
    <section class="section"><h2><b>12</b>Current boundary</h2><div class="limit"><strong>Whole-genome PAV is not solved yet.</strong> Its uncompressed node-set/window index—not traversal decoding—exceeds memory on HPRC v2.0/v2.1. Compressing that working index is the next target.</div></section>
    <section class="takeaway"><h2>A pangenome graph does not have to be <em>unpacked</em> to be queried.</h2><p>GFAz turns the largest part of GFA into a representation that is simultaneously compact, fast to decode, and useful for direct analysis.</p><ul><li>18–84× lossless compression</li><li>up to 5.4 GiB/s CPU decompression</li><li>one to three orders of magnitude faster analysis</li></ul></section>
  </div>
</section>
<footer class="footer"><img src="qr-code.png"><div class="foot"><strong>github.com/babyplutokurt/GFAz</strong> · taolue.yang@temple.edu<br>Yang, Liu, Jiang, Shi, Jin. <em>GFAz: Order-of-Magnitude Pangenome Analytics by Computing over Compression.</em> ICS ’26. DOI: 10.1145/3797905.3807870</div><div class="right"><img src="qr-paper.png"></div></footer>
</main></body></html>"""

OUT.write_text(HTML.replace("__RATIO__", ratio_plot).replace("__COMP__", comp_plot).replace("__DECOMP__", decomp_plot).replace("__SPEEDUP__", speedup_plot), encoding="utf-8")
print(f"wrote {OUT}")
