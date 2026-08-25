#!/usr/bin/env python3
"""Convert the straight-line-only Temple T SVG into a vector PDF via TikZ.

The source (Wikimedia Commons, File:Temple_T_logo.svg) uses only m/l/h/v/z
commands, so every subpath is a polygon and no curve handling is needed.
"""
import re, subprocess, sys, xml.etree.ElementTree as ET

SVG, OUT = "temple-t.svg", "temple-logo"
NS = "{http://www.w3.org/2000/svg}"

root = ET.parse(SVG).getroot()
H = float(root.get("height"))
TOK = re.compile(r"([MmLlHhVvZz])|(-?\d*\.?\d+(?:[eE]-?\d+)?)")


def subpaths(d):
    toks = [(c, n) for c, n in TOK.findall(d)]
    i, cmd, x, y, sx, sy = 0, None, 0.0, 0.0, 0.0, 0.0
    out, cur = [], []
    def num():
        nonlocal i
        while not toks[i][1]:
            i += 1
        v = float(toks[i][1]); i += 1; return v
    while i < len(toks):
        if toks[i][0]:
            cmd = toks[i][0]; i += 1
            if cmd in "Zz":
                if cur: out.append(cur); cur = []
                x, y = sx, sy
                continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            nx, ny = num(), num()
            x, y = (x + nx, y + ny) if rel else (nx, ny)
            sx, sy = x, y
            if cur: out.append(cur)
            cur = [(x, y)]
            cmd = "l" if rel else "L"          # implicit lineto after moveto
        elif c == "L":
            nx, ny = num(), num()
            x, y = (x + nx, y + ny) if rel else (nx, ny)
            cur.append((x, y))
        elif c == "H":
            nx = num(); x = x + nx if rel else nx
            cur.append((x, y))
        elif c == "V":
            ny = num(); y = y + ny if rel else ny
            cur.append((x, y))
        else:
            sys.exit(f"unsupported command {c!r} - file is not straight-line only")
    if cur: out.append(cur)
    return out


body = []
for el in root.iter(f"{NS}path"):
    fill = (el.get("fill") or "#000000").lstrip("#")
    if len(fill) == 3:                          # xcolor HTML needs 6 digits
        fill = "".join(ch * 2 for ch in fill)
    polys = subpaths(el.get("d"))
    body.append(f"\\definecolor{{c{fill}}}{{HTML}}{{{fill.upper()}}}")
    seg = " ".join(
        "".join(f"({px:.5f},{H - py:.5f}) -- " for px, py in poly) + "cycle"
        for poly in polys)                      # flip y: SVG grows down, TikZ up
    body.append(f"\\fill[c{fill}] {seg};")

tex = ("\\documentclass[tightpage]{standalone}\n"
       "\\usepackage{tikz}\n\\usepackage{xcolor}\n\\begin{document}\n"
       "\\begin{tikzpicture}[x=1pt,y=1pt]\n" + "\n".join(body) +
       "\n\\end{tikzpicture}\n\\end{document}\n")
open(OUT + ".tex", "w").write(tex)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", OUT + ".tex"],
                   capture_output=True, text=True)
print("pdflatex rc =", r.returncode)
if r.returncode:
    print(r.stdout[-1500:])
