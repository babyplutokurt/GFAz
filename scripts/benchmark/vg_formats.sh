#!/bin/bash
# The same questions asked three ways: gfaz on a .gfaz, vg on the .gfaz read in
# place, and vg on its own indexed format (XG or GBZ) built from the same GFA.
#
# usage: vg_formats.sh <graph.gfa> <ref path> <region> <tag> <workdir> [xg|gbz]
#   e.g. vg_formats.sh chr6.gfa 'grch38#chr6' 'grch38#chr6:30000000-30100000' \
#            chr6 /scratch/vs_vg xg
#
# Environment: GFAZ and VG name the binaries (default: build/bin/gfaz of this
# checkout, and vg on PATH; vg must be built with the GFAz handle graph);
# THREADS (default 16); REPS (default 1); GFAZ_INDEX_STEPS, when set, saves a
# path index with that block size before the runs (otherwise none is used).
#
# Prints one line per run: wall seconds, peak RSS in KB, output size, and the
# 1-minute load average, which says whether the machine was otherwise busy.
# Outputs are kept as <workdir>/out.<step>.<tool> for comparison. The formats
# are built once and reused; delete them from <workdir> to rebuild.
set -u
if [ $# -lt 5 ]; then
  sed -n '2,6p' "$0"
  exit 1
fi
HERE=$(cd "$(dirname "$0")" && pwd)
Z=${GFAZ:-$HERE/../../build/bin/gfaz}
V=${VG:-vg}
THREADS=${THREADS:-16}
REPS=${REPS:-1}
GFA=$(realpath "$1"); REF=$2; REGION=$3; TAG=$4; W=$5; FMT=${6:-xg}
mkdir -p "$W" && cd "$W" || exit 1
export OMP_NUM_THREADS=$THREADS
unset VG_GFAZ_CONVERT

t() { # step, tool, command...
  local step=$1 tool=$2; shift 2
  /usr/bin/time -f "%e %M" -o time.tmp timeout 14400 "$@" \
    > "out.$step.$tool" 2> "err.$step.$tool"
  echo "$TAG $step $tool rc=$? wall_rss=$(tail -1 time.tmp)" \
    "bytes=$(stat -c %s "out.$step.$tool") load=$(cut -d' ' -f1 /proc/loadavg)"
}

[ -f "$TAG.gfaz" ] || t build gfaz "$Z" compress -j "$THREADS" "$GFA" "$TAG.gfaz"
if [ ! -f "$TAG.$FMT" ]; then
  if [ "$FMT" = gbz ]; then
    t build gbz "$V" gbwt -G "$GFA" --gbz-format -g "$TAG.gbz" --num-jobs "$THREADS"
  else
    t build xg bash -c "'$V' convert -g '$GFA' -x -t $THREADS > '$TAG.xg'"
  fi
fi
ls -l "$TAG.gfaz" "$TAG.$FMT" | awk -v tag="$TAG" '{print tag, "size", $5, $9}'

rm -f "$TAG.gfaz.pidx"
if [ -n "${GFAZ_INDEX_STEPS:-}" ]; then
  t index gfaz "$Z" index -i "$TAG.gfaz" -b "$GFAZ_INDEX_STEPS" -t "$THREADS"
fi

for rep in $(seq "$REPS"); do
  t stats.$rep   gfaz     "$Z" stats -i "$TAG.gfaz"
  t stats.$rep   vg-$FMT  "$V" stats -z -l "$TAG.$FMT"
  t stats.$rep   vg-gfaz  "$V" stats -z -l "$TAG.gfaz"
  t pathsL.$rep  vg-$FMT  "$V" paths -L -x "$TAG.$FMT"
  t pathsL.$rep  vg-gfaz  "$V" paths -L -x "$TAG.gfaz"
  t pathseq.$rep vg-$FMT  "$V" paths -F -Q "$REF" -x "$TAG.$FMT"
  t pathseq.$rep vg-gfaz  "$V" paths -F -Q "$REF" -x "$TAG.gfaz"
  t region.$rep  gfaz     "$Z" extract-region -i "$TAG.gfaz" -r "$REGION" -t "$THREADS"
  t region.$rep  vg-$FMT  "$V" chunk -x "$TAG.$FMT" -p "$REGION" -c 10 -t "$THREADS"
  t region.$rep  vg-gfaz  "$V" chunk -x "$TAG.gfaz" -p "$REGION" -c 10 -t "$THREADS"
  t snarls.$rep  vg-$FMT  "$V" snarls -t "$THREADS" "$TAG.$FMT"
  t snarls.$rep  vg-gfaz  "$V" snarls -t "$THREADS" "$TAG.gfaz"
  t decon.$rep   gfaz     "$Z" deconstruct -i "$TAG.gfaz" -r "$REF" -t "$THREADS"
  t decon.$rep   vg-$FMT  "$V" deconstruct -t "$THREADS" -p "$REF" "$TAG.$FMT"
  t decon.$rep   vg-gfaz  "$V" deconstruct -t "$THREADS" -p "$REF" "$TAG.gfaz"
done
rm -f time.tmp
