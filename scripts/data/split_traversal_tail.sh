#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <input.gfa> [output_prefix]" >&2
  exit 1
fi

input="$1"
prefix="${2:-output}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${script_dir}/split_traversal_tail.py" \
  "$input" \
  50 \
  --trimmed-output "${prefix}_main.gfa" \
  --extracted-output "${prefix}_tail50.txt"
