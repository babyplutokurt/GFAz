if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <file1> <file2>"
  exit 1
fi

f1="$1"
f2="$2"

extract_type () {
  local type="$1" file="$2"
  grep -P "^${type}\t" "$file"
}

for t in H S L P J C; do
  echo "=== Comparing $t records (no sorting) ==="
  extract_type "$t" "$f1" > "/tmp/${t}.1"
  extract_type "$t" "$f2" > "/tmp/${t}.2"

  if diff -w -q "/tmp/${t}.1" "/tmp/${t}.2" >/dev/null; then
    echo "OK✅: $t records match"
  else
    echo "DIFF: $t records differ (first diff location):"
    diff -w "/tmp/${t}.1" "/tmp/${t}.2" | awk 'NR==1{print; exit}'
    echo
    echo "Tip: show first 20 diff lines:"
    echo "diff -w /tmp/${t}.1 /tmp/${t}.2 | sed -n '1,20p'"
    exit 1
  fi
done

echo "All H/S/L/P records match (no sorting)."

