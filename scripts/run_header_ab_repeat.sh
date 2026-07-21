#!/bin/bash
# Run scripts/run_header_ab.sh N times, archiving each into
# header_ab_results.run{N}. Mirrors run_selfverify_ab_repeat.sh.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"

last=0
for d in header_ab_results.run*; do
  [ -d "$d" ] || continue
  n="${d#header_ab_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out="header_ab_results.run${next}"
  if [ -d "$out" ]; then
    echo "[hdr_full_repeat] SKIP $out (already exists)"
    continue
  fi
  echo "=================================================="
  echo "[hdr_full_repeat] RUN $i/$N -> $out"
  echo "=================================================="
  rm -rf header_ab_results
  ./scripts/run_header_ab.sh
  mv header_ab_results "$out"
  echo "[hdr_full_repeat] archived $out"
done

echo "[hdr_full_repeat] DONE."
