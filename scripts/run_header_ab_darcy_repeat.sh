#!/bin/bash
# Run scripts/run_header_ab_darcy.sh N times, archiving each into
# header_ab_darcy_results.run{N}. Mirrors run_selfverify_ab_repeat.sh.
#
# Skips runs whose archive directory already exists. Numbering is monotonic
# across invocations so re-running picks up where the previous left off.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"

last=0
for d in header_ab_darcy_results.run*; do
  [ -d "$d" ] || continue
  n="${d#header_ab_darcy_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out="header_ab_darcy_results.run${next}"
  if [ -d "$out" ]; then
    echo "[hdr_repeat] SKIP $out (already exists)"
    continue
  fi
  echo "=================================================="
  echo "[hdr_repeat] RUN $i/$N -> $out"
  echo "=================================================="
  rm -rf header_ab_darcy_results
  ./scripts/run_header_ab_darcy.sh
  mv header_ab_darcy_results "$out"
  echo "[hdr_repeat] archived $out"
done

echo "[hdr_repeat] DONE."
