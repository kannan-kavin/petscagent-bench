#!/bin/bash
# Run scripts/run_multiplan_ab.sh N times, archiving each into
# multiplan_ab_results.run{N}. Mirrors run_plan_ab_repeat.sh.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"

last=0
for d in multiplan_ab_results.run*; do
  [ -d "$d" ] || continue
  n="${d#multiplan_ab_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out="multiplan_ab_results.run${next}"
  if [ -d "$out" ]; then
    echo "[multiplan_repeat] SKIP $out (already exists)"
    continue
  fi
  echo "=================================================="
  echo "[multiplan_repeat] RUN $i/$N -> $out"
  echo "=================================================="
  rm -rf multiplan_ab_results
  ./scripts/run_multiplan_ab.sh
  mv multiplan_ab_results "$out"
  echo "[multiplan_repeat] archived $out"
done

echo "[multiplan_repeat] DONE."
