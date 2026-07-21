#!/bin/bash
# Run scripts/run_plan_ab.sh N times, archiving each into
# plan_ab_results.run{N}. Mirrors run_header_ab_repeat.sh.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"

last=0
for d in plan_ab_results.run*; do
  [ -d "$d" ] || continue
  n="${d#plan_ab_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out="plan_ab_results.run${next}"
  if [ -d "$out" ]; then
    echo "[plan_repeat] SKIP $out (already exists)"
    continue
  fi
  echo "=================================================="
  echo "[plan_repeat] RUN $i/$N -> $out"
  echo "=================================================="
  rm -rf plan_ab_results
  ./scripts/run_plan_ab.sh
  mv plan_ab_results "$out"
  echo "[plan_repeat] archived $out"
done

echo "[plan_repeat] DONE."
