#!/bin/bash
# Run the self-verify A/B benchmark N times, archiving each into
# selfverify_ab_results.run{1,2,...}. Mirrors run_rag_ab_repeat.sh.
#
# Skips runs whose archive directory already exists.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"   # default: 3 runs

# Highest existing run index, so we keep numbering monotonic across invocations.
last=0
for d in selfverify_ab_results.run*; do
  [ -d "$d" ] || continue
  n="${d#selfverify_ab_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out="selfverify_ab_results.run${next}"
  if [ -d "$out" ]; then
    echo "[repeat] SKIP $out (already exists)"
    continue
  fi
  echo "=================================================="
  echo "[repeat] RUN $i/$N -> $out"
  echo "=================================================="
  rm -rf selfverify_ab_results
  ./scripts/run_selfverify_ab.sh
  mv selfverify_ab_results "$out"
  echo "[repeat] archived $out"
done

echo "[repeat] DONE."
