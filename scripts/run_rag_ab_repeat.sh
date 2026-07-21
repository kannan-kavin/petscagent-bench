#!/bin/bash
# Run the RAG A/B benchmark N times in a row, archiving each into
# rag_ab_results.run{1,2,...}. Used to estimate run-to-run noise on the
# composite-score delta.
#
# Skips runs whose archive directory already exists.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-2}"   # default: 2 additional runs

# Highest existing run index, so we keep numbering monotonic across invocations.
last=0
for d in rag_ab_results.run*; do
  [ -d "$d" ] || continue
  n="${d#rag_ab_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out="rag_ab_results.run${next}"
  if [ -d "$out" ]; then
    echo "[repeat] SKIP $out (already exists)"
    continue
  fi
  echo "=================================================="
  echo "[repeat] RUN $i/$N -> $out"
  echo "=================================================="
  rm -rf rag_ab_results
  ./scripts/run_rag_ab.sh
  mv rag_ab_results "$out"
  echo "[repeat] archived $out"
done

echo "[repeat] DONE."
