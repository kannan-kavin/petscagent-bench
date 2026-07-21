#!/bin/bash
# Run purple_agent v1 (baseline, single-call, no self-verify) N times.
# v1 has no config knobs to toggle, so we just loop and archive into
# v1_baseline_results.run{1,2,...}. Re-invocations auto-increment.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"

# Highest existing run index, so re-invocations keep numbering monotonic.
last=0
for d in v1_baseline_results.run*; do
  [ -d "$d" ] || continue
  n="${d#v1_baseline_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out_dir="v1_baseline_results.run${next}"
  log="${out_dir}.log"

  if [ -d "$out_dir" ]; then
    echo "[v1] SKIP $out_dir (already exists)"
    continue
  fi

  echo "=================================================="
  echo "[v1] RUN $i/$N -> $out_dir"
  echo "=================================================="

  rm -rf output
  uv run main.py launch --purple-variant v1 2>&1 | tee "$log" | tail -8

  mkdir -p "$out_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$out_dir/"
    mv "$log" "$out_dir/run.log"
    echo "[v1] saved $out_dir/benchmark_summary.json"
  else
    echo "[v1] WARNING: no summary produced for $out_dir"
    mv "$log" "$out_dir/run.log" 2>/dev/null || true
  fi
done

echo "[v1] DONE."
