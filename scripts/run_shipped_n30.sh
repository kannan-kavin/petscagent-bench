#!/usr/bin/env bash
# Run the current shipped v2 stack N times, archive each into
# shipped_n30_results.run{N}/, and append a one-line progress marker to
# shipped_n30.progress.log after each run so a Monitor can watch for
# "RUN N/30 DONE" lines.
#
# Usage: bash scripts/run_shipped_n30.sh [N]
set -u
N=${1:-30}
PROGRESS=shipped_n30.progress.log
: > "$PROGRESS"
echo "$(date -u +%FT%TZ) START n=$N" >> "$PROGRESS"

for i in $(seq 1 "$N"); do
  outdir="shipped_n30_results.run${i}"
  log="shipped_n30_trial.run${i}.driver.log"
  mkdir -p "$outdir"
  rm -f output/benchmark_summary.json
  echo "$(date -u +%FT%TZ) RUN ${i}/${N} START" >> "$PROGRESS"
  uv run main.py launch --purple-variant v2 > "$log" 2>&1
  rc=$?
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$outdir/"
    cp "$log" "$outdir/"
    cp config/purple_agent_v2_config.yaml "$outdir/"
    avg=$(uv run python -c "import json; d=json.load(open('output/benchmark_summary.json')); print(f\"{d['summary']['avg_composite_score']:.2f}\")" 2>/dev/null || echo "n/a")
    echo "$(date -u +%FT%TZ) RUN ${i}/${N} DONE rc=$rc avg=$avg" >> "$PROGRESS"
  else
    echo "$(date -u +%FT%TZ) RUN ${i}/${N} FAILED rc=$rc no benchmark_summary.json" >> "$PROGRESS"
  fi
done

echo "$(date -u +%FT%TZ) ALL DONE" >> "$PROGRESS"
