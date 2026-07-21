#!/usr/bin/env bash
# Continue the ragon_n3 series, appending runs starting at index START.
# Usage: bash scripts/run_ragon_more.sh START COUNT
set -u
START=${1:-4}
COUNT=${2:-3}
END=$((START + COUNT - 1))
PROGRESS=ragon_more.progress.log
: > "$PROGRESS"
echo "$(date -u +%FT%TZ) START runs ${START}..${END}" >> "$PROGRESS"

for i in $(seq "$START" "$END"); do
  outdir="ragon_n3_results.run${i}"
  log="ragon_n3_trial.run${i}.driver.log"
  mkdir -p "$outdir"
  rm -f output/benchmark_summary.json
  echo "$(date -u +%FT%TZ) RUN ${i} START" >> "$PROGRESS"
  uv run main.py launch --purple-variant v2 > "$log" 2>&1
  rc=$?
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$outdir/"
    cp "$log" "$outdir/"
    cp config/purple_agent_v2_config.yaml "$outdir/"
    avg=$(uv run python -c "import json; d=json.load(open('output/benchmark_summary.json')); print(f\"{d['summary']['avg_composite_score']:.2f}\")" 2>/dev/null || echo "n/a")
    echo "$(date -u +%FT%TZ) RUN ${i} DONE rc=$rc avg=$avg" >> "$PROGRESS"
  else
    echo "$(date -u +%FT%TZ) RUN ${i} FAILED rc=$rc no benchmark_summary.json" >> "$PROGRESS"
  fi
done

echo "$(date -u +%FT%TZ) ALL DONE" >> "$PROGRESS"
