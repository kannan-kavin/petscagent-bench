#!/usr/bin/env bash
# Same as run_shipped_n3.sh but archives into ragon_n3_results.run{N}/
# so we don't clobber the non-RAG shipped_n3 results.
set -u
N=${1:-3}
PROGRESS=ragon_n3.progress.log
: > "$PROGRESS"
echo "$(date -u +%FT%TZ) START n=$N (rag on)" >> "$PROGRESS"

for i in $(seq 1 "$N"); do
  outdir="ragon_n3_results.run${i}"
  log="ragon_n3_trial.run${i}.driver.log"
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
