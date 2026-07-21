#!/bin/bash
set -e
cd /Users/kavinkannan/petscagent-bench

MODELS=(gpt52 claudeopus47 gemini25pro)
RESULTS_DIR=poisson1d_3x3
mkdir -p "$RESULTS_DIR"

for purple in "${MODELS[@]}"; do
  for green in "${MODELS[@]}"; do
    cell="${purple}_x_${green}"
    if [ -f "$RESULTS_DIR/$cell/benchmark_summary.json" ]; then
      echo "SKIP $cell (already done)"
      continue
    fi
    echo "=================================================="
    echo "RUN: purple=$purple, green=$green"
    echo "=================================================="
    python3 -c "
import re
for p, m in [('config/purple_agent_config.yaml','$purple'), ('config/green_agent_config.yaml','$green')]:
    s = open(p).read()
    s = re.sub(r'model: \"openai/\\w+\"', f'model: \"openai/{m}\"', s, count=1)
    open(p, 'w').write(s)
"
    rm -rf output
    uv run main.py launch 2>&1 | tail -8
    mkdir -p "$RESULTS_DIR/$cell"
    cp output/benchmark_summary.json "$RESULTS_DIR/$cell/" 2>/dev/null || echo "no summary written for $cell"
  done
done
echo "ALL DONE"
