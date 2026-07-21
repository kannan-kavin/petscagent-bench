#!/bin/bash
set -e
cd /Users/kavinkannan/petscagent-bench

MODELS=(gpt52 claudeopus47 gemini25pro)
RESULTS_DIR=ns2d_3x3_retry
mkdir -p "$RESULTS_DIR"

# Belt-and-suspenders: kill any .DS_Store that crept in.
find data -name '.DS_Store' -delete 2>/dev/null || true

# Stash all problems except NS2D so this run only evaluates NS2D.
STASH_DIR=data_tmp_stash_ns2d
rm -rf "$STASH_DIR" && mkdir "$STASH_DIR"
for f in data/*.json; do
  base=$(basename "$f")
  if [ "$base" != "NS2D.json" ]; then
    mv "$f" "$STASH_DIR/"
  fi
done
trap 'echo "Restoring stashed problems..."; mv '"$STASH_DIR"'/*.json data/ 2>/dev/null || true; rmdir '"$STASH_DIR"' 2>/dev/null || true' EXIT

rm -rf purple_agent_cache && mkdir -p purple_agent_cache

export PETSCAGENT_RETRIES=1
echo "PETSCAGENT_RETRIES=$PETSCAGENT_RETRIES"

for purple in "${MODELS[@]}"; do
  for green in "${MODELS[@]}"; do
    cell="${purple}_x_${green}"
    if [ -f "$RESULTS_DIR/$cell/benchmark_summary.json" ]; then
      echo "SKIP $cell (already done)"
      continue
    fi
    echo "=================================================="
    echo "RUN: purple=$purple, green=$green   (NS2D + 1 retry)"
    echo "=================================================="
    python3 -c "
import re
for p, m in [('config/purple_agent_config.yaml','$purple'), ('config/green_agent_config.yaml','$green')]:
    s = open(p).read()
    s = re.sub(r'model: \"openai/\\w+\"', f'model: \"openai/{m}\"', s, count=1)
    open(p, 'w').write(s)
"
    rm -rf output
    uv run main.py launch 2>&1 | tail -20
    mkdir -p "$RESULTS_DIR/$cell"
    cp output/benchmark_summary.json "$RESULTS_DIR/$cell/" 2>/dev/null || echo "no summary written for $cell"
  done
done
echo "ALL DONE"
