#!/bin/bash
# Capture per-purple code snapshots for blind cross-eval.
# Sets green=gpt52 once (fast, doesn't matter for capture), runs each purple, copies .c files.
set -e
cd /Users/kavinkannan/petscagent-bench

# Pin green to gpt52 for capture runs (purple is the only thing we care about)
python3 -c "
import re
p = 'config/green_agent_config.yaml'
s = open(p).read()
s = re.sub(r'    model: \"openai/\\w+\"', '    model: \"openai/gpt52\"', s, count=1)
open(p, 'w').write(s)
"

WORK_DIR=/opt/homebrew/Cellar/petsc/3.24.6/work
SNAP=/Users/kavinkannan/petscagent-bench/snapshots
mkdir -p "$SNAP"

for purple in gpt52 claudeopus47 gemini25pro; do
  echo "=================================================="
  echo "Capturing purple=$purple"
  echo "=================================================="
  # Set purple model
  python3 -c "
import re
p = 'config/purple_agent_config.yaml'
s = open(p).read()
s = re.sub(r'  model: \"openai/\\w+\"', f'  model: \"openai/$purple\"', s, count=1)
open(p, 'w').write(s)
"
  # Clear output and work dir
  rm -rf output
  rm -f $WORK_DIR/*.c $WORK_DIR/makefile
  # Run benchmark
  uv run main.py launch 2>&1 | tail -20
  # Snapshot the .c files and the per-problem cli_args from benchmark_summary.json
  mkdir -p "$SNAP/$purple"
  cp $WORK_DIR/*.c "$SNAP/$purple/" 2>/dev/null || echo "no .c files for $purple"
  cp output/benchmark_summary.json "$SNAP/$purple/benchmark_summary.json"
  echo "Snapshotted purple=$purple:"
  ls "$SNAP/$purple/"
done

echo "=================================================="
echo "All captures complete."
echo "=================================================="
