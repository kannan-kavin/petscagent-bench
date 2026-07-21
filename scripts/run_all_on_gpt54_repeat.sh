#!/bin/bash
# Same as run_all_on_repeat.sh but with Green Agent's model flipped to
# Argo's gpt54 (GPT-5.4). Holds both purple and green configs in the
# desired state for the whole run, restoring originals on exit.
#
# Output: all_on_gpt54_results.run{1,2,...}/benchmark_summary.json

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"
PURPLE_CFG=config/purple_agent_v2_config.yaml
GREEN_CFG=config/green_agent_config.yaml
PURPLE_BACKUP="${PURPLE_CFG}.bak.$$"
GREEN_BACKUP="${GREEN_CFG}.bak.$$"

cp "$PURPLE_CFG" "$PURPLE_BACKUP"
cp "$GREEN_CFG"  "$GREEN_BACKUP"
restore_configs() {
  cp "$PURPLE_BACKUP" "$PURPLE_CFG"
  cp "$GREEN_BACKUP"  "$GREEN_CFG"
  rm -f "$PURPLE_BACKUP" "$GREEN_BACKUP"
}
trap restore_configs EXIT INT TERM

if [ ! -f src/petsc_rag/index/faiss.bin ]; then
  echo "ERROR: RAG index not found at src/petsc_rag/index/faiss.bin"
  exit 1
fi

# Purple: turn everything on.
python3 -c "
import re, sys
path = '$PURPLE_CFG'
s = open(path).read()

def patch(pat, repl, label):
    global s
    s, n = re.subn(pat, repl, s, count=1)
    if n != 1:
        sys.exit(f'failed to patch {label} in {path}')

patch(r'(\n\s*max_iters:\s*)\d+',                  r'\g<1>3',    'max_iters')
patch(r'(\n\s*do_smoke_run:\s*)(true|false)',      r'\g<1>true', 'do_smoke_run')
patch(r'(\n\s*check_diagnostics:\s*)(true|false)', r'\g<1>true', 'check_diagnostics')
patch(r'(\n\s*enabled:\s*)(true|false)',           r'\g<1>true', 'rag.enabled')

open(path, 'w').write(s)
print('[all_on_gpt54] purple: rag=on, max_iters=3, do_smoke_run=true, check_diagnostics=true')
"

# Green: swap model to gpt54. The model line in green_agent_config.yaml is
# 'model: \"openai/<name>\"' under llm:.
python3 -c "
import re, sys
path = '$GREEN_CFG'
s = open(path).read()
# Match first 'model: \"openai/<word>\"' (the llm.model entry, before scoring section).
new, n = re.subn(r'(\n\s*model:\s*\")openai/[a-zA-Z0-9._-]+(\")', r'\g<1>openai/gpt54\g<2>', s, count=1)
if n != 1:
    sys.exit(f'failed to patch green model in {path}')
open(path, 'w').write(new)
print('[all_on_gpt54] green: model=openai/gpt54')
"

last=0
for d in all_on_gpt54_results.run*; do
  [ -d "$d" ] || continue
  n="${d#all_on_gpt54_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out_dir="all_on_gpt54_results.run${next}"
  log="${out_dir}.log"

  if [ -d "$out_dir" ]; then
    echo "[all_on_gpt54] SKIP $out_dir (already exists)"
    continue
  fi

  echo "=================================================="
  echo "[all_on_gpt54] RUN $i/$N -> $out_dir"
  echo "=================================================="

  rm -rf output
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -8

  mkdir -p "$out_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$out_dir/"
    mv "$log" "$out_dir/run.log"
    echo "[all_on_gpt54] saved $out_dir/benchmark_summary.json"
  else
    echo "[all_on_gpt54] WARNING: no summary produced for $out_dir"
    mv "$log" "$out_dir/run.log" 2>/dev/null || true
  fi
done

echo "[all_on_gpt54] DONE."
