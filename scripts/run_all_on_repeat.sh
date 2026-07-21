#!/bin/bash
# Run purple_agent_v2 with ALL features on (RAG + self-verify loop +
# diagnostic-flag parsing), N times in a row. Archives each run into
# all_on_results.run{1,2,...}.
#
# Holds the config in the "all-on" state for the whole run, restoring the
# original on exit. Skips runs whose archive directory already exists.

set -e
cd /Users/kavinkannan/petscagent-bench

N="${1:-3}"
CFG=config/purple_agent_v2_config.yaml
CFG_BACKUP="${CFG}.bak.$$"

cp "$CFG" "$CFG_BACKUP"
restore_config() {
  cp "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT INT TERM

if [ ! -f src/petsc_rag/index/faiss.bin ]; then
  echo "ERROR: RAG index not found at src/petsc_rag/index/faiss.bin"
  echo "       Build it with: uv run python -m petsc_rag.build_index"
  exit 1
fi

# Flip all four knobs at once. Each subst is required (count check) so a
# silently-skipped patch can't ship the wrong config.
python3 -c "
import re, sys
path = '$CFG'
s = open(path).read()

def patch(pat, repl, label):
    global s
    s, n = re.subn(pat, repl, s, count=1)
    if n != 1:
        sys.exit(f'failed to patch {label} in {path}')

patch(r'(\n\s*max_iters:\s*)\d+',           r'\g<1>3',     'max_iters')
patch(r'(\n\s*do_smoke_run:\s*)(true|false)', r'\g<1>true', 'do_smoke_run')
patch(r'(\n\s*check_diagnostics:\s*)(true|false)', r'\g<1>true', 'check_diagnostics')
patch(r'(\n\s*enabled:\s*)(true|false)',    r'\g<1>true',  'rag.enabled')

open(path, 'w').write(s)
print('[all_on] config: rag=on, max_iters=3, do_smoke_run=true, check_diagnostics=true')
"

# Highest existing run index, so re-invocations keep numbering monotonic.
last=0
for d in all_on_results.run*; do
  [ -d "$d" ] || continue
  n="${d#all_on_results.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

for i in $(seq 1 "$N"); do
  next=$((last + i))
  out_dir="all_on_results.run${next}"
  log="${out_dir}.log"

  if [ -d "$out_dir" ]; then
    echo "[all_on] SKIP $out_dir (already exists)"
    continue
  fi

  echo "=================================================="
  echo "[all_on] RUN $i/$N -> $out_dir"
  echo "=================================================="

  rm -rf output
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -8

  mkdir -p "$out_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$out_dir/"
    mv "$log" "$out_dir/run.log"
    echo "[all_on] saved $out_dir/benchmark_summary.json"
  else
    echo "[all_on] WARNING: no summary produced for $out_dir"
    mv "$log" "$out_dir/run.log" 2>/dev/null || true
  fi
done

echo "[all_on] DONE."
