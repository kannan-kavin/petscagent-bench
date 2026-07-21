#!/bin/bash
# A/B benchmark: purple_agent_v2 with RAG off vs. RAG on.
#
# Holds the model, MCP server, self-fix loop, and problem set constant.
# Only difference between arms is config/purple_agent_v2_config.yaml: rag.enabled.
#
# Output:
#   rag_ab_results/rag_off/benchmark_summary.json
#   rag_ab_results/rag_on/benchmark_summary.json
#   rag_ab_results/run_<arm>.log   (full stdout for each arm)
#   rag_ab_results/diff.txt        (side-by-side score comparison)

set -e
cd /Users/kavinkannan/petscagent-bench

RESULTS_DIR=rag_ab_results
CFG=config/purple_agent_v2_config.yaml
CFG_BACKUP="${CFG}.bak.$$"
ARMS=(off on)

mkdir -p "$RESULTS_DIR"

# Restore the original config on any exit path so we never leave the repo
# with a half-flipped flag.
cp "$CFG" "$CFG_BACKUP"
restore_config() {
  cp "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT INT TERM

# Sanity-check: the RAG index must already be built. Otherwise the rag_on arm
# would silently fall back to no-RAG behavior, defeating the experiment.
if [ ! -f src/petsc_rag/index/faiss.bin ]; then
  echo "ERROR: RAG index not found at src/petsc_rag/index/faiss.bin"
  echo "Run: uv run python -m src.petsc_rag.build_index"
  exit 1
fi

set_rag() {
  local enabled="$1"
  # Target rag.enabled specifically. Other blocks (e.g. plan.enabled) also have
  # `enabled:` fields, so we anchor on `\nrag:\n` and consume only the indented
  # lines that follow until we hit `enabled:`.
  python3 -c "
import re, sys
path = '$CFG'
val = '$enabled'
s = open(path).read()
pat = r'(\\nrag:\\s*\\n(?:[ \\t]+[^\\n]*\\n)*?[ \\t]+enabled:\\s*)(true|false)'
new, n = re.subn(pat, f'\\\\g<1>{val}', s, count=1)
if n != 1:
    sys.exit(f'failed to patch rag.enabled in {path}')
open(path, 'w').write(new)
print(f'[rag_ab] set rag.enabled = {val}')
"
}

for arm in "${ARMS[@]}"; do
  arm_dir="$RESULTS_DIR/rag_${arm}"
  log="$RESULTS_DIR/run_${arm}.log"

  if [ -f "$arm_dir/benchmark_summary.json" ]; then
    echo "[rag_ab] SKIP rag_${arm} (already exists)"
    continue
  fi

  echo "=================================================="
  echo "ARM: rag_${arm}"
  echo "=================================================="
  if [ "$arm" = "on" ]; then
    set_rag true
  else
    set_rag false
  fi

  rm -rf output
  echo "[rag_ab] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -8

  mkdir -p "$arm_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$arm_dir/"
    echo "[rag_ab] saved $arm_dir/benchmark_summary.json"
  else
    echo "[rag_ab] WARNING: no summary produced for rag_${arm}"
  fi
done

# Build the side-by-side comparison.
echo "[rag_ab] writing diff..."
uv run python scripts/diff_rag_ab.py "$RESULTS_DIR/rag_off/benchmark_summary.json" \
                                     "$RESULTS_DIR/rag_on/benchmark_summary.json" \
  | tee "$RESULTS_DIR/diff.txt"

echo "[rag_ab] DONE. Per-problem diff in $RESULTS_DIR/diff.txt"
