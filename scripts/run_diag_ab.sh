#!/bin/bash
# A/B benchmark: purple_agent_v2 with PETSc diagnostic-flag self-correction
# OFF vs ON. Holds RAG (with reranker) on in both arms, model and self-fix
# budget constant. Only difference: self_fix.check_diagnostics.
#
# Output:
#   diag_ab_results/diag_off/benchmark_summary.json
#   diag_ab_results/diag_on/benchmark_summary.json
#   diag_ab_results/run_<arm>.log
#   diag_ab_results/diff.txt

set -e
cd /Users/kavinkannan/petscagent-bench

RESULTS_DIR=diag_ab_results
CFG=config/purple_agent_v2_config.yaml
CFG_BACKUP="${CFG}.bak.$$"
ARMS=(off on)

mkdir -p "$RESULTS_DIR"

cp "$CFG" "$CFG_BACKUP"
restore_config() {
  cp "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT INT TERM

if [ ! -f src/petsc_rag/index/faiss.bin ]; then
  echo "ERROR: RAG index not found at src/petsc_rag/index/faiss.bin"
  exit 1
fi

# Both arms: RAG on (so we measure diagnostics in isolation against the
# best-known-current configuration, not against the worse RAG-off baseline).
# Anchored on `\nrag:\n` so we hit the right `enabled:` even with other
# config blocks (plan:, etc.) above it that also use `enabled:`.
python3 -c "
import re, sys
path = '$CFG'
s = open(path).read()
pat = r'(\nrag:\s*\n(?:[ \t]+[^\n]*\n)*?[ \t]+enabled:\s*)(true|false)'
new, n = re.subn(pat, r'\1true', s, count=1)
if n != 1:
    sys.exit(f'failed to force rag.enabled = true in {path}')
open(path, 'w').write(new)
print('[diag_ab] forced rag.enabled = true')
"

set_diag() {
  local enabled="$1"
  python3 -c "
import re, sys
path = '$CFG'
val = '$enabled'
s = open(path).read()
new, n = re.subn(r'(\n\s*check_diagnostics:\s*)(true|false)', f'\\\\g<1>{val}', s, count=1)
if n != 1:
    sys.exit(f'failed to patch check_diagnostics in {path}')
open(path, 'w').write(new)
print(f'[diag_ab] set check_diagnostics = {val}')
"
}

for arm in "${ARMS[@]}"; do
  arm_dir="$RESULTS_DIR/diag_${arm}"
  log="$RESULTS_DIR/run_${arm}.log"

  if [ -f "$arm_dir/benchmark_summary.json" ]; then
    echo "[diag_ab] SKIP diag_${arm} (already exists)"
    continue
  fi

  echo "=================================================="
  echo "ARM: diag_${arm}"
  echo "=================================================="
  if [ "$arm" = "on" ]; then
    set_diag true
  else
    set_diag false
  fi

  rm -rf output
  echo "[diag_ab] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -8

  mkdir -p "$arm_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$arm_dir/"
    echo "[diag_ab] saved $arm_dir/benchmark_summary.json"
  else
    echo "[diag_ab] WARNING: no summary produced for diag_${arm}"
  fi
done

echo "[diag_ab] writing diff..."
uv run python scripts/diff_rag_ab.py "$RESULTS_DIR/diag_off/benchmark_summary.json" \
                                     "$RESULTS_DIR/diag_on/benchmark_summary.json" \
  | tee "$RESULTS_DIR/diff.txt"

echo "[diag_ab] DONE. Per-problem diff in $RESULTS_DIR/diff.txt"
