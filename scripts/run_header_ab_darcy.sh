#!/bin/bash
# A/B benchmark: purple_agent_v2 header_lookup OFF vs ON, restricted to DarcyFlow.
#
# Hypothesis: the LLM keeps emitting DMPlexSetSNESLocalFEM with 4 args (wrong
# arity) and stderr-alone feedback isn't enough signal to unlearn it. Injecting
# the canonical 3-arg signature from $PETSC_DIR/include into the fix-it turn
# should let the loop recover.
#
# Both arms keep self-verify ON (max_iters=3, do_smoke_run=true, RAG off).
# The only knob that moves is header_lookup.enabled.
#
# Output:
#   header_ab_darcy_results/hdr_off/benchmark_summary.json
#   header_ab_darcy_results/hdr_on/benchmark_summary.json
#   header_ab_darcy_results/run_<arm>.log
#   header_ab_darcy_results/diff.txt

set -e
cd /Users/kavinkannan/petscagent-bench

RESULTS_DIR=header_ab_darcy_results
CFG=config/purple_agent_v2_config.yaml
CFG_BACKUP="${CFG}.bak.$$"
DATA_DIR=data
DATA_HOLD=".data_hold.$$"
KEEP_PROBLEM=Darcyflow.json
ARMS=(off on)

mkdir -p "$RESULTS_DIR"

cp "$CFG" "$CFG_BACKUP"
mkdir -p "$DATA_HOLD"
for f in "$DATA_DIR"/*.json; do
  base="$(basename "$f")"
  if [ "$base" != "$KEEP_PROBLEM" ]; then
    mv "$f" "$DATA_HOLD/"
  fi
done
echo "[header_ab_darcy] data restricted to: $(ls $DATA_DIR/*.json)"

restore_all() {
  cp "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
  if [ -d "$DATA_HOLD" ]; then
    mv "$DATA_HOLD"/*.json "$DATA_DIR/" 2>/dev/null || true
    rmdir "$DATA_HOLD" 2>/dev/null || true
  fi
  echo "[header_ab_darcy] restored config + data dir"
}
trap restore_all EXIT INT TERM

# Both arms: self-verify ON, RAG off, diagnostics on.
python3 -c "
import re
path = '$CFG'
s = open(path).read()
# rag.enabled -> false (first match is rag)
s = re.sub(r'(\n\s*enabled:\s*)(true|false)', r'\1false', s, count=1)
# max_iters -> 3
s, n1 = re.subn(r'(\n\s*max_iters:\s*)\d+', r'\g<1>3', s, count=1)
# do_smoke_run -> true
s, n2 = re.subn(r'(\n\s*do_smoke_run:\s*)(true|false)', r'\g<1>true', s, count=1)
open(path, 'w').write(s)
print(f'[header_ab_darcy] base config: rag=off, max_iters=3, smoke=true (patches: {n1},{n2})')
"

set_arm() {
  local arm="$1"
  python3 -c "
import re, sys
path = '$CFG'
arm = '$arm'
s = open(path).read()
val = 'true' if arm == 'on' else 'false'

# Patch header_lookup.enabled if present; if not, append the block.
if re.search(r'\nheader_lookup:\s*\n', s):
    # Replace the enabled: under header_lookup specifically.
    s = re.sub(
        r'(header_lookup:\s*\n(?:\s*#[^\n]*\n)*\s*enabled:\s*)(true|false)',
        r'\g<1>' + val,
        s,
        count=1,
    )
else:
    if not s.endswith('\n'):
        s += '\n'
    s += f'\nheader_lookup:\n  enabled: {val}\n  limit: 8\n'

open(path, 'w').write(s)
print(f'[header_ab_darcy] arm={arm}: header_lookup.enabled={val}')
"
}

for arm in "${ARMS[@]}"; do
  arm_dir="$RESULTS_DIR/hdr_${arm}"
  log="$RESULTS_DIR/run_${arm}.log"

  if [ -f "$arm_dir/benchmark_summary.json" ]; then
    echo "[header_ab_darcy] SKIP hdr_${arm} (already exists)"
    continue
  fi

  echo "=================================================="
  echo "ARM: hdr_${arm}"
  echo "=================================================="
  set_arm "$arm"

  rm -rf output
  echo "[header_ab_darcy] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -10

  mkdir -p "$arm_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$arm_dir/"
    echo "[header_ab_darcy] saved $arm_dir/benchmark_summary.json"
  else
    echo "[header_ab_darcy] WARNING: no summary produced for hdr_${arm}"
  fi
done

echo "[header_ab_darcy] writing diff..."
uv run python scripts/diff_rag_ab.py \
  "$RESULTS_DIR/hdr_off/benchmark_summary.json" \
  "$RESULTS_DIR/hdr_on/benchmark_summary.json" \
  | tee "$RESULTS_DIR/diff.txt" || true

echo "[header_ab_darcy] DONE. Per-problem diff in $RESULTS_DIR/diff.txt"
echo "[header_ab_darcy] grep run_on.log for '@@@ Purple agent v2: header lookup injected' to confirm the path fired"
