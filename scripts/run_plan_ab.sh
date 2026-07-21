#!/bin/bash
# A/B benchmark: purple_agent_v2 plan.enabled OFF vs ON, full 6-problem suite.
#
# Both arms keep self-verify ON (max_iters=3, do_smoke_run=true), RAG OFF,
# header_lookup ON. The only knob that moves is plan.enabled.
#
# Output:
#   plan_ab_results/plan_off/benchmark_summary.json
#   plan_ab_results/plan_on/benchmark_summary.json
#   plan_ab_results/run_<arm>.log
#   plan_ab_results/diff.txt

set -e
cd /Users/kavinkannan/petscagent-bench

RESULTS_DIR=plan_ab_results
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

# Force base config: rag.enabled=false, max_iters=3, do_smoke_run=true,
# header_lookup.enabled=true. The only varying knob is plan.enabled.
python3 -c "
import re
path = '$CFG'
s = open(path).read()
s = re.sub(r'(\n\s*enabled:\s*)(true|false)', r'\1false', s, count=1)  # first 'enabled:' is rag
s, _ = re.subn(r'(\n\s*max_iters:\s*)\d+', r'\g<1>3', s, count=1)
s, _ = re.subn(r'(\n\s*do_smoke_run:\s*)(true|false)', r'\g<1>true', s, count=1)
s = re.sub(
    r'(header_lookup:\s*\n(?:\s*#[^\n]*\n)*\s*enabled:\s*)(true|false)',
    r'\1true',
    s,
    count=1,
)
open(path, 'w').write(s)
print('[plan_ab] base config: rag=off, max_iters=3, smoke=true, header_lookup=on')
"

set_arm() {
  local arm="$1"
  python3 -c "
import re
path = '$CFG'
arm = '$arm'
s = open(path).read()
val = 'true' if arm == 'on' else 'false'

if re.search(r'\nplan:\s*\n', s):
    s = re.sub(
        r'(plan:\s*\n(?:\s*#[^\n]*\n)*\s*enabled:\s*)(true|false)',
        r'\g<1>' + val,
        s,
        count=1,
    )
else:
    if not s.endswith('\n'):
        s += '\n'
    s += f'\nplan:\n  enabled: {val}\n  allow_plan_revision: true\n  emit_to_green: true\n'

open(path, 'w').write(s)
print(f'[plan_ab] arm={arm}: plan.enabled={val}')
"
}

for arm in "${ARMS[@]}"; do
  arm_dir="$RESULTS_DIR/plan_${arm}"
  log="$RESULTS_DIR/run_${arm}.log"

  if [ -f "$arm_dir/benchmark_summary.json" ]; then
    echo "[plan_ab] SKIP plan_${arm} (already exists)"
    continue
  fi

  echo "=================================================="
  echo "ARM: plan_${arm}"
  echo "=================================================="
  set_arm "$arm"

  rm -rf output
  echo "[plan_ab] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -10

  mkdir -p "$arm_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$arm_dir/"
    echo "[plan_ab] saved $arm_dir/benchmark_summary.json"
  else
    echo "[plan_ab] WARNING: no summary produced for plan_${arm}"
  fi
done

echo "[plan_ab] writing diff..."
uv run python scripts/diff_rag_ab.py \
  "$RESULTS_DIR/plan_off/benchmark_summary.json" \
  "$RESULTS_DIR/plan_on/benchmark_summary.json" \
  | tee "$RESULTS_DIR/diff.txt" || true

echo "[plan_ab] DONE. Per-problem diff in $RESULTS_DIR/diff.txt"
echo "[plan_ab] grep run_on.log for '@@@ Purple agent v2: plan generated' / 'revising plan' for plan firings"
