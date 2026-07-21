#!/bin/bash
# A/B benchmark: purple_agent_v2 plan.num_plans=1 (SINGLE) vs num_plans=3
# (MULTI: 3 candidate plans + judge), full 6-problem suite.
#
# Both arms keep:
#   - plan.enabled: true (this A/B is conditional on plan-then-code being on)
#   - header_lookup.enabled: true
#   - rag.enabled: false, max_iters: 3, do_smoke_run: true
# The only knob that moves is plan.num_plans.

set -e
cd /Users/kavinkannan/petscagent-bench

RESULTS_DIR=multiplan_ab_results
CFG=config/purple_agent_v2_config.yaml
CFG_BACKUP="${CFG}.bak.$$"
ARMS=(single multi)

mkdir -p "$RESULTS_DIR"

cp "$CFG" "$CFG_BACKUP"
restore_config() {
  cp "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT INT TERM

# Force base config: rag=off, max_iters=3, smoke=true, header_lookup=on, plan=on.
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
s = re.sub(
    r'(plan:\s*\n(?:\s*#[^\n]*\n)*\s*enabled:\s*)(true|false)',
    r'\1true',
    s,
    count=1,
)
open(path, 'w').write(s)
print('[multiplan_ab] base config: rag=off, max_iters=3, smoke=true, header_lookup=on, plan=on')
"

set_arm() {
  local arm="$1"
  local n
  if [ "$arm" = "multi" ]; then n=3; else n=1; fi
  python3 - "$CFG" "$n" "$arm" <<'PYEOF'
import re, sys
path, n, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
s = open(path).read()
if re.search(r'\n\s*num_plans:\s*\d+', s):
    s = re.sub(r'(\n\s*num_plans:\s*)\d+', lambda m: m.group(1) + str(n), s, count=1)
else:
    s = re.sub(
        r'(plan:\s*\n(?:\s*#[^\n]*\n|\s+[a-z_]+:[^\n]*\n)+)',
        lambda m: m.group(1) + f'  num_plans: {n}\n',
        s,
        count=1,
    )
open(path, 'w').write(s)
print(f'[multiplan_ab] arm={arm}: plan.num_plans={n}')
PYEOF
}

for arm in "${ARMS[@]}"; do
  arm_dir="$RESULTS_DIR/$arm"
  log="$RESULTS_DIR/run_${arm}.log"

  if [ -f "$arm_dir/benchmark_summary.json" ]; then
    echo "[multiplan_ab] SKIP $arm (already exists)"
    continue
  fi

  echo "=================================================="
  echo "ARM: $arm"
  echo "=================================================="
  set_arm "$arm"

  rm -rf output
  echo "[multiplan_ab] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -10

  mkdir -p "$arm_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$arm_dir/"
    echo "[multiplan_ab] saved $arm_dir/benchmark_summary.json"
  else
    echo "[multiplan_ab] WARNING: no summary produced for $arm"
  fi
done

echo "[multiplan_ab] writing diff..."
uv run python scripts/diff_rag_ab.py \
  "$RESULTS_DIR/single/benchmark_summary.json" \
  "$RESULTS_DIR/multi/benchmark_summary.json" \
  | tee "$RESULTS_DIR/diff.txt" || true

echo "[multiplan_ab] DONE. Per-problem diff in $RESULTS_DIR/diff.txt"
echo "[multiplan_ab] grep run_multi.log for '@@@ Purple agent v2: judge picked' for judge outcomes"
