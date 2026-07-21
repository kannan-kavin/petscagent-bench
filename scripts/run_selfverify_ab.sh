#!/bin/bash
# A/B benchmark: purple_agent_v2 self-verification loop OFF vs ON.
#
# OFF arm: self_fix.max_iters = 1, self_fix.do_smoke_run = false
#   -> single LLM call, no compile-loop, no smoke-run, no diagnostic scan.
#   The v2 codepath still runs (per-problem build isolation, code shape),
#   but no retry feedback. Effectively v1-like behavior on top of v2's plumbing.
#
# ON arm: self_fix.max_iters = 3, self_fix.do_smoke_run = true
#   -> the current production v2 self-verify loop.
#
# RAG is held OFF in both arms (current config default) to isolate the loop.
# check_diagnostics stays true in config — moot in OFF arm (no smoke run).
#
# Output:
#   selfverify_ab_results/sv_off/benchmark_summary.json
#   selfverify_ab_results/sv_on/benchmark_summary.json
#   selfverify_ab_results/run_<arm>.log
#   selfverify_ab_results/diff.txt

set -e
cd /Users/kavinkannan/petscagent-bench

RESULTS_DIR=selfverify_ab_results
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

# Force RAG off in both arms (matches current default; explicit so this script
# is idempotent regardless of prior config state).
python3 -c "
import re
path = '$CFG'
s = open(path).read()
s = re.sub(r'(\n\s*enabled:\s*)(true|false)', r'\1false', s, count=1)
open(path, 'w').write(s)
print('[selfverify_ab] forced rag.enabled = false')
"

set_arm() {
  local arm="$1"
  python3 -c "
import re, sys
path = '$CFG'
arm = '$arm'
s = open(path).read()

if arm == 'off':
    iters_val = '1'
    smoke_val = 'false'
elif arm == 'on':
    iters_val = '3'
    smoke_val = 'true'
else:
    sys.exit(f'unknown arm: {arm}')

new, n1 = re.subn(r'(\n\s*max_iters:\s*)\d+', f'\\\\g<1>{iters_val}', s, count=1)
if n1 != 1:
    sys.exit(f'failed to patch max_iters in {path}')
new, n2 = re.subn(r'(\n\s*do_smoke_run:\s*)(true|false)', f'\\\\g<1>{smoke_val}', new, count=1)
if n2 != 1:
    sys.exit(f'failed to patch do_smoke_run in {path}')
open(path, 'w').write(new)
print(f'[selfverify_ab] arm={arm}: max_iters={iters_val}, do_smoke_run={smoke_val}')
"
}

for arm in "${ARMS[@]}"; do
  arm_dir="$RESULTS_DIR/sv_${arm}"
  log="$RESULTS_DIR/run_${arm}.log"

  if [ -f "$arm_dir/benchmark_summary.json" ]; then
    echo "[selfverify_ab] SKIP sv_${arm} (already exists)"
    continue
  fi

  echo "=================================================="
  echo "ARM: sv_${arm}"
  echo "=================================================="
  set_arm "$arm"

  rm -rf output
  echo "[selfverify_ab] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -8

  mkdir -p "$arm_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$arm_dir/"
    echo "[selfverify_ab] saved $arm_dir/benchmark_summary.json"
  else
    echo "[selfverify_ab] WARNING: no summary produced for sv_${arm}"
  fi
done

echo "[selfverify_ab] writing diff..."
uv run python scripts/diff_rag_ab.py "$RESULTS_DIR/sv_off/benchmark_summary.json" \
                                     "$RESULTS_DIR/sv_on/benchmark_summary.json" \
  | tee "$RESULTS_DIR/diff.txt"

echo "[selfverify_ab] DONE. Per-problem diff in $RESULTS_DIR/diff.txt"
