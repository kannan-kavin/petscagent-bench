#!/bin/bash
# A/B benchmark: purple_agent_v2 header_lookup OFF vs ON, full 6-problem suite.
#
# Both arms keep self-verify ON (max_iters=3, do_smoke_run=true) and RAG OFF.
# The only knob that moves is header_lookup.enabled.
#
# Output:
#   header_ab_results/hdr_off/benchmark_summary.json
#   header_ab_results/hdr_on/benchmark_summary.json
#   header_ab_results/run_<arm>.log
#   header_ab_results/diff.txt

set -e
cd /Users/kavinkannan/petscagent-bench

RESULTS_DIR=header_ab_results
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

# Force base config: rag.enabled=false, max_iters=3, do_smoke_run=true.
python3 -c "
import re
path = '$CFG'
s = open(path).read()
s = re.sub(r'(\n\s*enabled:\s*)(true|false)', r'\1false', s, count=1)  # first 'enabled:' is rag
s, _ = re.subn(r'(\n\s*max_iters:\s*)\d+', r'\g<1>3', s, count=1)
s, _ = re.subn(r'(\n\s*do_smoke_run:\s*)(true|false)', r'\g<1>true', s, count=1)
open(path, 'w').write(s)
print('[header_ab] base config: rag=off, max_iters=3, smoke=true')
"

set_arm() {
  local arm="$1"
  python3 -c "
import re
path = '$CFG'
arm = '$arm'
s = open(path).read()
val = 'true' if arm == 'on' else 'false'

if re.search(r'\nheader_lookup:\s*\n', s):
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
print(f'[header_ab] arm={arm}: header_lookup.enabled={val}')
"
}

for arm in "${ARMS[@]}"; do
  arm_dir="$RESULTS_DIR/hdr_${arm}"
  log="$RESULTS_DIR/run_${arm}.log"

  if [ -f "$arm_dir/benchmark_summary.json" ]; then
    echo "[header_ab] SKIP hdr_${arm} (already exists)"
    continue
  fi

  echo "=================================================="
  echo "ARM: hdr_${arm}"
  echo "=================================================="
  set_arm "$arm"

  rm -rf output
  echo "[header_ab] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -10

  mkdir -p "$arm_dir"
  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "$arm_dir/"
    echo "[header_ab] saved $arm_dir/benchmark_summary.json"
  else
    echo "[header_ab] WARNING: no summary produced for hdr_${arm}"
  fi
done

echo "[header_ab] writing diff..."
uv run python scripts/diff_rag_ab.py \
  "$RESULTS_DIR/hdr_off/benchmark_summary.json" \
  "$RESULTS_DIR/hdr_on/benchmark_summary.json" \
  | tee "$RESULTS_DIR/diff.txt" || true

echo "[header_ab] DONE. Per-problem diff in $RESULTS_DIR/diff.txt"
echo "[header_ab] grep run_on.log for '@@@ Purple agent v2: header lookup injected' to see which problems triggered"
