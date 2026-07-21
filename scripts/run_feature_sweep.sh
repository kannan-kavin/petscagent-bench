#!/bin/bash
# Per-feature marginal-effect sweep for purple_agent_v2.
#
# Runs ONE arm for N reps (default 3), archiving each rep into
#   feature_sweep_results.run{N}/<arm>/benchmark_summary.json
# and snapshotting the exact config used into
#   feature_sweep_results.run{N}/<arm>/config_snapshot.yaml
#
# Seven arms, each measures one feature block against a common all-off
# baseline (see plan for interpretation caveats — some features have prereqs):
#
#   baseline       max_iters=1 do_smoke=false diag=false rag=off head=off plan=off n_plans=1
#   self_fix       max_iters=3 do_smoke=false diag=false rag=off head=off plan=off n_plans=1
#   smoke_run      max_iters=3 do_smoke=true  diag=false rag=off head=off plan=off n_plans=1
#   check_diag     max_iters=3 do_smoke=true  diag=true  rag=off head=off plan=off n_plans=1
#   rag            max_iters=1 do_smoke=false diag=false rag=on  head=off plan=off n_plans=1
#   header_lookup  max_iters=3 do_smoke=false diag=false rag=off head=on  plan=off n_plans=1
#   plan_multi     max_iters=1 do_smoke=false diag=false rag=off head=on  plan=on  n_plans=3
#
# Usage:
#   ./scripts/run_feature_sweep.sh <arm> [reps]
#
# Skip-if-exists: reruns are safe — arms with an existing
# benchmark_summary.json are skipped, so a crash mid-sweep can resume by
# re-invoking the same command.

set -e
cd /Users/kavinkannan/petscagent-bench

ARM="${1:-}"
REPS="${2:-3}"

VALID_ARMS=(baseline self_fix smoke_run check_diag rag header_lookup plan_multi)
if [ -z "$ARM" ]; then
  echo "usage: $0 <arm> [reps]"
  echo "arms:  ${VALID_ARMS[*]}"
  exit 1
fi
if ! printf '%s\n' "${VALID_ARMS[@]}" | grep -qx "$ARM"; then
  echo "error: unknown arm '$ARM'"
  echo "arms:  ${VALID_ARMS[*]}"
  exit 1
fi

CFG=config/purple_agent_v2_config.yaml
CFG_BACKUP="${CFG}.bak.$$"
RESULTS_ROOT_PREFIX=feature_sweep_results

# Pre-flight: RAG index must exist for arms that use it.
if [ "$ARM" = "rag" ] || [ "$ARM" = "plan_multi" ]; then
  # plan_multi doesn't strictly need the RAG tutorial index, but keep the check
  # off for it. Only rag actually reads src/petsc_rag/index/.
  if [ "$ARM" = "rag" ] && [ ! -d src/petsc_rag/index ] ; then
    echo "error: arm '$ARM' requires the RAG index at src/petsc_rag/index/"
    echo "       build it with: uv run python -m petsc_rag.build_index"
    exit 1
  fi
fi

cp "$CFG" "$CFG_BACKUP"
restore_config() {
  cp "$CFG_BACKUP" "$CFG"
  rm -f "$CFG_BACKUP"
}
trap restore_config EXIT INT TERM

# Apply full 7-field config state for the requested arm. Starts from the
# pristine backup every time so this is idempotent regardless of prior state
# (unlike scripts that layer diffs).
apply_arm_config() {
  local arm="$1"
  python3 - "$CFG_BACKUP" "$CFG" "$arm" <<'PYEOF'
import re, sys
src_path, dst_path, arm = sys.argv[1], sys.argv[2], sys.argv[3]

# arm -> (max_iters, do_smoke_run, check_diagnostics, rag_enabled,
#         header_lookup_enabled, plan_enabled, num_plans)
ARMS = {
    'baseline':      (1, 'false', 'false', 'false', 'false', 'false', 1),
    'self_fix':      (3, 'false', 'false', 'false', 'false', 'false', 1),
    'smoke_run':     (3, 'true',  'false', 'false', 'false', 'false', 1),
    'check_diag':    (3, 'true',  'true',  'false', 'false', 'false', 1),
    'rag':           (1, 'false', 'false', 'true',  'false', 'false', 1),
    'header_lookup': (3, 'false', 'false', 'false', 'true',  'false', 1),
    'plan_multi':    (1, 'false', 'false', 'false', 'true',  'true',  3),
}
mi, smoke, diag, rag, head, plan, nplans = ARMS[arm]

s = open(src_path).read()

def sub_or_die(pat, repl, s, label, count=1):
    new, n = re.subn(pat, repl, s, count=count)
    if n != count:
        sys.exit(f'failed to patch {label} (matched {n}, expected {count})')
    return new

# self_fix.max_iters — anchor on '\n  max_iters:' inside self_fix block.
s = sub_or_die(r'(\n\s*max_iters:\s*)\d+', rf'\g<1>{mi}', s, 'self_fix.max_iters')

# self_fix.do_smoke_run
s = sub_or_die(r'(\n\s*do_smoke_run:\s*)(true|false)', rf'\g<1>{smoke}', s, 'self_fix.do_smoke_run')

# self_fix.check_diagnostics
s = sub_or_die(r'(\n\s*check_diagnostics:\s*)(true|false)', rf'\g<1>{diag}', s, 'self_fix.check_diagnostics')

# rag.enabled — scope to the rag: block.
s = sub_or_die(
    r'(\nrag:\s*\n(?:\s*#[^\n]*\n)*\s*enabled:\s*)(true|false)',
    rf'\g<1>{rag}', s, 'rag.enabled',
)

# header_lookup.enabled — scope to the header_lookup: block.
s = sub_or_die(
    r'(\nheader_lookup:\s*\n(?:\s*#[^\n]*\n)*\s*enabled:\s*)(true|false)',
    rf'\g<1>{head}', s, 'header_lookup.enabled',
)

# plan.enabled — scope to the plan: block.
s = sub_or_die(
    r'(\nplan:\s*\n(?:\s*#[^\n]*\n)*\s*enabled:\s*)(true|false)',
    rf'\g<1>{plan}', s, 'plan.enabled',
)

# plan.num_plans — the field exists in the shipped config, so a straight
# substitution works. Guard with count=1 in case it's ever missing.
if re.search(r'\n\s*num_plans:\s*\d+', s):
    s = sub_or_die(r'(\n\s*num_plans:\s*)\d+', rf'\g<1>{nplans}', s, 'plan.num_plans')
else:
    sys.exit('plan.num_plans field missing from config — add a default before rerunning')

open(dst_path, 'w').write(s)
print(f'[feature_sweep] arm={arm}: max_iters={mi} do_smoke={smoke} diag={diag} '
      f'rag={rag} header_lookup={head} plan={plan} num_plans={nplans}')
PYEOF
}

# Find the highest existing rep index so numbering is monotonic across invocations.
last=0
for d in ${RESULTS_ROOT_PREFIX}.run*; do
  [ -d "$d" ] || continue
  n="${d#${RESULTS_ROOT_PREFIX}.run}"
  case "$n" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$n" -gt "$last" ]; then last="$n"; fi
done

# Count how many reps for this arm already exist so we only add missing ones.
existing=0
for i in $(seq 1 "$last"); do
  if [ -f "${RESULTS_ROOT_PREFIX}.run${i}/${ARM}/benchmark_summary.json" ]; then
    existing=$((existing + 1))
  fi
done

if [ "$existing" -ge "$REPS" ]; then
  echo "[feature_sweep] arm=${ARM} already has ${existing}/${REPS} reps; nothing to do."
  echo "[feature_sweep] delete a run directory to force regeneration."
  exit 0
fi

echo "[feature_sweep] arm=${ARM}: ${existing}/${REPS} reps already present; running $((REPS - existing)) more."

apply_arm_config "$ARM"

# Fill in the missing reps. Only run (REPS - existing) times; each iteration
# picks the lowest empty slot.
needed=$((REPS - existing))
for rep in $(seq 1 "$needed"); do
  # Pick the lowest run index whose <arm>/ directory is missing.
  rep_dir=""
  for i in $(seq 1 $((last + REPS))); do
    candidate="${RESULTS_ROOT_PREFIX}.run${i}/${ARM}"
    if [ ! -f "${candidate}/benchmark_summary.json" ]; then
      rep_dir="$candidate"
      run_root="${RESULTS_ROOT_PREFIX}.run${i}"
      break
    fi
  done

  if [ -z "$rep_dir" ]; then
    echo "[feature_sweep] no more slots to fill (unexpected); stopping."
    break
  fi

  mkdir -p "$rep_dir"
  log="${rep_dir}/run.log"

  echo "=================================================="
  echo "ARM: ${ARM}   REP -> ${rep_dir}"
  echo "=================================================="

  # Snapshot the exact config being used for post-hoc verification.
  cp "$CFG" "${rep_dir}/config_snapshot.yaml"

  rm -rf output
  echo "[feature_sweep] launching..."
  uv run main.py launch --purple-variant v2 2>&1 | tee "$log" | tail -10

  if [ -f output/benchmark_summary.json ]; then
    cp output/benchmark_summary.json "${rep_dir}/"
    echo "[feature_sweep] saved ${rep_dir}/benchmark_summary.json"
  else
    echo "[feature_sweep] WARNING: no summary produced for ${rep_dir}"
  fi
done

echo "[feature_sweep] DONE. Aggregate with: uv run python scripts/agg_feature_sweep.py"
