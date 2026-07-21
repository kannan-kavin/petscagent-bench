"""Aggregate feature_sweep_results.run*/ across arms and reps.

Companion to scripts/run_feature_sweep.sh. Reads every
feature_sweep_results.runN/<arm>/benchmark_summary.json in the repo root and
prints per-arm mean/stddev of the average composite score, plus Δ vs baseline
and (where meaningful) isolated marginal deltas for stacked features.

Reading guide baked into the output — noise floor on this suite is ≈±2
composite points from historical A/B σ; anything smaller than 2× the run-to-
run σ should be treated as inconclusive.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARMS = [
    ("baseline",      "v1-equivalent floor"),
    ("self_fix",      "compile-fix loop alone"),
    ("smoke_run",     "+runtime feedback (needs self_fix)"),
    ("check_diag",    "+PETSc diag parsing (needs smoke_run)"),
    ("rag",           "retrieval alone"),
    ("header_lookup", "header hints on compile failure (needs self_fix)"),
    ("plan_multi",    "plan-then-code w/ 3+judge (uses header_lookup)"),
]

# For features that stack on prereqs, report the isolated marginal
# by subtracting the prereq arm rather than baseline.
PREREQ = {
    "smoke_run":     "self_fix",
    "check_diag":    "smoke_run",
    "header_lookup": "self_fix",
}


def load_arm(path: Path) -> dict[str, float]:
    """Return {problem_name: composite_score} for one benchmark_summary.json.

    Missing file → {}. Empty dict tells callers this rep didn't run.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {r["problem_name"]: r.get("composite_score", 0.0) for r in data.get("results", [])}


def fmt_mean_std(vals: list[float]) -> str:
    if not vals:
        return "     —      "
    if len(vals) == 1:
        return f"{vals[0]:>6.2f}      "
    return f"{statistics.mean(vals):>6.2f} ±{statistics.stdev(vals):>5.2f}"


def main() -> None:
    runs = sorted(ROOT.glob("feature_sweep_results.run*"))
    if not runs:
        sys.exit("no feature_sweep_results.run* directories found")

    # arm -> list of per-rep avg composite scores (one entry per rep that
    # actually produced a benchmark_summary.json).
    # arm -> problem -> list of per-rep composite scores (for the per-problem table).
    arm_reps: dict[str, list[float]] = {name: [] for name, _ in ARMS}
    arm_per_problem: dict[str, dict[str, list[float]]] = {name: {} for name, _ in ARMS}

    for run in runs:
        for arm_name, _ in ARMS:
            scores = load_arm(run / arm_name / "benchmark_summary.json")
            if not scores:
                continue
            arm_reps[arm_name].append(statistics.mean(scores.values()))
            for prob, val in scores.items():
                arm_per_problem[arm_name].setdefault(prob, []).append(val)

    print(f"aggregating {len(runs)} run dirs: {[r.name for r in runs]}")
    print()

    # ---------- top-level per-arm table ----------
    header = (
        f"{'arm':16} {'reps':>4} "
        f"{'avg composite ± σ':>20} "
        f"{'Δ vs baseline':>14} "
        f"{'isolated Δ (vs prereq)':>24} "
        f"description"
    )
    print(header)
    print("-" * len(header))

    baseline_mean = (
        statistics.mean(arm_reps["baseline"]) if arm_reps["baseline"] else None
    )
    if baseline_mean is None:
        print("  (baseline arm not yet run — Δ vs baseline unavailable)")

    for arm_name, desc in ARMS:
        reps = arm_reps[arm_name]
        n = len(reps)
        mean_std = fmt_mean_std(reps)

        if arm_name == "baseline":
            delta_str = "     —      "
        elif not reps or baseline_mean is None:
            delta_str = "  (incomplete)"
        else:
            delta_str = f"{statistics.mean(reps) - baseline_mean:+7.2f}    "

        # Isolated marginal for arms that ride on a prereq.
        isolated_str = "          —            "
        prereq = PREREQ.get(arm_name)
        if prereq and reps and arm_reps[prereq]:
            iso_delta = statistics.mean(reps) - statistics.mean(arm_reps[prereq])
            isolated_str = f"  {iso_delta:+7.2f} vs {prereq:<12}"

        marker = "" if n >= 3 else f"  (only {n} rep{'s' if n != 1 else ''})"
        print(
            f"{arm_name:16} {n:>4} "
            f"{mean_std:>20} "
            f"{delta_str:>14} "
            f"{isolated_str:>24} "
            f"{desc}{marker}"
        )
    print()

    # ---------- per-problem breakdown vs baseline ----------
    if baseline_mean is not None:
        problems = sorted({p for arm in arm_per_problem.values() for p in arm})
        if problems:
            print("per-problem composite score, per arm (mean across reps):")
            col_arm_names = [a for a, _ in ARMS]
            head = f"{'problem':32} " + " ".join(f"{a[:12]:>12}" for a in col_arm_names)
            print(head)
            print("-" * len(head))
            for prob in problems:
                row = f"{prob[:32]:32} "
                for a in col_arm_names:
                    vals = arm_per_problem[a].get(prob, [])
                    if vals:
                        row += f"{statistics.mean(vals):>12.2f} "
                    else:
                        row += f"{'—':>12} "
                print(row)
            print()

    # ---------- interpretation guide ----------
    print("interpretation:")
    print("  - 'Δ vs baseline' is cumulative — stacked features (smoke_run,")
    print("    check_diag, header_lookup) also include contributions from")
    print("    their prereqs. Use 'isolated Δ' for the real marginal.")
    print("  - noise floor on this suite ≈ ±2 composite points (from")
    print("    historical A/B σ). |Δ| < 2σ ≈ inconclusive.")
    print("  - plan_multi runs at temp=0 with 3 prompt variants + judge;")
    print("    reruns are deterministic per problem so σ across reps under-")
    print("    represents true variance for this arm.")


if __name__ == "__main__":
    main()
