"""Aggregate A/B results across multiple runs, parameterized by archive prefix
and arm directory names.

Generalization of scripts/agg_rag_ab.py (which is hardcoded to RAG). Use this
for any A/B that follows the *_results.runN/<arm_a>|<arm_b>/benchmark_summary.json
layout.

Examples:
    # Header lookup A/B (this session)
    python scripts/agg_ab.py --prefix header_ab_darcy_results --arm-a hdr_off --arm-b hdr_on

    # Self-verify A/B (existing archives)
    python scripts/agg_ab.py --prefix selfverify_ab_results --arm-a sv_off --arm-b sv_on
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_arm(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {r["problem_name"]: r.get("composite_score", 0.0) for r in data.get("results", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="archive prefix, e.g. header_ab_darcy_results")
    ap.add_argument("--arm-a", required=True, help="arm A directory name, e.g. hdr_off")
    ap.add_argument("--arm-b", required=True, help="arm B directory name, e.g. hdr_on")
    args = ap.parse_args()

    runs = sorted(ROOT.glob(f"{args.prefix}.run*"))
    if not runs:
        sys.exit(f"no {args.prefix}.run* directories found")
    print(f"found {len(runs)} runs: {[r.name for r in runs]}")
    print(f"arms: a={args.arm_a}  b={args.arm_b}  (Δ = b − a)")
    print()

    by_problem: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        a = load_arm(r / args.arm_a / "benchmark_summary.json")
        b = load_arm(r / args.arm_b / "benchmark_summary.json")
        for name in set(a) | set(b):
            by_problem.setdefault(name, []).append((a.get(name, 0.0), b.get(name, 0.0)))

    n_runs = len(runs)

    def fmt_stat(vals: list[float]) -> str:
        if len(vals) <= 1:
            return f"{vals[0]:>6.2f}     "
        return f"{statistics.mean(vals):>6.2f} ±{statistics.stdev(vals):>5.2f}"

    header = (
        f"{'problem':35} "
        f"{args.arm_a + ' mean±std':>16} {args.arm_b + ' mean±std':>16} "
        f"{'Δ mean':>8} {'Δ std':>7} {'Δ range':>14}"
    )
    print(header)
    print("-" * len(header))

    sums = {"a": [], "b": [], "delta": []}
    for name in sorted(by_problem):
        runs_for_p = by_problem[name]
        a_vals = [a for a, _ in runs_for_p]
        b_vals = [b for _, b in runs_for_p]
        deltas = [b - a for a, b in runs_for_p]
        delta_mean = statistics.mean(deltas)
        delta_std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        delta_range = f"[{min(deltas):+.1f}, {max(deltas):+.1f}]"
        sums["a"].append(statistics.mean(a_vals))
        sums["b"].append(statistics.mean(b_vals))
        sums["delta"].append(delta_mean)
        print(
            f"{name[:35]:35} "
            f"{fmt_stat(a_vals):>16} {fmt_stat(b_vals):>16} "
            f"{delta_mean:>+8.2f} {delta_std:>7.2f} {delta_range:>14}"
        )

    print("-" * len(header))
    print()
    print(f"Aggregate across {len(by_problem)} problems × {n_runs} runs:")
    print(f"  mean {args.arm_a} composite : {statistics.mean(sums['a']):.2f}")
    print(f"  mean {args.arm_b} composite : {statistics.mean(sums['b']):.2f}")
    print(f"  mean {args.arm_b} − {args.arm_a}      : {statistics.mean(sums['delta']):+.2f}")
    print()
    print("Reading guide:")
    print("  - |Δ mean| > 2 × Δ std  →  probably real signal")
    print("  - Δ std large vs Δ mean →  noise; need more runs to call it")


if __name__ == "__main__":
    main()
