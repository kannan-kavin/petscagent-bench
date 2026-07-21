"""Aggregate RAG A/B results across multiple runs.

Reads every rag_ab_results.runN/ directory in the repo root and prints per-
problem mean / stddev / range for the OFF arm, ON arm, and the delta.

Used to tell signal from noise after running the A/B more than once.
"""

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
    runs = sorted(ROOT.glob("rag_ab_results.run*"))
    if not runs:
        sys.exit("no rag_ab_results.run* directories found")
    print(f"found {len(runs)} runs: {[r.name for r in runs]}")
    print()

    # problem -> list of (off_score, on_score) across runs
    by_problem: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        off = load_arm(r / "rag_off" / "benchmark_summary.json")
        on = load_arm(r / "rag_on" / "benchmark_summary.json")
        for name in set(off) | set(on):
            by_problem.setdefault(name, []).append((off.get(name, 0.0), on.get(name, 0.0)))

    n_runs = len(runs)

    def fmt_stat(vals: list[float]) -> str:
        if len(vals) <= 1:
            return f"{vals[0]:>6.2f}     "
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        return f"{mean:>6.2f} ±{std:>5.2f}"

    header = (
        f"{'problem':35} "
        f"{'off mean±std':>14} {'on mean±std':>14} "
        f"{'Δ mean':>8} {'Δ std':>7} {'Δ range':>14}"
    )
    print(header)
    print("-" * len(header))

    sums = {"off": [], "on": [], "delta": []}
    for name in sorted(by_problem):
        runs_for_p = by_problem[name]
        offs = [o for o, _ in runs_for_p]
        ons = [n for _, n in runs_for_p]
        deltas = [n - o for o, n in runs_for_p]

        delta_mean = statistics.mean(deltas)
        delta_std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        delta_range = f"[{min(deltas):+.1f}, {max(deltas):+.1f}]"

        sums["off"].append(statistics.mean(offs))
        sums["on"].append(statistics.mean(ons))
        sums["delta"].append(delta_mean)

        print(
            f"{name[:35]:35} "
            f"{fmt_stat(offs):>14} {fmt_stat(ons):>14} "
            f"{delta_mean:>+8.2f} {delta_std:>7.2f} {delta_range:>14}"
        )

    print("-" * len(header))
    print()
    print(f"Aggregate across {len(by_problem)} problems × {n_runs} runs:")
    print(f"  mean OFF composite : {statistics.mean(sums['off']):.2f}")
    print(f"  mean ON  composite : {statistics.mean(sums['on']):.2f}")
    print(f"  mean ON − OFF      : {statistics.mean(sums['delta']):+.2f}")
    print()
    print("Reading guide:")
    print("  - |Δ mean| > 2 × Δ std  →  probably real signal")
    print("  - Δ std large vs Δ mean →  noise; need more runs to call it")


if __name__ == "__main__":
    main()
