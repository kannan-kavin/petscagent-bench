"""Pretty-print a side-by-side comparison of two benchmark_summary.json files.

Used by scripts/run_rag_ab.sh to show the RAG-off vs. RAG-on delta after both
arms have been run.

Usage:
    python scripts/diff_rag_ab.py rag_ab_results/rag_off/benchmark_summary.json \
                                  rag_ab_results/rag_on/benchmark_summary.json
"""

import json
import sys
from pathlib import Path


TIER_RANK = {"GOLD": 3, "SILVER": 2, "BRONZE": 1, "FAIL": 0}


def load(path: Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    return {r["problem_name"]: r for r in data.get("results", [])}, data.get("summary", {})


def fmt_delta(d: float) -> str:
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.2f}"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: diff_rag_ab.py <off.json> <on.json>")
    off_path, on_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not off_path.exists() or not on_path.exists():
        sys.exit(f"missing summary file(s): off={off_path.exists()} on={on_path.exists()}")

    off, off_summary = load(off_path)
    on, on_summary = load(on_path)

    problems = sorted(set(off) | set(on))

    header = f"{'problem':35} {'off_score':>10} {'on_score':>10} {'delta':>8}  off→on tier"
    print(header)
    print("-" * len(header))

    wins = losses = unchanged = 0
    tier_changes = []

    for name in problems:
        o = off.get(name, {})
        n = on.get(name, {})
        os_, ns_ = o.get("composite_score", 0.0), n.get("composite_score", 0.0)
        ot, nt = o.get("tier", "FAIL"), n.get("tier", "FAIL")
        delta = ns_ - os_
        if delta > 0.5:
            wins += 1
        elif delta < -0.5:
            losses += 1
        else:
            unchanged += 1
        tier_arrow = f"{ot} → {nt}" if ot != nt else ot
        if ot != nt:
            tier_changes.append((name, ot, nt))
        print(f"{name[:35]:35} {os_:>10.2f} {ns_:>10.2f} {fmt_delta(delta):>8}  {tier_arrow}")

    print("-" * len(header))
    print()
    print(f"Per-problem composite score (delta > 0.5 = win):")
    print(f"  RAG wins:      {wins}")
    print(f"  RAG losses:    {losses}")
    print(f"  Unchanged:     {unchanged}")
    print()
    print(f"Average composite score:")
    print(f"  RAG off: {off_summary.get('avg_composite_score', 0):.2f}")
    print(f"  RAG on:  {on_summary.get('avg_composite_score', 0):.2f}")
    print()
    print(f"Tier distribution:")
    print(f"  RAG off: {off_summary.get('tier_distribution', {})}")
    print(f"  RAG on:  {on_summary.get('tier_distribution', {})}")
    if tier_changes:
        print()
        print("Tier changes:")
        for name, ot, nt in tier_changes:
            mark = "↑" if TIER_RANK.get(nt, 0) > TIER_RANK.get(ot, 0) else "↓"
            print(f"  {mark} {name}: {ot} → {nt}")


if __name__ == "__main__":
    main()
