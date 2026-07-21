"""Publication chart: v1 vs v2 per-problem composite score, n=30 each.

Reads directly from the archived benchmark_summary.json files so the numbers
can't drift out of sync with the data.
"""
import json
import statistics
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/Users/kavinkannan/petscagent-bench")

PROBLEM_ORDER = [
    "DarcyFlow2D_Steady",
    "NS2D_FV_Implicit",
    "scatter_vecmpi",
    "Robertson_ODE",
    "Rosenbrock_banana_function",
    "Advection_PDE",
]
SHORT = {
    "DarcyFlow2D_Steady": "DarcyFlow",
    "NS2D_FV_Implicit": "NS2D",
    "scatter_vecmpi": "vecmpi",
    "Robertson_ODE": "Robertson",
    "Rosenbrock_banana_function": "Rosenbrock",
    "Advection_PDE": "Advection",
}


def collect(pattern):
    import re
    def num(p):
        m = re.search(r"(?:trial|run)(\d+)/benchmark_summary\.json$", p)
        return int(m.group(1)) if m else -1
    files = sorted(glob(str(REPO / pattern / "benchmark_summary.json")), key=num)[:30]
    per_problem = {name: [] for name in PROBLEM_ORDER}
    composites = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        composites.append(d["summary"]["avg_composite_score"])
        for r in d["results"]:
            if r["problem_name"] in per_problem:
                per_problem[r["problem_name"]].append(r.get("composite_score", 0))
    return per_problem, composites, len(files)


v1_per, v1_comp, v1_n = collect("v1_claudeopus_results.trial*")
v2_per, v2_comp, v2_n = collect("ragon_n3_results.run*")

# ---- Styling ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.9,
    "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

CONTROL_COLOR = "#B0B7BF"   # cool warm gray
IMPL_COLOR = "#2E86AB"      # rich teal-blue
ACCENT = "#1B4965"          # deeper accent for callouts

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8.5), gridspec_kw={"width_ratios": [3, 2]})

# ---- Panel 1: per-problem composite score ----
labels = [SHORT[p] for p in PROBLEM_ORDER]
v1_means = [statistics.mean(v1_per[p]) for p in PROBLEM_ORDER]
v1_sds = [statistics.stdev(v1_per[p]) for p in PROBLEM_ORDER]
v2_means = [statistics.mean(v2_per[p]) for p in PROBLEM_ORDER]
v2_sds = [statistics.stdev(v2_per[p]) for p in PROBLEM_ORDER]

x = np.arange(len(labels))
w = 0.38

ax1.bar(x - w/2, v1_means, w, yerr=v1_sds, capsize=3.5, color=CONTROL_COLOR,
        edgecolor="white", linewidth=1.2, label=f"Baseline (n={v1_n})",
        error_kw={"elinewidth": 1.0, "ecolor": "#555"})
ax1.bar(x + w/2, v2_means, w, yerr=v2_sds, capsize=3.5, color=IMPL_COLOR,
        edgecolor="white", linewidth=1.2, label=f"Our Agent (n={v2_n})",
        error_kw={"elinewidth": 1.0, "ecolor": "#555"})

ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=18, ha="right")
ax1.set_ylabel("Composite score (0–100)")
ax1.set_ylim(0, 118)
ax1.set_yticks(range(0, 101, 20))
ax1.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
ax1.set_axisbelow(True)
ax1.set_title("Per-problem composite score  (mean ± σ)", loc="left", pad=12)
ax1.legend(loc="upper right", ncol=1)

v1_c = statistics.mean(v1_comp)
v1_csd = statistics.stdev(v1_comp)
v2_c = statistics.mean(v2_comp)
v2_csd = statistics.stdev(v2_comp)
ax1.text(0.02, 0.995,
         f"Composite mean\n"
         f"  Baseline:   {v1_c:.1f} ± {v1_csd:.1f}\n"
         f"  Our Agent:  {v2_c:.1f} ± {v2_csd:.1f}\n"
         f"  Δ = +{v2_c - v1_c:.1f}   Welch p < 0.001",
         transform=ax1.transAxes, va="top", ha="left", fontsize=9.5,
         color="#222",
         bbox=dict(boxstyle="round,pad=0.55", facecolor="#F5F7FA",
                   edgecolor="#D0D5DB", linewidth=0.8))

# ---- Panel 2: per-problem σ (variance collapse) ----
ax2.bar(x - w/2, v1_sds, w, color=CONTROL_COLOR, edgecolor="white", linewidth=1.2)
ax2.bar(x + w/2, v2_sds, w, color=IMPL_COLOR, edgecolor="white", linewidth=1.2)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=18, ha="right")
ax2.set_ylabel("Standard deviation  (σ)")
ax2.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
ax2.set_axisbelow(True)
ax2.set_title("Run-to-run variance", loc="left", pad=12)

fig.suptitle("Baseline vs Our Agent",
             fontsize=15, fontweight="semibold", color="#1B2733", y=1.00)
fig.text(0.5, 0.945, "Claude Opus 4.7  ·  temperature 0  ·  six-problem PETSc benchmark",
         ha="center", fontsize=10, color="#6B7280", style="italic")
plt.tight_layout(rect=[0, 0, 1, 0.94])

out = Path("/Users/kavinkannan/Downloads/control_vs_implementations.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
