"""Blind cross-evaluation: each captured snapshot scored by every grader model.

For each (purple_snapshot, grader_model) cell, instantiate the LLM-based
quality evaluators with the grader model and run them on the snapshot code.
Output a per-dimension and aggregate matrix to detect self-preference bias.
"""

import asyncio
import json
import os
from pathlib import Path

import dotenv
dotenv.load_dotenv()

from src.evaluators.quality.code_quality.readability import ReadabilityQuality
from src.evaluators.quality.code_quality.code_style import CodeStyleQuality
from src.evaluators.quality.code_quality.documentation import DocumentationQuality
from src.evaluators.quality.algorithm_quality.algorithm_appropriateness import AlgorithmAppropriatenessQuality
from src.evaluators.quality.algorithm_quality.solver_choice import SolverChoiceQuality
from src.evaluators.quality.petsc_quality.best_practices import PETScBestPracticesQuality

SNAP_ROOT = Path("snapshots")
DATA_DIR = Path("data")
ARGO_URL = "https://apps-dev.inside.anl.gov/argoapi/v1"
PURPLES = ["gpt52", "claudeopus47", "gemini25pro"]
GRADERS = ["gpt52", "claudeopus47", "gemini25pro"]

PROBLEM_BY_C = {
    "Advection_PDE": "Advection.json",
    "DarcyFlow2D_Steady": "Darcyflow.json",
    "NS2D_FV_Implicit": "NS2D.json",
    "Robertson_ODE": "Robertson.json",
    "Rosenbrock_banana_function": "Rosenbrock.json",
    "scatter_vecmpi": "vecmpi.json",
}


def load_problems():
    problems = {}
    for cname, jname in PROBLEM_BY_C.items():
        with open(DATA_DIR / jname) as f:
            problems[cname] = json.load(f)
    return problems


def make_evaluators(grader_model: str):
    cfg = {
        "llm_model": f"openai/{grader_model}",
        "llm_api_base_url": ARGO_URL,
        "llm_temperature": 0,
        "use_llm": True,
    }
    return [
        ReadabilityQuality({**cfg}),
        CodeStyleQuality({**cfg}),
        DocumentationQuality({**cfg}),
        AlgorithmAppropriatenessQuality({**cfg}),
        SolverChoiceQuality({**cfg}),
        PETScBestPracticesQuality({**cfg}),
    ]


async def score_one(code: str, problem: dict, evaluators):
    out = {}
    for ev in evaluators:
        try:
            r = await ev.evaluate(code, problem, execution_result=None)
            out[ev.name] = r.quality_score if r.quality_score is not None else 0.0
        except Exception as e:
            out[ev.name] = None
            out[f"{ev.name}_err"] = str(e)[:200]
    return out


async def main():
    problems = load_problems()
    results = {}  # results[purple][grader][problem] = {dim: score}

    for purple in PURPLES:
        results[purple] = {}
        snap_dir = SNAP_ROOT / purple
        if not snap_dir.exists():
            print(f"missing snapshot dir {snap_dir}, skipping")
            continue
        for grader in GRADERS:
            print(f"\n=== purple={purple} grader={grader} ===")
            results[purple][grader] = {}
            evaluators = make_evaluators(grader)
            for cname in PROBLEM_BY_C:
                c_path = snap_dir / f"{cname}.c"
                if not c_path.exists():
                    print(f"  {cname}: no .c file in snapshot, skipping")
                    continue
                code = c_path.read_text()
                problem = problems[cname]
                scores = await score_one(code, problem, evaluators)
                results[purple][grader][cname] = scores
                avg = sum(v for v in scores.values() if isinstance(v, (int, float))) / max(
                    1, sum(1 for v in scores.values() if isinstance(v, (int, float)))
                )
                print(f"  {cname}: avg={avg:.2f}")

    out_path = Path("blind_cross_eval_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")

    # Per-purple per-grader mean across problems and dimensions
    print("\n=== Mean quality score (0-1) across all problems/dimensions ===")
    print(f"{'purple ↓ / grader →':28s}", end="")
    for g in GRADERS:
        print(f"  {g:14s}", end="")
    print()
    for purple in PURPLES:
        print(f"{purple:28s}", end="")
        for grader in GRADERS:
            vals = []
            for cname, dims in results.get(purple, {}).get(grader, {}).items():
                for v in dims.values():
                    if isinstance(v, (int, float)):
                        vals.append(v)
            mean = sum(vals) / len(vals) if vals else float("nan")
            print(f"  {mean:14.3f}", end="")
        print()


if __name__ == "__main__":
    asyncio.run(main())
