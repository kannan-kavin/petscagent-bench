# PETSc Agent Benchmark

An agentified evaluation framework for testing PETSc code generation agents using A2A (Agent-to-Agent) and MCP (Model Context Protocol) standards.

## Overview

This repository implements a multi-agent benchmark for evaluating code generation agents that produce PETSc (Portable, Extensible Toolkit for Scientific Computation) programs.
> [!IMPORTANT]
> 📖 See [MOTIVATION.md](MOTIVATION.md) for the motivation and design rationale behind this project.

## What's in this fork

This fork adds a **v2 Purple Agent** (`src/purple_agent_v2/`) that layers six independently-toggleable features on top of the baseline single-shot v1 agent. Each feature was designed to help the agent for *any* PETSc problem, not just the ones in the benchmark suite, and every feature ships with an A/B script under `scripts/` so the delta over baseline can be measured empirically.

| # | Feature | Config knob | What it does |
|---|---------|-------------|--------------|
| 1 | Self-verify loop | `self_fix.max_iters` | Compile + smoke-run the generated code via MCP; feed stderr back to the LLM on failure |
| 2 | Diagnostic-flag checks | `self_fix.check_diagnostics` | Inject `-ksp_converged_reason`, `-snes_converged_reason`, `-ts_monitor`; parse output for silent solver failures |
| 3 | RAG over PETSc tutorials | `rag.enabled` | FAISS vector search + flashrank cross-encoder rerank over ~295 tutorials (`src/petsc_rag/`) |
| 4 | Header lookup | `header_lookup.enabled` | Reactive PETSc signature injection on compile failure |
| 5 | Plan-then-code | `plan.enabled` | Separate "numerical analyst" LLM turn emits a structured `NumericalPlan` before code generation |
| 6 | Multi-plan + judge | `plan.num_plans > 1` | Generate N candidate plans from different angles (simplest / accurate / robust), judge LLM picks the winner |

**Multi-plan + judge (feature 6)** is currently the strongest configuration in this fork, delivering a ~3.2-point composite score improvement over the v1 baseline on the 6-problem active suite. Toggle any feature off in `config/purple_agent_v2_config.yaml` to reproduce the A/B.

Also new:
- `src/petsc_rag/` — the FAISS + flashrank retrieval pipeline (build the index once with `python -m petsc_rag.build_index`)
- `scripts/run_*_ab.sh` — one A/B harness per feature; each backs up the config, toggles the knob, runs both arms, and diffs
- `scripts/agg_*.py` — pure-Python aggregators for multi-repeat A/B runs (noise estimation)
- `snapshots/` — reference outputs from `claudeopus47`, `gemini25pro`, and `gpt52` on all 6 active problems
- `CLAUDE.md` — engineering notes covering the v1/v2 split, config nuances, and A/B workflow in depth

Core building blocks:

- **A2A Protocol**: standardized agent-to-agent communication over HTTP.
- **MCP Protocol**: tool access for compilation and execution.
- **Evaluation pipeline**: gates + metrics + LLM-based quality evaluators, aggregated into a composite score and tier.

High-level flow:

1. The **Green Agent** loads benchmark problems from `data/*.json`.
2. For each problem, it asks the **Purple Agent** to generate PETSc code.
3. It compiles and runs the returned code via MCP tools.
4. It evaluates results and writes reports to `output/`.

> Note: Running the benchmark can consume significant LLM tokens depending on the model and number of problems.

## Architecture

![petscagent-bench workflow](assets/workflow_dia.png)

The system consists of three components:

1. **Green Agent** (assessment manager)
   - Loads benchmark problems from `data/*.json`
   - Sends each problem description to the Purple Agent via A2A
   - Compiles and runs returned code via MCP tools
   - Scores results (gates + metrics + quality) and aggregates into a composite score + tier
   - Writes reports to `output/`

2. **Purple Agent** (target under test)
   - Receives a problem description via A2A
   - Uses an LLM to generate PETSc code
   - Returns:
     - a status text that includes `cli_args`
     - one or more code files

3. **MCP Server** (tool provider)
   - Provides compilation and execution tools for PETSc code (used by the Green Agent)

### Why PETSc?

PETSc is an ideal benchmark for evaluating LLM capabilities in scientific computing because it demands:

- **Domain expertise**: Numerical methods, PDEs, linear algebra, and parallel computing
- **Large API surface**: 1000+ functions across solvers (TS, SNES, KSP), data structures (Vec, Mat, DM), and optimizers (TAO)
- **Correctness and performance**: Solutions must be mathematically accurate *and* computationally efficient
- **Parallel programming**: MPI, domain decomposition, GPU acceleration (CUDA/HIP)

Unlike toy benchmarks, PETSc code generation tests whether LLMs can produce **scientifically valid, performant, and maintainable** solutions for real-world HPC applications. See [PETSc applications](https://petsc.org/main/miscellaneous/applications_publications/) for examples spanning climate modeling, CFD, astrophysics, and more.


## Evaluation System

At a high level, evaluation is organized into:

- **Gates**: binary pass/fail checks (e.g., compilation/execution/API usage)
- **Metrics**: quantitative measurements (e.g., numerical accuracy, execution time)
- **Quality**: LLM-based qualitative assessment (e.g., code style, algorithm choice, PETSc best practices)

> [!IMPORTANT]
> For full details on the evaluation design, scoring, and components, see [EVALUATION_SYSTEM_SUMMARY.md](EVALUATION_SYSTEM_SUMMARY.md).

## Benchmark Problems

Benchmark problems are defined as JSON files under `data/`. The Green Agent loads **all** JSON files in that directory. 

Each problem file is expected to contain (at minimum):

- `problem_name`
- `problem_id`
- `problem_description`

Current suite (see `data/` for full definitions):

- Robertson ODE
- 1D Advection
- Rosenbrock optimization
- Darcy flow
- 2D Navier–Stokes
- Vec/MPI tests

`gpu_data` contains problems that run on GPUs. Since our Github runners do not support GPU at the moment, they are not included in the default setting, but can be activated manually. 

### Evaluation Criteria

Each problem is evaluated across multiple dimensions (see `config/green_agent_config.yaml` for weights):

- Correctness
- Performance
- Code quality
- Algorithm choice
- PETSc best practices
- Semantic correctness

## Output

### Files written to disk

The Green Agent writes a single file to `output/`:

- `output/benchmark_summary.json`: overall summary + per-problem results

### Task artifacts

The Green Agent also emits A2A task artifacts (via `TaskUpdater.add_artifact`). Depending on your runner/integration, these may be downloadable from logs/UI but are not written to `output/` by default:

### Tier System

Codes are assigned to tiers based on composite scores:

- 🥇 **GOLD** (≥85): Excellent code quality and correctness
- 🥈 **SILVER** (≥70): Good code with minor issues
- 🥉 **BRONZE** (≥50): Functional but needs improvement
- ❌ **FAIL** (<50 or gate failure): Significant issues

## Project Structure

```
├── data/                           # Benchmark problems (JSON files)
│   └── archive/                    # Historical problems (still readable, not in default suite)
├── config/
│   ├── green_agent_config.yaml     # Green agent evaluation + scoring + LLM settings
│   └── purple_agent_v2_config.yaml # v2 Purple: self-fix, RAG, header lookup, plan, multi-plan
├── src/
│   ├── launcher.py                 # Spawns Green/Purple/MCP locally (end-to-end)
│   ├── green_agent/                # Assessment manager agent
│   ├── purple_agent/               # v1 target agent (single-shot baseline)
│   ├── purple_agent_v2/            # v2 target agent (self-verify + RAG + plan + judge)
│   ├── petsc_rag/                  # FAISS index + flashrank rerank over PETSc tutorials
│   ├── evaluators/                 # Gates / metrics / quality evaluators
│   └── util/                       # A2A helpers + LLM client
├── scripts/
│   ├── run_*_ab.sh                 # One A/B harness per v2 feature
│   ├── run_*_ab_repeat.sh          # N-repeat variants for noise estimation
│   └── agg_*.py, diff_*.py         # Aggregators and per-problem diff tools
├── snapshots/                      # Reference outputs from claudeopus47, gemini25pro, gpt52
├── main.py                         # CLI entry point (green/purple/purple-v2/launch)
├── pyproject.toml                  # Python project configuration
├── CLAUDE.md                       # Detailed engineering notes (v1/v2 split, config, A/B)
└── output/                         # Generated reports (gitignored; regenerated per run)
```

## Installation

### Prerequisites

1. **PETSc Installation**: Install PETSc from [https://petsc.org/](https://petsc.org/) for local compilation/execution.
2. **Python 3.12+**: Required (see `pyproject.toml`).
3. **uv**: Python package manager used by this repo: https://github.com/astral-sh/uv

### Setup

1. Install dependencies using `uv`:

```bash
uv sync
```

2. Create a `.env` file in the root directory with the following variables:

```bash
# LLM API Keys (only those actually used by your configured models are required)
OPENAI_API_KEY="<your_openai_key>"             # OpenAI or Argo/OpenAI-compatible proxy
GEMINI_API_KEY="<your_gemini_key>"             # only if you use gemini25pro directly
ASKSAGE_API_KEY="<your_asksage_key>"           # only if api_base_url starts with api.asksage.anl.gov
ASKSAGE_SSL_CERT_FILE="<path/to/cert.pem>"     # AskSage requires a custom SSL cert

# PETSc Configuration (required for compilation/execution)
PETSC_DIR="<path_to_petsc_installation>"
PETSC_ARCH="<petsc_architecture>"              # e.g., arch-darwin-c-debug
```

The repo is configured against ANL's **Argo proxy** by default (`https://apps-dev.inside.anl.gov/argoapi/v1`) — VPN required. See `CLAUDE.md` for endpoint-specific setup notes.

## Usage

### Quick Start

For local testing, launch the complete evaluation workflow. Pick which Purple variant to grade:

```bash
uv run main.py launch --purple-variant v2   # v2 agent (self-verify + RAG + plan + judge)
uv run main.py launch --purple-variant v1   # v1 baseline agent (single-shot)
```

Either command will:
1. Start the Green Agent (assessment manager, port 9001)
2. Start the chosen Purple Agent (port 9002)
3. Start the MCP server (compilation/execution tools, port 8080)
4. Run all benchmark problems in `data/`
5. Write `output/benchmark_summary.json`

**Before running v2 with RAG enabled**, build the tutorial index once:

```bash
uv run python -m petsc_rag.build_index
```

### Deploying Individual Components

You can run the components separately (useful when deploying services on different machines or restarting a single component during development).

```bash
uv run main.py green         # Green Agent only
uv run main.py purple        # v1 Purple Agent only
uv run main.py purple-v2     # v2 Purple Agent only
```

For MCP server deployment, refer to https://gitlab.com/petsc/petsc_mcp_servers.

### Reproducing the A/B experiments

Each v2 feature can be toggled off in isolation and compared against v2-with-that-feature-on. The `scripts/` directory contains one A/B harness per feature; each backs up the config, flips one knob, runs both arms, and prints a per-problem diff:

```bash
scripts/run_rag_ab.sh              # RAG off vs. on (holds everything else constant)
scripts/run_diag_ab.sh             # Diagnostic-flag checks off vs. on
scripts/run_plan_ab.sh             # Plan-then-code off vs. on
scripts/run_multiplan_ab.sh        # num_plans=1 vs. num_plans=3 (with judge)
scripts/run_header_ab.sh           # Header lookup off vs. on
scripts/run_selfverify_ab.sh       # Self-verify off vs. on
```

For noise estimation, the `*_repeat.sh` variants run each A/B N times into `<name>_results.run{1..N}/` (gitignored — regenerated locally). Aggregate with `scripts/agg_*.py`.


### Configuration

The system uses separate configuration files for each agent:

- `config/green_agent_config.yaml` — Green agent LLM model and evaluation settings
- `config/purple_agent_v2_config.yaml` — v2 Purple agent LLM + feature toggles (self-fix, RAG, plan, etc.)

Example `config/green_agent_config.yaml`:

```yaml
evaluation:
  enable_gates: true          # Enable binary pass/fail checks
  enable_metrics: true        # Enable quantitative measurements
  enable_quality: true        # Enable quality assessments
  parallel_evaluation: true   # Run evaluators in parallel
  
  llm:
    model: "openai/gpt52"            # LLM for quality evaluation
    api_base_url: "https://apps-dev.inside.anl.gov/argoapi/v1"  # Optional API base URL (e.g., Argo/AskSage)
    temperature: 0                    # Set to 0 only for reproducibility
    max_concurrent_calls: 3           # Rate limiting for LLM calls

scoring:
  weights:
    correctness: 0.35     # Weight for correctness score
    performance: 0.15     # Weight for performance metrics
    code_quality: 0.15    # Weight for code quality
    algorithm: 0.15       # Weight for algorithm choice
    petsc: 0.10          # Weight for PETSc best practices
    semantic: 0.10       # Weight for semantic correctness
  
  tiers:
    gold: 85      # Minimum score for GOLD tier
    silver: 70    # Minimum score for SILVER tier
    bronze: 50    # Minimum score for BRONZE tier
```

Example `config/purple_agent_v2_config.yaml` (abbreviated — see the file for all knobs and inline docs):

```yaml
llm:
  model: "openai/claudeopus47"
  api_base_url: "https://apps-dev.inside.anl.gov/argoapi/v1"
  temperature: 0                    # Set to 0 only for reproducibility

self_fix:
  max_iters: 3                      # Regeneration attempts on compile/run failure
  do_smoke_run: true
  check_diagnostics: true           # Parse solver output for silent failures

rag:
  enabled: true                     # Requires `python -m petsc_rag.build_index` first
  k: 4
  rerank: true
  k_initial: 12

header_lookup:
  enabled: true

plan:
  enabled: true
  num_plans: 3                      # 1 = single plan; >1 enables multi-plan + judge
  allow_plan_revision: true
```

**Note**:
- Use a LiteLLM-style name, e.g. `<provider_name>/<model_name>`. For models provided with an OpenAI-compatible endpoint, use `openai` as the provider name.
- Leave `api_base_url` `null` to use each provider’s default (e.g. `https://api.openai.com/v1`). Set this to use a custom or proxy endpoint (e.g. `https://apps-dev.inside.anl.gov/argoapi/v1` for Argo). For OpenAI-compatible APIs the URL should end with `/v1`; the client will use the appropriate LiteLLM provider prefix.
- The system auto-detects AskSage endpoints when `api_base_url` starts with `https://api.asksage.anl.gov` and configures SSL and API keys accordingly.


## Development

### Adding custom evaluators

Evaluators live under `src/evaluators/` and are wired into the pipeline in `src/evaluators/pipeline.py`.

To add a new evaluator:

1. Create a class inheriting from `src.evaluators.base.Evaluator`
2. Implement `name`, `evaluator_type`, and `evaluate(...)`
3. Add the evaluator to the pipeline

Example:

```python
from src.evaluators.base import Evaluator, EvaluatorType, EvaluationResult

class MyCustomEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "my_custom_check"
    
    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.QUALITY

    async def evaluate(self, code: str, problem: dict, execution_result: dict | None = None) -> EvaluationResult:
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_type=self.evaluator_type,
            quality_score=0.8,
            feedback="Custom evaluation passed",
            evaluation_method="deterministic",
            confidence=1.0,
        )
```

### Caching

The Green Agent can cache Purple Agent responses (pickled per problem) to speed up development iteration. Cached responses are stored in `purple_agent_cache/`.

## Troubleshooting

### Common Issues

1. **Wrong Python version**: This repo requires Python **3.12+** (see `pyproject.toml`).
2. **PETSc not found**: Ensure `PETSC_DIR` and `PETSC_ARCH` are set correctly in `.env`.
3. **LLM API/proxy errors**:
   - Verify API keys are valid and have sufficient quota.
   - If using an OpenAI-compatible proxy (Argo/AskSage), ensure `api_base_url` is set correctly in the relevant config.
   - For AskSage endpoints, ensure `ASKSAGE_API_KEY` and `ASKSAGE_SSL_CERT_FILE` are set.
4. **Agent connectivity / timeouts**:
   - Confirm the Green and Purple URLs/ports match your deployment.
   - If agents are slow to start, you may need to increase timeouts in `src/util/a2a_comm.py`.
5. **Port conflicts**: Modify ports in `src/launcher.py` if defaults are in use (Green `9001`, Purple `9002`, MCP `8080`).
6. **Missing output files**: Only `output/benchmark_summary.json` is written to disk by default; other reports are emitted as task artifacts.
