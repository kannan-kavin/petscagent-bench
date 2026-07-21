# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Background reading — read before proposing structural changes

Two PDFs in `~/Downloads/` describe (a) the larger DOE project this Purple Agent will eventually slot into, and (b) the benchmark methodology this repo implements. Read the relevant one before suggesting any architectural change — proposals that ignore the project's hierarchical decomposition or the benchmark's scoring categories will be rejected.

- **`~/Downloads/Proposal.18B.PDEs.2026.04-final-narrative.pdf`** — *Automated Problem-to-Solution Generation for PDE-Based Simulation Science* (Argonne, McInnes et al., DE-FOA-0003612, Phase I, 9-month plan). Describes a **hierarchical multi-agent system** for end-to-end PDE simulation: a problem-formulation specialist, a **Numerical Analysis specialist agent** (discretization selection, solver identification, configuration generation, performance-aware adaptation — retrieval-augmented in Phase I, telemetry-learning in Phase II), and an **HPC Code Generation and Execution specialist** (API selection, code synthesis, scalability-aware generation, build/runtime checks with embedded verification — MMS, convergence studies, conservation checks). **The user's role in this larger project is building the Numerical Analysis specialist agent.** The Phase I go/no-go targets (85% build+execute+verify success, efficiency measured in wall-clock and tokens, baseline = human + Claude Code) frame what "good enough" means.
- **`~/Downloads/2603.15976v1.pdf`** — *An Agentic Evaluation Framework for AI-Generated Scientific Code in PETSc* (Zhang et al., arXiv 2603.15976, Mar 2026). The paper for **this repo**: defines the Green/Purple agents-evaluating-agents architecture, the A2A + MCP protocols, the 14-evaluator pipeline (3 stages: gates → metrics → quality assessments) across 5 scoring categories (correctness, performance, code, appropriateness, library-specific), and the 6-problem suite. Anything you change about scoring, gates, or the wire protocol must be checked against this paper; the paper's key empirical finding — frontier LLMs produce readable code but universally fail on library-specific metrics — is the gap the Purple Agent improvements are trying to close.

When reasoning about Purple Agent improvements: tie each proposal to (i) a scoring category from the benchmark paper that it should move and (ii) a capability the Numerical Analysis or HPC Code Generation specialist needs in the proposal's architecture. Improvements that don't map to both are likely benchmark hacks or local optimizations that won't transfer to the larger system.

## What this repo is

A two-agent benchmark for evaluating LLMs that generate PETSc (HPC numerical solvers) C/C++/CUDA code. A **Green Agent** loads problems from `data/*.json`, hands each to a **Purple Agent** over A2A, compiles + runs the returned code via an MCP server, and scores it with gates / metrics / LLM-judged quality. README.md and MOTIVATION.md cover the high-level design. EVALUATION_SYSTEM_SUMMARY.md is the authoritative scoring reference.

The active suite is **6 problems**: Advection, Darcyflow, NS2D, Robertson, Rosenbrock, vecmpi. Poisson1D and FuelRod2D were moved to `data/archive/` (still readable for historical comparison; Green's loader uses `Path.iterdir() + isfile()` so the subdir is silently skipped). Older A/B archives, the 3-rep RAG and diag aggregates, and `scripts/archive/run_3x3{,_convergence,_fuelrod}.sh` were generated against the 8-problem suite — when comparing to new runs, account for the two dropped problems.

## Running and developing

Python ≥3.12 (<3.14), managed by `uv`. After `uv sync`, the canonical entry point is:

```bash
uv run main.py launch --purple-variant v2   # full end-to-end against v2 (or v1)
```

`launch` spawns: Green Agent (port **9001**), the chosen Purple variant (**9002**), and the PETSc compile-run MCP server (**8080**) in one process. `--purple-variant` is the most important knob — `v1` is the baseline, `v2` is the self-verifying fork (see below). Both write the same `output/benchmark_summary.json`.

For component-level testing:

```bash
uv run main.py green       # Green only
uv run main.py purple      # v1 Purple only
uv run main.py purple-v2   # v2 Purple only
uv run src/client_cli.py --green-url ... --purple-url ... --mcp-server-url ...
```

There is no test suite. Validation happens by running the benchmark and reading `output/benchmark_summary.json` (or any `rag_ab_results*/diff.txt` — see the A/B scripts below).

## The v1 / v2 split — the main thing to know

`src/purple_agent/petsc_agent.py` is the **baseline (v1)** agent: single LLM call, returns `{codes, nsize, cli_args}` over A2A.

`src/purple_agent_v2/petsc_agent.py` is a **fork** of v1 that adds three things, all controlled by `config/purple_agent_v2_config.yaml`:

1. **Self-verification loop**: after generation, the agent uploads its code to the MCP server, runs `make`, then `run_executable`. Compile failure → fix-it user turn with compiler stderr → regenerate (up to `self_fix.max_iters`, default 3). Same for runtime failures. Same `max_iters` budget is shared with all failure classes.
2. **Diagnostic-flag parsing** (`self_fix.check_diagnostics`): injects PETSc flags like `-ksp_converged_reason`, `-snes_converged_reason`, `-ts_monitor` into the smoke run, then parses output for `DIVERGED_*`, NaN/Inf, and `[N]PETSC ERROR:` blocks. Hits trigger another fix-it turn. **`-options_left` was deliberately dropped** — it false-fires on programs that pass a flag only one of several solver objects consumes, and the resulting fix pressure corrupted working code (see the inline comment in `petsc_agent.py` and `diag_ab_results.with_options_left/`). **Empirical note from `diag_ab_results.run{1,2,3}/`** (run against the now-historical 8-problem suite): the narrowed feature is a no-op on that bench (Δ = -0.86 ±~2 across 3 reps; the failure modes it catches don't occur often enough in Claude's output here). It's kept because its real-user value — surfacing silent solver failures when no oracle exists — doesn't show up in bench scoring. Don't re-litigate this without a broader problem set; instead, find a different signal for the FAIL/BRONZE problems (NS2D, DarcyFlow, Robertson), which fail by silently computing wrong answers, not by emitting PETSc diagnostics.
3. **RAG over the PETSc tutorial corpus** (`rag.enabled`): see next section.

The two agents share the A2A wire protocol exactly, so the Green Agent grades them identically. Per-problem build isolation: v2 renames the first generated file to `{sanitized_context_id}.c`, so concurrent problems don't collide on the MCP server.

## RAG layer (`src/petsc_rag/`)

Used only by v2. One-time setup:

```bash
uv run python -m petsc_rag.build_index   # indexes ~295 PETSc tutorials with sentence-transformers + FAISS
```

`retrieve.py` does two-stage retrieval: vector search returns `k_initial` (default 12) candidates, then an optional flashrank cross-encoder reranker picks the top `k` (default 4). The reranker was added specifically because vanilla vector search ranked the literal Robertson tutorial at #5 for the Robertson problem. RAG hits are spliced into the system prompt on the **first turn only** per A2A `context_id`, so subsequent fix-it turns don't re-retrieve based on stderr.

If the index isn't built, the `scripts/run_rag_ab.sh` sanity-check bails out before silently running RAG-off in both arms.

## Configuration nuances

Both `config/green_agent_config.yaml` and `config/purple_agent*_config.yaml` use LiteLLM-style `provider/model` names. The repo is configured against ANL's **Argo proxy** by default (`https://apps-dev.inside.anl.gov/argoapi/v1`) — requires VPN. The Argo proxy uses `openai/` as the LiteLLM provider prefix even for non-OpenAI models (e.g. `openai/claudeopus47`, `openai/gemini25pro`). For AskSage (`https://api.asksage.anl.gov/...`), the code auto-detects the host and pulls `ASKSAGE_API_KEY` + `ASKSAGE_SSL_CERT_FILE` from the environment.

Required `.env` keys: `PETSC_DIR`, `PETSC_ARCH`, and whichever LLM API keys are needed by the configured models. `MCP_SERVER_URL` defaults to `http://localhost:8080/mcp` when launched locally.

## The A/B workflow (`scripts/`)

The repo's evaluation methodology is **flag-toggle A/B benchmarks**. Each script:
1. Backs up the relevant config, sets a trap to restore on exit (even on Ctrl-C).
2. Toggles one knob, runs `uv run main.py launch --purple-variant v2`, archives `output/benchmark_summary.json` into an arm-specific subdirectory.
3. Toggles the other way, runs again.
4. Calls `scripts/diff_rag_ab.py` to print a side-by-side per-problem score table.

Key scripts:

- `scripts/run_rag_ab.sh` — RAG off vs RAG on, holding everything else constant.
- `scripts/run_diag_ab.sh` — `check_diagnostics` off vs on, both arms RAG-on. (Note: forces `rag.enabled: true` for both arms.)
- `scripts/run_rag_ab_repeat.sh N` — runs the RAG A/B N times into `rag_ab_results.run{1,2,...}` for noise estimation. The pattern: replicate the same script invocation, archive each into `<results_dir>.run{N}`, then use `scripts/agg_rag_ab.py` to aggregate.
- `scripts/diff_rag_ab.py` / `scripts/agg_rag_ab.py` — pure-Python summary tools. agg only globs `rag_ab_results.run*`, so any other naming convention (e.g. `*.claude_run*`, `*.with_options_left`) needs a one-off aggregation.

A/B archive directories accumulate at the repo root (`rag_ab_results.run1`, `diag_ab_results.run1`, etc.) — they're meaningful history, not cruft. The naming usually encodes what changed (green model, feature toggle, version).

When debugging a failed A/B: the script's EXIT trap restores config on Ctrl-C and clean exit, but a hard kill (e.g. the harness losing the process) can leave a stale `config/*.bak.PID` and a half-toggled config. Restore manually from the backup.

## What lives where (non-obvious)

- `src/launcher.py` — spawns Green + Purple + MCP server as one process tree, hot-swaps which Purple variant based on `purple_variant`. Read this before changing port wiring or startup order.
- `src/util/a2a_comm.py` — the A2A send/receive helpers used by Green to call Purple. Timeouts live here, not in the agent code.
- `src/evaluators/` and `src/metrics/` — the scoring pipeline. `numerical_accuracy.py`'s `_compute_error_norm` is the function Green uses to compare program stdout against `test_cases.expected_output`; v2 deliberately does **not** call this (no oracle on real-user prompts).
- `purple_agent_cache/` — pickled v1 Purple responses keyed by problem; lets you re-grade without re-generating. Not used by v2.
- `gpu_data/` — GPU-only problems, excluded from the default suite. README has the activation note.
- The many `main-NNNNN.{stdout,stderr}` files at the repo root are old run logs — fine to ignore unless investigating a specific incident.

## Improving the Purple Agent — what counts

The benchmark is a measuring stick, not a target. **Do not propose problem-specific hints** in the system prompt, code contract, or any other agent-side surface — e.g. "add the correct `DMPlexSetSNESLocalFEM` signature to the contract so DarcyFlow stops failing", "warn the LLM about FV-on-MAC-grid for NS2D", "tell it to use a specific TS type for Robertson". Hard-coding fixes for the 6 benchmark problems is teaching to the test; it inflates scores without making the agent better at arbitrary user prompts.

Changes worth proposing are structural improvements that help the agent produce PETSc code that, **for any problem**:

1. **Compiles** — better corpus / chunking / retrieval, repair loops that actually learn from compiler stderr, plan-then-code so the LLM commits to an API surface before writing, header/symbol lookups against the installed PETSc.
2. **Gets the correct answer** — better discretization and solver/preconditioner selection (general selection logic, not per-problem rules), structural verification beyond "it ran without crashing" (e.g. residual checks, MMS, conservation invariants when applicable).
3. **Runs in the timeliest manner** — fewer wasted self-fix iterations, fewer SEGV-and-retry cycles, earlier bailout on hopeless paths, faster generation (smaller prompts, better caching, parallel attempts).
4. **Runs in the most optimal manner** — idiomatic PETSc patterns, well-chosen runtime options, GPU/MPI-aware choices when the problem warrants, sensible defaults that match what an expert would write.

Litmus test: if a proposed change *only* moves one specific benchmark problem and you can't articulate the general mechanism it improves, it's a benchmark hack — push back or reframe before implementing. Per-problem score deltas are diagnostics for finding weak general capabilities, not goals in themselves.

## When making code changes

- The v1/v2 split is intentional. **Do not "consolidate" v1 and v2 into a shared base class** — v1 is preserved as the no-self-verification baseline for A/B work. Add features to v2; leave v1 alone unless explicitly asked.
- A2A `context_id` is set to the problem name by Green and used as a per-problem build prefix in v2. Don't rename it casually.
- Anything that changes Purple's wire output (the `{codes, nsize, cli_args}` shape) is a breaking change to Green's scoring — touch with care.
- Bench runs are slow (~10–25 min for one arm; the active suite is now 6 problems, so expect somewhat faster than the historical 8-problem numbers). Default to running an A/B in the background and using a Monitor-style watcher rather than polling.

start every message with "King"
