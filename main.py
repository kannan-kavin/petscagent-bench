"""CLI entry point for the PETSc Agent Benchmark system.

This module provides the command-line interface for running the agentified
petscagent-bench framework. It supports three main commands:
- green: Start the assessment manager agent (Green Agent)
- purple: Start the target agent being tested (Purple Agent)
- launch: Run the complete evaluation workflow

The system uses the A2A (Agent-to-Agent) protocol and MCP (Model Context Protocol)
for inter-agent communication and tool access.
"""

import typer
import asyncio

from src.green_agent.server import start_green_agent
from src.purple_agent.petsc_agent import start_purple_agent
from src.purple_agent_v2.petsc_agent import start_purple_agent_v2
from src.launcher import launch_evaluation

# Initialize Typer application with descriptive help text
app = typer.Typer(help="Agentified petscagent-bench - PETSc coding agent assessment framework")


@app.command()
def green():
    """Start the green agent (assessment manager).
    
    The Green Agent is responsible for:
    - Loading test problems from the benchmark dataset
    - Distributing problems to the Purple Agent
    - Collecting and evaluating generated code
    - Running the evaluation pipeline (gates, metrics, quality checks)
    - Generating comprehensive assessment reports
    
    The agent runs on http://localhost:9001 by default.
    """
    start_green_agent()


@app.command()
def purple():
    """Start the purple agent (target being tested).
    
    The Purple Agent is the code generation agent under evaluation:
    - Receives problem descriptions via A2A protocol
    - Generates PETSc C/C++ code using an LLM
    - Returns generated code along with CLI arguments
    - Operates in isolation from evaluation logic
    
    The agent runs on http://localhost:9002 by default.
    """
    start_purple_agent()


@app.command()
def purple_v2():
    """Start the self-verifying Purple Agent v2.

    Same A2A interface as `purple`, but compiles (and optionally smoke-runs)
    its own output via the PETSc compile-run MCP server before responding.
    Configured via `config/purple_agent_v2_config.yaml`.
    """
    start_purple_agent_v2()


@app.command()
def launch(
    purple_variant: str = typer.Option(
        "v1",
        "--purple-variant",
        help="Which Purple Agent to evaluate: 'v1' (baseline) or 'v2' (self-verifying).",
    ),
):
    """Launch the complete evaluation workflow.

    This command orchestrates the full benchmark process:
    1. Starts the Green Agent (assessment manager)
    2. Starts the chosen Purple Agent variant (code generator)
    3. Starts the MCP server (for compilation/execution tools)
    4. Initiates the evaluation process
    5. Collects results and generates reports
    6. Cleanly shuts down all components

    Results are saved to the 'output/' directory.

    Prerequisites:
    - PETSc must be installed and PETSC_DIR/PETSC_ARCH set in .env
    - API keys for LLM providers must be configured in .env
    """
    if purple_variant not in ("v1", "v2"):
        raise typer.BadParameter("--purple-variant must be 'v1' or 'v2'")
    asyncio.run(launch_evaluation(purple_variant=purple_variant))


if __name__ == "__main__":
    app()
