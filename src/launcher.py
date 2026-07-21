"""Evaluation launcher module - orchestrates the complete benchmark workflow.

This module is responsible for:
- Starting all required agents (Green, Purple) and servers (MCP)
- Coordinating inter-process communication
- Managing the evaluation lifecycle
- Ensuring clean shutdown of all components

The launcher uses multiprocessing to run agents in separate processes,
allowing them to communicate via HTTP using the A2A protocol.
"""

import multiprocessing
import asyncio
import json
import mcp
from src.green_agent.server import start_green_agent, load_green_agent_config
from src.purple_agent.petsc_agent import start_purple_agent, load_purple_agent_config
from src.purple_agent_v2.petsc_agent import (
    start_purple_agent_v2,
    load_purple_agent_v2_config,
)
from src.util.a2a_comm import wait_agent_ready, send_message
import os
import dotenv

# Load environment variables before importing server code
# This ensures PETSC_DIR, PETSC_ARCH, and API keys are available
dotenv.load_dotenv()
from petsc_compile_run_mcp_server import main as start_mcp_server


def run_green_agent(agent_llm, api_base_url=None):
    """Execute the Green Agent in a separate process.
    
    This wrapper function is needed for multiprocessing.Process,
    which requires a synchronous entry point. The function creates
    a new asyncio event loop and runs the async green agent server.
    """
    asyncio.run(start_green_agent(agent_llm=agent_llm, api_base_url=api_base_url))


def run_purple_agent(agent_llm, api_base_url=None):
    """Execute the Purple Agent in a separate process.

    Starts the Purple Agent with a specific LLM configuration.
    The LLM model can be changed here to test different models.

    Currently configured to use: openai/gpt-5.2
    Other options: gemini/gemini-2.5-flash, openai/gpt-4o, etc.
    """
    asyncio.run(start_purple_agent(agent_llm=agent_llm, api_base_url=api_base_url))
    # asyncio.run(start_purple_agent(agent_llm="openai/google-claude-45-opus")) # test AskSage


def run_purple_agent_v2(agent_llm, api_base_url=None, mcp_server_url=None):
    """Execute the self-verifying Purple Agent v2 in a separate process."""
    asyncio.run(start_purple_agent_v2(
        agent_llm=agent_llm,
        api_base_url=api_base_url,
        mcp_server_url=mcp_server_url,
    ))


async def launch_evaluation(purple_variant: str = "v1"):
    """Main launcher function - initiates and coordinates the evaluation process.
    
    This function orchestrates the complete benchmark workflow:
    
    1. Process Initialization:
       - Spawns the Green Agent process (assessment manager)
       - Spawns the Purple Agent process (code generator under test)
       - Spawns the MCP server process (PETSc compilation/execution tools)
    
    2. Health Checks:
       - Waits for each agent to become ready (HTTP health check)
       - Ensures all components are operational before proceeding
    
    3. Task Execution:
       - Sends the evaluation task to the Green Agent
       - Green Agent autonomously manages the evaluation workflow
       - Waits for evaluation to complete
    
    4. Cleanup:
       - Terminates all spawned processes
       - Ensures clean shutdown
    
    The evaluation results are automatically saved by the Green Agent
    to the 'output/' directory.
    
    Raises:
        AssertionError: If any agent fails to become ready within timeout
        Exception: If communication or execution errors occur
    """
    # Define service endpoints
    green_url = "http://localhost:9001"    # Green Agent A2A server
    purple_url = "http://localhost:9002"   # Purple Agent A2A server
    mcp_server_url = "http://localhost:8080/mcp"  # MCP tools server
    green_id = "019bb856-c8bf-7390-8c4f-bced52276932" # AgentBeats ID
    purple_id = ""

    green_cfg = load_green_agent_config()
    green_llm_cfg = green_cfg.get('evaluation', {}).get('llm', {})
    green_model = green_llm_cfg.get('model', 'openai/gpt-4o-mini')
    green_api_base_url = green_llm_cfg.get('api_base_url')

    if purple_variant == "v2":
        purple_cfg = load_purple_agent_v2_config()
    else:
        purple_cfg = load_purple_agent_config()
    purple_llm_cfg = purple_cfg.get('llm') or {}
    purple_model = purple_llm_cfg.get('model', 'openai/gpt-4o-mini')
    purple_api_base_url = purple_llm_cfg.get('api_base_url')
    purple_mcp_server_url = (purple_cfg.get('mcp') or {}).get('server_url') if purple_variant == "v2" else None
    # Step 1: Start Green Agent (assessment manager)
    print("Launching green agent...")
    p_green = multiprocessing.Process(target=run_green_agent, args=(green_model, green_api_base_url))
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # Step 2: Start Purple Agent (code generator being tested)
    if purple_variant == "v2":
        print("Launching purple agent v2 (self-verifying)...")
        p_purple = multiprocessing.Process(
            target=run_purple_agent_v2,
            args=(purple_model, purple_api_base_url, purple_mcp_server_url),
        )
    else:
        print("Launching purple agent (baseline)...")
        p_purple = multiprocessing.Process(
            target=run_purple_agent,
            args=(purple_model, purple_api_base_url),
        )
    p_purple.start()
    assert await wait_agent_ready(purple_url), "purple agent not ready in time"
    print("purple agent is ready.")

    # Step 3: Start MCP server (provides PETSc compilation/execution tools)
    print("Launching MCP server for green agent...")
    petsc_mcp_server = multiprocessing.Process(target=start_mcp_server)
    petsc_mcp_server.start()
    # Poll until the MCP port accepts TCP connections — otherwise short bench
    # runs (single-problem A/Bs) can finish before Purple's verify loop can
    # reach MCP, silently degenerating to no-self-verify in every arm.
    import socket as _socket
    _mcp_host, _mcp_port = "localhost", 8080
    for _i in range(60):
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
            _s.settimeout(0.5)
            try:
                _s.connect((_mcp_host, _mcp_port))
                print(f"PETSc MCP server is ready (port {_mcp_port} accepting connections).")
                break
            except OSError:
                await asyncio.sleep(1)
    else:
        print(f"WARNING: MCP server did not become ready within 60s on port {_mcp_port}; continuing anyway.")

    # Step 4: Send evaluation task to Green Agent
    print("Sending task description to green agent...")
    task_text = f"""
Your task is to instantiate petscagent-bench to test the agent located at:
<purple_agent_url>
{purple_url}/
</purple_agent_url>
You can use MCP tools from:
<mcp_server_url>
{mcp_server_url}/
</mcp_server_url>
Green agent's AgentBeats ID is
<green_id>
{green_id}
</green_id>
Purple agent's AgentBeats ID is
<purple_id>
{purple_id}
</purple_id>
    """
    print("Task description:")
    print(task_text)
    print("Sending...")
    
    # Send message and wait for completion
    # The Green Agent will autonomously manage the entire evaluation workflow
    response = await send_message(green_url, task_text)

    # Step 5: Cleanup - terminate all processes
    print("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_purple.terminate()
    p_purple.join()
    print("Agents terminated.")
    petsc_mcp_server.terminate()
    petsc_mcp_server.join()
    print("PETSc MCP server terminated.")
