"""Standalone driver that exercises the purple-agent LLM flow without A2A/MCP.

Mirrors PetscAgentExecutor.execute() from src/purple_agent/petsc_agent.py:
same system prompt, same response schema, same model/config — but no server,
no compile, no run. Useful for poking the model directly.

Usage: uv run scripts/exercise_purple.py [problem_file]
       (defaults to data/vecmpi.json)
"""
import json
import sys
from pathlib import Path

import dotenv
import litellm
import yaml
from litellm import completion
from pydantic import BaseModel

dotenv.load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_CODE_CONTRACT = (
    "You are a code-generation agent.\n"
    "Return ONLY a single raw JSON object. No markdown, no backticks, no code blocks, no explanation outside the JSON.\n"
    "Top-level JSON keys MUST be exactly: 'codes', 'nsize', 'cli_args' (no additional keys).\n\n"
    "Rules:\n"
    "- 'codes': a list of objects with 'filename' and 'code'. Code must be valid C/C++/CUDA.\n"
    "- 'nsize': the number of MPI processes (use 1 for sequential).\n"
    "- 'cli_args': command line arguments string.\n"
    "- First file in 'codes' is the main file.\n"
    "- Any explanations MUST be inside C block comments /* ... */ within the code strings.\n"
)


class CodeFile(BaseModel):
    filename: str
    code: str


class ProblemResponse(BaseModel):
    codes: list[CodeFile]
    nsize: int
    cli_args: str


def main() -> int:
    problem_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / "vecmpi.json"
    with open(problem_path) as f:
        problem = json.load(f)

    with open(REPO_ROOT / "config" / "purple_agent_config.yaml") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg["llm"]
    model = llm_cfg["model"]
    api_base = llm_cfg.get("api_base_url")
    temperature = float(llm_cfg.get("temperature", 0))

    print(f"Problem: {problem['problem_name']} (id={problem['problem_id']})")
    print(f"Model:   {model}   api_base={api_base}")
    print("-" * 70)
    print(problem["problem_description"])
    print("-" * 70)

    litellm.ssl_verify = False
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_CODE_CONTRACT},
            {"role": "user", "content": problem["problem_description"]},
        ],
        "temperature": temperature,
        "response_format": ProblemResponse,
        "timeout": 300,
    }
    if api_base:
        kwargs["api_base"] = api_base
        # Argo validates this field against ANL usernames; provide it via
        # OPENAI_API_KEY in .env (litellm picks it up automatically).

    response = completion(**kwargs)
    content = response.choices[0].message.content
    if content.startswith("```"):
        content = content.split("```", 2)[1].lstrip("json").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("===== RAW LLM RESPONSE (failed to parse as JSON) =====")
        print(content)
        return 1

    print(f"nsize:    {data['nsize']}")
    print(f"cli_args: {data['cli_args']}")
    print(f"files:    {[c['filename'] for c in data['codes']]}")
    for entry in data["codes"]:
        print(f"\n===== {entry['filename']} =====")
        print(entry["code"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
