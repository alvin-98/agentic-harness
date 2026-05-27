import asyncio
import json
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_gateway"))
from client import LLM

from . import perception as perception_module
from . import decision as decision_module
from . import memory as memory_module
from . import action as action_module
from .artifacts import ArtifactStore
from .schemas import Goal, Observation
from .decision import ToolDef

MAX_ITERATIONS = 10

perception = perception_module.Perception()
decision = decision_module.Decision()
memory = memory_module.Memory()
action = action_module.Action()
artifacts = ArtifactStore()

GATEWAY_URL = "http://localhost:8100"
MCP_SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp_server.py"


def ensure_gateway() -> None:
    """Verify the LLM gateway is running, or raise an error."""
    import urllib.request
    import urllib.error
    
    try:
        req = urllib.request.Request(f"{GATEWAY_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        pass
    
    raise RuntimeError(
        f"LLM Gateway not running at {GATEWAY_URL}. "
        "Start it with: python -m llm_gateway.server"
    )


@asynccontextmanager
async def mcp_session():
    """Context manager that yields a live MCP ClientSession."""
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession) -> list:
    """Fetch the list of available tools from the MCP server."""
    result = await session.list_tools()
    return result.tools


def mcp_tools_for_decision(mcp_tools: list) -> list[ToolDef]:
    """Convert MCP tool objects to ToolDef list for Decision module."""
    return [
        ToolDef(
            name=t.name,
            description=t.description or "",
            input_schema=t.inputSchema or {"type": "object", "properties": {}},
        )
        for t in mcp_tools
    ]


def final_answer_from(history: list[dict]) -> str:
    """Synthesize a final answer from the run history."""
    answers = [h["text"] for h in history if h.get("kind") == "answer"]
    if answers:
        return "\n\n".join(answers)
    
    actions = [h for h in history if h.get("kind") == "action"]
    if actions:
        last = actions[-1]
        return f"Completed {len(actions)} action(s). Last: {last.get('tool', 'unknown')} - {last.get('result_descriptor', '')[:200]}"
    
    return "No results produced."


async def run(query: str) -> str:
    ensure_gateway()
    run_id = uuid.uuid4().hex
    history: list[dict] = []
    prior_goals: list[Goal] = []

    # Durable memory: classify the user's query so facts/preferences
    # in it survive into future runs.
    memory.remember(query, source="user_query", run_id=run_id)

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        tools = mcp_tools_for_decision(mcp_tools)

        for it in range(1, MAX_ITERATIONS + 1):
            hits = memory.read(query, history)
            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals
            if obs.all_done:
                break

            goal = obs.next_unfinished()
            attached = []
            if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                attached.append((
                    goal.attach_artifact_id,
                    artifacts.get_bytes(goal.attach_artifact_id),
                ))

            out = decision.next_step(goal, hits, attached, history, tools)

            if out.is_answer:
                history.append({"iter": it, "kind": "answer",
                                "goal_id": goal.id, "text": out.answer})
                continue

            result_text, art_id = await action.execute(session, out.tool_call)
            memory.record_outcome(
                tool_call=out.tool_call,
                result_text=result_text,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            )
            history.append({"iter": it, "kind": "action",
                            "goal_id": goal.id, "tool": out.tool_call.name,
                            "arguments": out.tool_call.arguments,
                            "result_descriptor": result_text[:300],
                            "artifact_id": art_id})

    return final_answer_from(history)

def main() -> None:
    query = "What is the capital of France?"
    result = asyncio.run(run(query))
    print(result)


if __name__ == "__main__":
    main()
