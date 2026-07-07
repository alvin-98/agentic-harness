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
from .logging_config import get_logger, LogContext, set_context

logger = get_logger(__name__)

MAX_ITERATIONS = 10
DECISION_MAX_RETRIES = 2

# When an artifact is too small to have a chunk sidecar, attach at most this many
# raw bytes so a fallback attach can never reintroduce a context blow-up.
ATTACH_FALLBACK_BYTES = 8000

perception = perception_module.Perception()
decision = decision_module.Decision()
memory = memory_module.Memory()
action = action_module.Action()
artifacts = ArtifactStore()

GATEWAY_URL = "http://localhost:8101"
MCP_SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"


def ensure_gateway() -> None:
    """Verify the LLM gateway is running, or raise an error."""
    import urllib.request
    import urllib.error
    
    try:
        req = urllib.request.Request(f"{GATEWAY_URL}/v1/status", method="GET")
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
        command=sys.executable,
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
        return f"Completed {len(actions)} action(s). Last: {last.get('tool', 'unknown')} - {last.get('result_descriptor', '')}"
    
    return "No results produced."


async def run(query: str, run_id: str | None = None) -> str:
    ensure_gateway()
    run_id = run_id or uuid.uuid4().hex
    history: list[dict] = []
    prior_goals: list[Goal] = []

    with LogContext(run_id=run_id):
        logger.info("run_start", query=query, max_iterations=MAX_ITERATIONS)
        start_time = time.time()

        # Durable memory: classify the user's query so facts/preferences
        # in it survive into future runs.
        memory.remember(descriptor="user query", source="user_query", run_id=run_id, query=query)

        async with mcp_session() as session:
            mcp_tools = await load_tools(session)
            tools = mcp_tools_for_decision(mcp_tools)
            logger.debug("tools_loaded", tool_count=len(tools), tool_names=[t.name for t in tools])

            for it in range(1, MAX_ITERATIONS + 1):
                set_context(iteration=it)
                logger.info("iteration_start", iteration=it)
                iter_start = time.time()

                hits = memory.read(query, history)
                logger.debug("memory_read", hit_count=len(hits))

                obs = perception.observe(query, hits, history, prior_goals, run_id)
                prior_goals = obs.goals
                logger.info("perception_complete",
                           goal_count=len(obs.goals),
                           all_done=obs.all_done,
                           goals=[{"id": g.id, "text": g.text, "done": g.done} for g in obs.goals])

                if obs.all_done:
                    logger.info("all_goals_done", iteration=it)
                    break

                goal = obs.next_unfinished()
                logger.info("goal_selected", goal_id=goal.id, goal_text=goal.text)

                attached = []
                if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                    aid = goal.attach_artifact_id
                    # Retrieve only the chunks relevant to this goal instead of
                    # attaching the whole artifact (which can be 100k+ tokens and
                    # overflow the model context). Falls back to a bounded slice
                    # when the artifact was too small to be chunk-indexed.
                    chunks = memory.retrieve_artifact_chunks(aid, query=goal.text)
                    if chunks is not None:
                        attached.append((aid, chunks.encode("utf-8")))
                        logger.info("artifact_chunks_attached",
                                   artifact_id=aid, chars=len(chunks))
                    else:
                        raw = artifacts.get_bytes(aid)[:ATTACH_FALLBACK_BYTES]
                        attached.append((aid, raw))
                        logger.debug("artifact_attached_fallback",
                                    artifact_id=aid, bytes=len(raw))

                out = None
                for attempt in range(1, DECISION_MAX_RETRIES + 1):
                    out = decision.next_step(goal, hits, history, attached, tools)
                    if out.is_answer and not out.answer:
                        logger.warning("decision_malformed",
                                      goal_id=goal.id,
                                      attempt=attempt,
                                      detail="is_answer=True but answer is empty")
                        continue
                    if not out.is_answer and out.tool_call is None:
                        logger.warning("decision_malformed",
                                      goal_id=goal.id,
                                      attempt=attempt,
                                      detail="neither answer nor tool_call populated")
                        continue
                    break

                logger.info("decision_complete",
                           is_answer=out.is_answer,
                           tool_name=out.tool_call.name if out.tool_call else None)

                if out.is_answer and out.answer:
                    history.append({"iter": it, "kind": "answer",
                                    "goal_id": goal.id, "text": out.answer})
                    logger.info("answer_produced",
                               goal_id=goal.id,
                               answer_preview=out.answer)      
                    prior_goals = [
                        Goal(id=g.id, text=g.text, done=True, attach_artifact_id=g.attach_artifact_id)
                        if g.id == goal.id else g
                        for g in prior_goals
                    ]
                    continue

                if out.tool_call is None:
                    logger.warning("decision_empty", goal_id=goal.id,
                                   detail="DecisionOutput has neither answer nor tool_call after retries")
                    history.append({"iter": it, "kind": "answer",
                                    "goal_id": goal.id,
                                    "text": "Unable to determine next action."})
                    prior_goals = [
                        Goal(id=g.id, text=g.text, done=True, attach_artifact_id=g.attach_artifact_id)
                        if g.id == goal.id else g
                        for g in prior_goals
                    ]
                    continue

                logger.info("action_start",
                           tool=out.tool_call.name,
                           arguments=out.tool_call.arguments)
                action_start = time.time()

                result_text, art_id = await action.execute(session, out.tool_call)

                logger.info("action_complete",
                           tool=out.tool_call.name,
                           duration_ms=int((time.time() - action_start) * 1000),
                           result_preview=result_text,
                           artifact_id=art_id,
                           has_artifact=art_id is not None)

                # Chunk + embed large artifacts into a per-artifact sidecar so
                # later goals can retrieve only the relevant slices instead of
                # attaching the whole blob.
                if art_id and artifacts.exists(art_id):
                    raw = artifacts.get_bytes(art_id).decode("utf-8", errors="replace")
                    memory.index_artifact(art_id, raw)

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
                                "result_descriptor": result_text,
                                "artifact_id": art_id})

                logger.debug("iteration_complete",
                            iteration=it,
                            duration_ms=int((time.time() - iter_start) * 1000))

        memory.expire_run(run_id)

        final = final_answer_from(history)
        logger.info("run_complete",
                   total_iterations=len([h for h in history if h.get("kind") in ("action", "answer")]),
                   total_actions=len([h for h in history if h.get("kind") == "action"]),
                   total_answers=len([h for h in history if h.get("kind") == "answer"]),
                   duration_ms=int((time.time() - start_time) * 1000),
                   final_answer_preview=final)

    return final

def main() -> None:
    # query = "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
#     query = """Find 3 family-friendly things to do in Tokyo this weekend.
# Check Saturday's weather forecast there and tell me which one
# is most appropriate."""
    # query = """My mom's birthday is 15 May 2026. Remember that and give me
    #    a calendar reminder for two weeks before and on the day."""
    # query = "When is mom's birthday?"
    query = """Search for 'Python asyncio best practices', read the top 3 results,
and give me a short numbered list of the advice they agree on."""
    result = asyncio.run(run(query))
    print(result)


if __name__ == "__main__":
    main()
