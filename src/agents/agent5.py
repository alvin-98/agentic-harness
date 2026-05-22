"""
agent5.py — Session 5 (2026) shape, paired side-by-side with agent.py.

Same MCP server, same task (compute (a+b)+(c-d) for 4 numbers), but the
loop is rebuilt on the upgrades V2 of the gateway exposes:

  agent.py (Session 4 style)            agent5.py (Session 5 style)
  ─────────────────────────────────     ─────────────────────────────────
  prompted JSON + regex parser     →    native tool-use, no parser
  hand-rolled normalize_action()   →    canonical tool_calls[] from gateway
  8-turn few-shot scaffold         →    short system prompt only
  list[dict] message history       →    Pydantic AgentTrace
  sequential tool dispatch         →    asyncio.TaskGroup parallel dispatch
  no system-prompt caching         →    cache_system=True (one flag)
  no reasoning knob                →    reasoning="off" on executor
  no verifier                      →    typed Pydantic Verdict via
                                        response_format=
  llm_gateway (V1, port 8099)      →    llm_gatewayV2     (port 8100)

The MCP server (mcp_server.py) is unchanged — that's the point. Session 5
keeps the server, changes the loop.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# V2 client lives one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_gatewayV2"))
from client import LLM  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Pydantic schemas — one source of truth for every boundary
# ────────────────────────────────────────────────────────────────────────────

class ToolDef(BaseModel):
    """Canonical tool envelope — what V2 expects on the request."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """One row in the structured event log."""
    kind: Literal["llm_call", "tool_call", "verdict"]
    turn: int
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read: int | None = None
    cache_create: int | None = None
    dialect: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    text: str | None = None
    payload: dict | None = None


class AgentTrace(BaseModel):
    goal: str
    events: list[TraceEvent] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)

    def add(self, **kw) -> None:
        self.events.append(TraceEvent(**kw))

    def summary(self) -> dict:
        llm_calls = [e for e in self.events if e.kind == "llm_call"]
        tool_calls = [e for e in self.events if e.kind == "tool_call"]
        return {
            "llm_turns": len(llm_calls),
            "tool_calls": len(tool_calls),
            "total_in_tokens": sum(e.input_tokens or 0 for e in llm_calls),
            "total_out_tokens": sum(e.output_tokens or 0 for e in llm_calls),
            "cache_reads": sum(e.cache_read or 0 for e in llm_calls),
            "wall_clock_s": round(time.time() - self.started_at, 2),
        }


class Verdict(BaseModel):
    """Verifier's typed contract."""
    passed: bool
    reason: str
    final_answer: float


# ────────────────────────────────────────────────────────────────────────────
# MCP ↔ V2 bridge (one function)
# ────────────────────────────────────────────────────────────────────────────

def mcp_tool_to_v2(t) -> dict:
    """The whole 'protocol bridge' between MCP and the gateway is this reshape."""
    return ToolDef(
        name=t.name,
        description=t.description or "",
        input_schema=t.inputSchema or {"type": "object", "properties": {}},
    ).model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Parallel MCP dispatcher — when the model emits multiple independent
# tool_calls in one turn, run them concurrently inside a TaskGroup.
# ────────────────────────────────────────────────────────────────────────────

async def dispatch_tool_calls(session, tool_calls: list[dict]) -> list[dict]:
    async def run_one(tc: dict) -> dict:
        result = await session.call_tool(tc["name"], tc.get("arguments") or {})
        text = result.content[0].text if result.content else ""
        # Echo provider_meta back unchanged on the assistant turn (Gemini
        # requires its thoughtSignature; other providers ignore it).
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "tool_name": tc["name"],
            "content": text,
        }

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run_one(tc)) for tc in tool_calls]
    return [t.result() for t in tasks]


# ────────────────────────────────────────────────────────────────────────────
# The agent loop — native tool-use, no parser
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an arithmetic agent. Use the add and subtract tools for every "
    "calculation — never compute mentally. When you have the final number, "
    "reply in plain text with just the number."
)


async def run_native_loop(
    session: ClientSession,
    tools: list[dict],
    user_task: str,
    trace: AgentTrace,
    provider: str | None = None,
    max_turns: int = 6,
) -> str:
    llm = LLM()
    messages: list[dict] = [{"role": "user", "content": user_task}]

    for turn in range(1, max_turns + 1):
        print(f"\n─── turn {turn}  →  LLM ───────────────────────────────────────────────")
        reply = llm.chat(
            messages=messages,
            system=SYSTEM_PROMPT,
            cache_system=True,           # mark the system prompt cacheable
            tools=tools,                 # native tool-use
            tool_choice="auto",
            reasoning="off",             # executor stays cheap
            provider=provider,           # None = capability-aware failover
            temperature=0,
            max_tokens=1024,
        )

        trace.add(
            kind="llm_call",
            turn=turn,
            provider=reply["provider"],
            model=reply["model"],
            latency_ms=reply["latency_ms"],
            input_tokens=reply["input_tokens"],
            output_tokens=reply["output_tokens"],
            cache_read=reply.get("cache_read_input_tokens"),
            cache_create=reply.get("cache_creation_input_tokens"),
            dialect=reply.get("tool_call_dialect"),
            text=reply.get("text"),
            payload={"tool_calls": reply.get("tool_calls", [])},
        )
        print(f"  provider : {reply['provider']}  model: {reply['model']}")
        print(f"  latency  : {reply['latency_ms']} ms")
        print(f"  tokens   : in={reply['input_tokens']}  out={reply['output_tokens']}  "
              f"cache_read={reply.get('cache_read_input_tokens', 0)}  "
              f"cache_create={reply.get('cache_creation_input_tokens', 0)}")
        print(f"  dialect  : {reply.get('tool_call_dialect')}  "
              f"reasoning_applied={reply.get('reasoning_applied')}")
        print(f"  stop     : {reply.get('stop_reason')}")
        print(f"  text     : {reply.get('text')!r}")

        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return reply.get("text", "").strip()

        # Echo the assistant turn (incl. tool_calls + provider_meta) back into history.
        messages.append({
            "role": "assistant",
            "content": reply.get("text", "") or "",
            "tool_calls": tool_calls,
        })

        print(f"\n─── turn {turn}  →  MCP   ({len(tool_calls)} calls"
              + (", parallel via TaskGroup" if len(tool_calls) > 1 else "") + ") ───")
        results = await dispatch_tool_calls(session, tool_calls)
        for tc, r in zip(tool_calls, results):
            print(f"  {tc['name']}({json.dumps(tc.get('arguments', {}))}) -> {r['content']}")
            trace.add(
                kind="tool_call",
                turn=turn,
                tool_name=tc["name"],
                tool_args=tc.get("arguments"),
                tool_result=r["content"],
            )
        messages.extend(results)

    raise RuntimeError(f"agent exceeded max_turns={max_turns}")


# ────────────────────────────────────────────────────────────────────────────
# Verifier — separate call, typed Pydantic output via response_format
# ────────────────────────────────────────────────────────────────────────────

def verify(trace: AgentTrace, expected: float, executor_answer: str) -> Verdict:
    """Independent typed-output check — does the executor's answer match what
    the actual MCP tool calls produced? No model arithmetic, just inspection
    of the trace.

    We use the gateway's structured-output feature so the model returns a
    validated `Verdict`. Reasoning="medium" because verification is the
    place to spend a little budget per Session 5.
    """
    last_tool_result = next(
        (e.tool_result for e in reversed(trace.events) if e.kind == "tool_call"),
        None,
    )
    schema = Verdict.model_json_schema()

    llm = LLM()
    reply = llm.chat(
        prompt=(
            f"You are a verifier. The expected answer is {expected}. "
            f"The agent reported: {executor_answer!r}. "
            f"The last tool call returned: {last_tool_result!r}. "
            "Decide if the agent's reported answer matches the expected number "
            "(numerically — '20' and '20.0' both match 20). Return a Verdict."
        ),
        system="Return a single Verdict object. Be terse.",
        cache_system=True,
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "Verdict",
            "strict": True,
        },
        reasoning="medium",
        temperature=0,
        max_tokens=512,
    )

    if reply.get("parsed"):
        return Verdict.model_validate(reply["parsed"])
    # Fallback if structured output wasn't honoured by the chosen provider.
    return Verdict(
        passed=str(expected) in (executor_answer or ""),
        reason="structured-output not honoured; fell back to substring check",
        final_answer=float(expected),
    )


# ────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────────

async def run(numbers: list[float], provider: str | None = "gr") -> None:
    a, b, c, d = numbers
    expected = (a + b) + (c - d)
    user_task = (
        f"Numbers: a={a}, b={b}, c={c}, d={d}. "
        f"Compute (a + b) + (c - d). "
        f"Note: (a+b) and (c-d) are independent — call both tools in parallel "
        f"in your first turn if your API supports parallel tool calls."
    )

    print("═" * 78)
    print(f"agent5.py — Session 5 native tool-use loop")
    print(f"inputs   : a={a}  b={b}  c={c}  d={d}")
    print(f"expected : ({a}+{b}) + ({c}-{d}) = {expected}")
    print(f"provider : {provider or 'auto-failover'}")
    print("═" * 78)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = [mcp_tool_to_v2(t) for t in mcp_tools]
            print(f"[mcp] tools from server: {[t.name for t in mcp_tools]}")

            trace = AgentTrace(goal=user_task)

            # ── Act ────────────────────────────────────────────────────────
            answer = await run_native_loop(session, tools, user_task, trace, provider=provider)
            print(f"\n[executor] answer: {answer!r}")

            # ── Verify ─────────────────────────────────────────────────────
            print("\n─── VERIFY (structured output) ─────────────────────────────────────")
            verdict = verify(trace, expected, answer)
            trace.add(kind="verdict", turn=0, payload=verdict.model_dump())
            print(f"  passed       : {verdict.passed}")
            print(f"  final_answer : {verdict.final_answer}")
            print(f"  reason       : {verdict.reason}")

            # ── Trace summary ─────────────────────────────────────────────
            print("\n─── TRACE SUMMARY ──────────────────────────────────────────────────")
            for k, v in trace.summary().items():
                print(f"  {k:<22}: {v}")
            print("\n─── EVENTS (Pydantic AgentTrace) ───────────────────────────────────")
            for i, e in enumerate(trace.events):
                line = e.model_dump(exclude_none=True)
                # truncate noisy payloads
                if "payload" in line and isinstance(line["payload"], dict):
                    line["payload"] = {k: v for k, v in line["payload"].items() if v}
                print(f"  #{i:02d} {line}")

            print("\n" + "═" * 78)
            print(f"FINAL: {verdict.final_answer}  (passed={verdict.passed})")
            print("═" * 78)


def main() -> None:
    numbers = [10, 20, 30, 40]   # (10+20) + (30-40) = 20
    asyncio.run(run(numbers, provider="gr"))   # groq llama-3.3-70b: native tools, parallel


if __name__ == "__main__":
    main()
