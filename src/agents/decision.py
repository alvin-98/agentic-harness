"""
Decision module selects the next action to take based on the current state of the agent.
It receives one goal, the relevant memory hits, the recent history, optionally the raw
bytes of an artifact and a list of available MCP tools.

It returns a DecisionOutput object containing the ToolCall required or the final answer.
Decision does not pick more than one tool and does not narrate.

Two-attempt strategy:
  1. Native tool-calling: pass tools= + tool_choice='auto' to the gateway. Capable
     models return tool_calls directly. The gateway's prompted_fallback handles
     providers that lack native tool support.
  2. JSON fallback: if the first attempt returns neither tool_calls nor usable text,
     retry with response_format=DecisionOutput schema and no tools=, asking the model
     to emit JSON. This is the universal fallback for any provider.

Decision routes through the gateway with auto_route="decision". The router pool
classifies the call and picks a tier. Most Decision calls land on the LARGE-tier
Gemini model. Smaller Decision calls land on TINY-tier workers. The router decision
is visible in the gateway's response under router_decision.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Any

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_gateway"))

from .schemas import Goal, MemoryItem, DecisionOutput, ToolCall, Kind
from .instrumented_llm import InstrumentedLLM
from .logging_config import get_logger
from . import config

logger = get_logger(__name__)

# How much attached content to send to the model per turn. Most LARGE-tier
# workers handle 30 KB comfortably; truncate above that and let the model
# work with a head-and-tail window.
ATTACH_HEAD = 20_000
ATTACH_TAIL = 10_000


class ToolDef(BaseModel):
    """Canonical tool envelope — what the gateway expects on the request."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class Decision:
    def __init__(self):
        self.llm = InstrumentedLLM()
        self.DECISION_SYSTEM_PROMPT = config.DECISION_SYSTEM_PROMPT

    def next_step(
        self,
        goal: Goal,
        hits: List[MemoryItem],
        history: List[dict],
        attached: Optional[list[tuple[str, bytes]]] = None,
        tools: Optional[List[ToolDef]] = None,
    ) -> DecisionOutput:
        """Determine the next action or final answer for the current goal.

        First attempt uses native tool-calling (tools= + tool_choice='auto').
        Falls back to JSON response_format if the first attempt is empty.
        """
        attached_text = self._format_attached(attached)
        hits_text = self._format_hits(hits)
        history_text = self._format_history(history)
        tools_list = tools or []

        prompt = (
            f"GOAL:\n  {goal.text}\n\n"
            f"MEMORY HITS:\n{hits_text}\n\n"
            f"RECENT HISTORY:\n{history_text}"
            f"{attached_text}"
        )

        logger.debug("decision_llm_start",
                    goal_id=goal.id,
                    goal_text=goal.text,
                    memory_hits=len(hits) if hits else 0,
                    history_len=len(history) if history else 0,
                    attached_count=len(attached) if attached else 0,
                    tool_count=len(tools_list))

        _c = config.DECISION_LLM

        # ── Attempt 1: native tool-calling ────────────────────────────────
        output = None
        try:
            output = self._try_native(prompt, tools_list, _c, goal_id=goal.id)
        except Exception as e:
            logger.warning("decision_native_failed",
                          goal_id=goal.id, error=str(e))

        # ── Attempt 2: JSON fallback ──────────────────────────────────────
        if output is None:
            logger.debug("decision_json_fallback", goal_id=goal.id)
            try:
                output = self._try_json(prompt, _c, goal_id=goal.id)
            except Exception as e:
                logger.error("decision_json_failed",
                            goal_id=goal.id, error=str(e))
                # Last resort: empty output so the loop's retry handles it.
                output = DecisionOutput(answer=None, tool_call=None)

        return output

    # ── Attempt implementations ──────────────────────────────────────────

    def _try_native(self, prompt: str, tools: list[ToolDef], _c, goal_id: str = "") -> Optional[DecisionOutput]:
        """Native tool-calling: pass tools= and tool_choice='auto'. The gateway
        returns tool_calls if the model picks a tool, or text if it answers."""
        start_time = time.time()
        reply = self.llm.chat(
            call_label="decision.next_step",
            prompt=prompt,
            system=self.DECISION_SYSTEM_PROMPT,
            cache_system=_c.cache_system,
            tools=[t.model_dump() for t in tools],
            tool_choice="auto",
            reasoning=_c.reasoning,
            temperature=_c.temperature,
            max_tokens=_c.max_tokens,
            provider=_c.provider,
            model=_c.model,
            auto_route=_c.auto_route,
        )
        self._log_complete("native", goal_id=goal_id, reply=reply, start_time=start_time)

        tool_calls = reply.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            return DecisionOutput(
                answer=None,
                tool_call=ToolCall(name=tc["name"], arguments=tc.get("arguments") or {}),
            )
        text = (reply.get("text") or "").strip()
        if text:
            return DecisionOutput(answer=text, tool_call=None)
        return None  # neither tool_calls nor text — fall through to JSON

    def _try_json(self, prompt: str, _c, goal_id: str = "") -> Optional[DecisionOutput]:
        """JSON fallback: response_format=DecisionOutput schema, no tools=."""
        start_time = time.time()
        reply = self.llm.chat(
            call_label="decision.next_step_json",
            prompt=prompt,
            system=self.DECISION_SYSTEM_PROMPT,
            cache_system=_c.cache_system,
            response_format={
                "type": "json_schema",
                "schema": DecisionOutput.model_json_schema(),
                "name": "DecisionOutput",
                "strict": True,
            },
            reasoning=_c.reasoning,
            temperature=_c.temperature,
            max_tokens=_c.max_tokens,
            provider=_c.provider,
            model=_c.model,
            auto_route=_c.auto_route,
        )
        self._log_complete("json", goal_id=goal_id, reply=reply, start_time=start_time)

        parsed = reply.get("parsed")
        if parsed:
            return DecisionOutput.model_validate(parsed)
        return None

    def _log_complete(self, mode: str, goal_id: str, reply: dict, start_time: float):
        duration_ms = int((time.time() - start_time) * 1000)
        tool_calls = reply.get("tool_calls") or []
        logger.info("decision_llm_complete",
                    mode=mode,
                    goal_id=goal_id,
                    duration_ms=duration_ms,
                    is_answer=bool(reply.get("text")),
                    tool_name=tool_calls[0]["name"] if tool_calls else None,
                    input_tokens=reply.get("input_tokens", 0),
                    output_tokens=reply.get("output_tokens", 0),
                    provider=reply.get("provider"),
                    model=reply.get("model"),
                    attempted=reply.get("attempted", []),
                    router_decision=reply.get("router_decision"))

    # ── Prompt formatting (adapted from S7) ──────────────────────────────

    @staticmethod
    def _format_hits(hits: list[MemoryItem]) -> str:
        """Render memory hits with inline raw/chunk previews so Decision can
        synthesize directly from indexed chunks without a separate
        search_knowledge call."""
        if not hits:
            return "  (none)"
        out = []
        for h in hits[:10]:
            kind = h.kind.value if isinstance(h.kind, Kind) else h.kind
            line = f"  - [{kind}] {h.descriptor}"
            val = h.value or {}
            if val:
                raw = val.get("raw")
                chunk = val.get("chunk")
                if isinstance(raw, str) and raw.strip():
                    raw_more = "…" if len(raw) > 2000 else ""
                    line += f"\n      raw: {raw[:2000]}{raw_more}"
                elif isinstance(chunk, str) and chunk.strip():
                    src = val.get("source") or ""
                    preview = chunk[:2000].replace("\n", " ")
                    more = "…" if len(chunk) > 2000 else ""
                    line += f"\n      chunk ({src}): {preview}{more}"
                else:
                    compact = {
                        k: v for k, v in val.items()
                        if k != "chunk" and not (isinstance(v, str) and len(v) > 200)
                    }
                    if compact:
                        line += f"\n      value: {json.dumps(compact)[:240]}"
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """Clip to last 6 events and format compactly with 300-char ceiling."""
        if not history:
            return "  (empty)"
        lines = []
        for h in history[-6:]:
            kind = h.get("kind", "?")
            if kind == "answer":
                lines.append(f"  - iter {h.get('iter')}: ANSWER → {(h.get('text') or '')[:140]}")
            elif kind == "action":
                tool = h.get("tool")
                desc = h.get("result_descriptor", "")[:300]
                art = f" (artifact {h['artifact_id']})" if h.get("artifact_id") else ""
                lines.append(f"  - iter {h.get('iter')}: {tool}{art} → {desc}")
            else:
                lines.append(f"  - iter {h.get('iter')}: {kind} {h}")
        return "\n".join(lines)

    @staticmethod
    def _format_attached(attached: Optional[list[tuple[str, bytes]]]) -> str:
        """Head+tail truncate large artifacts to keep within model context."""
        if not attached:
            return ""
        parts = ["\n\nATTACHED ARTIFACTS:"]
        for art_id, data in attached:
            text = data.decode("utf-8", errors="replace")
            if len(text) > ATTACH_HEAD + ATTACH_TAIL + 50:
                text = (
                    text[:ATTACH_HEAD]
                    + f"\n\n...[truncated; full size {len(data)} bytes]...\n\n"
                    + text[-ATTACH_TAIL:]
                )
            parts.append(f"--- {art_id} ---\n{text}")
        return "\n".join(parts)

    @staticmethod
    def mcp_tool_to_gateway(t) -> dict:
        """The whole 'protocol bridge' between MCP and the gateway is this reshape."""
        return ToolDef(
            name=t.name,
            description=t.description or "",
            input_schema=t.inputSchema or {"type": "object", "properties": {}},
        ).model_dump()
