"""
Perception module for agents — single-call design.

Perception tracks state across agentic loop iterations. Each iteration it
receives the user's original query, the current memory hits, the run history
so far, and the prior goal list. It emits a fresh Observation containing the
current goals list with done flags and optional artifact attachments.

All four responsibilities — decomposition (iter 1), goal updates (iter 2+),
artifact-attach decisions, and goal extension (appending new goals when
discovery actions reveal concrete items) — are handled in a SINGLE LLM call
per iteration. This is the S7 design: one rich prompt, one structured
response, one post-processing pass.

Goals are identified by POSITION in the LLM output array (no `id` field on
the schema). The LLM cannot drift identity across iterations because there
is no identity field to drift. Post-processing assigns deterministic ids
(`g::1`, `g::2`, …) and preserves them across iterations via a defensive
merge that only updates `done` flags and appends new goals at the end.
"""

import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_gateway"))

from .schemas import MemoryItem, Goal, Observation, Kind
from .instrumented_llm import InstrumentedLLM
from .logging_config import get_logger
from . import config

logger = get_logger(__name__)


# ── LLM output schema (position-based, no id field) ──────────────────────────

class _GoalDelta(BaseModel):
    """What the Perception LLM emits per goal. No `id` field — goals are
    identified by their position in the output list. The LLM cannot drift
    identity across iterations because there is no identity field to drift."""
    text: str = Field(max_length=240)
    done: bool = False
    send_artifact: bool = False
    artifact_index: int | None = None


class _PerceptionOutput(BaseModel):
    goals: list[_GoalDelta] = Field(default_factory=list, max_length=10)


# Synthesis-type goals require Decision to actually produce a substantive
# answer; Perception must not declare them done on the strength of a
# tool-call alone.
SYNTHESIS_KW = (
    "evaluate", "select", "synthes", "compare", "decide", "recommend",
    "tell me which", "most appropriate", "analy", "pick", "choose",
    "summarise", "summarize", "answer", "identify", "find", "determine",
    "extract", "list", "report", "tell", "explain", "describe", "name",
)


class Perception:
    def __init__(self):
        self.llm = InstrumentedLLM()
        self.PERCEPTION_SYSTEM_PROMPT = config.PERCEPTION_SYSTEM_PROMPT

    def reset(self) -> None:
        """Reset perception state for a new run."""
        pass

    # ── Single-call observe ──────────────────────────────────────────────────

    def observe(
        self,
        query: str,
        hits: list[MemoryItem],
        history: list[dict],
        prior_goals: list[Goal],
        run_id: str,
    ) -> Observation:
        """One LLM call per iteration. Decomposes (iter 1), updates done
        flags (iter 2+), decides artifact attachment, and may append new
        goals — all in the same structured response."""
        logger.debug("perception_observe_start",
                    has_prior_goals=bool(prior_goals),
                    prior_goal_count=len(prior_goals) if prior_goals else 0,
                    memory_hits=len(hits) if hits else 0,
                    history_len=len(history) if history else 0)

        # Artifact ids in hit order, so the LLM can point at them by integer.
        art_ids_in_order = [h.artifact_id for h in hits[:12] if h.artifact_id]

        prior_snapshot = [g.model_dump() for g in prior_goals] if prior_goals else []
        prompt = (
            f"USER QUERY:\n  {query}\n\n"
            f"PRIOR GOALS:\n{json.dumps(prior_snapshot, indent=2)}\n\n"
            f"MEMORY HITS (handles + descriptors only, no raw bytes; `i` is the\n"
            f"artifact_index to pass back when send_artifact is true):\n"
            f"{json.dumps(self._snapshot_hits(hits), indent=2)}\n\n"
            f"RUN HISTORY (last 10 events):\n"
            f"{json.dumps(self._snapshot_history(history), indent=2, default=str)}\n\n"
            f"Return the current goal list as JSON matching the schema."
        )

        _c = config.PERCEPTION_LLM
        start_time = time.time()
        reply = self.llm.chat(
            call_label="perception.observe",
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=_c.cache_system,
            response_format={
                "type": "json_schema",
                "schema": _PerceptionOutput.model_json_schema(),
                "name": "PerceptionOutput",
                "strict": True,
            },
            reasoning=_c.reasoning,
            temperature=_c.temperature,
            max_tokens=_c.max_tokens,
            provider=_c.provider,
            model=_c.model,
            auto_route=_c.auto_route,
        )

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("perception_llm_complete",
                    call="observe",
                    duration_ms=duration_ms,
                    input_tokens=reply.get("input_tokens", 0),
                    output_tokens=reply.get("output_tokens", 0),
                    provider=reply.get("provider"),
                    model=reply.get("model"),
                    attempted=reply.get("attempted", []),
                    router_decision=reply.get("router_decision"))

        parsed = reply.get("parsed")
        if not parsed or not parsed.get("goals"):
            # Fallback: single goal echoing the query.
            return Observation(goals=[Goal(id="g::1", text=query, done=False, attach_artifact_id=None)])

        out_goals = self._post_process(
            parsed["goals"],
            prior_goals,
            history,
            art_ids_in_order,
        )

        done_count = sum(1 for g in out_goals if g.done)
        logger.info("perception_complete",
                    goal_count=len(out_goals),
                    done=done_count,
                    remaining=len(out_goals) - done_count,
                    goals=[{"id": g.id, "text": g.text, "done": g.done,
                            "attach_artifact_id": g.attach_artifact_id} for g in out_goals])
        return Observation(goals=out_goals)

    # ── Post-processing ──────────────────────────────────────────────────────

    def _post_process(
        self,
        raw_goals: list[dict],
        prior_goals: list[Goal],
        history: list[dict],
        art_ids_in_order: list[str],
    ) -> list[Goal]:
        """Convert the LLM's position-based output into Goal objects with
        stable ids, preserving prior goals and appending new ones with dedup."""
        n_prior = len(prior_goals)
        out_goals: list[Goal] = []

        # Phase 1: merge existing goals (preserve id, text; update done only).
        prior_texts = set()
        for i in range(n_prior):
            delta = _GoalDelta.model_validate(raw_goals[i]) if i < len(raw_goals) else None
            prior = prior_goals[i]
            prior_texts.add(prior.text.strip().lower())

            new_done = prior.done
            if delta and not prior.done:
                new_done = self._check_done(prior.id, delta.text, delta.done, history)

            attach = None
            if delta and delta.send_artifact and delta.artifact_index is not None:
                if 0 <= delta.artifact_index < len(art_ids_in_order):
                    attach = art_ids_in_order[delta.artifact_index]

            out_goals.append(Goal(
                id=prior.id,
                text=prior.text,
                done=new_done,
                attach_artifact_id=attach,
            ))

        # Phase 2: append new goals (dedup against existing texts).
        for i in range(n_prior, len(raw_goals)):
            delta = _GoalDelta.model_validate(raw_goals[i])
            t = delta.text.strip().lower()
            if not t or t in prior_texts:
                continue
            prior_texts.add(t)

            attach = None
            if delta.send_artifact and delta.artifact_index is not None:
                if 0 <= delta.artifact_index < len(art_ids_in_order):
                    attach = art_ids_in_order[delta.artifact_index]

            gid = f"g::{len(out_goals) + 1}"
            done = self._check_done(gid, delta.text, delta.done, history) if delta.done else False
            out_goals.append(Goal(
                id=gid,
                text=delta.text,
                done=done,
                attach_artifact_id=attach,
            ))
            logger.info("goal_appended", goal_id=gid, goal_text=delta.text)

        return out_goals

    @staticmethod
    def _check_done(gid: str, text: str, llm_done: bool, history: list[dict]) -> bool:
        """Apply the SYNTHESIS_KW guard: don't let Perception declare a
        synthesis-type goal done unless history has a substantive answer."""
        if not llm_done:
            return False
        gtext_lc = text.lower()
        if any(kw in gtext_lc for kw in SYNTHESIS_KW):
            has_answer = any(
                h.get("kind") == "answer"
                and h.get("goal_id") == gid
                and len((h.get("text") or "")) > 60
                for h in history
            )
            if not has_answer:
                return False
        return True

    # ── Prompt helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _snapshot_hits(hits: list[MemoryItem]) -> list[dict]:
        """Render the memory hits the LLM sees. Artifacts are indexed (i) so
        Perception can point at them by integer; non-artifact hits show i=null."""
        art_pos = 0
        out = []
        for h in hits[:12]:
            i = None
            if h.artifact_id:
                i = art_pos
                art_pos += 1
            out.append({
                "i": i,
                "kind": h.kind.value if isinstance(h.kind, Kind) else h.kind,
                "descriptor": h.descriptor,
                "keywords": h.keywords,
                "artifact_id": h.artifact_id,
            })
        return out

    @staticmethod
    def _snapshot_history(history: list[dict]) -> list[dict]:
        """Clip to last 10 events and truncate long string fields to 2000 chars."""
        out = []
        for h in history[-10:]:
            clipped = {}
            for k, v in h.items():
                if isinstance(v, str) and len(v) > 2000:
                    clipped[k] = v[:2000] + "..."
                else:
                    clipped[k] = v
            out.append(clipped)
        return out
