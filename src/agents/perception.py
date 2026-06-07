"""
Perception module for agents

> Perception tracks state across agentic loop iterations. It runs the iterations themselves
and updates the state accordingly. Each iteration, it receives 4 inputs: users original query,
the current memory hits, run history so far, and prior goal list.

> Perception emits a fresh observation containing the current goals list with done flags and optional artifact attachments.
The decomposition into goals happens the first time perception runs and every other iteration the goal list is preserved 
and updated the done flags and attach_artifact_id on the next unfinished goal.

Objectives of Perception:
1. If the prior goal list is empty, decompose the query into one or more
   bounded goals, each a short imperative statement.

2. For each prior goal, examine the run history. Mark the goal `done: true`
   the moment the history contains an action that satisfies it. Once done,
   the goal remains done in every subsequent iteration.

3. For the first unfinished goal in the list, decide whether it needs raw
   bytes from a previously fetched artifact. If yes, set the goal's
   attach_artifact_id to one of the artifact handles in MEMORY HITS.

4. Preserve goal order. Do not reorder, do not insert in the middle, do
   not drop a goal.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_gateway"))

from .schemas import MemoryItem, Goal, Observation
from client import LLM
from .logging_config import get_logger
from . import config

logger = get_logger(__name__)


class Perception:
    def __init__(self):
        self.llm = LLM()

        self.DECOMPOSE_QUERY_PROMPT = config.DECOMPOSE_QUERY_PROMPT
        self.UPDATE_GOALS_PROMPT = config.UPDATE_GOALS_PROMPT
        self.ATTACH_ARTIFACT_PROMPT = config.ATTACH_ARTIFACT_PROMPT
        self.PERCEPTION_SYSTEM_PROMPT = config.PERCEPTION_SYSTEM_PROMPT
        self.CHECK_IF_ARTIFACT_NEEDED_PROMPT = config.CHECK_IF_ARTIFACT_NEEDED_PROMPT


    def reset(self) -> None:
        """Reset perception state for a new run."""
        pass


    def observe(self, query: str, hits: list[MemoryItem], history: list[dict], prior_goals: list[Goal], run_id: str) -> Observation:
        """
        Track state across agentic loop iterations. Run the iterations themselves
        and update the state accordingly. Each iteration, receive 4 inputs: users
        original query, the current memory hits, run history so far, and prior
        goal list.
        
        Emit a fresh observation containing the current goals list with done flags
        and optional artifact attachments.

        query: The user's original query
        hits: The current memory hits
        history: The run history so far
        prior_goals: The prior goal list
        run_id: The run ID
        """
        logger.debug("perception_observe_start",
                    has_prior_goals=bool(prior_goals),
                    prior_goal_count=len(prior_goals) if prior_goals else 0,
                    memory_hits=len(hits) if hits else 0,
                    history_len=len(history) if history else 0)

        if not prior_goals:
            goals = self._decompose_query(query)
            logger.info("goals_decomposed",
                       goal_count=len(goals),
                       goals=[{"id": g.id, "text": g.text} for g in goals])
        else:
            goals = self._update_goals(prior_goals, history)
            done_count = sum(1 for g in goals if g.done)
            logger.info("goals_updated",
                       total=len(goals),
                       done=done_count,
                       remaining=len(goals) - done_count)
        
        # Attach artifact to the first unfinished goal only
        for i, goal in enumerate(goals):
            if not goal.done:
                goals[i] = self._attach_artifacts(goal, hits)
                if goals[i].attach_artifact_id:
                    logger.info("artifact_attached_to_goal",
                               goal_id=goals[i].id,
                               artifact_id=goals[i].attach_artifact_id)
                break  # Only attach to first unfinished goal
        
        return Observation(goals=goals)


    def _decompose_query(self, query: str) -> list[Goal]:
        """
        Decompose the query into one or more bounded goals, each a short imperative statement.
        """
        prompt = self.DECOMPOSE_QUERY_PROMPT + "\n" + query
        
        logger.debug("decompose_query_start", query_len=len(query))
        start_time = time.time()
        
        _c = config.PERCEPTION_DECOMPOSE_LLM
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=_c.cache_system,
            response_format={
                "type": "json_schema",
                "schema": Observation.model_json_schema(),
                "name": "Goals",
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
                    call="decompose_query",
                    duration_ms=duration_ms,
                    input_tokens=reply.get("input_tokens", 0),
                    output_tokens=reply.get("output_tokens", 0),
                    provider=reply.get("provider"),
                    model=reply.get("model"),
                    attempted=reply.get("attempted", []),
                    router_decision=reply.get("router_decision"))
        
        obs = Observation.model_validate(reply["parsed"])

        # Overwrite LLM-generated IDs with deterministic sequential IDs.
        # TINY-tier models often produce duplicates (e.g., all "?" or all "g").
        goals = []
        for idx, g in enumerate(obs.goals, start=1):
            goals.append(Goal(
                id=f"g::{idx}",
                text=g.text,
                done=False,
                attach_artifact_id=None,
            ))
        return goals


    def _update_goals(self, goals: list[Goal], history: list[dict]) -> list[Goal]:
        """
        For each prior goal, examine the run history. Mark the goal `done: true`
        the moment the history contains an action that satisfies it. Once done,
        the goal remains done in every subsequent iteration.
        """
        prompt = f"""{self.UPDATE_GOALS_PROMPT}
        
Goals:
{str(goals)}
        
History:
{str(history)}
        """
        
        logger.debug("update_goals_start", goal_count=len(goals), history_len=len(history))
        start_time = time.time()
        
        _c = config.PERCEPTION_UPDATE_LLM
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=_c.cache_system,
            response_format={
                "type": "json_schema",
                "schema": Observation.model_json_schema(),
                "name": "Goals",
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
                    call="update_goals",
                    duration_ms=duration_ms,
                    input_tokens=reply.get("input_tokens", 0),
                    output_tokens=reply.get("output_tokens", 0),
                    provider=reply.get("provider"),
                    model=reply.get("model"),
                    attempted=reply.get("attempted", []),
                    router_decision=reply.get("router_decision"))
        obs = Observation.model_validate(reply["parsed"])
        
        # Defensive merge: preserve original goals, only update done flags.
        # This prevents the LLM from dropping, reordering, or mutating goals.
        llm_done_by_id = {g.id: g.done for g in obs.goals}
        updated = []
        for original in goals:
            new_done = original.done or llm_done_by_id.get(original.id, original.done)
            if not original.done and new_done:
                logger.info("goal_marked_done", goal_id=original.id, goal_text=original.text)
            updated.append(Goal(
                id=original.id,
                text=original.text,
                done=new_done,
                attach_artifact_id=original.attach_artifact_id,
            ))
        
        logger.debug("update_goals_complete", duration_ms=duration_ms)
        return updated


    def _attach_artifacts(self, goal: Goal, hits: list[MemoryItem]) -> Goal:
        """
        For the first unfinished goal in the list, decide whether it needs raw
        bytes from a previously fetched artifact. If yes, set the goal's
        attach_artifact_id to one of the artifact handles in MEMORY HITS.
        """

        # Short-circuit: no memory hits means no artifacts to attach
        artifact_hits = [h for h in hits if h.artifact_id]
        if not artifact_hits:
            return goal

        # STEP1: Check if artifact is needed
        prompt = f"""{self.CHECK_IF_ARTIFACT_NEEDED_PROMPT}
        
Goal:
{str(goal)}
        
Memory Hits:
{str(hits)}
        """
        
        _c = config.PERCEPTION_ARTIFACT_LLM
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=_c.cache_system,
            response_format={
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"needs_artifact": {"type": "boolean"}},
                    "required": ["needs_artifact"],
                    "additionalProperties": False,
                },
                "name": "NeedsArtifact",
                "strict": True,
            },
            reasoning=_c.reasoning,
            temperature=_c.temperature,
            max_tokens=_c.max_tokens,
            provider=_c.provider,
            model=_c.model,
            auto_route=_c.auto_route,
        )

        needs_artifact = reply["parsed"]["needs_artifact"]
        logger.debug("artifact_check_result", goal_id=goal.id, needs_artifact=needs_artifact)

        if not needs_artifact:
            return goal

        # STEP2: Pick which artifact to attach.
        # memory hits should be sorted by relevance score and must only contain artifacts with description and id
        # TODO: add ability for the LLM to choose to explore what each artifacts hold and decide instead of relying on description.
        prompt = f"""{self.ATTACH_ARTIFACT_PROMPT}
        
Goal:
{str(goal)}
        
Memory Hits (with artifacts):
{str(artifact_hits)}
        """
        
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=_c.cache_system,
            response_format={
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "attach_artifact_id": {
                            "enum": [h.artifact_id for h in artifact_hits if h.artifact_id] + [None],
                        },
                    },
                    "required": ["attach_artifact_id"],
                    "additionalProperties": False,
                },
                "name": "ArtifactAttachment",
                "strict": True,
            },
            reasoning=_c.reasoning,
            temperature=_c.temperature,
            max_tokens=_c.max_tokens,
            provider=_c.provider,
            model=_c.model,
            auto_route=_c.auto_route,
        )
        
        artifact_id = reply["parsed"].get("attach_artifact_id")
        return Goal(
            id=goal.id,
            text=goal.text,
            done=goal.done,
            attach_artifact_id=artifact_id,
        )
