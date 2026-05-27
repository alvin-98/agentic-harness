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

from .memory import MemoryItem
from .goal import Goal
from .observation import Observation
from client import LLM


class Perception:
    def __init__(self):
        self.llm = LLM()

        self.DECOMPOSE_QUERY_PROMPT = """
        Decompose the user's query into one or more bounded goals, each a short imperative statement.
        """

        self.UPDATE_GOALS_PROMPT = """
        For each prior goal, examine the run history. Mark the goal `done: true`
        the moment the history contains an action that satisfies it. Once done,
        the goal remains done in every subsequent iteration.
        """

        self.ATTACH_ARTIFACT_PROMPT = """
        For the first unfinished goal in the list, decide whether it needs raw
        bytes from a previously fetched artifact. If yes, set the goal's
        attach_artifact_id to one of the artifact handles in MEMORY HITS.
        """

        self.PERCEPTION_SYSTEM_PROMPT = """
        You are a perception module for an agentic AI system. Your job is to analyze
        the user's query, the current memory hits, the run history, and the prior
        goal list to produce a fresh observation containing the current goals list
        with done flags and optional artifact attachments.
        """

        self._iteration = 0


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
        if self._iteration == 0:
            goals = self._decompose_query(query)
        
            for goal in goals:
                if not goal.done:
                    # TODO: check if artifact is needed for this goal based on goals and hits  
                    goal = self._attach_artifacts(goal, hits)
                    # TODO: update goal back to goals list
        else:
            goals = self._update_goals(prior_goals, history)
        self._iteration += 1
        return Observation(goals=goals)


    def _decompose_query(self, query: str) -> list[Goal]:
        """
        Decompose the query into one or more bounded goals, each a short imperative statement.
        """
        prompt = self.DECOMPOSE_QUERY_PROMPT + "\n" + query
        
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=True,
            response_format={
                "type": "json_schema",
                "schema": Observation.model_json_schema(),
                "name": "Goals",
                "strict": True,
            },
            reasoning="medium",
            temperature=0,
            max_tokens=1024,
        )
        
        return reply["parsed"]


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
        
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=True,
            response_format={
                "type": "json_schema",
                "schema": Observation.model_json_schema(),
                "name": "Goals",
                "strict": True,
            },
            reasoning="medium",
            temperature=0,
            max_tokens=1024,
        )
        
        return reply["parsed"]


    def _attach_artifacts(self, goal: Goal, hits: list[MemoryItem]) -> Goal:
        """
        For the first unfinished goal in the list, decide whether it needs raw
        bytes from a previously fetched artifact. If yes, set the goal's
        attach_artifact_id to one of the artifact handles in MEMORY HITS.
        """

        # STEP1: Check if artifact is needed
        prompt = f"""{self.CHECK_IF_ARTIFACT_NEEDED_PROMPT}
        
Goal:
{str(goal)}
        
Memory Hits:
{str(hits)}
        """
        
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=True,
            response_format={
                "type": "json_schema",
                "schema": {"type": "boolean"}, # should be simple boolean holding true / false
                "name": "NeedsArtifact",
                "strict": True,
            },
            reasoning="medium",
            temperature=0,
            max_tokens=1024,
        )

        needs_artifact = reply["parsed"]

        if not needs_artifact:
            return goal

        # STEP3: Attach relevant artifact / if multiple artifacts then compose relevant information from them.
        # memory hits should be sorted by relevance score and must only contain artifacts with description and id
        # TODO: add ability for the LLM to choose to explore what each artifacts hold and decide instead of relying on description.
        prompt = f"""{self.ATTACH_ARTIFACT_PROMPT}
        
Goal:
{str(goal)}
        
Memory Hits:
{str(hits)}
        """
        
        reply = self.llm.chat(
            prompt=prompt,
            system=self.PERCEPTION_SYSTEM_PROMPT,
            cache_system=True,
            response_format={
                "type": "json_schema",
                "schema": Goal.model_json_schema(),
                "name": "Goal",
                "strict": True,
            },
            reasoning="medium",
            temperature=0,
            max_tokens=1024,
        )
        
        return reply["parsed"]
