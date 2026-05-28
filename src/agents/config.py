"""
Central configuration for the agent system.

Change the LLM provider/model and all prompts here.
Each use case gets its own LLMConfig so you can route different
tasks to different providers, models, or parameter sets.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: Optional[str] = None      # "sg", "openai", "gemini", etc. None = let router decide
    model: Optional[str] = None         # None = provider/router default
    temperature: float = 0
    max_tokens: int = 1024
    reasoning: str = "medium"           # "off", "low", "medium", "high"
    cache_system: bool = True
    auto_route: Optional[str] = None    # "perception", "memory", "decision" — gateway router picks model
    #                                     Setting provider overrides auto_route (explicit wins).


# ── Per-Use-Case LLM Settings ────────────────────────────────────────────────
# Set auto_route to let the gateway router classify each call and pick the
# best model/tier automatically. Or set provider (+ optional model) to pin
# a specific backend. provider overrides auto_route when both are set.
#
# Router mode (default):
#   LLMConfig(auto_route="decision")
# Manual mode:
#   LLMConfig(provider="gemini", model="gemini-2.5-flash")

# Memory: lightweight extraction & ranking — reasoning off
MEMORY_EXTRACTION_LLM  = LLMConfig(auto_route="memory", reasoning="off")
MEMORY_RELEVANCE_LLM   = LLMConfig(auto_route="memory", reasoning="off")
# MEMORY_EXTRACTION_LLM  = LLMConfig(provider="sglang")
# MEMORY_RELEVANCE_LLM   = LLMConfig(provider="sglang")


# # Perception: goal decomposition, goal updates, artifact attachment
PERCEPTION_DECOMPOSE_LLM  = LLMConfig(auto_route="perception")
PERCEPTION_UPDATE_LLM     = LLMConfig(auto_route="perception")
PERCEPTION_ARTIFACT_LLM   = LLMConfig(auto_route="perception")
# PERCEPTION_DECOMPOSE_LLM  = LLMConfig(provider="sglang")
# PERCEPTION_UPDATE_LLM     = LLMConfig(provider="sglang")
# PERCEPTION_ARTIFACT_LLM   = LLMConfig(provider="sglang")

# # Decision: the heaviest call — main action selection
DECISION_LLM = LLMConfig(auto_route="decision")
# DECISION_LLM = LLMConfig(provider="sglang")

# ── Memory Prompts ────────────────────────────────────────────────────────────

MEMORY_RELEVANCE_SYSTEM_PROMPT = """You are a memory relevance ranker. Given a query and a list of memories, 
return the IDs of the most relevant memories as a JSON list of strings, ordered by relevance (most relevant first).
Only return the JSON list, nothing else."""

MEMORY_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. Given a user query, extract ONLY information explicitly stated in the query text. Do NOT hallucinate facts, dates, names, or answers that are not literally present.

Return a JSON object with these fields:
- "should_store": boolean. Set to false if the user is only asking a question or requesting retrieval of external data (e.g., "fetch", "tell me", "find", "look up"). Set to true if the query contains explicit facts or preferences worth remembering.
- "kind": one of "preference", "fact", "tool_outcome", "scratchpad" (use "preference" for user likes/dislikes/wants, "fact" only for objective information explicitly in the query)
- "keywords": list of relevant keywords/tags for retrieval
- "descriptor": a short one-line human-readable summary
- "value": a dict with structured data extracted ONLY from the query text. If should_store is false, leave "value" empty ({}).
- "confidence": float 0.0-1.0 indicating how confident you are in the extraction

Only return valid JSON, nothing else.

Example input: "I prefer dark mode and use Python for most projects"
Example output:
{"should_store": true, "kind": "preference", "keywords": ["dark mode", "python", "preferences", "coding"], "descriptor": "User prefers dark mode and primarily uses Python", "value": {"ui_preference": "dark mode", "primary_language": "Python"}, "confidence": 0.95}

Example input: "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date"
Example output:
{"should_store": false, "kind": "fact", "keywords": ["Claude Shannon", "Wikipedia", "birth date"], "descriptor": "User wants to fetch Claude Shannon Wikipedia page and find birth date", "value": {}, "confidence": 1.0}
"""


# ── Perception Prompts ────────────────────────────────────────────────────────

PERCEPTION_SYSTEM_PROMPT = """
You are a perception module for an agentic AI system. Respond in JSON.
Your job is to analyze the user's query, the current memory hits, the run
history, and the prior goal list to produce a fresh observation containing
the current goals list with done flags and optional artifact attachments.

Rules:
- Preserve goal order. Do not reorder, insert, or drop goals.
- Once a goal is marked done, it stays done.
- Only attach artifacts to the first unfinished goal.
"""

DECOMPOSE_QUERY_PROMPT = """
Decompose the user's query into one or more bounded goals, each a short imperative statement.
"""

UPDATE_GOALS_PROMPT = """
For each prior goal, examine the run history. Mark the goal `done: true`
the moment the history contains an action or answer that satisfies it. Once done,
the goal remains done in every subsequent iteration.
"""

ATTACH_ARTIFACT_PROMPT = """
For the first unfinished goal in the list, respond in JSON.
Decide whether it needs raw bytes from a previously fetched artifact.
If yes, set the goal's attach_artifact_id to one of the artifact handles in MEMORY HITS.
"""

CHECK_IF_ARTIFACT_NEEDED_PROMPT = """
Examine the goal and the available memory hits. Respond in JSON.
Determine if this goal requires raw bytes from a previously fetched artifact to proceed.

Return true only if:
- The goal explicitly references content that exists in an artifact
- The artifact's descriptor indicates it contains data needed for this goal

Return false if the goal can be accomplished without artifact data.
"""


# ── Decision Prompts ──────────────────────────────────────────────────────────

DECISION_SYSTEM_PROMPT = """You are a decision module. Respond in JSON with exactly ONE of two outputs:
1. A final ANSWER if you can satisfy the goal from the available context, OR
2. A single TOOL CALL if external action is required.

Rules:
- If the goal contains a URL like "https://...", you MUST call fetch_url. Do NOT answer.
- If the goal says "Fetch", "Get", "Download", or "Retrieve" a webpage or file, you MUST call the matching tool. Do NOT answer from memory.
- Strings starting with "art:" are internal artifact handles, NOT file paths or URLs. Never pass them to tools like read_file or fetch_url. When artifact bytes are needed, they appear under ATTACHED ARTIFACTS.
- When the goal asks for extraction, listing, comparison, or selection, your answer must be substantive: at least 3 sentences or a list of items. Do not return meta-answers like "the page has been fetched".
- Pick exactly one tool. Do not narrate.

Examples:
Goal: "Fetch https://en.wikipedia.org/wiki/Claude_Shannon"
Output: {"tool_call": {"name": "fetch_url", "arguments": {"url": "https://en.wikipedia.org/wiki/Claude_Shannon"}}}

Goal: "Extract birth date from the fetched Wikipedia page" (with attached artifact)
Output: {"answer": "Claude Shannon was born on April 30, 1916."}
"""
