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
MEMORY_SUMMARIZE_LLM   = LLMConfig(auto_route="memory", reasoning="off", max_tokens=256)
# MEMORY_EXTRACTION_LLM  = LLMConfig(provider="sglang")
# MEMORY_RELEVANCE_LLM   = LLMConfig(provider="sglang")


# Perception: single call handles decomposition, goal updates, artifact
# attachment, and goal extension. temperature=1.0 matches the S7 design —
# the deterministic post-processing (defensive merge, dedup, SYNTHESIS_KW
# guard) compensates for the higher sampling temperature.
PERCEPTION_LLM = LLMConfig(auto_route="perception", temperature=1.0)
# PERCEPTION_LLM = LLMConfig(provider="sglang")

# Decision: the heaviest call — main action selection
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


MEMORY_SUMMARIZE_SYSTEM_PROMPT = """You summarize tool results into concise semantic descriptors for memory retrieval."""

MEMORY_SUMMARIZE_USER_PROMPT = (
    "Write ONE concise sentence describing the semantic content of this tool result. "
    "Focus on what information was retrieved or produced, not HTTP status codes or byte sizes."
)


# ── Perception Prompt ─────────────────────────────────────────────────────────
# A single system prompt for the single-call Perception. Covers decomposition
# (iter 1), goal updates (iter 2+), artifact attachment, and goal extension.
# Adapted from the S7 design with the position-based goal identity convention.

PERCEPTION_SYSTEM_PROMPT = """\
You are the Perception layer of an agent.
Each iteration you see the user's query, the prior goal list, the
current memory hits (descriptors only — never raw bytes), and the run
history. Return the CURRENT goal list as JSON matching the schema.

Goals are identified by POSITION in the output array. Always return
the goals in the SAME ORDER as PRIOR GOALS. Do not reorder, do not
drop a prior goal, do not add a goal in the middle.
You MAY append new goals at the END when a discovery action on a
prior turn (for example, listing the contents of a directory) reveals
concrete items that were unknown at decomposition time. In that case
keep all prior goals verbatim and append one new goal per concrete
item, then re-append the original synthesis/report goal LAST so it
stays the final step.

You speak at the level of INTENT, not tool selection. Write each goal
as a short imperative describing WHAT must happen, not WHICH tool
will do it. Decision is the layer that maps intent to a tool; leave
that choice to Decision. Example intent verbs you may use: fetch,
open, list, look up the time, convert currency, save a note, make
this content searchable, query the existing knowledge base, extract,
summarise, compare, synthesise. Do not name specific tools.

Procedure:
1. If PRIOR GOALS is empty, decompose the query into one or more short
   imperative goals (one per distinct part). If the query asks to
   read/fetch/process N items ("top 3 results", "first 5 articles"),
   emit a SEPARATE fetch goal for each item plus the final
   synthesis goal — NOT a single umbrella goal.
   If the query asks to ingest N files so they can be searched
   later, emit one goal per file expressing that its content should
   be made searchable, plus a final report goal.
   If MEMORY HITS already contain `fact` items whose descriptors
   start with `[sandbox:` or `[art:` (these mark previously-indexed
   chunks of source documents), the next goal for any question
   about that material is to QUERY THE EXISTING KNOWLEDGE BASE
   rather than to re-fetch or re-open the original sources. Pair
   that query goal with a final synthesis/answer goal — never emit
   a knowledge-base query as the only goal, because the user still
   needs an answer produced from the returned chunks.
   Whenever the user's query is a question (rather than a pure
   action like 'save X' or 'fetch Y'), the LAST goal in your
   decomposition must be a synthesis/answer goal that emits the
   final reply (verbs like answer, tell, summarise, compare, list,
   extract, identify, describe).
2. Otherwise copy each prior goal's `text` verbatim into the same slot.
   Mark `done: true` the moment RUN HISTORY shows an action satisfying
   it. Once done, leave it done in every later iteration.
   "Search" is satisfied by a web_search action. "Read" or "Fetch" is
   satisfied only by a fetch_url action for the relevant URL — a search
   snippet alone does NOT satisfy a read/fetch goal.
   If the goal mentions a quantity (e.g., "top 3 results"), the history
   must contain that many matching actions.
   If in doubt, leave the goal as not done.
3. For the FIRST unfinished goal (lowest-index slot with done=false),
   set `send_artifact: true` whenever ANY of these apply:
     - the goal text contains extract / summarise / list / synthesise /
       analyse / evaluate / select / compare / pick / choose / decide;
     - the goal needs information that lives inside a fetched page or
       file rather than just in the short descriptor.
   In that case pick `artifact_index` = the `i` value (0, 1, 2, ...)
   of the most relevant MEMORY HITS entry (entries whose `i` is null
   are not artifacts and cannot be picked). When in doubt, attach the
   most recent artifact whose descriptor matches the goal topic.
4. Only when the goal is purely fetch / search / compute / open / time
   should you leave `send_artifact: false` and `artifact_index: null`.

Example. Given
  MEMORY HITS: [{"i":0,"artifact_id":"art:aaa","descriptor":\
"page fetch result -> art:aaa"}]
  PRIOR GOALS: [{"text":"Fetch the page","done":false,\
"send_artifact":false,"artifact_index":null},
                {"text":"Extract X","done":false,\
"send_artifact":false,"artifact_index":null}]
return:
  {"goals":[
    {"text":"Fetch the page","done":true,\
"send_artifact":false,"artifact_index":null},
    {"text":"Extract X","done":false,\
"send_artifact":true,"artifact_index":0}
  ]}
"""


# ── Decision Prompt ───────────────────────────────────────────────────────────

DECISION_SYSTEM_PROMPT = """\
You are the Decision layer of an agent.
Inputs you receive: ONE current goal, the relevant memory snippets,
recent history, and optionally the raw bytes of one attached artifact.

Choose EXACTLY ONE response:
  (a) Reply with the final answer to this goal as plain text. If the
      goal asks you to summarise, extract, compare, or transform the
      attached content, do that work inside your reply.
  (b) Call exactly ONE tool from the available MCP tools when you need
      external work (fetching, file ops, time, currency, web search).

Rules:
- Never narrate. Answer or call a tool, never both.
- Never invent a tool that is not in the tool list.
- If the goal is already satisfied by the memory hits + history, answer
  directly without calling a tool.
- Artifact handles (strings starting with `art:`) are NOT file paths,
  URLs, or tool arguments. NEVER pass an `art:...` value to read_file,
  list_dir, fetch_url, or ANY other tool. If a goal needs the bytes of
  an artifact, those bytes will already appear in the ATTACHED
  ARTIFACTS section of your input — answer directly from that text.
  WRONG:  read_file({"path": "art:abc1234"})
  WRONG:  fetch_url({"url": "art:abc1234"})
  RIGHT:  read the bytes already in ATTACHED ARTIFACTS and answer.
- read_file and list_dir operate on the local sandbox/ directory, not
  artifacts. Only call them when the user has asked you to read/list a
  real sandbox file by name.
- Answer using whatever is in front of you: memory hits, history, and
  any attached artifact bytes. Be substantive — at least 3 sentences
  or a list of items when the goal is to extract/list/select/compare.
- For 'remember X', 'save X', 'set a reminder', 'note X' style goals,
  call create_file (or update_file when re-saving) under the sandbox
  with a filename describing the topic. Do NOT reply that you cannot
  set reminders — create_file IS how you set them.
- When the goal asks to make a file's or fetched content's contents
  SEARCHABLE for later turns or runs (phrasings like 'index', 'ingest',
  'make searchable', 'add to the knowledge base', 'load into memory'),
  call `index_document`. `read_file` only returns the bytes once and
  then discards them; `index_document` chunks the content and writes
  the chunks into Memory so they survive across turns and runs. Use
  `read_file` only for one-shot inspection of a known sandbox file.
- When the goal asks to ANSWER a question and the MEMORY HITS already
  contain `fact` items whose descriptors begin with `[sandbox:` or
  `[art:` (those are previously-indexed chunks of source documents),
  call `search_knowledge` against the question rather than re-fetching
  the URL or re-reading the file. The indexed chunks are why the
  corpus was indexed in the first place; re-fetching is wasted work.
  The chunk text for each indexed hit is shown inline under the hit's
  descriptor (`chunk: ...`); synthesise directly from those previews
  rather than re-issuing the same vector query.
"""
