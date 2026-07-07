from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

 
class Kind(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    TOOL_OUTCOME = "tool_outcome"
    SCRATCHPAD = "scratchpad"


class MemoryItem(BaseModel):
    should_store: bool = True
    id: str = ""
    kind: Kind
    keywords: list[str]
    descriptor: str            # one short human-readable line
    value: dict                # structured payload
    artifact_id: str | None = None
    embedding: list[float] | None = None  # 768-dim nomic vector; None for scratchpad
    source: str = ""
    run_id: str = ""
    goal_id: str | None = None
    confidence: float
    created_at: datetime = Field(default_factory=datetime.now)
    expiry_date: datetime | None = None  # None means never expires

    def to_prompt(self) -> dict:
        """Return a prompt-safe projection of this memory item.

        Excludes the `embedding` field (a 768-dim float vector used only by the
        FAISS index) so that stringifying memory hits into LLM prompts does not
        pollute the context with thousands of meaningless floats. The embedding
        remains on the model for index rebuild/append paths.
        """
        return {
            "id": self.id,
            "kind": self.kind.value if isinstance(self.kind, Kind) else self.kind,
            "keywords": self.keywords,
            "descriptor": self.descriptor,
            "value": self.value,
            "artifact_id": self.artifact_id,
            "source": self.source,
            "run_id": self.run_id,
            "goal_id": self.goal_id,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MemoryExtraction(BaseModel):
    """LLM output schema for memory attribute extraction.

    This is the minimal contract the LLM must satisfy — only the 6 fields the
    extraction code actually reads. The other 8 fields on ``MemoryItem``
    (``id``, ``artifact_id``, ``embedding``, ``source``, ``run_id``,
    ``goal_id``, ``created_at``, ``expiry_date``) are system-prefilled by
    ``remember()`` after extraction and must NOT appear in the LLM schema.

    Using ``MemoryItem.model_json_schema()`` directly was problematic:
    - The 768-dim ``embedding`` float array appeared in the schema, inviting
      the model to hallucinate it or, under strict mode, forcing it to emit
      768 floats.
    - System-managed fields (``id``, ``run_id``, ``created_at``) were exposed,
      letting the model drift values that are then overwritten anyway.
    - Under ``_groq_strict_schema``, all 14 fields became ``required``,
      forcing the model to produce every one including the embedding vector.
    """
    should_store: bool = True
    kind: Kind
    keywords: list[str]
    descriptor: str
    value: dict
    confidence: float = Field(ge=0.0, le=1.0)


class Artifact(BaseModel):
    id: str                    # "art:<sha256-prefix>"
    content_type: str
    size_bytes: int
    source: str
    descriptor: str


class Goal(BaseModel):
    id: str
    text: str                  # short imperative description
    done: bool
    attach_artifact_id: str | None


class Observation(BaseModel):
    goals: list[Goal]

    @property
    def all_done(self) -> bool:
        """Return True if all goals are done."""
        return all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        """Return the first goal that is not done, or None if all done."""
        for g in self.goals:
            if not g.done:
                return g
        return None


class ToolCall(BaseModel):
    name: str
    arguments: dict


class DecisionOutput(BaseModel):
    answer: str | None         # exactly one of these two is populated
    tool_call: ToolCall | None

    @property
    def is_answer(self) -> bool:
        return self.answer is not None