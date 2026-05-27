from pydantic import BaseModel
from datetime import datetime
from enum import Enum

 
class Kind(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    TOOL_OUTCOME = "tool_outcome"
    SCRATCHPAD = "scratchpad"


class MemoryItem(BaseModel):
    id: str
    kind: Kind
    keywords: list[str]
    descriptor: str            # one short human-readable line
    value: dict                # structured payload
    artifact_id: str | None    # handle into the artifact store
    source: str
    run_id: str
    goal_id: str | None
    confidence: float
    created_at: datetime
    expiry_date: datetime | None = None  # None means never expires


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