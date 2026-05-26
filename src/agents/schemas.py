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


class ToolCall(BaseModel):
    name: str
    arguments: dict


class DecisionOutput(BaseModel):
    answer: str | None         # exactly one of these two is populated
    tool_call: ToolCall | None