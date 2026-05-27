"""
Durable memory for the agent.

- Can add, read, filter, and clear memories.
- Can use kind to tag and set automatic expiry dates, for example, scratchpad memories expire after run completes.
- Memories can also be added with an expiry date.
- Memories can be filtered by kind, goal_id, and recency.
- Relevance of memory can be determined by query and history.
"""

import os
import uuid
import json
import ast
from datetime import datetime

import pandas as pd

from .schemas import MemoryItem, Kind
from client import LLM

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(PARENT_DIR, "memory.csv")

CSV_COLUMNS = [
    "id", "kind", "keywords", "descriptor", "value", "artifact_id",
    "source", "run_id", "goal_id", "confidence", "created_at", "expiry_date"
]

RELEVANCE_SYSTEM_PROMPT = """You are a memory relevance ranker. Given a query and a list of memories, 
return the IDs of the most relevant memories as a JSON list of strings, ordered by relevance (most relevant first).
Only return the JSON list, nothing else."""


class Memory:
    def __init__(self):
        self._ensure_csv_exists()
        self._cleanup_expired()

    def __len__(self):
        df = self._load_csv()
        return len(df)

    def __repr__(self):
        return f"Memory(items={len(self)})"

    def __str__(self):
        return self.__repr__()

    def _ensure_csv_exists(self):
        """Create the CSV file if it doesn't exist."""
        if not os.path.exists(CSV_FILE):
            df = pd.DataFrame(columns=CSV_COLUMNS)
            df.to_csv(CSV_FILE, index=False)

    def _load_csv(self) -> pd.DataFrame:
        """Load the CSV file into a DataFrame."""
        self._ensure_csv_exists()
        return pd.read_csv(CSV_FILE)

    def _save_csv(self, df: pd.DataFrame):
        """Save the DataFrame to the CSV file."""
        df.to_csv(CSV_FILE, index=False)

    def _cleanup_expired(self):
        """Remove all expired memories from the CSV file."""
        df = self._load_csv()
        if df.empty:
            return
        
        now = datetime.now()
        
        def is_not_expired(expiry_val):
            if pd.isna(expiry_val) or expiry_val == "" or expiry_val is None:
                return True  # None means never expires
            try:
                expiry_dt = datetime.fromisoformat(str(expiry_val))
                return expiry_dt > now
            except (ValueError, TypeError):
                return True  # Keep if we can't parse
        
        mask = df["expiry_date"].apply(is_not_expired)
        df_cleaned = df[mask]
        
        if len(df_cleaned) < len(df):
            self._save_csv(df_cleaned)

    def _df_to_memory_items(self, df: pd.DataFrame) -> list[MemoryItem]:
        """Convert DataFrame rows to MemoryItem objects, filtering out expired."""
        self._cleanup_expired()
        df = self._load_csv()
        
        if df.empty:
            return []
        
        items = []
        for _, row in df.iterrows():
            try:
                keywords = row.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = ast.literal_eval(keywords) if keywords else []
                
                value = row.get("value", {})
                if isinstance(value, str):
                    value = ast.literal_eval(value) if value else {}
                
                expiry_val = row.get("expiry_date")
                item = MemoryItem(
                    id=str(row["id"]),
                    kind=Kind(row["kind"]) if row["kind"] else Kind.FACT,
                    keywords=keywords,
                    descriptor=str(row.get("descriptor", "")),
                    value=value,
                    artifact_id=str(row["artifact_id"]) if pd.notna(row.get("artifact_id")) else None,
                    source=str(row["source"]),
                    run_id=str(row["run_id"]),
                    goal_id=str(row["goal_id"]) if pd.notna(row.get("goal_id")) else None,
                    confidence=float(row.get("confidence", 0.0)),
                    created_at=datetime.fromisoformat(str(row["created_at"])) if row.get("created_at") else datetime.now(),
                    expiry_date=datetime.fromisoformat(str(expiry_val)) if pd.notna(expiry_val) and expiry_val != "" else None,
                )
                items.append(item)
            except Exception:
                continue
        
        return items

    def _delete_by_ids(self, ids: list[str]):
        """Delete memories by their IDs from the CSV."""
        df = self._load_csv()
        df = df[~df["id"].astype(str).isin([str(i) for i in ids])]
        self._save_csv(df)

    def remember(
        self,
        descriptor: str,
        source: str,
        run_id: str,
        expiry_date: datetime | None = None,
        kind: Kind = Kind.FACT,
        keywords: list[str] = None,
        value: dict = None,
        goal_id: str | None = None,
        confidence: float = 0.0,
    ) -> str:
        """
        Add a memory to the agent's memory.
        
        Args:
            descriptor: Short human-readable description of the memory.
            source: The source of the memory.
            run_id: The run ID of the memory.
            expiry_date: The expiry date of the memory (None = never expires).
            kind: The kind of the memory - fact, preference, tool_outcome, scratchpad.
            keywords: Keywords for the memory.
            value: Structured payload dict.
            goal_id: Optional goal ID.
            confidence: Confidence score (0.0-1.0).
        
        Returns:
            The ID of the created memory.
        """
        memory_id = str(uuid.uuid4())
        memory_item = MemoryItem(
            id=memory_id,
            kind=kind,
            keywords=keywords or [],
            descriptor=descriptor,
            value=value or {},
            artifact_id=None,
            source=source,
            run_id=str(run_id),
            goal_id=goal_id,
            confidence=confidence,
            created_at=datetime.now(),
            expiry_date=expiry_date,
        )
        
        df = self._load_csv()
        row_dict = memory_item.model_dump()
        row_dict["kind"] = row_dict["kind"].value if isinstance(row_dict["kind"], Kind) else row_dict["kind"]
        row_dict["keywords"] = json.dumps(row_dict["keywords"])
        row_dict["value"] = json.dumps(row_dict["value"])
        row_dict["created_at"] = row_dict["created_at"].isoformat() if row_dict["created_at"] else None
        row_dict["expiry_date"] = row_dict["expiry_date"].isoformat() if row_dict["expiry_date"] else None
        
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
        self._save_csv(df)
        
        return memory_id

    def recollect(self, query: str, history: list[dict] = None, kinds: list[Kind] = None, top_k: int = 10) -> list[MemoryItem]:
        """
        Read memories that are relevant to the query and history.
        
        Args:
            query: The query to read.
            history: The history of the agent (used for context).
            kinds: The kinds of memories to read - fact, preference, tool_outcome, scratchpad.
            top_k: The number of memories to read.
        """
        return self.relevant(query, kinds, top_k)

    def filter(
        self,
        kinds: list[Kind] = None,
        goal_id: str = None,
        recency: int = None,
    ) -> list[MemoryItem]:
        """
        Filter memories by kinds, goal_id, and recency.
        
        Args:
            kinds: The kinds of memories to filter by - fact, preference, tool_outcome, scratchpad.
            goal_id: The goal ID to filter by.
            recency: The number of most recent memories to return.
        """
        df = self._load_csv()
        
        if kinds:
            kind_values = [k.value if isinstance(k, Kind) else k for k in kinds]
            df = df[df["kind"].isin(kind_values)]
        
        if goal_id:
            df = df[df["goal_id"] == goal_id]
        
        # Sort by created_at descending
        df = df.sort_values("created_at", ascending=False)
        
        if recency and recency > 0:
            df = df.head(recency)
        
        return self._df_to_memory_items(df)

    def relevant(self, query: str, kinds: list[Kind] = None, top_k: int = 10) -> list[MemoryItem]:
        """
        Get relevant memories for a query using LLM ranking.
        
        Args:
            query: The query to get relevant memories for.
            kinds: The kinds of memories to get - fact, preference, tool_outcome, scratchpad.
            top_k: The number of memories to get.
        """
        candidates = self.filter(kinds=kinds)
        
        if not candidates:
            return []
        
        if len(candidates) <= top_k:
            return candidates
        
        memories_summary = "\n".join([
            f"- ID: {m.id}, Descriptor: {m.descriptor}, Kind: {m.kind.value}"
            for m in candidates
        ])
        
        messages = [{
            "role": "user",
            "content": f"Query: {query}\n\nMemories:\n{memories_summary}\n\nReturn the top {top_k} most relevant memory IDs."
        }]
        
        llm = LLM()
        reply = llm.chat(
            messages=messages,
            system=RELEVANCE_SYSTEM_PROMPT,
            cache_system=True,
            reasoning="off",
            provider=None,
            temperature=0,
            max_tokens=1024,
        )
        
        try:
            relevant_ids = json.loads(reply.content if hasattr(reply, 'content') else str(reply))
            id_to_memory = {m.id: m for m in candidates}
            return [id_to_memory[mid] for mid in relevant_ids if mid in id_to_memory][:top_k]
        except (json.JSONDecodeError, AttributeError):
            return candidates[:top_k]

    def edit(self, memory_id: str, descriptor: str = None, value: dict = None, keywords: list[str] = None):
        """
        Edit a memory.
        
        Args:
            memory_id: The ID of the memory to edit.
            descriptor: New descriptor (if provided).
            value: New value dict (if provided).
            keywords: New keywords (if provided).
        """
        df = self._load_csv()
        mask = df["id"].astype(str) == str(memory_id)
        
        if not mask.any():
            return
        
        if descriptor is not None:
            df.loc[mask, "descriptor"] = descriptor
        if value is not None:
            df.loc[mask, "value"] = json.dumps(value)
        if keywords is not None:
            df.loc[mask, "keywords"] = json.dumps(keywords)
        
        self._save_csv(df)

    def delete(self, memory_id: str):
        """
        Delete a memory.
        
        Args:
            memory_id: The ID of the memory to delete.
        """
        self._delete_by_ids([memory_id])

    def reset(self):
        """
        Reset all memories by clearing the CSV file.
        """
        df = pd.DataFrame(columns=CSV_COLUMNS)
        self._save_csv(df)

    def get_all(self) -> list[MemoryItem]:
        """
        Get all non-expired memories.
        """
        df = self._load_csv()
        return self._df_to_memory_items(df)

    def read(self, query: str, history: list[dict] = None) -> list[MemoryItem]:
        """
        Alias for recollect - read memories relevant to the query and history.
        """
        return self.recollect(query, history)

    def record_outcome(
        self,
        tool_call,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str,
    ) -> str:
        """
        Record the outcome of a tool execution as a memory.
        
        Args:
            tool_call: The ToolCall that was executed.
            result_text: The result descriptor text.
            artifact_id: Optional artifact ID if result was stored.
            run_id: The current run ID.
            goal_id: The goal ID this action was for.
        
        Returns:
            The ID of the created memory.
        """
        descriptor = f"Tool '{tool_call.name}' executed: {result_text[:200]}"
        
        memory_id = str(uuid.uuid4())
        memory_item = MemoryItem(
            id=memory_id,
            kind=Kind.TOOL_OUTCOME,
            keywords=[tool_call.name],
            descriptor=descriptor,
            value={
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:500],
            },
            artifact_id=artifact_id,
            source="action",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.now(),
            expiry_date=None,
        )
        
        df = self._load_csv()
        row_dict = memory_item.model_dump()
        row_dict["kind"] = row_dict["kind"].value if isinstance(row_dict["kind"], Kind) else row_dict["kind"]
        row_dict["keywords"] = json.dumps(row_dict["keywords"])
        row_dict["value"] = json.dumps(row_dict["value"])
        row_dict["created_at"] = row_dict["created_at"].isoformat() if row_dict["created_at"] else None
        row_dict["expiry_date"] = row_dict["expiry_date"].isoformat() if row_dict["expiry_date"] else None
        
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
        self._save_csv(df)
        
        return memory_id

