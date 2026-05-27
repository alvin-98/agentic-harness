"""
Durable memory for the agent.

- Can add, read, filter, and clear memories.
- Memories can also be added with an expiry date.
- Memories can be filtered by kind, goal_id, and recency.
- Relevance of memory can be determined by query and history.
"""

import os
import uuid
from datetime import datetime
from .schemas import MemoryItem, Kind
from client import LLM

class Memory:
    def __init__(self):
        self.items: list[MemoryItem] = []

    def __len__(self):
        return len(self.items)


    def __repr__(self):
        return f"Memory(items={len(self.items)})"

    def __str__(self):
        return self.__repr__()

        
    def remember(self, query: str, source: str, run_id: uuid.UUID, expiry_date: datetime | None = datetime.max, kinds: Kind | list[Kind] = "fact"):
        """
        Add a memory to the agent's memory.
        
        Args:
            query: The query to remember.
            source: The source of the memory.
            run_id: The run ID of the memory.
            expiry_date: The expiry date of the memory.
            kinds: The kind of the memory - fact, preference, tool_outcome, scratchpad
        """
        # implementation notes: Currently implemented as csv file
        # create a csv if its doesn't exist
        # append MemoryItem to the csv
        PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        CSV_FILE = os.path.join(PARENT_DIR, "memory.csv")
        memory_item = MemoryItem(
            id=uuid.uuid4(),
            kind=kinds,
            keywords=[],
            descriptor="",
            value={},
            artifact_id=None,
            source=source,
            run_id=run_id,
            goal_id=None,
            confidence=0.0,
            created_at=datetime.now(),
        )
        if not os.path.exists(CSV_FILE):
            import pandas as pd
            df = pd.DataFrame(columns=["id", "kind", "keywords", "descriptor", "value", "artifact_id", "source", "run_id", "goal_id", "confidence", "created_at"])
            df.to_csv(CSV_FILE, index=False)
        df = pd.read_csv(CSV_FILE)
        df = df.append(memory_item.model_dump(), ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        
    def read(self, query: str, history: list[dict], kinds: list[Kind] = ["fact"], top_k: int = 10) -> list[MemoryItem]:
        """
        Read memories that are relevant to the query and history.
        
        Args:
            query: The query to read.
            history: The history of the agent.
            kinds: The kinds of memories to read - fact, preference, tool_outcome, scratchpad.
            top_k: The number of memories to read.
        """
        # use the relevance function to rank memories by relevance to the query and history
        # return the top_k memories
        return self.relevant(query, kinds, top_k)

    def filter(self, query: str, kinds: list[Kind] = None, goal_id: str = None, recency: int = 10) -> list[MemoryItem]:
        """
        Filter memories by query, kinds, goal_id, and recency.
        
        Args:
            query: The query to filter by.
            kinds: The kinds of memories to filter by - fact, preference, tool_outcome, scratchpad.
            goal_id: The goal ID to filter by.
            recency: The recency of memories to filter by.
        """
        df = pd.read_csv(CSV_FILE)
        df.filter() # TODO: implement filtering
        return df.to_dict(orient="records")

    
    def relevant(self, query: str, kinds: list[Kind] = None, top_k: int = 10) -> list[MemoryItem]:
        """
        Get relevant memories for a query.
        
        Args:
            query: The query to get relevant memories for.
            kinds: The kinds of memories to get - fact, preference, tool_outcome, scratchpad.
            top_k: The number of memories to get.
        """
        # the relevance function ranks memories by relevance to the query
        # return the top_k memories
        # calls LLM to do the ranking, calls a ranking model ideally
        llm = LLM()
        reply = llm.chat(
            messages=messages,
            system=SYSTEM_PROMPT,
            cache_system=True,           # mark the system prompt cacheable
            reasoning="off",             # executor stays cheap
            provider=None,           # TODO: Check for ranking model and add
            temperature=0,
            max_tokens=1024,
        )
        print(reply) # need to check if reply returns a list of memories
        return 0
        
    
    def clear(self):
        """
        Clear all memories.
        """
        self.items = []

    
