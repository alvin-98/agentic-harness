"""
Durable memory for the agent — Session 7.

Persistence:
  - state/memory.json   — source of truth for all MemoryItems
  - state/index.faiss   — FAISS IndexFlatIP (768-dim, cosine via L2-norm)
  - state/index_ids.json — ordered list of memory IDs parallel to FAISS rows

Read path:
  1. Embed query via gateway (task_type="retrieval_query")
  2. FAISS.search → top-k by cosine similarity
  3. If FAISS is empty or gateway unreachable → keyword overlap fallback

Write paths:
  - remember()       — LLM classifier + embed descriptor → persist
  - record_outcome() — embed descriptor → persist
  - add_fact()       — direct insertion (no classifier), embed → persist
  Scratchpad items skip embedding (embedding=None).

Cross-process consistency: The FAISS index is reloaded from disk on every
read call. The MCP subprocess writes chunks via index_document which updates
the on-disk files. The agent process sees those writes on its next read.
"""

import re
import sys
import uuid
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_gateway"))

from .schemas import MemoryItem, Kind
from .instrumented_llm import InstrumentedLLM
from .logging_config import get_logger
from . import config

logger = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
STATE_DIR = Path(__file__).resolve().parent / "state"
MEMORY_JSON = STATE_DIR / "memory.json"
FAISS_INDEX_PATH = STATE_DIR / "index.faiss"
FAISS_IDS_PATH = STATE_DIR / "index_ids.json"

# Per-artifact chunk sidecars live here, one JSON file per artifact:
#   [{"index": int, "text": str, "embedding": [float, ...]}, ...]
# These are kept OUT of memory.json/FAISS so they never surface in the general
# memory.read() pool — they are only queried on demand for a specific artifact.
ARTIFACT_CHUNKS_DIR = STATE_DIR / "artifact_chunks"

# Chunking / retrieval defaults for artifact content.
ARTIFACT_CHUNK_WORDS = 300      # words per chunk
ARTIFACT_CHUNK_OVERLAP = 60     # overlapping words between adjacent chunks
ARTIFACT_RETRIEVE_TOP_K = 6     # relevant chunks returned per goal

# Upper bounds on how much raw tool text flows into LLM calls / memory storage.
SUMMARIZE_SNIPPET_CHARS = 4000  # head of a result fed to the descriptor summarizer
OUTCOME_PREVIEW_CHARS = 2000    # preview stored in a tool_outcome memory value

EXTRACTION_SYSTEM_PROMPT = config.MEMORY_EXTRACTION_SYSTEM_PROMPT

EMBED_DIM = 768  # nomic-embed-text-v1.5 output dimension

# ── Keyword fallback helpers ─────────────────────────────────────────────────
# Stopwords and tokenization for the keyword-overlap fallback path. The vector
# path is primary; this only runs when FAISS is empty or the gateway is
# unreachable. Folding recent history into the query tokens lets the fallback
# match on what the agent just did, not just the original user query.

_STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "or", "in", "on", "for", "at",
    "with", "by", "from", "what", "how", "when", "where", "why", "this", "that",
    "it", "be", "as", "are", "was", "were", "i", "you", "me", "my", "your",
}


def _tokens(text: str) -> set[str]:
    """Tokenize text: lowercase word tokens, stopwords removed, len > 2."""
    return {
        w for w in re.findall(r"\w+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def _new_id(prefix: str = "mem") -> str:
    return f"{prefix}:{uuid.uuid4().hex[:12]}"


class Memory:
    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_json_exists()
        self._cleanup_expired()
        self.llm = InstrumentedLLM()

    # ── JSON persistence ─────────────────────────────────────────────────────

    def _ensure_json_exists(self):
        if not MEMORY_JSON.exists():
            MEMORY_JSON.write_text("[]", encoding="utf-8")

    def _load_items(self) -> list[dict]:
        """Load raw dicts from memory.json."""
        self._ensure_json_exists()
        try:
            data = json.loads(MEMORY_JSON.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_items(self, items: list[dict]):
        """Write raw dicts to memory.json."""
        MEMORY_JSON.write_text(
            json.dumps(items, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    def _dict_to_item(self, d: dict) -> MemoryItem | None:
        """Convert a raw dict to a MemoryItem, returning None on failure."""
        try:
            return MemoryItem(
                id=d["id"],
                kind=Kind(d["kind"]),
                keywords=d.get("keywords", []),
                descriptor=d.get("descriptor", ""),
                value=d.get("value", {}),
                artifact_id=d.get("artifact_id"),
                embedding=d.get("embedding"),
                source=d.get("source", ""),
                run_id=d.get("run_id", ""),
                goal_id=d.get("goal_id"),
                confidence=float(d.get("confidence", 1.0)),
                created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(),
                expiry_date=datetime.fromisoformat(d["expiry_date"]) if d.get("expiry_date") else None,
            )
        except Exception:
            return None

    def _item_to_dict(self, item: MemoryItem) -> dict:
        """Serialize a MemoryItem to a JSON-safe dict."""
        return {
            "id": item.id,
            "kind": item.kind.value if isinstance(item.kind, Kind) else item.kind,
            "keywords": item.keywords,
            "descriptor": item.descriptor,
            "value": item.value,
            "artifact_id": item.artifact_id,
            "embedding": item.embedding,
            "source": item.source,
            "run_id": item.run_id,
            "goal_id": item.goal_id,
            "confidence": item.confidence,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "expiry_date": item.expiry_date.isoformat() if item.expiry_date else None,
        }

    # ── FAISS index ──────────────────────────────────────────────────────────

    def _load_faiss_index(self):
        """Reload FAISS index from disk. Returns (index, id_list) or (None, [])."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss_not_installed")
            return None, []

        if not FAISS_INDEX_PATH.exists() or not FAISS_IDS_PATH.exists():
            return None, []

        try:
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            ids = json.loads(FAISS_IDS_PATH.read_text(encoding="utf-8"))
            return index, ids
        except Exception as e:
            logger.warning("faiss_load_failed", error=str(e))
            return None, []

    def _save_faiss_index(self, index, ids: list[str]):
        """Write FAISS index and id list to disk."""
        import faiss
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        FAISS_IDS_PATH.write_text(json.dumps(ids), encoding="utf-8")

    def _append_to_faiss(self, memory_id: str, embedding: list[float]):
        """Append a single vector to the on-disk FAISS index."""
        import faiss

        index, ids = self._load_faiss_index()
        if index is None:
            index = faiss.IndexFlatIP(EMBED_DIM)
            ids = []

        vec = np.array([embedding], dtype=np.float32)
        # L2-normalize so inner product = cosine similarity
        faiss.normalize_L2(vec)
        index.add(vec)
        ids.append(memory_id)
        self._save_faiss_index(index, ids)

    def _rebuild_faiss_from_items(self, items: list[dict]):
        """Rebuild the entire FAISS index from memory items that have embeddings."""
        import faiss

        ids = []
        vectors = []
        for item in items:
            emb = item.get("embedding")
            if emb and len(emb) == EMBED_DIM:
                ids.append(item["id"])
                vectors.append(emb)

        index = faiss.IndexFlatIP(EMBED_DIM)
        if vectors:
            vecs = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vecs)
            index.add(vecs)

        self._save_faiss_index(index, ids)

    # ── Embedding ────────────────────────────────────────────────────────────

    def _try_embed(self, text: str, task_type: str = "retrieval_document") -> list[float] | None:
        """Embed text via the gateway. Returns None on failure (graceful degradation)."""
        try:
            result = self.llm.embed(text, task_type=task_type)
            emb = result.get("embedding")
            if emb and len(emb) == EMBED_DIM:
                return emb
            return None
        except Exception as e:
            logger.warning("embed_failed", error=str(e), task_type=task_type)
            return None

    # ── Core persist helper ──────────────────────────────────────────────────

    def _persist_item(self, item: MemoryItem) -> MemoryItem:
        """Append item to memory.json and update FAISS index. Synchronous."""
        items = self._load_items()
        items.append(self._item_to_dict(item))
        self._save_items(items)

        if item.embedding:
            self._append_to_faiss(item.id, item.embedding)

        return item

    # ── Expiry ───────────────────────────────────────────────────────────────

    def _cleanup_expired(self):
        """Remove all expired memories and rebuild FAISS if any were removed."""
        items = self._load_items()
        if not items:
            return

        now = datetime.now()
        cleaned = []
        removed = 0
        for d in items:
            expiry = d.get("expiry_date")
            if expiry:
                try:
                    if datetime.fromisoformat(expiry) <= now:
                        removed += 1
                        continue
                except (ValueError, TypeError):
                    pass
            cleaned.append(d)

        if removed:
            self._save_items(cleaned)
            self._rebuild_faiss_from_items(cleaned)

    # ── Keyword overlap fallback ─────────────────────────────────────────────

    def _keyword_search(
        self,
        query: str,
        items: list[dict],
        top_k: int,
        history: list[dict] | None = None,
    ) -> list[MemoryItem]:
        """Score items by keyword overlap with query tokens. Fallback retrieval.

        Uses stopword-aware tokenization and folds the last 3 history events
        into the query tokens so the fallback can match on recent tool calls
        and answers, not just the original user query."""
        query_tokens = _tokens(query)
        if history:
            for h in history[-3:]:
                query_tokens |= _tokens(json.dumps(h, default=str))

        scored = []
        for d in items:
            keywords = {w.lower() for w in d.get("keywords", [])}
            desc_tokens = _tokens(d.get("descriptor", ""))
            overlap = len(query_tokens & (keywords | desc_tokens))
            if overlap > 0:
                scored.append((overlap, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, d in scored[:top_k]:
            item = self._dict_to_item(d)
            if item:
                results.append(item)
        return results

    # ── Public API ───────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._load_items())

    def __repr__(self):
        return f"Memory(items={len(self)})"

    def __str__(self):
        return self.__repr__()

    def remember(
        self,
        descriptor: str,
        source: str,
        run_id: str,
        query: Optional[str] = None,
        expiry_date: datetime | None = None,
        kind: Kind = Kind.FACT,
        keywords: list[str] = None,
        value: dict = None,
        goal_id: str | None = None,
        confidence: float = 0.0,
    ) -> str:
        """
        Add a memory to the agent's memory.
        If you receive a user_query, you should parse it using an LLM 
        and extract facts and preferences, and other relevant information 
        to populate the descriptor, keywords, and value.
        
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
        memory_id = _new_id("mem")
        if query and source == "user_query":
            messages = [{"role": "user", "content": f"Extract memory attributes from this user input:\n\n{query}"}]
            _c = config.MEMORY_EXTRACTION_LLM
            reply = self.llm.chat(
                call_label="memory.extract_attributes",
                messages=messages,
                system=EXTRACTION_SYSTEM_PROMPT,
                cache_system=_c.cache_system,
                reasoning=_c.reasoning,
                provider=_c.provider,
                model=_c.model,
                temperature=_c.temperature,
                max_tokens=_c.max_tokens,
                auto_route=_c.auto_route,
                response_format={
                    "type": "json_schema",
                    "schema": MemoryItem.model_json_schema(),
                    "name": "MemoryItem",
                    "strict": True,
                },
            )
            try:
                extracted = json.loads(reply["text"])
                if not extracted.get("should_store", True):
                    logger.debug("memory_skipped", source="user_query", reason="should_store_false")
                    return None
                kind = Kind(extracted.get("kind", "fact"))
                keywords = extracted.get("keywords", keywords or [])
                descriptor = extracted.get("descriptor", descriptor)
                value = extracted.get("value", value or {})
                confidence = float(extracted.get("confidence", confidence))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # Use the original provided values if extraction fails

        # Embed for non-scratchpad items
        embedding = None
        if kind != Kind.SCRATCHPAD:
            embedding = self._try_embed(descriptor, task_type="retrieval_document")

        memory_item = MemoryItem(
            id=memory_id,
            kind=kind,
            keywords=[k.lower() for k in (keywords or [])],
            descriptor=descriptor,
            value=value or {},
            artifact_id=None,
            embedding=embedding,
            source=source,
            run_id=str(run_id),
            goal_id=goal_id,
            confidence=confidence,
            created_at=datetime.now(),
            expiry_date=expiry_date,
        )

        self._persist_item(memory_item)

        logger.info("memory_created",
                   memory_id=memory_id,
                   kind=kind.value if isinstance(kind, Kind) else kind,
                   source=source,
                   has_embedding=embedding is not None,
                   descriptor=descriptor if descriptor else None)
        return memory_id

    def add_fact(
        self,
        descriptor: str,
        *,
        value: dict,
        keywords: list[str],
        source: str,
        run_id: str,
        goal_id: str | None = None,
    ) -> MemoryItem:
        """Write a fact chunk directly — no LLM classifier. Used by index_document."""
        embedding = self._try_embed(descriptor, task_type="retrieval_document")
        item = MemoryItem(
            id=_new_id("mem"),
            kind=Kind.FACT,
            keywords=[k.lower() for k in keywords],
            descriptor=descriptor,
            value=value,
            embedding=embedding,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.now(),
            expiry_date=None,
        )
        return self._persist_item(item)

    def recollect(self, query: str, history: list[dict] = None, kinds: list[Kind] = None, top_k: int = 10) -> list[MemoryItem]:
        """
        Read memories that are relevant to the query and history.
        Vector-first retrieval with keyword overlap fallback.
        
        Args:
            query: The query to read.
            history: The history of the agent (used for context).
            kinds: The kinds of memories to read - fact, preference, tool_outcome, scratchpad.
            top_k: The number of memories to read.
        """
        items = self._load_items()
        if not items:
            return []

        # Filter by kinds if specified
        if kinds:
            kind_values = {k.value if isinstance(k, Kind) else k for k in kinds}
            items = [d for d in items if d.get("kind") in kind_values]

        # Vector path: embed query, search FAISS
        query_embedding = self._try_embed(query, task_type="retrieval_query")
        if query_embedding:
            index, ids = self._load_faiss_index()
            if index is not None and index.ntotal > 0:
                vec = np.array([query_embedding], dtype=np.float32)
                import faiss
                faiss.normalize_L2(vec)
                k_search = min(top_k * 2, index.ntotal)
                scores, indices = index.search(vec, k_search)

                # Map FAISS results back to items, respecting kind filter
                id_to_item = {d["id"]: d for d in items}
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(ids):
                        continue
                    mid = ids[idx]
                    if mid in id_to_item:
                        mem = self._dict_to_item(id_to_item[mid])
                        if mem:
                            results.append(mem)
                    if len(results) >= top_k:
                        break

                if results:
                    logger.debug("memory_vector_search",
                                query=query,
                                hits=len(results))
                    return results

        # Fallback: keyword overlap
        logger.debug("memory_keyword_fallback", query=query)
        return self._keyword_search(query, items, top_k, history=history)

    def read(self, query: str, history: list[dict] = None, top_k: int = 10) -> list[MemoryItem]:
        """
        Primary read interface. Vector-first with keyword fallback.
        """
        return self.recollect(query, history, top_k=top_k)

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
        items = self._load_items()

        if kinds:
            kind_values = {k.value if isinstance(k, Kind) else k for k in kinds}
            items = [d for d in items if d.get("kind") in kind_values]

        if goal_id:
            items = [d for d in items if d.get("goal_id") == goal_id]

        # Sort by created_at descending
        items.sort(key=lambda d: d.get("created_at", ""), reverse=True)

        if recency and recency > 0:
            items = items[:recency]

        results = []
        for d in items:
            item = self._dict_to_item(d)
            if item:
                results.append(item)
        return results

    def relevant(self, query: str, kinds: list[Kind] = None, top_k: int = 10, history: list[dict] = None) -> list[MemoryItem]:
        """
        Get relevant memories — delegates to vector search with fallback.
        Replaces the old LLM-ranking approach.
        """
        return self.recollect(query, history=history, kinds=kinds, top_k=top_k)

    def edit(self, memory_id: str, descriptor: str = None, value: dict = None, keywords: list[str] = None):
        """
        Edit a memory.
        
        Args:
            memory_id: The ID of the memory to edit.
            descriptor: New descriptor (if provided).
            value: New value dict (if provided).
            keywords: New keywords (if provided).
        """
        items = self._load_items()
        changed = False
        for d in items:
            if d["id"] == memory_id:
                if descriptor is not None:
                    d["descriptor"] = descriptor
                    # Re-embed on descriptor change
                    if d.get("kind") != Kind.SCRATCHPAD.value:
                        d["embedding"] = self._try_embed(descriptor, task_type="retrieval_document")
                if value is not None:
                    d["value"] = value
                if keywords is not None:
                    d["keywords"] = [k.lower() for k in keywords]
                changed = True
                break

        if changed:
            self._save_items(items)
            self._rebuild_faiss_from_items(items)

    def delete(self, memory_id: str):
        """
        Delete a memory.
        
        Args:
            memory_id: The ID of the memory to delete.
        """
        items = self._load_items()
        items = [d for d in items if d["id"] != memory_id]
        self._save_items(items)
        self._rebuild_faiss_from_items(items)

    def reset(self):
        """
        Reset all memories — deletes memory.json, index.faiss, index_ids.json.
        """
        self._save_items([])
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        if FAISS_IDS_PATH.exists():
            FAISS_IDS_PATH.unlink()

    def get_all(self) -> list[MemoryItem]:
        """
        Get all non-expired memories.
        """
        items = self._load_items()
        results = []
        for d in items:
            item = self._dict_to_item(d)
            if item:
                results.append(item)
        return results

    def expire_run(self, run_id: str) -> int:
        """
        Mark all TOOL_OUTCOME and SCRATCHPAD memories for a given run as expired.
        They will be removed on the next _cleanup_expired() call.

        Returns the number of entries expired.
        """
        items = self._load_items()
        if not items:
            return 0

        run_scoped_kinds = {Kind.TOOL_OUTCOME.value, Kind.SCRATCHPAD.value}
        count = 0
        for d in items:
            if str(d.get("run_id")) == str(run_id) and d.get("kind") in run_scoped_kinds:
                d["expiry_date"] = datetime.now().isoformat()
                count += 1

        if count:
            self._save_items(items)
            logger.info("run_memories_expired", run_id=run_id, count=count)

        return count

    def _index_search_results(
        self,
        result_text: str,
        query_args: dict,
        run_id: str,
        goal_id: str,
    ) -> None:
        """
        Parse web_search JSON results and store each result as its own
        memory item so downstream goals can reference concrete URLs
        without the LLM having to dig through a JSON blob.
        """
        try:
            results = json.loads(result_text)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(results, list):
            return

        search_query = query_args.get("query", "")
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            if not url:
                continue

            # Scratchpad — no embedding
            mem = MemoryItem(
                id=_new_id("mem"),
                kind=Kind.SCRATCHPAD,
                keywords=["search_result", "url", search_query.lower()],
                descriptor=f"Search result #{idx} for '{search_query}': {title} — {url}",
                value={"index": idx, "url": url, "title": title, "snippet": snippet},
                artifact_id=None,
                embedding=None,
                source="search_index",
                run_id=run_id,
                goal_id=goal_id,
                confidence=1.0,
                created_at=datetime.now(),
                expiry_date=None,
            )
            self._persist_item(mem)

        logger.info("search_results_indexed",
                   query=search_query,
                   goal_id=goal_id)

    # ── Artifact chunk index (per-artifact sidecar) ──────────────────────────

    @staticmethod
    def _chunk_artifact_id(artifact_id: str) -> str:
        """Filesystem-safe form of an artifact handle (mirrors ArtifactStore)."""
        return artifact_id.replace(":", "_").replace("/", "_")

    def _chunk_sidecar_path(self, artifact_id: str) -> Path:
        return ARTIFACT_CHUNKS_DIR / f"{self._chunk_artifact_id(artifact_id)}.json"

    @staticmethod
    def _extract_artifact_text(raw: str) -> str:
        """Pull the human-readable body out of a stored artifact.

        fetch_url artifacts are JSON dicts with a "text" field; read_file uses
        "content". Anything else is treated as plain text.
        """
        raw = raw.strip()
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    for key in ("text", "content", "markdown"):
                        if isinstance(obj.get(key), str) and obj[key].strip():
                            return obj[key]
            except (json.JSONDecodeError, ValueError):
                pass
        return raw

    @staticmethod
    def _chunk_words(
        text: str,
        chunk_size: int = ARTIFACT_CHUNK_WORDS,
        overlap: int = ARTIFACT_CHUNK_OVERLAP,
    ) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        step = max(1, chunk_size - overlap)
        while start < len(words):
            chunks.append(" ".join(words[start:start + chunk_size]))
            start += step
        return chunks

    def index_artifact(self, artifact_id: str, raw_content: str) -> int:
        """Chunk an artifact's content, embed each chunk, and persist to a
        per-artifact sidecar for later on-demand retrieval.

        Chunks live outside memory.json/FAISS so they never pollute the general
        memory.read() pool. Returns the number of chunks indexed (0 if the body
        is small enough to attach whole, or if embedding is unavailable).
        """
        text = self._extract_artifact_text(raw_content)
        chunks = self._chunk_words(text)
        # Small bodies aren't worth chunking — the caller can attach them whole.
        if len(chunks) <= 1:
            return 0

        records = []
        for i, chunk in enumerate(chunks):
            emb = self._try_embed(chunk, task_type="retrieval_document")
            if emb is None:
                # Without embeddings we can't rank chunks; abort and let the
                # caller fall back to a bounded slice of the full artifact.
                logger.warning("artifact_index_embed_failed", artifact_id=artifact_id, chunk=i)
                return 0
            records.append({"index": i, "text": chunk, "embedding": emb})

        ARTIFACT_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        path = self._chunk_sidecar_path(artifact_id)
        path.write_text(json.dumps(records), encoding="utf-8")

        logger.info("artifact_indexed",
                   artifact_id=artifact_id,
                   chunks=len(records),
                   chars=len(text))
        return len(records)

    def retrieve_artifact_chunks(
        self,
        artifact_id: str,
        query: str,
        top_k: int = ARTIFACT_RETRIEVE_TOP_K,
    ) -> str | None:
        """Return the top-k chunks of an artifact most relevant to `query`,
        joined in original document order. Returns None when no sidecar exists
        (caller should fall back to attaching a bounded slice of the artifact).
        """
        path = self._chunk_sidecar_path(artifact_id)
        if not path.exists():
            return None
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not records:
            return None

        q_emb = self._try_embed(query, task_type="retrieval_query")
        if q_emb is None:
            # No query embedding — return the leading chunks in order.
            selected = sorted(records, key=lambda r: r["index"])[:top_k]
        else:
            q = np.array(q_emb, dtype=np.float32)
            q /= (np.linalg.norm(q) + 1e-8)
            scored = []
            for r in records:
                v = np.array(r["embedding"], dtype=np.float32)
                v /= (np.linalg.norm(v) + 1e-8)
                scored.append((float(np.dot(q, v)), r))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [r for _, r in scored[:top_k]]
            # Re-order the winning chunks by their position in the document so
            # the attached text reads coherently.
            selected.sort(key=lambda r: r["index"])

        logger.info("artifact_chunks_retrieved",
                   artifact_id=artifact_id,
                   returned=len(selected),
                   total=len(records))
        return "\n\n[...]\n\n".join(r["text"] for r in selected)

    def _summarize_outcome(self, tool_name: str, arguments: dict, result_text: str) -> str:
        """Generate a one-line semantic descriptor for a tool outcome via LLM."""
        # Cap the snippet: the summarizer only needs the head of a large result to
        # write a one-line descriptor. Sending the full artifact (100k+ tokens)
        # here is what previously blew past the model's context window.
        snippet = result_text[:SUMMARIZE_SNIPPET_CHARS]
        prompt = (
            f"Tool: {tool_name}\nArguments: {arguments}\n"
            f"Result snippet:\n{snippet}\n\n"
            f"{config.MEMORY_SUMMARIZE_USER_PROMPT}"
        )
        _c = config.MEMORY_SUMMARIZE_LLM
        try:
            reply = self.llm.chat(
                call_label="memory.summarize_outcome",
                prompt=prompt,
                system=config.MEMORY_SUMMARIZE_SYSTEM_PROMPT,
                cache_system=_c.cache_system,
                reasoning=_c.reasoning,
                provider=_c.provider,
                model=_c.model,
                temperature=_c.temperature,
                max_tokens=_c.max_tokens,
                auto_route=_c.auto_route,
            )
            summary = reply["text"].strip()
            if summary:
                return summary
        except Exception as e:
            logger.warning("summarize_outcome_failed", error=str(e))
        # Fallback: use tool name + arguments
        return f"Tool '{tool_name}' called with {arguments}"

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
        descriptor = self._summarize_outcome(tool_call.name, tool_call.arguments, result_text)
        embedding = self._try_embed(descriptor, task_type="retrieval_document")

        memory_id = _new_id("mem")
        memory_item = MemoryItem(
            id=memory_id,
            kind=Kind.TOOL_OUTCOME,
            keywords=[tool_call.name],
            descriptor=descriptor,
            value={
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                # Store only a bounded preview — the full bytes live in the
                # artifact store (recoverable via artifact_id) and, for large
                # content, in the per-artifact chunk sidecar. Persisting the
                # whole result here bloated memory.json and leaked back into
                # prompts on subsequent reads.
                "result_preview": result_text[:OUTCOME_PREVIEW_CHARS],
                "artifact_id": artifact_id,
            },
            artifact_id=artifact_id,
            embedding=embedding,
            source="action",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.now(),
            expiry_date=None,
        )

        self._persist_item(memory_item)

        logger.info("tool_outcome_recorded",
                   memory_id=memory_id,
                   tool_name=tool_call.name,
                   goal_id=goal_id,
                   has_artifact=artifact_id is not None,
                   has_embedding=embedding is not None)

        if tool_call.name == "web_search":
            self._index_search_results(
                result_text=result_text,
                query_args=tool_call.arguments,
                run_id=run_id,
                goal_id=goal_id,
            )

        return memory_id

