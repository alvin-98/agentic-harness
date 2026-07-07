"""Embedding providers for llm_gateway.

Two concrete providers, both async, both returning the same dict shape:
    {"embedding": list[float], "model": str, "dim": int}

PROVIDER 1: NomicEmbedder — local, default. Runs nomic-ai/nomic-embed-text-v1.5
via HuggingFace transformers. 768-dim. No key. Prepends nomic's required task
prefix ("search_document: " or "search_query: "). Model loaded once at startup;
inference dispatched to a thread to avoid blocking the event loop.

PROVIDER 2: GeminiEmbedder — free non-local fallback. Hits
generativelanguage.googleapis.com gemini-embedding-001 with
outputDimensionality=768 so the vector space matches nomic's output dimension.
Task type is passed natively (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY).

Both providers produce 768-dim vectors so a project can fall over from
nomic to Gemini without invalidating its FAISS index. Changing the
provider pair, or the configured fallback model, is a one-way trip that
invalidates every index — see README.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Literal

import httpx
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


TaskType = Literal["retrieval_document", "retrieval_query"]
EMBED_DIM = 768  # both providers are pinned to this

# Hard input ceiling. gemini-embedding-001 caps text inputs at ~2048 tokens
# (≈8000 chars at 4 chars/token). The gateway rejects oversize inputs with
# 413 rather than silently truncating — callers must chunk before embedding.
MAX_INPUT_CHARS = 8000

# Exponential backoff schedule for embedder failures. Resets to step 0 on
# the first success. After step 2 the wait stays at 15s (sticky cap).
BACKOFF_STEPS = [5, 10, 15]  # seconds per step


class EmbedderError(Exception):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


class EmbedRateState:
    """Per-embedder rate state.

    Enforces:
      - RPM   sliding 60s window of records; refused when full
      - cooldown   minimum seconds between successful records
      - backoff   exponential wait on failure, sticky-capped at 15s

    Set rpm=0 to disable the RPM check (used for local providers where
    rate-limiting at the gateway buys nothing)."""

    def __init__(self, rpm: int, cooldown: float):
        self.rpm = rpm
        self.cooldown = cooldown
        self.calls_minute: deque[float] = deque()
        self.last_call = 0.0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""
        self.backoff_step = 0  # 0 = no current backoff

    def _gc(self) -> None:
        cutoff = time.time() - 60
        while self.calls_minute and self.calls_minute[0] < cutoff:
            self.calls_minute.popleft()

    def can_use(self) -> tuple[bool, str]:
        self._gc()
        now = time.time()
        if now < self.unavailable_until:
            return False, f"backoff: {self.unavailable_reason} ({self.unavailable_until - now:.0f}s left)"
        if self.cooldown > 0:
            wait = self.cooldown - (now - self.last_call)
            if wait > 0:
                return False, f"cooldown ({wait:.1f}s)"
        if self.rpm > 0 and len(self.calls_minute) >= self.rpm:
            return False, f"RPM limit ({self.rpm}/min)"
        return True, ""

    def record(self) -> None:
        """Call on success. Resets any active backoff."""
        now = time.time()
        self.calls_minute.append(now)
        self.last_call = now
        self.backoff_step = 0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""

    def mark_failure(self, reason: str) -> None:
        """Call on failure. Pushes the backoff window forward by one step."""
        idx = min(self.backoff_step, len(BACKOFF_STEPS) - 1)
        secs = BACKOFF_STEPS[idx]
        self.backoff_step += 1
        self.unavailable_until = time.time() + secs
        self.unavailable_reason = reason

    def snapshot(self) -> dict:
        self._gc()
        now = time.time()
        return {
            "rpm_used": len(self.calls_minute),
            "rpm_limit": self.rpm,
            "cooldown_s": self.cooldown,
            "cooldown_remaining": max(0.0, self.cooldown - (now - self.last_call)) if self.last_call else 0.0,
            "backoff_remaining": max(0.0, self.unavailable_until - now),
            "backoff_reason": self.unavailable_reason if now < self.unavailable_until else "",
            "backoff_step": self.backoff_step,
        }


class EmbeddingProvider:
    name: str = ""
    model: str = ""
    state: EmbedRateState

    async def embed(self, text: str, task_type: TaskType) -> dict:
        raise NotImplementedError


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class NomicEmbedder(EmbeddingProvider):
    name = "nomic"

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self.model = model_name
        # Local — no upstream rate limit to defend against.
        self.state = EmbedRateState(rpm=0, cooldown=0.0)
        # Load tokenizer and model once
        self._tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", model_max_length=8192
        )
        rope_parameters = {"rope_theta": 1000.0, "rope_type": "dynamic", "factor": 2.0}
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, rope_parameters=rope_parameters
        )
        self._model.eval()
        # Move to GPU if available
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)

    def _embed_sync(self, text: str, task_type: TaskType) -> dict:
        # nomic-embed-text requires a task prefix for retrieval quality.
        prefix = "search_query: " if task_type == "retrieval_query" else "search_document: "
        prefixed = prefix + text
        encoded = self._tokenizer(
            [prefixed], padding=True, truncation=True, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output = self._model(**encoded)
        embeddings = _mean_pooling(output, encoded["attention_mask"])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        vec = embeddings[0].cpu().tolist()
        return {"embedding": vec, "model": self.model_name, "dim": len(vec)}

    async def embed(self, text: str, task_type: TaskType) -> dict:
        return await asyncio.to_thread(self._embed_sync, text, task_type)


class GeminiEmbedder(EmbeddingProvider):
    name = "gemini"
    _TASK_MAP = {
        "retrieval_document": "RETRIEVAL_DOCUMENT",
        "retrieval_query": "RETRIEVAL_QUERY",
    }

    def __init__(self, api_key: str, model: str, output_dim: int = EMBED_DIM,
                 rpm: int = 5, cooldown: float = 5.0):
        self.api_key = api_key
        self.model = model
        self.output_dim = output_dim
        self.state = EmbedRateState(rpm=rpm, cooldown=cooldown)

    async def embed(self, text: str, task_type: TaskType) -> dict:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:embedContent?key={self.api_key}"
        )
        body = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": self._TASK_MAP[task_type],
            "outputDimensionality": self.output_dim,
        }
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(url, json=body)
        if r.status_code != 200:
            raise EmbedderError(f"gemini HTTP {r.status_code}: {r.text}", status=r.status_code)
        d = r.json()
        vec = ((d.get("embedding") or {}).get("values")) or []
        if not vec:
            raise EmbedderError(f"gemini returned no embedding: {str(d)}")
        return {"embedding": vec, "model": self.model, "dim": len(vec)}


def build_embedders() -> tuple[list[EmbeddingProvider], list[str]]:
    """Return (ordered list of available embedders, ordered list of names).

    Order is read from EMBED_ORDER env var (comma-separated names) and defaults
    to ['nomic', 'gemini']. The nomic embedder is always available (local).
    The Gemini embedder is included only if GEMINI_API_KEY is set.
    """
    nomic_model = os.getenv("EMBED_NOMIC_MODEL", "nomic-ai/nomic-embed-text-v1.5")

    fallback_provider = os.getenv("EMBED_FALLBACK_PROVIDER", "gemini").lower()
    fallback_model = os.getenv("EMBED_FALLBACK_MODEL", "gemini-embedding-001")

    registry: dict[str, EmbeddingProvider] = {
        "nomic": NomicEmbedder(nomic_model),
    }
    if fallback_provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if key:
            registry["gemini"] = GeminiEmbedder(key, fallback_model)

    default_order = ["nomic", fallback_provider]
    order_env = os.getenv("EMBED_ORDER", ",".join(default_order))
    order = [n.strip() for n in order_env.split(",") if n.strip()]
    embedders = [registry[n] for n in order if n in registry]
    return embedders, [e.name for e in embedders]


async def embed_with_failover(
    embedders: list[EmbeddingProvider],
    text: str,
    task_type: TaskType,
    explicit: str | None = None,
):
    """Run the failover ring with per-provider rate-state gating.

    Returns (name, result_dict, attempts, latency_ms).

    For each candidate:
      - call `state.can_use()` first; if rate-limited / cooled-down / in backoff,
        skip to the next candidate (and record the reason in `attempts`)
      - on a real call success: `state.record()`  (resets backoff)
      - on a real call failure: `state.mark_failure(reason)`  (bumps backoff)

    If `explicit` is set, only that provider is tried — failure becomes a 502 /
    rate-limit becomes a 429; the gateway does not silently fall back when the
    caller pinned a provider.
    """
    attempts: list[dict] = []
    candidates = embedders
    if explicit:
        candidates = [e for e in embedders if e.name == explicit]
        if not candidates:
            raise EmbedderError(f"unknown embedder '{explicit}'", status=400)

    last_err: Exception | None = None
    t0 = time.time()
    for e in candidates:
        ok, why = e.state.can_use()
        if not ok:
            attempts.append({"provider": e.name, "reason": why})
            if explicit:
                raise EmbedderError(f"{e.name} unavailable: {why}", status=429)
            continue
        try:
            out = await e.embed(text, task_type)
            e.state.record()
            latency = int((time.time() - t0) * 1000)
            return e.name, out, attempts, latency
        except Exception as exc:
            last_err = exc
            reason = str(exc)
            e.state.mark_failure(reason)
            attempts.append({"provider": e.name, "reason": reason})
            if explicit:
                raise
    raise EmbedderError(
        f"all embedders unavailable. attempts={attempts}. last_error={last_err}",
        status=503,
    )
