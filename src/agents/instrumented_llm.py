"""
Instrumented LLM wrapper for full-transparency observability.

The agent's modules (perception, decision, memory) all route LLM traffic through
the shared gateway client (``llm_gateway.client.LLM``). That client only persists
*metadata* about each call (token counts, char counts, provider/model). For the
Agent Run Viewer we want the *complete* input and output of every LLM call.

``InstrumentedLLM`` is a drop-in replacement for ``LLM``. It forwards ``.chat()``
and ``.embed()`` to the real client, but additionally records the full request
(system, prompt/messages, every parameter, response_format, tools) and the full
response (raw text, parsed JSON, tokens, provider/model, router decision, fallback
attempts, latency) to a per-run sidecar file::

    src/logs/runs/<run_id>.llm.jsonl

One JSON object per line, keyed by run_id + iteration + call_label so the viewer
can associate each LLM call with the exact step in the agent loop that produced it.

The sidecar is intentionally separate from the noisy ``<run_id>.jsonl`` trace
(which contains httpcore/httpx debug spam); this keeps full-I/O records clean and
easy to parse.
"""

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_gateway"))
from client import LLM

from .logging_config import RUNS_DIR, _log_context, get_logger

logger = get_logger(__name__)

_seq_lock = threading.Lock()
_seq_counters: dict[str, int] = {}


def _next_seq(run_id: str) -> int:
    with _seq_lock:
        n = _seq_counters.get(run_id, 0)
        _seq_counters[run_id] = n + 1
        return n


def _safe(value: Any) -> Any:
    """Make a value JSON-serializable, falling back to str()."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _failover_trace(reply: dict, err_obj: Optional[BaseException]) -> dict:
    """Extract the model-failover trace (worker + router attempts) from either a
    successful reply or a GatewayError. This is what the observability UI uses to
    reconstruct the breadth/depth flow, retries, and cooldown behaviour.

    On success the trace comes from the reply's ``attempted`` list and the router
    decision. On failure the gateway's structured 503 body is carried on the
    ``GatewayError`` (``.attempts`` / ``.router_decision``)."""
    worker_attempts: list = []
    router_decision = None
    if reply:
        worker_attempts = reply.get("attempted") or []
        router_decision = reply.get("router_decision")
    elif err_obj is not None:
        worker_attempts = getattr(err_obj, "attempts", None) or []
        router_decision = getattr(err_obj, "router_decision", None)

    router_attempts: list = []
    if isinstance(router_decision, dict):
        router_attempts = router_decision.get("router_attempts") or []

    return {
        "worker_attempts": _safe(worker_attempts),
        "router_attempts": _safe(router_attempts),
        "router_decision": _safe(router_decision),
    }


def _write_record(record: dict) -> None:
    """Append one full-I/O record to the per-run sidecar file."""
    run_id = record.get("run_id")
    if not run_id:
        return  # No run context — nothing to associate this call with.
    try:
        path = RUNS_DIR / f"{run_id}.llm.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:  # never let observability break the agent
        logger.warning("llm_record_write_failed", call_label=record.get("call_label"))


class InstrumentedLLM:
    """Drop-in wrapper around ``LLM`` that captures full request/response I/O."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 600):
        self._llm = LLM(base_url, timeout) if base_url else LLM(timeout=timeout)

    def chat(self, prompt: Optional[str] = None, *, call_label: str = "unlabeled", **kwargs) -> dict:
        ctx = _log_context.get()
        run_id = ctx.get("run_id")
        iteration = ctx.get("iteration")

        request = {"prompt": prompt}
        for key in (
            "messages", "system", "provider", "model", "max_tokens",
            "temperature", "tools", "tool_choice", "cache_system",
            "reasoning", "response_format", "auto_route",
        ):
            if key in kwargs:
                request[key] = _safe(kwargs[key])

        t0 = time.time()
        error = None
        err_obj: Optional[BaseException] = None
        reply: dict = {}
        try:
            reply = self._llm.chat(prompt=prompt, **kwargs)
            return reply
        except Exception as e:
            error = str(e)
            err_obj = e
            raise
        finally:
            latency_ms = int((time.time() - t0) * 1000)
            response = None
            if reply:
                response = {
                    "text": reply.get("text"),
                    "parsed": _safe(reply.get("parsed")),
                    "tool_calls": _safe(reply.get("tool_calls")),
                    "input_tokens": reply.get("input_tokens"),
                    "output_tokens": reply.get("output_tokens"),
                    "provider": reply.get("provider"),
                    "model": reply.get("model"),
                    "latency_ms": reply.get("latency_ms"),
                    "router_decision": _safe(reply.get("router_decision")),
                    "attempted": _safe(reply.get("attempted")),
                }
            _write_record({
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "iteration": iteration,
                "seq": _next_seq(run_id) if run_id else None,
                "call_label": call_label,
                "request": request,
                "response": response,
                "error": error,
                "latency_ms": latency_ms,
                "failover": _failover_trace(reply, err_obj),
            })

    def embed(self, *args, **kwargs) -> dict:
        return self._llm.embed(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Forward anything else (stream, capabilities, ...) to the real client.
        return getattr(self._llm, name)
