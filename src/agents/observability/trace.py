"""
Parse per-run agent logs into a structured trace for the Agent Run Viewer.

Two files per run live in ``src/logs/runs/``:
  - ``<run_id>.jsonl``      structured event trace (noisy: also contains httpcore/httpx)
  - ``<run_id>.llm.jsonl``  full LLM request/response I/O (written by InstrumentedLLM)

The viewer consumes:
  - ``list_runs()``       -> lightweight summary of every run on disk
  - ``load_trace(id)``    -> full structured trace (iterations, events, LLM calls)
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from ..logging_config import RUNS_DIR

# Only records from these loggers represent agent reasoning. Everything else
# (httpcore, httpx, faiss.loader, ...) is transport/library noise.
AGENT_LOGGER_PREFIXES = ("src.agents", "agents", "__main__")


def _is_noise(record: dict) -> bool:
    logger = record.get("logger", "")
    return not any(logger.startswith(p) for p in AGENT_LOGGER_PREFIXES)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _event_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.jsonl"


def _llm_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.llm.jsonl"


def list_runs() -> list[dict]:
    """Return a summary for every run trace on disk, newest first."""
    runs = []
    for path in RUNS_DIR.glob("*.jsonl"):
        if path.name.endswith(".llm.jsonl"):
            continue
        run_id = path.name[: -len(".jsonl")]
        runs.append(_summarize_run(run_id, path))
    runs.sort(key=lambda r: r.get("mtime", 0), reverse=True)
    return runs


def _summarize_run(run_id: str, path: Path) -> dict:
    records = _read_jsonl(path)
    agent_records = [r for r in records if not _is_noise(r)]
    query = ""
    start_ts = None
    end_ts = None
    status = "unknown"
    counts = {"iterations": 0, "actions": 0, "answers": 0}

    for r in agent_records:
        ev = r.get("event", "").split(" ")[0]
        if ev == "run_start":
            query = r.get("query", "")
            start_ts = r.get("timestamp")
        elif ev == "iteration_start":
            counts["iterations"] = max(counts["iterations"], r.get("iteration", 0))
        elif ev == "action_complete":
            counts["actions"] += 1
        elif ev == "answer_produced":
            counts["answers"] += 1
        elif ev == "run_complete":
            end_ts = r.get("timestamp")
            status = "complete"

    if status != "complete":
        # Recently-touched files with no run_complete are likely still running.
        age = _file_age_seconds(path)
        status = "running" if age is not None and age < 120 else "incomplete"

    return {
        "run_id": run_id,
        "query": query,
        "status": status,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "counts": counts,
        "has_llm_io": _llm_path(run_id).exists(),
        "mtime": path.stat().st_mtime,
    }


def _file_age_seconds(path: Path) -> Optional[float]:
    try:
        import time
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def load_trace(run_id: str) -> Optional[dict]:
    """Parse a run into a structured trace, or None if it doesn't exist."""
    event_path = _event_path(run_id)
    if not event_path.exists():
        return None

    records = [r for r in _read_jsonl(event_path) if not _is_noise(r)]
    llm_records = _read_jsonl(_llm_path(run_id))
    llm_by_iter: dict[int, list[dict]] = {}
    for rec in llm_records:
        llm_by_iter.setdefault(rec.get("iteration") or 0, []).append(rec)
    for calls in llm_by_iter.values():
        calls.sort(key=lambda c: (c.get("seq") is None, c.get("seq") or 0))

    summary = _summarize_run(run_id, event_path)

    setup_events: list[dict] = []
    iterations: list[dict] = []
    current: Optional[dict] = None

    def _bucket_events() -> list[dict]:
        return current["events"] if current is not None else setup_events

    for r in records:
        ev = r.get("event", "").split(" ")[0]
        if ev == "iteration_start":
            it = r.get("iteration", len(iterations) + 1)
            current = {"iteration": it, "events": [], "llm_calls": llm_by_iter.get(it, [])}
            iterations.append(current)
            continue
        _bucket_events().append(_clean_event(r))

    # LLM calls that never matched an iteration (e.g. iteration 0 / setup).
    if 0 in llm_by_iter:
        setup_llm = llm_by_iter[0]
    else:
        setup_llm = []

    return {
        **summary,
        "setup_events": setup_events,
        "setup_llm_calls": setup_llm,
        "iterations": iterations,
        "llm_calls": llm_records,
        "model_flow": build_model_flow(llm_records),
    }


# ---------------------------------------------------------------------------
# Model-failover flow aggregation
#
# Each LLM call carries a `failover` block (worker_attempts + router_attempts)
# written by InstrumentedLLM. Here we flatten every attempt across the whole run
# into one ordered timeline, compute per-model stats, and flag cooldown
# violations so the UI can render "complete visibility on the flow through the
# models" without the frontend having to know the gateway's internals.
# ---------------------------------------------------------------------------

_ROUTER_STATUS_TO_OUTCOME = {
    "ok": "success",
    "error": "error",
    "skipped": "skipped",
    "unparseable": "unparseable",
}

# An actual upstream call was made (and thus consumes a cooldown slot) only for
# these outcomes. "skipped" attempts never touch the provider.
_CALLED_OUTCOMES = ("success", "error", "selected", "unparseable")

_COOLDOWN_EPSILON = 0.05  # tolerance (s) to avoid flagging float rounding


def _norm_worker_attempt(a: dict, call: dict) -> dict:
    return {
        "kind": "worker",
        "call_label": call.get("call_label"),
        "iteration": call.get("iteration"),
        "seq": call.get("seq"),
        "provider": a.get("provider"),
        "model": a.get("model"),
        "slot_index": a.get("slot_index"),
        "strategy": a.get("strategy"),
        "outcome": a.get("outcome"),
        "reason": a.get("reason"),
        "latency_ms": a.get("latency_ms"),
        "error": a.get("error"),
        "status": a.get("status"),
        "ts": a.get("ts"),
        "rate": a.get("rate"),
    }


def _norm_router_attempt(a: dict, call: dict) -> dict:
    status = a.get("status")
    return {
        "kind": "router",
        "call_label": call.get("call_label"),
        "iteration": call.get("iteration"),
        "seq": call.get("seq"),
        "provider": a.get("provider"),
        "model": a.get("model"),
        "slot_index": None,
        "strategy": "router",
        "outcome": _ROUTER_STATUS_TO_OUTCOME.get(status, status),
        "reason": a.get("reason") or (f"tier={a.get('tier')}" if a.get("tier") else status),
        "latency_ms": a.get("latency_ms"),
        "error": a.get("error"),
        "status": None,
        "ts": a.get("ts"),
        "rate": a.get("rate"),
    }


def build_model_flow(llm_records: list[dict]) -> dict:
    """Flatten all model attempts (worker + router) into an ordered timeline
    with per-model stats and cooldown-violation flags."""
    attempts: list[dict] = []
    for idx, call in enumerate(llm_records):
        fo = call.get("failover") or {}
        for a in (fo.get("router_attempts") or []):
            if isinstance(a, dict):
                rec = _norm_router_attempt(a, call)
                rec["_order"] = idx
                attempts.append(rec)
        for a in (fo.get("worker_attempts") or []):
            if isinstance(a, dict):
                rec = _norm_worker_attempt(a, call)
                rec["_order"] = idx
                attempts.append(rec)

    # Stable order: by timestamp when present, else by call order then insertion.
    attempts.sort(key=lambda r: (r.get("ts") is None, r.get("ts") or 0, r.get("_order", 0)))
    for r in attempts:
        r.pop("_order", None)

    stats: dict[tuple, dict] = {}
    violations: list[dict] = []
    last_call_ts: dict[tuple, float] = {}
    counts = {"total_attempts": len(attempts), "worker": 0, "router": 0,
              "success": 0, "error": 0, "skipped": 0, "other": 0}

    for r in attempts:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
        outcome = r.get("outcome")
        if outcome in ("success",):
            counts["success"] += 1
        elif outcome in ("error",):
            counts["error"] += 1
        elif outcome in ("skipped",):
            counts["skipped"] += 1
        else:
            counts["other"] += 1

        # Key on (kind, provider, model): the router pool and worker pool can
        # share a provider+model but have independent rate state / cooldowns, so
        # they must be tracked separately to avoid false cooldown violations.
        key = (r.get("kind"), r.get("provider"), r.get("model"))
        s = stats.get(key)
        if s is None:
            s = {"provider": r.get("provider"), "model": r.get("model"),
                 "kind": r.get("kind"), "success": 0, "error": 0, "skipped": 0,
                 "other": 0, "total": 0, "latency_total": 0, "latency_count": 0,
                 "reasons": {}}
            stats[key] = s
        s["total"] += 1
        s[outcome if outcome in ("success", "error", "skipped") else "other"] += 1
        lat = r.get("latency_ms")
        if isinstance(lat, (int, float)):
            s["latency_total"] += lat
            s["latency_count"] += 1
        reason = r.get("reason")
        if reason:
            s["reasons"][reason] = s["reasons"].get(reason, 0) + 1

        # Cooldown-respect check: consecutive real calls to the same model must
        # be >= the configured cooldown apart. Skips don't consume a slot.
        if outcome in _CALLED_OUTCOMES:
            ts = r.get("ts")
            rate = r.get("rate") or {}
            cooldown = rate.get("cooldown")
            prev = last_call_ts.get(key)
            if ts is not None and prev is not None and isinstance(cooldown, (int, float)) and cooldown > 0:
                gap = ts - prev
                if gap < cooldown - _COOLDOWN_EPSILON:
                    violations.append({
                        "kind": r.get("kind"),
                        "provider": r.get("provider"), "model": r.get("model"),
                        "gap_s": round(gap, 3), "cooldown_s": cooldown,
                        "call_label": r.get("call_label"), "iteration": r.get("iteration"),
                    })
            if ts is not None:
                last_call_ts[key] = ts

    for s in stats.values():
        s["avg_latency_ms"] = round(s["latency_total"] / s["latency_count"]) if s["latency_count"] else None

    stat_list = sorted(stats.values(), key=lambda s: (-s["total"], s["provider"] or "", s["model"] or ""))

    return {
        "attempts": attempts,
        "stats": stat_list,
        "violations": violations,
        "counts": counts,
    }


def _clean_event(record: dict) -> dict:
    """Trim a raw log record to the fields the UI cares about."""
    out = {k: v for k, v in record.items() if k not in ("logger",)}
    out["event"] = record.get("event", "").split(" ")[0]
    return out


def event_count(run_id: str) -> int:
    """Total raw (incl. noise) line count — used by the live tailer for offsets."""
    path = _event_path(run_id)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)
