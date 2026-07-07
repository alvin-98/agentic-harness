"""
Structured JSON logging for agent observability.

Usage:
    from .logging_config import get_logger, LogContext

    logger = get_logger(__name__)
    
    with LogContext(run_id="abc123"):
        logger.info("step_start", iteration=1, goal_id="g1", goal_text="Fetch data")
        logger.info("tool_call", tool="web_search", arguments={"query": "test"})
        logger.info("step_complete", result_preview="Found 10 results")

Logs are written to:
    - logs/agent.log (rotating, max 10MB x 5 backups)
    - logs/runs/<run_id>.jsonl (per-run trace file)
"""

import json
import logging
import os
import sys
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional
import re

# Context variable for run-scoped metadata
_log_context: ContextVar[dict] = ContextVar("log_context", default={})

# Directory setup
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
RUNS_DIR = LOG_DIR / "runs"
LOG_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Merge context vars (run_id, etc.)
        ctx = _log_context.get()
        if ctx:
            log_entry.update(ctx)

        # Merge extra fields passed via logger.info("msg", extra={...})
        if hasattr(record, "__dict__"):
            for key, val in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "message", "taskName",
                ):
                    # Serialize non-JSON-safe types
                    try:
                        json.dumps(val)
                        log_entry[key] = val
                    except (TypeError, ValueError):
                        log_entry[key] = str(val)

        return json.dumps(log_entry, default=str)


class RunFileHandler(logging.Handler):
    """
    Writes logs to per-run JSONL files: logs/runs/<run_id>.jsonl
    Only emits records that have a run_id in context.
    """

    def __init__(self):
        super().__init__()
        self._file_handles: dict[str, Any] = {}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        ctx = _log_context.get()
        run_id = ctx.get("run_id")
        if not run_id:
            return

        try:
            msg = self.format(record)
            with self._lock:
                if run_id not in self._file_handles:
                    filepath = RUNS_DIR / f"{run_id}.jsonl"
                    self._file_handles[run_id] = open(filepath, "a", encoding="utf-8")
                fh = self._file_handles[run_id]
                fh.write(msg + "\n")
                fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            for fh in self._file_handles.values():
                fh.close()
            self._file_handles.clear()
        super().close()


class HumanReadableFormatter(logging.Formatter):
    """
    Formats log records as human-readable text for .log files.
    Filters out verbose httpcore/httpx DEBUG noise.
    """

    # Loggers to exclude from human-readable output
    EXCLUDED_LOGGERS = {"httpcore.connection", "httpcore.http11", "httpx"}

    # Events that are too verbose for human-readable logs
    EXCLUDED_EVENTS = {
        "connect_tcp.started", "connect_tcp.complete",
        "send_request_headers.started", "send_request_headers.complete",
        "send_request_body.started", "send_request_body.complete",
        "receive_response_headers.started", "receive_response_headers.complete",
        "receive_response_body.started", "receive_response_body.complete",
        "response_closed.started", "response_closed.complete",
        "close.started", "close.complete",
    }

    def __init__(self):
        super().__init__()
        self._run_headers_written: set[str] = set()

    def should_skip(self, record: logging.LogRecord) -> bool:
        """Return True if this record should be skipped in human-readable output."""
        if record.name in self.EXCLUDED_LOGGERS:
            return True
        event = record.getMessage()
        for excluded in self.EXCLUDED_EVENTS:
            if excluded in event:
                return True
        return False

    def format(self, record: logging.LogRecord) -> str:
        if self.should_skip(record):
            return ""

        ctx = _log_context.get()
        run_id = ctx.get("run_id", "")
        event = record.getMessage()
        ts = datetime.now(timezone.utc).strftime("%m/%d/%y %H:%M:%S")
        level = record.levelname

        # Extract extra fields
        extra = {}
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            ):
                extra[key] = val

        return self._format_event(event, extra, ts, level, run_id)

    @staticmethod
    def _format_attempts(extra: dict) -> str:
        """Format router_attempts and worker attempted lists into compact lines."""
        lines = ""
        router = extra.get("router_decision") or {}
        router_attempts = router.get("router_attempts", []) if isinstance(router, dict) else []
        for ra in router_attempts:
            st = ra.get("status", "")
            if st == "ok":
                continue  # successful router already shown in main line
            prov = ra.get("provider", "")
            model = ra.get("model", "")
            raw = ra.get("raw_reply", ra.get("error", ra.get("reason", "")))
            lines += f"                 ⚠ router {prov}/{model}: {st} — {raw}\n"
        attempted = extra.get("attempted", [])
        for att in attempted:
            prov = att.get("provider", "")
            reason = att.get("reason", "")
            lines += f"                 ⚠ worker {prov}: {reason}\n"
        return lines

    def _format_event(self, event: str, extra: dict, ts: str, level: str, run_id: str) -> str:
        """Format a specific event type into human-readable text."""
        short_id = run_id[:8] if run_id else "????????"

        # Run lifecycle events
        if event == "run_start":
            query = extra.get("query", "")
            max_iter = extra.get("max_iterations", 10)
            return f"run {short_id} - query: {query}\n"

        if event == "run_complete":
            iters = extra.get("total_iterations", 0)
            actions = extra.get("total_actions", 0)
            answers = extra.get("total_answers", 0)
            duration = extra.get("duration_ms", 0)
            return f"\nrun complete: {iters} iterations, {actions} actions, {answers} answers, {duration}ms\n"

        # Tools loaded
        if event == "tools_loaded":
            count = extra.get("tool_count", 0)
            names = extra.get("tool_names", [])
            tools_str = ", ".join(f"'{n}'" for n in names)
            return f"[{ts}] {level:<5}  [mcp] loaded {count} tools: [{tools_str}]\n"

        # Iteration events
        if event == "iteration_start":
            it = extra.get("iteration", 0)
            return f"\n--- iter {it} {'─' * 40}\n"

        if event == "iteration_complete":
            it = extra.get("iteration", 0)
            duration = extra.get("duration_ms", 0)
            return ""  # Skip, redundant with iter start

        # Memory events
        if event == "memory_read":
            hits = extra.get("hit_count", 0)
            return f"[memory.read]    {hits} hits\n"

        if event == "memory_created":
            kind = extra.get("kind", "")
            desc = extra.get("descriptor", "")
            return f"[memory.write]   {kind}: {desc}\n"

        if event == "tool_outcome_recorded":
            tool = extra.get("tool_name", "")
            goal_id = extra.get("goal_id", "")
            return f"[memory.write]   tool_outcome for {tool} (goal {goal_id})\n"

        # Perception events
        if event == "goals_decomposed":
            goals = extra.get("goals", [])
            lines = []
            for g in goals:
                gid = g.get("id", "?")[:12]
                text = g.get("text", "")
                lines.append(f"[perception]     ○ g:{gid} - {text}")
            return "\n".join(lines) + "\n" if lines else ""

        if event == "goals_updated":
            total = extra.get("total", 0)
            done = extra.get("done", 0)
            remaining = extra.get("remaining", 0)
            return f"[perception]     goals: {done}/{total} done, {remaining} remaining\n"

        if event == "goal_marked_done":
            gid = extra.get("goal_id", "?")
            text = extra.get("goal_text", "")
            return f"[perception]     ✓ g:{gid} done - {text}\n"

        if event == "artifact_attached_to_goal":
            gid = extra.get("goal_id", "?")
            art_id = extra.get("artifact_id", "")
            return f"[perception]     attached {art_id} → g:{gid}\n"

        if event == "perception_complete":
            goals = extra.get("goals", [])
            all_done = extra.get("all_done", False)
            if all_done:
                return "[perception]     all goals complete\n"
            return ""  # Skip if not all done, other events cover it

        if event == "goal_selected":
            gid = extra.get("goal_id", "?")
            text = extra.get("goal_text", "")
            return f"[perception]     → selected g:{gid}: {text}\n"

        # Perception LLM events
        if event == "perception_llm_complete":
            call = extra.get("call", "")
            duration = extra.get("duration_ms", 0)
            in_tok = extra.get("input_tokens", 0)
            out_tok = extra.get("output_tokens", 0)
            provider = extra.get("provider", "")
            model = extra.get("model", "")
            router = extra.get("router_decision", {})
            tier = router.get("tier", "") if router else ""
            tokens_str = f" ({in_tok}→{out_tok} tok)" if in_tok or out_tok else ""
            lines = f"[perception]     {call} [{tier} {provider}/{model}]{tokens_str} {duration}ms\n"
            lines += self._format_attempts(extra)
            return lines

        # Decision events
        if event == "decision_llm_complete":
            gid = extra.get("goal_id", "?")
            is_answer = extra.get("is_answer", False)
            tool = extra.get("tool_name", "")
            duration = extra.get("duration_ms", 0)
            in_tok = extra.get("input_tokens", 0)
            out_tok = extra.get("output_tokens", 0)
            provider = extra.get("provider", "")
            model = extra.get("model", "")
            router = extra.get("router_decision", {})
            tier = router.get("tier", "") if router else ""
            tokens_str = f" ({in_tok}→{out_tok} tok)" if in_tok or out_tok else ""
            if is_answer:
                line = f"[decision]       ANSWER (g:{gid}) [{tier} {provider}/{model}]{tokens_str} {duration}ms\n"
            else:
                line = f"[decision]       TOOL_CALL: {tool} (g:{gid}) [{tier} {provider}/{model}]{tokens_str} {duration}ms\n"
            line += self._format_attempts(extra)
            return line

        if event == "decision_complete":
            return ""  # Covered by decision_llm_complete

        # Action events
        if event == "action_start":
            tool = extra.get("tool", "")
            args = extra.get("arguments", {})
            args_str = json.dumps(args) if args else "{}"
            return f"[action]         → {tool}({args_str})\n"

        if event == "action_complete":
            tool = extra.get("tool", "")
            duration = extra.get("duration_ms", 0)
            preview = extra.get("result_preview", "").replace("\n", " ")
            art_id = extra.get("artifact_id")
            if art_id:
                return f"[action]         ← {tool} ({duration}ms) artifact:{art_id}\n"
            return f"[action]         ← {tool} ({duration}ms): {preview}\n"

        if event == "artifact_created":
            art_id = extra.get("artifact_id", "")
            size = extra.get("size_bytes", 0)
            source = extra.get("source", "")
            return f"[action]         artifact {art_id} ({size} bytes) from {source}\n"

        # Answer events
        if event == "answer_produced":
            gid = extra.get("goal_id", "?")
            preview = extra.get("answer_preview", "").replace("\n", " ")
            return f"[answer]         g:{gid}: {preview}...\n"

        if event == "all_goals_done":
            it = extra.get("iteration", 0)
            return f"\n--- all goals done (iter {it}) ---\n"

        # Warnings and errors
        if event == "decision_empty":
            gid = extra.get("goal_id", "")
            return f"[warning]        decision_empty for g:{gid}\n"

        if event == "artifact_handle_blocked":
            tool = extra.get("tool", "")
            return f"[warning]        artifact handle blocked for {tool}\n"

        if event == "mcp_call_failed":
            tool = extra.get("tool", "")
            error = extra.get("error", "")
            return f"[error]          {tool} failed: {error}\n"

        # Debug events we want to show
        if event == "perception_observe_start":
            return ""  # Skip, too verbose

        if event == "decompose_query_start" or event == "decompose_query_complete":
            return ""  # Skip

        if event == "update_goals_start" or event == "update_goals_complete":
            return ""  # Skip

        if event == "decision_llm_start":
            return ""  # Skip, decision_llm_complete covers it

        if event == "mcp_call_start" or event == "mcp_call_success":
            return ""  # Skip, action_start/complete covers it

        if event == "artifact_check_result":
            return ""  # Skip, too verbose

        if event == "memory_skipped":
            return ""  # Skip

        if event == "artifact_attached":
            art_id = extra.get("artifact_id", "")
            return f"[debug]          artifact attached: {art_id}\n"

        if event == "result_inline":
            return ""  # Skip

        # Generic fallback for unknown events
        if level in ("WARNING", "ERROR"):
            return f"[{ts}] {level:<5}  {event}\n"

        return ""  # Skip unknown DEBUG/INFO events


class HumanReadableRunHandler(logging.Handler):
    """
    Writes human-readable logs to per-run .log files: logs/runs/<run_id>.log
    Filters out verbose httpcore/httpx DEBUG messages.
    """

    def __init__(self):
        super().__init__()
        self._file_handles: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._formatter = HumanReadableFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        ctx = _log_context.get()
        run_id = ctx.get("run_id")
        if not run_id:
            return

        try:
            msg = self._formatter.format(record)
            if not msg:  # Skip empty messages (filtered out)
                return
            with self._lock:
                if run_id not in self._file_handles:
                    filepath = RUNS_DIR / f"{run_id}.log"
                    self._file_handles[run_id] = open(filepath, "a", encoding="utf-8")
                fh = self._file_handles[run_id]
                fh.write(msg)
                fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            for fh in self._file_handles.values():
                fh.close()
            self._file_handles.clear()
        super().close()


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that allows passing extra fields directly as kwargs.
    
    Usage:
        logger.info("tool_call", tool="web_search", query="test")
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        # Move non-standard kwargs into extra
        extra = kwargs.get("extra", {})
        standard_keys = {"exc_info", "stack_info", "stacklevel", "extra"}
        for key in list(kwargs.keys()):
            if key not in standard_keys:
                extra[key] = kwargs.pop(key)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> ContextLogger:
    """
    Get a configured logger for the given module name.
    Returns a ContextLogger that supports extra kwargs.
    """
    logger = logging.getLogger(name)

    # Only configure once (check for handlers)
    if not logger.handlers and not logging.getLogger().handlers:
        _configure_root_logger()

    return ContextLogger(logger, {})


def _configure_root_logger() -> None:
    """Configure the root logger with JSON file + console handlers."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    json_formatter = JSONFormatter()

    # Rotating file handler (10MB, 5 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR / "agent.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)
    root.addHandler(file_handler)

    # Per-run JSONL file handler (full verbose logs)
    run_handler = RunFileHandler()
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(json_formatter)
    root.addHandler(run_handler)

    # Per-run human-readable .log file handler (filtered, readable)
    human_handler = HumanReadableRunHandler()
    human_handler.setLevel(logging.DEBUG)
    root.addHandler(human_handler)

    # Console handler (INFO level, simpler format for dev)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(console_handler)


class LogContext:
    """
    Context manager for setting run-scoped logging context.
    
    Usage:
        with LogContext(run_id="abc123", user="alice"):
            logger.info("Starting run")  # Includes run_id and user
    """

    def __init__(self, **kwargs: Any):
        self._new_context = kwargs
        self._token: Optional[Any] = None

    def __enter__(self) -> "LogContext":
        current = _log_context.get().copy()
        current.update(self._new_context)
        self._token = _log_context.set(current)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _log_context.reset(self._token)


def set_context(**kwargs: Any) -> None:
    """Set context values without a context manager (useful for async)."""
    current = _log_context.get().copy()
    current.update(kwargs)
    _log_context.set(current)


def clear_context() -> None:
    """Clear all context values."""
    _log_context.set({})
