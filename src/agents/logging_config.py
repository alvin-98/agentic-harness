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

    # Per-run file handler
    run_handler = RunFileHandler()
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(json_formatter)
    root.addHandler(run_handler)

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
