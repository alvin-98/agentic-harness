"""
Agent Run Viewer — FastAPI server.

A standalone observability UI (separate from the LLM gateway's usage dashboard)
for inspecting the agent loop with full transparency. It:

  - lists agent runs found in src/logs/runs/
  - renders a single run's Perception -> Decision -> Action flow, iteration by iteration
  - shows the complete input and output of every LLM call (from the .llm.jsonl sidecar)
  - streams active runs live via SSE
  - launches new runs as subprocesses (the agent's LLM client is synchronous and
    would block this server's event loop if run in-process)

Run:
    python -m agents.observability.server          # serves on :8201
    AGENT_VIEWER_PORT=9000 python -m agents.observability.server
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import trace
from ..logging_config import RUNS_DIR

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PORT = int(os.getenv("AGENT_VIEWER_PORT", "8201"))
# Project src root, so the spawned subprocess can import `agents`.
SRC_ROOT = ROOT.parent.parent

app = FastAPI(title="Agent Run Viewer")


class LaunchRequest(BaseModel):
    query: str


@app.get("/api/runs")
def api_runs():
    return {"runs": trace.list_runs()}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    t = trace.load_trace(run_id)
    if t is None:
        raise HTTPException(404, f"run '{run_id}' not found")
    return t


@app.post("/api/runs")
async def api_launch(req: LaunchRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "query is required")
    run_id = uuid.uuid4().hex
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        await asyncio.create_subprocess_exec(
            sys.executable, "-m", "agents.run_cli", query, "--run-id", run_id,
            cwd=str(SRC_ROOT), env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except Exception as e:
        raise HTTPException(500, f"failed to launch run: {e}")
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/stream")
async def api_stream(run_id: str):
    """SSE stream: re-emits the full structured trace whenever the run files grow."""
    event_path = RUNS_DIR / f"{run_id}.jsonl"

    async def gen():
        last_sig = None
        # Wait briefly for the file to appear (freshly launched runs).
        for _ in range(40):
            if event_path.exists():
                break
            await asyncio.sleep(0.25)

        while True:
            t = trace.load_trace(run_id)
            if t is not None:
                # Signature changes when new lines are written or status flips.
                sig = (trace.event_count(run_id), len(t.get("llm_calls", [])), t.get("status"))
                if sig != last_sig:
                    last_sig = sig
                    yield f"data: {json.dumps(t, default=str)}\n\n"
                if t.get("status") == "complete":
                    yield "event: done\ndata: {}\n\n"
                    return
            await asyncio.sleep(0.8)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


# Static assets (app.js, style.css)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn
    print(f"Agent Run Viewer → http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
