#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

# ── Config (override via env vars) ──────────────────────────────────
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-Qwen/Qwen3.5-0.8B}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
GATEWAY_PORT="${GATEWAY_PORT:-8101}"
VIEWER_PORT="${VIEWER_PORT:-8201}"

# ── 1. Install Chromium via Playwright ──────────────────────────────
echo "[1/4] Installing Chromium via Playwright…"
"$PYTHON" -m playwright install chromium

# ── 2. Launch sglang server ─────────────────────────────────────────
echo "[2/4] Starting sglang server (model: $SGLANG_MODEL_PATH, port: $SGLANG_PORT)…"
"$PYTHON" -m sglang.launch_server \
    --model-path "$SGLANG_MODEL_PATH" \
    --port "$SGLANG_PORT" \
    --mem-fraction-static 0.8 \
    --host 0.0.0.0 &
SGLANG_PID=$!

echo "   Waiting for sglang to be ready…"
for _ in $(seq 1 60); do
    if curl -sf "http://localhost:$SGLANG_PORT/health" >/dev/null 2>&1; then
        echo "   sglang is ready."
        break
    fi
    sleep 2
done

# ── 3. Launch LLM gateway ───────────────────────────────────────────
echo "[3/4] Starting LLM gateway on port $GATEWAY_PORT…"
(cd "$REPO_ROOT/src/llm_gateway" && "$PYTHON" main.py) &
GATEWAY_PID=$!

# ── 4. Launch Agent Run Viewer (observability) ──────────────────────
echo "[4/4] Starting Agent Run Viewer on port $VIEWER_PORT…"
(cd "$REPO_ROOT/src" && PYTHONPATH=. "$PYTHON" -m agents.observability.server) &
VIEWER_PID=$!

# ── Cleanup on exit ─────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down all services…"
    kill "$SGLANG_PID" "$GATEWAY_PID" "$VIEWER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo ""
echo "All services running:"
echo "  sglang:         http://localhost:$SGLANG_PORT"
echo "  LLM gateway:    http://localhost:$GATEWAY_PORT"
echo "  Agent viewer:   http://localhost:$VIEWER_PORT"
echo ""
echo "Press Ctrl+C to stop all services."

wait