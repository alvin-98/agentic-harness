"""
CLI entry point for launching a single agent run with a known run_id.

Used by the Agent Run Viewer to spawn runs as subprocesses (the agent's LLM
client is synchronous and would otherwise block the observability server's event
loop). The viewer generates the run_id up front so it can immediately start
tailing ``src/logs/runs/<run_id>.jsonl`` for live updates.

Usage:
    python -m agents.run_cli "<query>" --run-id <hex>
"""

import argparse
import asyncio
import uuid

from .agent_loop import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent loop for a single query.")
    parser.add_argument("query", help="The user query to run.")
    parser.add_argument("--run-id", dest="run_id", default=None,
                        help="Optional run id. Generated if omitted.")
    args = parser.parse_args()

    run_id = args.run_id or uuid.uuid4().hex
    result = asyncio.run(run(args.query, run_id=run_id))
    print(result)


if __name__ == "__main__":
    main()
