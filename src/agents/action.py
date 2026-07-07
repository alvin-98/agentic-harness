"""
Action module receives a ToolCall and a live MCP session, dispatches the tool call to the MCP server,
and returns a tuple of (descriptor, artifact_id_or_none).

Action contains no LLM call.

When the tool returns a payload larger than ARTIFACT_THRESHOLD_BYTES (4 KB in this session), 
Action calls ArtifactStore.put(...) to persist the full bytes and returns a short descriptor of the 
form [artifact art:abc..., 263507 bytes] preview: .... When the payload is smaller than the threshold, 
Action returns the text directly and no artifact is created.

When tool_call.arguments contains a path or url value that starts with art:, Action refuses the call 
and returns an error string explaining that artifact handles are not paths. This guard exists because 
TINY-tier Decision models occasionally pass an artifact handle to read_file or fetch_url. The guard 
blocks the dispatch and returns a clear error that the history records, so the next Perception iteration 
can mark the goal accordingly.

When the tool call is a real MCP dispatch, Action awaits session.call_tool(name, arguments=...), 
collapses the result's content blocks into a single text string, and proceeds with the threshold check.

The MCP server for Session 6 (in mcp_server.py) exposes nine tools: web_search, fetch_url, get_time, 
currency_convert, read_file, list_dir, create_file, update_file, edit_file. The full inventory and 
contracts are documented in the server file itself. Decision sees these nine tools as a tool list and 
picks one when external work is required.
"""

import asyncio

from mcp import ClientSession

from .schemas import ToolCall
from .artifacts import ArtifactStore, ARTIFACT_THRESHOLD_BYTES
from .logging_config import get_logger

logger = get_logger(__name__)

# The artifact descriptor is what lands in the run history (and is therefore
# replayed into every subsequent Perception/Decision prompt). Keep its inline
# preview short — the full bytes live in the artifact store and are surfaced
# on demand via chunk retrieval, not by dumping the whole page into history.
ARTIFACT_PREVIEW_CHARS = 500


class Action:
    def __init__(self):
        self.artifact_store = ArtifactStore()
    
    async def execute(self, session: ClientSession, tool_call: ToolCall) -> tuple[str, str | None]:
        """
        Execute a tool call and return (descriptor, artifact_id_or_none).
        
        - Guards against artifact handles being passed as paths/URLs.
        - Stores large results (>4KB) in artifact store.
        """
        if self._contains_artifact_handle(tool_call.arguments):
            error_msg = (
                f"ERROR: Artifact handles (art:...) are not file paths or URLs. "
                f"Do not pass them to {tool_call.name}. "
                f"Artifact bytes appear under ATTACHED ARTIFACTS in the prompt."
            )
            logger.warning("artifact_handle_blocked",
                          tool=tool_call.name,
                          arguments=tool_call.arguments,
                          error=error_msg)
            return error_msg, None
        
        try:
            logger.debug("mcp_call_start", tool=tool_call.name, arguments=tool_call.arguments)
            result = await session.call_tool(tool_call.name, tool_call.arguments)
            logger.debug("mcp_call_success", tool=tool_call.name)
        except Exception as e:
            logger.error("mcp_call_failed", tool=tool_call.name, error=str(e), exc_info=True)
            return f"ERROR: Tool execution failed: {str(e)}", None
        
        text = self._collapse_content(result)
        text_bytes = text.encode("utf-8")
        
        if len(text_bytes) > ARTIFACT_THRESHOLD_BYTES:
            artifact_id = self.artifact_store.put(
                data=text_bytes,
                source=tool_call.name,
                content_type="text/plain",
                descriptor=f"Result from {tool_call.name}",
            )
            preview = text.replace("\n", " ")[:ARTIFACT_PREVIEW_CHARS]
            descriptor = f"[artifact {artifact_id}, {len(text_bytes)} bytes] preview: {preview}..."
            logger.info("artifact_created",
                       artifact_id=artifact_id,
                       size_bytes=len(text_bytes),
                       source=tool_call.name)
            return descriptor, artifact_id
        else:
            logger.debug("result_inline", tool=tool_call.name, size_bytes=len(text_bytes))
            return text, None

    def _contains_artifact_handle(self, arguments: dict) -> bool:
        """Check if any argument value starts with 'art:'."""
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("art:"):
                return True
        return False

    def _collapse_content(self, result) -> str:
        """Collapse MCP result content blocks into a single string."""
        if not result.content:
            return ""
        
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(f"[binary data: {len(block.data)} bytes]")
            else:
                parts.append(str(block))
        
        return "\n".join(parts)

    @staticmethod
    async def dispatch_tool_calls(session: ClientSession, tool_calls: list[dict]) -> list[dict]:
        """Dispatch multiple tool calls in parallel via TaskGroup."""
        async def run_one(tc: dict) -> dict:
            result = await session.call_tool(tc["name"], tc.get("arguments") or {})
            text = result.content[0].text if result.content else ""
            return {
                "role": "tool",
                "tool_call_id": tc["id"],
                "tool_name": tc["name"],
                "content": text,
            }

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_one(tc)) for tc in tool_calls]
        return [t.result() for t in tasks]