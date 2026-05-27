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

class Action:
    def __init__(self):
        pass
    
    async def execute(self, session: ClientSession, tool_call: ToolCall) -> tuple[str, str | None]:
        pass