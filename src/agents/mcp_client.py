"""
MCP Client for Simple Image Creation - Using OpenAI Function Calling
"""

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import asyncio
from openai import OpenAI
import json


def mcp_tools_to_openai_format(tools_result) -> list[dict]:
    """Convert MCP tools to OpenAI function calling format."""
    openai_tools = []
    for tool in tools_result.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        })
    return openai_tools


SYSTEM_PROMPT = """You create images by calling tools step by step.

To create an image with text inside a rectangle:
1. First call create_canvas to create a blank canvas
2. Then call add_rectangle to draw a rectangle
3. Then call write_text to add text INSIDE the rectangle bounds
4. When done, respond with a text message confirming completion

Always call tools one at a time and wait for the result before proceeding."""


async def main():
    client = OpenAI(
        base_url="http://localhost:30000/v1",
        api_key="EMPTY"
    )

    server_params = StdioServerParameters(
        command="python",
        args=["-m", "agents.mcp_server"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server")
            
            # List available tools and convert to OpenAI format
            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai_format(tools_result)
            print(f"Available tools: {[t['function']['name'] for t in openai_tools]}")
            
            user_query = "Create an image with a red rectangle and the text 'Hello World' inside it."
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ]

            max_iterations = 10
            for i in range(max_iterations):
                print(f"\n--- Iteration {i + 1} ---")
                
                response = client.chat.completions.create(
                    model="default",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                
                message = response.choices[0].message
                
                # Check if there are tool calls
                if message.tool_calls:
                    # Add assistant message with tool calls
                    messages.append(message)
                    
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        print(f"Calling tool: {tool_name} with {tool_args}")
                        
                        try:
                            result = await session.call_tool(tool_name, tool_args)
                            result_text = result.content[0].text if result.content else str(result)
                        except Exception as e:
                            result_text = f"Error: {e}"
                        
                        print(f"Tool result: {result_text}")
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        })
                else:
                    # No tool calls - this is a final text response
                    print(f"Final response: {message.content}")
                    break
            else:
                print(f"WARNING: Reached max iterations ({max_iterations})")


if __name__ == "__main__":
    asyncio.run(main())


