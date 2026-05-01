"""
__main__.py - stdio entrypoint for the FastMCP server.

Used by build systems (e.g. Glama) that wrap a stdio MCP via mcp-proxy.
For production HTTP serving, use `uvicorn src.main:app` directly; that
uses the streamable_http_app() exposed in src/main.py.
"""

from .main import mcp

if __name__ == "__main__":
    mcp.run()
