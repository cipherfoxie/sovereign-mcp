"""
main.py — Sovereign AI MCP Server
FastMCP app with Streamable HTTP transport.
MCP endpoint: POST /self-hosted-ai
Health endpoint: GET /health (custom Starlette route added to the app)
"""

import json
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import knowledge
from .tools.search import search_blog
from .tools.article import get_article
from .tools.diagnose import diagnose_sglang

# ── MCP Server ────────────────────────────────────────────────────────────────
# DNS rebinding protection: server binds to 127.0.0.1 inside Docker, only
# Caddy reverse-proxies internet traffic. Allow the public hostname Caddy
# forwards as the Host header. Localhost patterns kept for healthchecks.
mcp = FastMCP(
    name="sovereign-ai-blog",
    instructions=(
        "Search and retrieve articles from the Sovereign AI Blog — "
        "a practical engineering log of self-hosted AI on NVIDIA DGX Spark. "
        "Use search_blog to find articles, get_article for full content, "
        "and diagnose_sglang to validate SGLang configs for GB10/SM121A hardware."
    ),
    streamable_http_path="/self-hosted-ai",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "mcp.sovgrid.org",
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
        ],
        allowed_origins=[
            "https://mcp.sovgrid.org",
            "http://127.0.0.1:*",
            "http://localhost:*",
        ],
    ),
)

# Pure read-only / deterministic tools.
# readOnlyHint:    no side effects on any system
# idempotentHint:  same input always yields the same output
# openWorldHint:   tool does NOT touch external systems beyond a fixed local KB
mcp.tool(annotations=ToolAnnotations(
    title="Search Blog",
    readOnlyHint=True,
    idempotentHint=True,
    openWorldHint=False,
))(search_blog)

mcp.tool(annotations=ToolAnnotations(
    title="Get Article",
    readOnlyHint=True,
    idempotentHint=True,
    openWorldHint=False,
))(get_article)

mcp.tool(annotations=ToolAnnotations(
    title="Diagnose SGLang Config",
    readOnlyHint=True,
    idempotentHint=True,
    openWorldHint=False,
))(diagnose_sglang)


async def health_endpoint(request: Request) -> JSONResponse:
    try:
        meta = knowledge.get_meta()
        return JSONResponse({
            "status": "ok",
            "articles": meta["article_count"],
            "generated_at": meta["generated_at"],
        })
    except FileNotFoundError:
        return JSONResponse({"status": "degraded", "error": "knowledge base not found"}, status_code=503)


mcp.custom_route("/health", methods=["GET"])(health_endpoint)

app = mcp.streamable_http_app()
