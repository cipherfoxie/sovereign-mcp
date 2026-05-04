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
from .tools.tags import list_tags

# ── MCP Server ────────────────────────────────────────────────────────────────
# DNS rebinding protection: server binds to 127.0.0.1 inside Docker, only
# Caddy reverse-proxies internet traffic. Allow the public hostname Caddy
# forwards as the Host header. Localhost patterns kept for healthchecks.
mcp = FastMCP(
    name="sovereign-ai-blog",
    instructions=(
        "Search and retrieve articles from the Sovereign AI Blog, a practical "
        "engineering log of self-hosted AI on NVIDIA DGX Spark. "
        "Use list_tags to discover topic categories, search_blog to find "
        "articles (with optional tag filter and date_desc sort), get_article "
        "for full content by slug, and diagnose_sglang to validate SGLang "
        "configs for GB10/SM121A hardware."
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

mcp.tool(annotations=ToolAnnotations(
    title="List Tags",
    readOnlyHint=True,
    idempotentHint=True,
    openWorldHint=False,
))(list_tags)


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


# ── Tool-call HTTP wrapper for in-article widgets (BLOG-009 / Gitea #9) ──────
# The MCP Streamable HTTP protocol requires session init + JSON-RPC handshake,
# which is overkill for a static-site embedded form. This thin wrapper exposes
# the same tool functions as direct POST endpoints with plain JSON in/out.
# Same backend code, same KB, same Pydantic validation, no protocol overhead.
# Used by <MCPToolWidget> in blog articles via Caddy same-origin /api/* proxy.
def _make_tool_endpoint(tool_fn):
    """Wrap an MCP tool function as a plain JSON HTTP endpoint.

    Bypasses Streamable-HTTP session-init handshake. Same Pydantic
    validation (lives inside the tool function via Annotated[Field(...)]),
    same backend code, same KB. JSON in, JSON out.
    """
    async def endpoint(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        try:
            result = tool_fn(**body)
        except TypeError as exc:
            return JSONResponse({"error": f"invalid arguments: {exc}"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"tool call failed: {exc}"}, status_code=500)
        # Tools may return Pydantic models, lists of Pydantic models, or plain values.
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif isinstance(result, list):
            payload = [r.model_dump() if hasattr(r, "model_dump") else r for r in result]
        else:
            payload = result
        return JSONResponse(payload)
    return endpoint


mcp.custom_route("/health", methods=["GET"])(health_endpoint)
mcp.custom_route("/api/diagnose", methods=["POST"])(_make_tool_endpoint(diagnose_sglang))
mcp.custom_route("/api/search", methods=["POST"])(_make_tool_endpoint(search_blog))
mcp.custom_route("/api/tags", methods=["POST"])(_make_tool_endpoint(list_tags))
mcp.custom_route("/api/article", methods=["POST"])(_make_tool_endpoint(get_article))

app = mcp.streamable_http_app()
