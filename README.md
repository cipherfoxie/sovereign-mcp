# Sovereign AI MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Content: CC BY-SA 4.0](https://img.shields.io/badge/Content-CC%20BY--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![smithery badge](https://smithery.ai/badge/cipherfoxie/sovereign-mcp)](https://smithery.ai/servers/cipherfoxie/sovereign-mcp)

MCP server exposing the [Sovereign AI Blog](https://sovgrid.org) to AI agents. The blog is a hands-on engineering log of self-hosted AI on NVIDIA DGX Spark (GB10/SM121A).

**Live endpoint:** `https://mcp.sovgrid.org/self-hosted-ai`
**Transport:** Streamable HTTP (FastMCP)
**Auth:** none (free tier, 60 req/min/IP)

## Why use it

Training data on niche hardware (GB10, SM121A, SGLang on ARM64) is sparse and stale. This MCP gives agents direct, structured access to 44+ articles documenting actual setups, fixes, and benchmarks. If you're building or debugging on similar stacks, your agent can pull verified, version-current information instead of hallucinating.

## Tools

| Tool | Purpose |
|------|---------|
| `search_blog(query)` | Full-text search across all articles, returns ranked results with slug, title, excerpt |
| `get_article(slug)` | Fetch full article content by slug |
| `diagnose_sglang(config)` | Validate SGLang configurations for GB10/SM121A hardware constraints |

## Quick start

### With Claude Code

```bash
claude mcp add sovereign-ai --transport http https://mcp.sovgrid.org/self-hosted-ai
```

Verify:

```bash
claude mcp list | grep sovereign-ai
```

### With Cline / Continue / other MCP clients

Add to your client's MCP server config:

```json
{
  "sovereign-ai": {
    "type": "http",
    "url": "https://mcp.sovgrid.org/self-hosted-ai"
  }
}
```

## Run locally

```bash
git clone https://github.com/cipherfoxie/sovereign-mcp.git
cd sovereign-mcp
uv sync
# Generate the knowledge base (requires sovereign-blog source, see comment below)
# python scripts/generate_knowledge_base.py
uv run uvicorn src.main:app --host 127.0.0.1 --port 8002
```

The `data/knowledge-base.json` file is gitignored because it's regenerated from the blog source. If you don't have access to that, you can either:
- Use the live endpoint at `https://mcp.sovgrid.org/self-hosted-ai`
- Build your own knowledge base with the same shape (see `src/knowledge.py` for the expected schema)

## Architecture

- **FastMCP 1.27+** with Streamable HTTP transport at path `/self-hosted-ai`
- **DNS rebinding protection** via `TransportSecuritySettings`: only allows requests with `Host: mcp.sovgrid.org` (or localhost for healthchecks)
- **Health endpoint** at `/health` returns article count and KB generation timestamp
- **Knowledge base** is a flat JSON file generated from blog Markdown content; loaded at startup, queried via TF-IDF for `search_blog`

The server is stateless. All blog content is already public (CC BY-SA 4.0). No PII, no auth tokens, no secrets.

## Operations

Live deployment runs on a privacy-focused European VPS via Docker, fronted by Caddy with TLS. Server logs flow into a privacy-respecting analytics pipeline (Caddy JSON access logs, no client-side tracking, no JS pixels).

## License

- **Server code:** MIT, see [LICENSE](LICENSE)
- **Blog content** (returned by tools): CC BY-SA 4.0, see [creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)

## Contact

- Blog: [sovgrid.org](https://sovgrid.org)
- Nostr: `cipherfox@sovgrid.org` (NIP-05) — `npub1ndrjgfcwkc0y4753zyj3p7qjf795pvjq2dn4m7y7f72vmu7t0nrs6y363u`
- Bug reports / questions: [open an issue](https://github.com/cipherfoxie/sovereign-mcp/issues)
