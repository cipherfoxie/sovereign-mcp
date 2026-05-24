# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-05-24

First tagged release. The server has been live at
[mcp.sovgrid.org/self-hosted-ai](https://mcp.sovgrid.org/self-hosted-ai)
since 2026-04 and listed on the official MCP Registry, Smithery,
Glama, and awesome-mcp-servers.

### Added
- Four MCP tools exposed via Streamable HTTP (FastMCP 1.3+):
  - `search_blog(query, tag?, sort?, n?)` — TF-IDF full-text search
    with optional tag filter and relevance/date sorting
  - `list_tags(sort?)` — topic taxonomy with article counts
  - `get_article(slug)` — full article body and frontmatter
  - `diagnose_sglang(error_message)` — pattern-match runtime errors
    against curated GB10/SM121A failure modes
- Typed Pydantic `BaseModel` output schemas (`SearchResult`, `TagInfo`,
  `Article`, `Diagnosis`) with `ToolAnnotations` for MCP client trust
  signals
- `Annotated[type, Field(description=...)]` input descriptions for
  agent introspection
- Docker image and systemd service unit for self-hosting
- Placeholder `data/knowledge-base.json` (zero articles, valid schema)
  so the server starts out-of-the-box without source content
- MIT-licensed reference deployment at mcp.sovgrid.org/self-hosted-ai
  (60 req/min/IP rate limit, no auth)

[Unreleased]: https://github.com/cipherfoxie/sovereign-mcp/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/cipherfoxie/sovereign-mcp/releases/tag/v1.0.0
