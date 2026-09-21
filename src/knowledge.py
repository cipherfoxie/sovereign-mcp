"""
knowledge.py — Loads and caches the knowledge base from data/knowledge-base.json.

Env override: set SOVEREIGN_KB_PATH to point to an alternate KB file.
"""

import json
import os
from pathlib import Path
from typing import Optional

_cache: Optional[dict] = None


def _data_path() -> Path:
    env = os.environ.get("SOVEREIGN_KB_PATH", "")
    if env:
        return Path(env)
    return Path(__file__).parent.parent / "data" / "knowledge-base.json"


def _load() -> dict:
    global _cache
    if _cache is None:
        path = _data_path()
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        _cache = json.loads(path.read_text(encoding="utf-8"))
    return _cache


def reload() -> None:
    global _cache
    _cache = None
    _load()


def get_articles() -> list[dict]:
    return _load().get("articles", [])


def get_article_by_slug(slug: str) -> Optional[dict]:
    for a in get_articles():
        if a["slug"] == slug:
            return a
    return None


def get_meta() -> dict:
    data = _load()
    return {
        "generated_at": data.get("generated_at", ""),
        "site_url": data.get("site_url", ""),
        "article_count": data.get("article_count", len(get_articles())),
    }
