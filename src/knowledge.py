"""
knowledge.py — Loads and caches the knowledge base from data/knowledge-base.json.
"""

import json
from pathlib import Path
from typing import Optional

_DATA_PATH = Path(__file__).parent.parent / "data" / "knowledge-base.json"
_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is None:
        if not _DATA_PATH.exists():
            raise FileNotFoundError(f"Knowledge base not found: {_DATA_PATH}")
        _cache = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
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
