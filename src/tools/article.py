"""
article.py — Retrieve full article by slug.
"""

from .. import knowledge


def get_article(slug: str) -> dict:
    """
    Retrieve the full content of a blog article by its slug.

    Args:
        slug: Article slug (e.g. 'setup-mistral-sglang-setup')
    """
    article = knowledge.get_article_by_slug(slug)
    if article is None:
        return {"error": "article_not_found", "slug": slug}

    return {
        "slug": article["slug"],
        "title": article["title"],
        "url": article["url"],
        "date": article.get("date", ""),
        "tags": article.get("tags", []),
        "description": article.get("description", ""),
        "body": article.get("body", ""),
        "eeat": article.get("eeat", {}),
        "eeat_avg": article.get("eeat_avg", 0.0),
        "word_count": article.get("word_count", 0),
    }
