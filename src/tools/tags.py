"""
tags.py - List available topic tags across the corpus.
"""

from typing import Annotated, Literal
from pydantic import BaseModel, Field
from .. import knowledge


class TagInfo(BaseModel):
    """A topic tag with the number of articles using it."""
    tag: str = Field(description="The tag name (lower-case, snake_case)")
    article_count: int = Field(description="Number of articles tagged with this tag")


def list_tags(
    sort: Annotated[Literal["count_desc", "alpha"], Field(description=(
        "Result ordering. 'count_desc' lists most-used tags first (default). "
        "'alpha' sorts alphabetically."
    ))] = "count_desc",
) -> list[TagInfo]:
    """
    List all topic tags used across the Sovereign AI Blog corpus, with article
    counts. Use this to browse the topic space before calling search_blog with
    a tag filter.
    """
    counts: dict[str, int] = {}
    for a in knowledge.get_articles():
        for t in a.get("tags", []):
            t_norm = t.strip()
            if not t_norm:
                continue
            counts[t_norm] = counts.get(t_norm, 0) + 1

    items = [TagInfo(tag=t, article_count=c) for t, c in counts.items()]

    if sort == "alpha":
        items.sort(key=lambda x: x.tag.lower())
    else:
        items.sort(key=lambda x: (-x.article_count, x.tag.lower()))

    return items
