"""
article.py - Retrieve full article by slug.
"""

from typing import Annotated, Optional
from pydantic import BaseModel, Field
from .. import knowledge


class Article(BaseModel):
    """Full content of a single blog article."""
    slug: str = Field(description="Article slug")
    title: str = Field(default="", description="Article title")
    url: str = Field(default="", description="Public URL of the article")
    date: str = Field(default="", description="Publication date (ISO 8601)")
    tags: list[str] = Field(default_factory=list, description="Topic tags assigned to the article")
    description: str = Field(default="", description="Short article description")
    body: str = Field(default="", description="Full article body in Markdown")
    eeat: dict = Field(default_factory=dict, description="EEAT score breakdown by signal")
    eeat_avg: float = Field(default=0.0, description="Average EEAT score across all signals")
    word_count: int = Field(default=0, description="Word count of the article body")
    error: Optional[str] = Field(default=None, description="Set to 'article_not_found' if no article matches the slug")


def get_article(
    slug: Annotated[str, Field(description="Article slug as returned by search_blog (e.g. 'setup-mistral-sglang-setup'). Lower-case, hyphenated.")],
) -> Article:
    """
    Retrieve the full content of a blog article by its slug.

    Returns the article body (Markdown) plus metadata. If the slug does not
    match any article, returns an Article with `error='article_not_found'`
    and other fields at their defaults.
    """
    article = knowledge.get_article_by_slug(slug)
    if article is None:
        return Article(slug=slug, error="article_not_found")

    return Article(
        slug=article["slug"],
        title=article["title"],
        url=article["url"],
        date=article.get("date", ""),
        tags=article.get("tags", []),
        description=article.get("description", ""),
        body=article.get("body", ""),
        eeat=article.get("eeat", {}),
        eeat_avg=article.get("eeat_avg", 0.0),
        word_count=article.get("word_count", 0),
    )
