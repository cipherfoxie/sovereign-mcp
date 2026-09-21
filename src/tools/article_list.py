"""
article_list.py - List all blog articles (pure listing, no TF-IDF).
"""

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from .. import knowledge


class ArticleSummary(BaseModel):
    """A single article entry returned by list_articles."""
    slug: str = Field(description="Article slug, use as input to get_article")
    title: str = Field(description="Article title")
    url: str = Field(description="Public URL of the article")
    date: str = Field(default="", description="Publication date (ISO 8601)")
    tags: list[str] = Field(description="Topic tags assigned to the article")
    description: str = Field(default="", description="Article description or summary")
    quality_score: float = Field(description="Editorial quality score")
    quality_style: str = Field(default="", description="Editorial style category")
    quality_class: str = Field(default="", description="Editorial quality class")


def list_articles(
    tag: Annotated[Optional[str], Field(description=(
        "Optional tag filter (e.g. 'setup', 'fixes', 'strategy'). "
        "Only articles with this tag are considered. "
        "Use list_tags to discover available tags."
    ))] = None,
    sort: Annotated[Literal["date_desc", "date_asc", "title_asc", "quality_desc"], Field(description=(
        "Result ordering. 'date_desc' newest first (default). "
        "'date_asc' oldest first. 'title_asc' alphabetical. "
        "'quality_desc' best quality first."
    ))] = "date_desc",
    limit: Annotated[int, Field(description="Number of results (1-50)", ge=1, le=50)] = 20,
    offset: Annotated[int, Field(description="Pagination offset (0-based)", ge=0)] = 0,
) -> list[ArticleSummary]:
    """
    List all blog articles. No TF-IDF computation — pure database listing.

    Use to browse the full corpus, paginate through articles, or filter by tag.
    For full-text semantic search use search_blog instead.
    """
    limit = min(max(1, limit), 50)
    offset = max(0, offset)

    articles = knowledge.get_articles()
    if not articles:
        return []

    if tag:
        tag_lc = tag.lower()
        articles = [a for a in articles if tag_lc in {t.lower() for t in a.get("tags", [])}]

    if sort == "date_desc":
        articles.sort(key=lambda a: a.get("date", ""), reverse=True)
    elif sort == "date_asc":
        articles.sort(key=lambda a: a.get("date", ""))
    elif sort == "title_asc":
        articles.sort(key=lambda a: a.get("title", "").lower())
    elif sort == "quality_desc":
        articles.sort(key=lambda a: a.get("quality_score", 0), reverse=True)

    return [
        ArticleSummary(
            slug=a["slug"],
            title=a.get("title", ""),
            url=a.get("url", ""),
            date=a.get("date", ""),
            tags=a.get("tags", []),
            description=a.get("description", ""),
            quality_score=a.get("quality_score", 0.0),
            quality_style=a.get("quality_style", ""),
            quality_class=a.get("quality_class", ""),
        )
        for a in articles[offset:offset + limit]
    ]