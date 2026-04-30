"""
search.py - TF-IDF blog search tool.
"""

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .. import knowledge


class SearchResult(BaseModel):
    """One ranked article result from search_blog."""
    slug: str = Field(description="Article slug, use as input to get_article")
    title: str = Field(description="Article title")
    url: str = Field(description="Public URL of the article")
    description: str = Field(description="Article description or summary")
    tags: list[str] = Field(description="Topic tags assigned to the article")
    date: str = Field(default="", description="Publication date (ISO 8601)")
    relevance_score: float = Field(description="TF-IDF cosine similarity score, 0.0 to 1.0 (1.0 = direct date-sorted hit when query is empty)")
    quality_score: float = Field(description="Build-time editorial quality score")
    quality_style: str = Field(default="", description="Editorial style category")


def _to_result(a: dict, score: float) -> SearchResult:
    return SearchResult(
        slug=a["slug"],
        title=a["title"],
        url=a["url"],
        description=a.get("description", ""),
        tags=a.get("tags", []),
        date=a.get("date", ""),
        relevance_score=round(score, 4),
        quality_score=a.get("quality_score", 0.0),
        quality_style=a.get("quality_style", ""),
    )


def search_blog(
    query: Annotated[str, Field(description=(
        "Natural language search query (e.g. 'flashinfer OOM on GB10'). "
        "Multi-word queries are tokenized and TF-IDF ranked. "
        "Pass empty string to list articles without ranking by relevance."
    ))] = "",
    tag: Annotated[Optional[str], Field(description=(
        "Optional tag filter (e.g. 'setup', 'fixes', 'strategy'). "
        "Only articles with this tag are considered. "
        "Use list_tags to discover available tags."
    ))] = None,
    sort: Annotated[Literal["relevance", "date_desc"], Field(description=(
        "Result ordering. 'relevance' uses TF-IDF score (default for non-empty query). "
        "'date_desc' sorts newest first (default behaviour when query is empty). "
        "When query is empty, 'relevance' is treated as 'date_desc'."
    ))] = "relevance",
    n: Annotated[int, Field(
        description="Maximum number of results to return", ge=1, le=20,
    )] = 5,
) -> list[SearchResult]:
    """
    Search the Sovereign AI Blog for articles matching a natural language query,
    optionally filtered by tag and sorted by relevance or date.

    Behaviour matrix:
      - query='', sort=*           -> list newest-first, optionally tag-filtered
      - query!='', sort=relevance  -> TF-IDF ranked, optionally tag-filtered
      - query!='', sort=date_desc  -> TF-IDF filtered (score > 0.001), then sorted by date

    Pure read-only, deterministic for a given KB snapshot.
    """
    n = min(max(1, n), 20)
    articles = knowledge.get_articles()
    if not articles:
        return []

    if tag:
        tag_lc = tag.lower()
        articles = [a for a in articles if tag_lc in {t.lower() for t in a.get("tags", [])}]
        if not articles:
            return []

    query_stripped = query.strip()

    # Empty query: pure listing, sorted by date_desc, optional tag filter already applied.
    if not query_stripped:
        sorted_articles = sorted(articles, key=lambda a: a.get("date", ""), reverse=True)
        return [_to_result(a, 1.0) for a in sorted_articles[:n]]

    # Non-empty query: TF-IDF over (filtered) corpus.
    corpus = []
    for a in articles:
        body_excerpt = a.get("body", "")[:500]
        tags_text = " ".join(a.get("tags", []))
        corpus.append(f"{a['title']} {a.get('description', '')} {tags_text} {body_excerpt}")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query_stripped])
    except ValueError:
        return []

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Filter score floor first.
    candidates = [(i, float(scores[i])) for i in range(len(articles)) if float(scores[i]) >= 0.001]
    if not candidates:
        return []

    if sort == "date_desc":
        candidates.sort(key=lambda p: articles[p[0]].get("date", ""), reverse=True)
    else:
        candidates.sort(key=lambda p: p[1], reverse=True)

    return [_to_result(articles[i], score) for i, score in candidates[:n]]
