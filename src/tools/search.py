"""
search.py - TF-IDF blog search tool.
"""

from typing import Annotated
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
    relevance_score: float = Field(description="TF-IDF cosine similarity score, range 0.0 to 1.0 (higher is more relevant)")
    eeat_avg: float = Field(description="Average EEAT quality score across 13 signals (expertise, experience, authority, trust)")


def search_blog(
    query: Annotated[str, Field(description="Natural language search query (e.g. 'flashinfer OOM on GB10'). Multi-word queries are tokenized and TF-IDF ranked.")],
    n: Annotated[int, Field(description="Maximum number of results to return", ge=1, le=10)] = 5,
) -> list[SearchResult]:
    """
    Search the Sovereign AI Blog for articles matching a natural language query.

    Uses TF-IDF over title, description, tags, and the first 500 chars of body.
    Returns up to n results ranked by cosine similarity. Pure read-only operation,
    deterministic for a given knowledge base snapshot.
    """
    n = min(max(1, n), 10)
    articles = knowledge.get_articles()
    if not articles:
        return []

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
        query_vec = vectorizer.transform([query])
    except ValueError:
        return []

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:n]

    results: list[SearchResult] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < 0.001:
            continue
        a = articles[idx]
        results.append(SearchResult(
            slug=a["slug"],
            title=a["title"],
            url=a["url"],
            description=a.get("description", ""),
            tags=a.get("tags", []),
            relevance_score=round(score, 4),
            eeat_avg=a.get("eeat_avg", 0.0),
        ))

    return results
