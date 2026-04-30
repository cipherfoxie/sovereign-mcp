"""
search.py — TF-IDF blog search tool.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .. import knowledge


def search_blog(query: str, n: int = 5) -> list[dict]:
    """
    Search the Sovereign AI Blog for articles matching the query.
    Returns up to n results with title, url, description, and EEAT scores.

    Args:
        query: Natural language search query
        n: Maximum number of results (default 5, max 10)
    """
    n = min(max(1, n), 10)
    articles = knowledge.get_articles()
    if not articles:
        return []

    # Build corpus: title + description + tags + body excerpt (first 500 chars)
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

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < 0.001:
            continue
        a = articles[idx]
        eeat = a.get("eeat", {})
        results.append({
            "slug": a["slug"],
            "title": a["title"],
            "url": a["url"],
            "description": a.get("description", ""),
            "tags": a.get("tags", []),
            "relevance_score": round(score, 4),
            "eeat_avg": a.get("eeat_avg", 0.0),
        })

    return results
