"""
test_tools.py — Basic tests for MCP tools.
Run: pytest tests/
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.search import search_blog
from src.tools.article import get_article
from src.tools.diagnose import diagnose_sglang
from src.tools.tags import list_tags


# ── diagnose_sglang (no KB required) ─────────────────────────────────────────

def test_diagnose_flashinfer_on_gb10():
    result = diagnose_sglang(attention_backend="flashinfer", hardware="DGX Spark GB10")
    assert result.verdict == "invalid"
    issues = result.issues
    assert any(i.param == "attention_backend" for i in issues)


def test_diagnose_high_mem_fraction():
    result = diagnose_sglang(mem_fraction=0.88)
    assert result.verdict == "invalid"
    assert any(i.param == "mem_fraction" for i in result.issues)


def test_diagnose_cuda_graph_too_high():
    result = diagnose_sglang(cuda_graph_max_bs=64)
    assert result.verdict == "valid_with_warnings"
    assert any(w.param == "cuda_graph_max_bs" for w in result.warnings)


def test_diagnose_stable_image_gb10():
    result = diagnose_sglang(image_tag="lmsysorg/sglang:stable", hardware="GB10")
    assert result.verdict == "invalid"


def test_diagnose_clean_config():
    result = diagnose_sglang(
        attention_backend="triton",
        mem_fraction=0.75,
        cuda_graph_max_bs=32,
        image_tag="lmsysorg/sglang:latest",
        hardware="DGX Spark GB10",
    )
    assert result.verdict == "valid"
    assert result.issues == []
    assert result.warnings == []


def test_diagnose_invalid_device_ordinal():
    result = diagnose_sglang(error_message="CUDA error: invalid device ordinal")
    assert any(i.param == "error_message" for i in result.issues)


def test_diagnose_no_input():
    result = diagnose_sglang()
    assert result.verdict == "unknown"


# ── search_blog (requires knowledge base) ────────────────────────────────────

@pytest.fixture
def kb_available():
    from pathlib import Path
    kb = Path(__file__).parent.parent / "data" / "knowledge-base.json"
    if not kb.exists():
        pytest.skip("knowledge-base.json not found")


def test_search_returns_results(kb_available):
    results = search_blog("SGLang mistral setup")
    assert isinstance(results, list)
    assert len(results) > 0
    assert hasattr(results[0], "slug")
    assert hasattr(results[0], "relevance_score")


def test_search_respects_n_limit(kb_available):
    results = search_blog("setup guide", n=3)
    assert len(results) <= 3


def test_search_n_capped_at_20(kb_available):
    results = search_blog("docker", n=99)
    assert len(results) <= 20


# ── get_article (requires knowledge base) ────────────────────────────────────

def test_get_article_not_found(kb_available):
    result = get_article("nonexistent-slug-xyz")
    assert result.error == "article_not_found"
    assert result.slug == "nonexistent-slug-xyz"


def test_get_article_sglang(kb_available):
    result = get_article("setup-mistral-sglang-setup")
    if result.error:
        pytest.skip("Article not in current KB")
    assert result.slug == "setup-mistral-sglang-setup"
    assert hasattr(result, "body")
    assert len(result.body) > 100


# ── search_blog new behaviours ───────────────────────────────────────────────

def test_search_empty_query_returns_listing(kb_available):
    """Empty query should return articles sorted by date_desc, not nothing."""
    results = search_blog(query="", n=5)
    assert len(results) > 0, "Empty query should return latest articles, not empty list"
    # Sorted by date desc
    dates = [r.date for r in results if r.date]
    if len(dates) >= 2:
        assert dates == sorted(dates, reverse=True), "Empty query results should be date_desc"


def test_search_with_tag_filter(kb_available):
    """Tag filter should narrow results to articles with that tag."""
    results = search_blog(query="", tag="setup", n=10)
    assert all("setup" in [t.lower() for t in r.tags] for r in results)


def test_search_with_unknown_tag_returns_empty(kb_available):
    results = search_blog(query="", tag="this-tag-does-not-exist-xyz", n=10)
    assert results == []


def test_search_sort_date_desc_with_query(kb_available):
    """When sort='date_desc' with a query, hits are filtered by relevance then sorted by date."""
    results = search_blog(query="docker", sort="date_desc", n=10)
    if len(results) >= 2:
        dates = [r.date for r in results if r.date]
        assert dates == sorted(dates, reverse=True)


# ── list_tags ────────────────────────────────────────────────────────────────

def test_list_tags_returns_count_desc(kb_available):
    tags = list_tags(sort="count_desc")
    assert len(tags) > 0
    # First tag should have highest count
    if len(tags) >= 2:
        assert tags[0].article_count >= tags[1].article_count


def test_list_tags_alpha(kb_available):
    tags = list_tags(sort="alpha")
    if len(tags) >= 2:
        names = [t.tag.lower() for t in tags]
        assert names == sorted(names)


def test_list_tags_no_empty_tag(kb_available):
    tags = list_tags()
    for t in tags:
        assert t.tag.strip() != ""
        assert t.article_count > 0
