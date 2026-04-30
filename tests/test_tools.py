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


def test_search_n_capped_at_10(kb_available):
    results = search_blog("docker", n=99)
    assert len(results) <= 10


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
