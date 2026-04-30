"""
diagnose.py — SGLang config validation rules for DGX Spark / GB10 / SM121A.
No inference — pure pattern matching against known failure modes from the blog.
"""

from .. import knowledge


def _article_url(slug: str) -> str:
    meta = knowledge.get_meta()
    base = meta.get("site_url", "").rstrip("/")
    return f"{base}/blog/{slug}"


def _urls() -> tuple[str, str, str]:
    return (
        _article_url("setup-mistral-sglang-setup"),
        _article_url("fixes-sglang-vibe-performance-benchmark"),
        _article_url("fixes-sglang-restart-oom-fix"),
    )

_STABLE_TAG_KEYWORDS = ("stable", "release", "v0.3", "v0.4")
_GB10_KEYWORDS = ("gb10", "sm121a", "dgx spark", "blackwell")


def _is_gb10(hardware: str) -> bool:
    h = hardware.lower()
    return any(k in h for k in _GB10_KEYWORDS)


def _is_stable_image(image_tag: str) -> bool:
    t = image_tag.lower()
    return any(k in t for k in _STABLE_TAG_KEYWORDS)


def diagnose_sglang(
    attention_backend: str = "",
    mem_fraction: float = 0.0,
    cuda_graph_max_bs: int = 0,
    image_tag: str = "",
    hardware: str = "",
    error_message: str = "",
) -> dict:
    """
    Validate an SGLang configuration for NVIDIA DGX Spark (GB10/SM121A).
    Returns known issues, recommended fixes, and links to relevant articles.

    Args:
        attention_backend: e.g. 'flashinfer' or 'triton'
        mem_fraction: --mem-fraction-static value (e.g. 0.88)
        cuda_graph_max_bs: --cuda-graph-max-bs value
        image_tag: Docker image tag used
        hardware: Hardware description (e.g. 'GB10', 'DGX Spark')
        error_message: Paste error output here for pattern matching
    """
    sglang_url, vibe_perf_url, oom_url = _urls()
    issues = []
    warnings = []
    gb10 = _is_gb10(hardware)

    # Rule 1: flashinfer on GB10/SM121A — critical
    if attention_backend.lower() == "flashinfer" and gb10:
        issues.append({
            "severity": "critical",
            "param": "attention_backend",
            "value": attention_backend,
            "problem": "SM121A architecture is not supported by flashinfer. Causes OOM on first batch.",
            "fix": "Use --attention-backend triton",
            "source": sglang_url,
        })

    # Rule 2: mem_fraction > 0.85 — critical
    if mem_fraction > 0.85:
        issues.append({
            "severity": "critical",
            "param": "mem_fraction",
            "value": mem_fraction,
            "problem": f"mem-fraction-static={mem_fraction} causes OOM during init on GB10 (confirmed failure at 0.88).",
            "fix": "Set --mem-fraction-static to 0.75 or lower",
            "source": sglang_url,
        })

    # Rule 3: cuda_graph_max_bs > 32 — warning
    if cuda_graph_max_bs > 32:
        warnings.append({
            "severity": "warning",
            "param": "cuda_graph_max_bs",
            "value": cuda_graph_max_bs,
            "problem": "Values above 32 increase CUDA graph compilation time with no throughput gain on GB10.",
            "fix": "Use --cuda-graph-max-bs 32",
            "source": vibe_perf_url,
        })

    # Rule 4: stable image on GB10 — critical
    if _is_stable_image(image_tag) and gb10:
        issues.append({
            "severity": "critical",
            "param": "image_tag",
            "value": image_tag,
            "problem": "Stable SGLang releases lack SM121A support. Model will not load.",
            "fix": "Use lmsysorg/sglang:latest or a nightly build",
            "source": sglang_url,
        })

    # Rule 5: error_message pattern — invalid device ordinal
    if "invalid device ordinal" in error_message.lower():
        issues.append({
            "severity": "critical",
            "param": "error_message",
            "value": "invalid device ordinal",
            "problem": "Driver or image mismatch. SM121A requires nightly SGLang with matching CUDA driver.",
            "fix": "Switch to lmsysorg/sglang:latest and verify nvidia-smi shows the GB10 GPU",
            "source": sglang_url,
        })

    # Rule 6: --rm + --restart both set — critical (inferred from error_message or config)
    err_lower = error_message.lower()
    if "--rm" in err_lower and "--restart" in err_lower:
        issues.append({
            "severity": "critical",
            "param": "docker_flags",
            "value": "--rm + --restart",
            "problem": "--rm removes the container on exit, making --restart ineffective. Container will not auto-restart.",
            "fix": "Remove --rm. Use --restart=unless-stopped alone.",
            "source": oom_url,
        })

    # Rule 7: SGLANG_ENABLE_SPEC_V2 missing when EAGLE flags present
    if "eagle" in err_lower or "speculative" in err_lower:
        if "sglang_enable_spec_v2" not in err_lower and "spec_v2" not in err_lower:
            warnings.append({
                "severity": "warning",
                "param": "SGLANG_ENABLE_SPEC_V2",
                "value": "missing",
                "problem": "EAGLE speculative decoding requires SGLANG_ENABLE_SPEC_V2=True env var.",
                "fix": "Add -e SGLANG_ENABLE_SPEC_V2=True to your docker run command",
                "source": vibe_perf_url,
            })

    # Determine verdict
    if issues:
        verdict = "invalid"
    elif warnings:
        verdict = "valid_with_warnings"
    elif not any([attention_backend, mem_fraction, cuda_graph_max_bs, image_tag, hardware, error_message]):
        verdict = "unknown"
    else:
        verdict = "valid"

    recommended_config = {
        "attention_backend": "triton",
        "mem_fraction": 0.75,
        "cuda_graph_max_bs": 32,
        "image_tag": "lmsysorg/sglang:latest",
        "env": {"SGLANG_ENABLE_SPEC_V2": "True"},
        "max_running_requests": 16,
    }

    return {
        "issues": issues,
        "warnings": warnings,
        "recommended_config": recommended_config,
        "verdict": verdict,
    }
