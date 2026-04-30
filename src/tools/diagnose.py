"""
diagnose.py - SGLang config validation rules for DGX Spark / GB10 / SM121A.
No inference, pure pattern matching against known failure modes from the blog.
"""

from typing import Annotated, Literal
from pydantic import BaseModel, Field
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


class DiagnosticIssue(BaseModel):
    """A single rule violation found in the SGLang config."""
    severity: Literal["critical", "warning"] = Field(description="critical = config will fail, warning = will run but suboptimal")
    param: str = Field(description="The config parameter or input field that triggered the rule")
    value: str = Field(description="The offending value (stringified)")
    problem: str = Field(description="Plain-English explanation of the failure mode")
    fix: str = Field(description="Concrete fix to apply")
    source: str = Field(description="URL of the blog article documenting this issue")


class RecommendedConfig(BaseModel):
    """A known-good SGLang config baseline for GB10/SM121A hardware."""
    attention_backend: str = Field(description="SGLang attention backend (triton is the verified default for GB10)")
    mem_fraction: float = Field(description="--mem-fraction-static value (0.75 is the upper safe bound for GB10)")
    cuda_graph_max_bs: int = Field(description="--cuda-graph-max-bs (32 is the optimum for GB10)")
    image_tag: str = Field(description="Docker image tag with SM121A support")
    env: dict = Field(description="Environment variables required for stable operation")
    max_running_requests: int = Field(description="Concurrent request cap, calibrated for GB10 memory budget")


class DiagnoseResult(BaseModel):
    """Outcome of a diagnose_sglang call."""
    issues: list[DiagnosticIssue] = Field(description="Critical issues that will prevent SGLang from running correctly")
    warnings: list[DiagnosticIssue] = Field(description="Non-fatal warnings (suboptimal but non-blocking)")
    recommended_config: RecommendedConfig = Field(description="Verified-good baseline config for GB10/SM121A")
    verdict: Literal["valid", "valid_with_warnings", "invalid", "unknown"] = Field(description="Overall verdict. 'unknown' = no inputs provided.")


def diagnose_sglang(
    attention_backend: Annotated[str, Field(description="SGLang --attention-backend value (e.g. 'flashinfer', 'triton'). Empty string = skip this check.")] = "",
    mem_fraction: Annotated[float, Field(description="SGLang --mem-fraction-static value (e.g. 0.88). 0.0 = skip this check.", ge=0.0, le=1.0)] = 0.0,
    cuda_graph_max_bs: Annotated[int, Field(description="SGLang --cuda-graph-max-bs value. 0 = skip this check.", ge=0)] = 0,
    image_tag: Annotated[str, Field(description="Docker image tag in use (e.g. 'lmsysorg/sglang:latest', 'lmsysorg/sglang:v0.4.0'). Empty = skip.")] = "",
    hardware: Annotated[str, Field(description="Hardware description (e.g. 'GB10', 'DGX Spark', 'SM121A'). Empty = skip GB10-specific rules.")] = "",
    error_message: Annotated[str, Field(description="Paste error log output here for pattern matching against known failure modes.")] = "",
) -> DiagnoseResult:
    """
    Validate an SGLang configuration for NVIDIA DGX Spark (GB10/SM121A).

    Pure pattern-matching against known failure modes documented in the
    Sovereign AI Blog. No inference, no external calls. Returns critical
    issues, non-fatal warnings, and a recommended baseline config.

    All parameters are optional; supply only what you have. With no inputs
    you get the recommended config and a 'unknown' verdict.
    """
    sglang_url, vibe_perf_url, oom_url = _urls()
    issues: list[DiagnosticIssue] = []
    warnings: list[DiagnosticIssue] = []
    gb10 = _is_gb10(hardware)

    # Rule 1: flashinfer on GB10/SM121A - critical
    if attention_backend.lower() == "flashinfer" and gb10:
        issues.append(DiagnosticIssue(
            severity="critical",
            param="attention_backend",
            value=attention_backend,
            problem="SM121A architecture is not supported by flashinfer. Causes OOM on first batch.",
            fix="Use --attention-backend triton",
            source=sglang_url,
        ))

    # Rule 2: mem_fraction > 0.85 - critical
    if mem_fraction > 0.85:
        issues.append(DiagnosticIssue(
            severity="critical",
            param="mem_fraction",
            value=str(mem_fraction),
            problem=f"mem-fraction-static={mem_fraction} causes OOM during init on GB10 (confirmed failure at 0.88).",
            fix="Set --mem-fraction-static to 0.75 or lower",
            source=sglang_url,
        ))

    # Rule 3: cuda_graph_max_bs > 32 - warning
    if cuda_graph_max_bs > 32:
        warnings.append(DiagnosticIssue(
            severity="warning",
            param="cuda_graph_max_bs",
            value=str(cuda_graph_max_bs),
            problem="Values above 32 increase CUDA graph compilation time with no throughput gain on GB10.",
            fix="Use --cuda-graph-max-bs 32",
            source=vibe_perf_url,
        ))

    # Rule 4: stable image on GB10 - critical
    if _is_stable_image(image_tag) and gb10:
        issues.append(DiagnosticIssue(
            severity="critical",
            param="image_tag",
            value=image_tag,
            problem="Stable SGLang releases lack SM121A support. Model will not load.",
            fix="Use lmsysorg/sglang:latest or a nightly build",
            source=sglang_url,
        ))

    # Rule 5: error_message pattern - invalid device ordinal
    if "invalid device ordinal" in error_message.lower():
        issues.append(DiagnosticIssue(
            severity="critical",
            param="error_message",
            value="invalid device ordinal",
            problem="Driver or image mismatch. SM121A requires nightly SGLang with matching CUDA driver.",
            fix="Switch to lmsysorg/sglang:latest and verify nvidia-smi shows the GB10 GPU",
            source=sglang_url,
        ))

    # Rule 6: --rm + --restart both set - critical
    err_lower = error_message.lower()
    if "--rm" in err_lower and "--restart" in err_lower:
        issues.append(DiagnosticIssue(
            severity="critical",
            param="docker_flags",
            value="--rm + --restart",
            problem="--rm removes the container on exit, making --restart ineffective. Container will not auto-restart.",
            fix="Remove --rm. Use --restart=unless-stopped alone.",
            source=oom_url,
        ))

    # Rule 7: SGLANG_ENABLE_SPEC_V2 missing when EAGLE flags present
    if "eagle" in err_lower or "speculative" in err_lower:
        if "sglang_enable_spec_v2" not in err_lower and "spec_v2" not in err_lower:
            warnings.append(DiagnosticIssue(
                severity="warning",
                param="SGLANG_ENABLE_SPEC_V2",
                value="missing",
                problem="EAGLE speculative decoding requires SGLANG_ENABLE_SPEC_V2=True env var.",
                fix="Add -e SGLANG_ENABLE_SPEC_V2=True to your docker run command",
                source=vibe_perf_url,
            ))

    # Verdict
    if issues:
        verdict: Literal["valid", "valid_with_warnings", "invalid", "unknown"] = "invalid"
    elif warnings:
        verdict = "valid_with_warnings"
    elif not any([attention_backend, mem_fraction, cuda_graph_max_bs, image_tag, hardware, error_message]):
        verdict = "unknown"
    else:
        verdict = "valid"

    recommended_config = RecommendedConfig(
        attention_backend="triton",
        mem_fraction=0.75,
        cuda_graph_max_bs=32,
        image_tag="lmsysorg/sglang:latest",
        env={"SGLANG_ENABLE_SPEC_V2": "True"},
        max_running_requests=16,
    )

    return DiagnoseResult(
        issues=issues,
        warnings=warnings,
        recommended_config=recommended_config,
        verdict=verdict,
    )
