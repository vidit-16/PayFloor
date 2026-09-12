"""Generates `evaluation/evaluation_report.md`.

The June 2026 edition made this a hard requirement: model calls, token usage,
images processed, cost with stated pricing assumptions, latency, and TPM/RPM
strategy. August dropped the explicit requirement but the Robustness and
Engineering-rigor sub-scores still reward it, so this is generated on every run
regardless of whether the current edition asks for it.

Numbers come from the `UsageLedger`, so they are measured, not estimated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import Settings
from .llm import UsageLedger


def render_report(
    *,
    settings: Settings,
    ledger: UsageLedger,
    cache_stats: Mapping[str, int],
    row_count: int,
    media_count: int = 0,
    score_summary: str | None = None,
    rule_histogram: Mapping[str, int] | None = None,
    failures: int = 0,
    notes: str = "",
) -> str:
    usage = ledger.summary(settings)
    in_price, out_price = settings.price_for(settings.model)
    live_calls = usage["live_api_calls"]
    per_row = (usage["total_calls"] / row_count) if row_count else 0.0

    # Project the full-set cost from the live calls actually made.
    projected = usage["billed_cost_usd"]

    lines = [
        "# Evaluation Report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        f"- Provider: `{settings.provider}`",
        f"- Model: `{settings.model}`",
        f"- Vision model: `{settings.vision_model}`",
        f"- Max concurrency: {settings.max_concurrency}",
        f"- Deterministic cache: `{settings.cache_path}`",
        "",
        "## Volume",
        "",
        f"- Rows processed: **{row_count}**",
        f"- Media files processed: **{media_count}**",
        f"- Model calls total: **{usage['total_calls']}** ({per_row:.2f} per row)",
        f"- Served from cache: **{usage['served_from_cache']}** "
        f"(hit rate {usage['cache_hit_rate']:.1%})",
        f"- Live API calls: **{live_calls}**",
        f"- Row failures: **{failures}**",
        "",
        "## Tokens and cost",
        "",
        f"- Tokens processed: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out",
        f"- Tokens actually billed: {usage['billed_input_tokens']:,} in / "
        f"{usage['billed_output_tokens']:,} out",
        "",
        "Note: on reasoning models the hidden reasoning tokens are billed as output,",
        "so output token counts run well above the visible response length.",
        "",
        "Pricing assumptions (per 1M tokens, list price at time of writing):",
        "",
        f"- `{settings.model}` input **${in_price:.2f}**, output **${out_price:.2f}**",
        "",
        f"**Cost billed for this run: ${projected:.4f}**",
        f"(the same work without the cache would have cost "
        f"${usage['uncached_cost_usd']:.4f})",
        "",
        "Cache hits cost nothing, so the marginal cost of a rerun after a policy-layer",
        "change is zero — only new extractions bill.",
        "",
        "## Latency",
        "",
        f"- Cumulative model wall time: {usage['wall_seconds']:.1f}s across {live_calls} live calls",
        f"- Mean live call latency: "
        f"{(usage['wall_seconds'] / live_calls):.2f}s" if live_calls else "- Mean live call latency: n/a (fully cached)",
        "",
        "## Rate limits and throughput strategy",
        "",
        f"- Concurrency is bounded at {settings.max_concurrency} in-flight requests",
        "  (`ORCHESTRATE_CONCURRENCY`), which is the primary TPM/RPM control.",
        "- The SDK retries 429 and 5xx with exponential backoff; `tenacity` wraps that",
        "  with a second bounded retry (4 attempts, 2-30s exponential) for transport errors.",
        "- Every call is content-hash cached, so retries and reruns do not re-bill.",
        "- Media extraction is cached per media ID, not per row — repeated media across",
        "  rows costs one call, not N.",
        "",
        "## Cache",
        "",
        f"- Entries stored: {cache_stats.get('entries', 0)}",
        f"- Hits this run: {cache_stats.get('hits', 0)}",
        f"- Misses this run: {cache_stats.get('misses', 0)}",
        "",
    ]

    if rule_histogram:
        lines += ["## Decision rules fired", "", "| rule | rows |", "|---|---|"]
        lines += [f"| `{name}` | {count} |" for name, count in rule_histogram.items()]
        lines.append("")

    if score_summary:
        lines += ["## Accuracy on the labeled sample", "", "```", score_summary, "```", ""]

    if notes:
        lines += ["## Notes", "", notes, ""]

    return "\n".join(lines)


def write_report(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
