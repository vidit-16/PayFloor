"""Reusable harness for HackerRank Orchestrate-style agent challenges.

Pipeline shape:

    load input + context  ->  structured extraction (model)  ->  deterministic
    policy  ->  schema validation  ->  output.csv  +  evaluation report

The model is asked only to *describe*; the policy layer *decides*.
"""

from .cache import CallCache, content_key
from .config import Settings, load_settings
from .context import BM25Index, ContextStore, Table, find_duplicate_values
from .llm import Extractor, ImagePart, TextPart, UsageLedger
from .policy import (
    Decision,
    PolicyEngine,
    ReasonLibrary,
    Rule,
    looks_like_chain_forward,
    looks_like_injection,
    looks_like_risk,
    matched_signals,
)
from .report import render_report, write_report
from .runner import Checkpoint, Runner, RunResult
from .schema import Column, OutputSpec, ValidationReport
from .score import ScoreReport, load_csv, score_predictions

__all__ = [
    "BM25Index",
    "CallCache",
    "Checkpoint",
    "Column",
    "ContextStore",
    "Decision",
    "Extractor",
    "ImagePart",
    "OutputSpec",
    "PolicyEngine",
    "ReasonLibrary",
    "Rule",
    "RunResult",
    "Runner",
    "ScoreReport",
    "Settings",
    "Table",
    "TextPart",
    "UsageLedger",
    "ValidationReport",
    "content_key",
    "find_duplicate_values",
    "load_csv",
    "load_settings",
    "looks_like_chain_forward",
    "looks_like_injection",
    "looks_like_risk",
    "matched_signals",
    "render_report",
    "score_predictions",
    "write_report",
]
