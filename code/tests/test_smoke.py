"""Smoke tests and mocked model-layer tests. No API key, no network.

Run:  python -m pytest code/tests/test_smoke.py   (from the repo root)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import main  # noqa: E402
from orchestrate.cache import CallCache  # noqa: E402
from orchestrate.config import load_settings  # noqa: E402
from orchestrate.llm import Extractor, TextPart, UsageLedger  # noqa: E402


class Fact(BaseModel):
    amount: float


class FakeBackend:
    """Stands in for the OpenAI client and counts how often it is reached."""

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, *, model, system, parts, schema, max_tokens):
        self.calls += 1
        return schema(amount=42.0), {"input_tokens": 10, "output_tokens": 2}


def _extractor(tmp: Path, monkeypatch, *, dry_run: bool = False) -> tuple[Extractor, CallCache]:
    monkeypatch.setenv("ORCHESTRATE_DRY_RUN", "1" if dry_run else "0")
    settings = load_settings(dataset_dir=tmp, work_dir=tmp)
    cache = CallCache(tmp / "c.sqlite3")
    return Extractor(settings, cache, UsageLedger()), cache


def test_cli_boots_and_lists_commands():
    result = subprocess.run([sys.executable, str(CODE / "main.py"), "--help"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    for command in ("run", "verify", "extract"):
        assert command in result.stdout


def test_extract_without_key_fails_fast_with_a_clear_message():
    assert main.missing_key_message({}) is not None
    assert "OPENAI_API_KEY" in main.missing_key_message({"OPENAI_API_KEY": "  "})
    assert main.missing_key_message({"OPENAI_API_KEY": "sk-test"}) is None
    assert main.missing_key_message({"ORCHESTRATE_DRY_RUN": "1"}) is None


def test_mocked_backend_is_called_once_then_served_from_cache(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    extractor, cache = _extractor(tmp, monkeypatch)
    fake = FakeBackend()
    extractor._backend = fake
    with cache:
        first = extractor.extract(namespace="t", system="s", parts=[TextPart("x")], schema=Fact)
        second = extractor.extract(namespace="t", system="s", parts=[TextPart("x")], schema=Fact)
    assert first == second == Fact(amount=42.0)
    assert fake.calls == 1
    assert extractor.ledger.cached_calls == 1


def test_dry_run_refuses_uncached_calls_without_touching_the_backend(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    extractor, cache = _extractor(tmp, monkeypatch, dry_run=True)
    fake = FakeBackend()
    extractor._backend = fake
    with cache, pytest.raises(RuntimeError, match="dry-run"):
        extractor.extract(namespace="t", system="s", parts=[TextPart("y")], schema=Fact)
    assert fake.calls == 0


@pytest.mark.parametrize("value", ["six", "0", "-3"])
def test_bad_integer_settings_name_the_variable(monkeypatch, value):
    monkeypatch.setenv("ORCHESTRATE_CONCURRENCY", value)
    with pytest.raises(ValueError, match="ORCHESTRATE_CONCURRENCY"):
        load_settings(dataset_dir=tempfile.mkdtemp())
