"""Runtime configuration, resolved from environment variables only.

No secret is ever read from a file tracked in git; see `.env.example` for the
variables this module expects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load `.env` from the project root if present, so keys live in a gitignored
# file rather than a shell profile.
#
# override=True is deliberate: the project-local .env is an explicit statement of
# intent for this run, and a stale key left in the shell or Windows user
# environment should not silently win over it. That failure mode is near
# impossible to diagnose under time pressure.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
except ImportError:  # dotenv is optional
    pass

# Per-million-token prices, used by the cost estimator in `report.py`.
# Keep in sync with the pricing page; these are assumptions, not quotes.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def _resolve_dataset(given: Path) -> Path:
    """Find the dataset directory even when run from a different cwd.

    The scaffold is installed at `<edition-repo>/code/`, so the dataset sits at
    `../dataset` from there but at `./dataset` from the repo root. Rather than
    making that a thing to remember at 18:00, check the obvious places.
    """
    if given.is_dir():
        return given
    for candidate in (
        given,
        Path("dataset"), Path("data"),
        Path("..") / "dataset", Path("..") / "data",
        Path(__file__).resolve().parents[2] / "dataset",
    ):
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate
    return given  # let the caller fail with a clear path in the message


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs to run, in one immutable object."""

    provider: str
    model: str
    vision_model: str
    max_concurrency: int
    max_tokens: int
    cache_path: Path
    checkpoint_dir: Path
    dataset_dir: Path
    output_path: Path
    dry_run: bool
    seed: int
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    @property
    def base_url(self) -> str:
        return os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    def price_for(self, model: str) -> tuple[float, float]:
        return PRICING.get(model, (0.0, 0.0))


def load_settings(
    *,
    dataset_dir: str | Path = "dataset",
    output_path: str | Path | None = None,
    work_dir: str | Path = ".orchestrate",
) -> Settings:
    """Build `Settings` from the environment, with sane defaults for a sprint.

    The model layer speaks the OpenAI API. Any OpenAI-compatible endpoint works
    unchanged by setting `OPENAI_BASE_URL` — that covers Gemini's compatibility
    layer, Groq, Together, OpenRouter and a local Ollama/LM Studio server, so a
    quota problem mid-sprint is an env-var change rather than a code change.
    `ORCHESTRATE_PROVIDER` is a free-text label for the run report only.
    """
    provider = os.environ.get("ORCHESTRATE_PROVIDER", "").strip().lower() or "openai"
    defaults = ("gpt-5-mini", "gpt-5-mini")

    work = Path(work_dir)
    dataset = _resolve_dataset(Path(dataset_dir))

    return Settings(
        provider=provider,  # type: ignore[arg-type]
        model=os.environ.get("ORCHESTRATE_MODEL", defaults[0]),
        vision_model=os.environ.get("ORCHESTRATE_VISION_MODEL", defaults[1]),
        max_concurrency=int(os.environ.get("ORCHESTRATE_CONCURRENCY", "6")),
        max_tokens=int(os.environ.get("ORCHESTRATE_MAX_TOKENS", "2000")),
        cache_path=work / "cache.sqlite3",
        checkpoint_dir=work / "checkpoints",
        dataset_dir=dataset,
        output_path=Path(output_path) if output_path else dataset / "output.csv",
        dry_run=_env_flag("ORCHESTRATE_DRY_RUN"),
        seed=int(os.environ.get("ORCHESTRATE_SEED", "0")),
    )
