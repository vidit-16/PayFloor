"""Model-call layer: one interface, two backends, every call cached and metered.

Design notes for the judge / reader:

* The pipeline never asks a model for a *decision*. It asks for **structured
  facts** against a Pydantic schema, and the policy layer decides. So the only
  entry point here returns a validated model instance, never free text.
* Every call goes through `CallCache` keyed on a content hash, which is what
  makes reruns reproducible (temperature alone does not).
* Token usage is accumulated in a `UsageLedger` so the cost/latency section of
  the evaluation report is measured, not guessed.
"""

from __future__ import annotations

import base64
import mimetypes
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .cache import CallCache, content_key
from .config import Settings

T = TypeVar("T", bound=BaseModel)


class ModelRefusal(RuntimeError):
    """Raised when the provider declines to answer. Callers fall back to a
    conservative default rather than retrying forever."""


@dataclass
class UsageLedger:
    """Thread-safe running total of tokens, calls, wall time and cost."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Tokens that were actually billed this run, i.e. excluding cache hits.
    billed_input_tokens: int = 0
    billed_output_tokens: int = 0
    seconds: float = 0.0
    by_namespace: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        namespace: str,
        usage: dict[str, Any],
        elapsed: float,
        *,
        cached: bool,
    ) -> None:
        with self._lock:
            self.calls += 1
            inp = int(usage.get("input_tokens", 0) or 0)
            out = int(usage.get("output_tokens", 0) or 0)
            self.input_tokens += inp
            self.output_tokens += out
            if cached:
                self.cached_calls += 1
            else:
                self.billed_input_tokens += inp
                self.billed_output_tokens += out
            self.seconds += elapsed
            self.by_namespace[namespace] = self.by_namespace.get(namespace, 0) + 1

    def _price(self, settings: Settings, inp: int, out: int, model: str | None) -> float:
        in_price, out_price = settings.price_for(model or settings.model)
        return (inp / 1e6) * in_price + (out / 1e6) * out_price

    def cost(self, settings: Settings, model: str | None = None) -> float:
        """What this run actually cost. Cache hits are free."""
        return self._price(settings, self.billed_input_tokens, self.billed_output_tokens, model)

    def uncached_cost(self, settings: Settings, model: str | None = None) -> float:
        """What the same work would have cost with no cache — the saving."""
        return self._price(settings, self.input_tokens, self.output_tokens, model)

    def summary(self, settings: Settings) -> dict[str, Any]:
        live = self.calls - self.cached_calls
        return {
            "total_calls": self.calls,
            "served_from_cache": self.cached_calls,
            "live_api_calls": live,
            "cache_hit_rate": round(self.cached_calls / self.calls, 3) if self.calls else 0.0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "billed_input_tokens": self.billed_input_tokens,
            "billed_output_tokens": self.billed_output_tokens,
            "wall_seconds": round(self.seconds, 1),
            "billed_cost_usd": round(self.cost(settings), 4),
            "uncached_cost_usd": round(self.uncached_cost(settings), 4),
        }


# --------------------------------------------------------------------------- #
# Content blocks: a provider-neutral representation the backends translate.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    path: Path

    def encode(self) -> tuple[str, str]:
        media_type = mimetypes.guess_type(str(self.path))[0] or "image/jpeg"
        data = base64.standard_b64encode(self.path.read_bytes()).decode("utf-8")
        return media_type, data


Part = TextPart | ImagePart


class Backend(Protocol):
    """What every provider adapter must implement."""

    name: str

    def parse(
        self, *, model: str, system: str, parts: list[Part], schema: type[T], max_tokens: int
    ) -> tuple[T, dict[str, Any]]: ...


class OpenAIBackend:
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(max_retries=3)

    def parse(
        self, *, model: str, system: str, parts: list[Part], schema: type[T], max_tokens: int
    ) -> tuple[T, dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                media_type, data = part.encode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )

        completion = self._client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format=schema,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise ModelRefusal(str(message.refusal))

        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
        }
        return message.parsed, usage  # type: ignore[return-value]


def build_backend(settings: Settings) -> Backend:
    return OpenAIBackend()


class Extractor:
    """The one object the pipeline calls. Cached, metered, retried, and it
    returns validated Pydantic instances — never raw text."""

    def __init__(self, settings: Settings, cache: CallCache, ledger: UsageLedger) -> None:
        self.settings = settings
        self.cache = cache
        self.ledger = ledger
        self._backend: Backend | None = None
        self._backend_lock = threading.Lock()

    @property
    def backend(self) -> Backend:
        # Built lazily so `--dry-run` and pure-policy tests need no API key.
        if self._backend is None:
            with self._backend_lock:
                if self._backend is None:
                    self._backend = build_backend(self.settings)
        return self._backend

    def extract(
        self,
        *,
        namespace: str,
        system: str,
        parts: list[Part],
        schema: type[T],
        model: str | None = None,
    ) -> T:
        """Run one structured extraction. Identical inputs always return the
        identical object, from cache, without touching the network."""
        model = model or self.settings.model
        payload = {
            "system": system,
            "schema": schema.model_json_schema(),
            "parts": [
                {"text": p.text} if isinstance(p, TextPart) else {"image": str(p.path)}
                for p in parts
            ],
        }
        key = content_key(namespace, model, payload)

        hit = self.cache.get(key)
        if hit is not None:
            value, usage = hit
            self.ledger.record(namespace, usage, 0.0, cached=True)
            return schema.model_validate(value)

        if self.settings.dry_run:
            raise RuntimeError(
                f"dry-run: uncached call in namespace {namespace!r}. "
                "Run without ORCHESTRATE_DRY_RUN to populate the cache."
            )

        started = time.perf_counter()
        parsed, usage = self._call_with_retry(
            model=model, system=system, parts=parts, schema=schema
        )
        elapsed = time.perf_counter() - started

        self.cache.put(key, namespace, model, parsed.model_dump(), usage)
        self.ledger.record(namespace, usage, elapsed, cached=False)
        return parsed

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
        reraise=True,
    )
    def _call_with_retry(
        self, *, model: str, system: str, parts: list[Part], schema: type[T]
    ) -> tuple[T, dict[str, Any]]:
        return self.backend.parse(
            model=model,
            system=system,
            parts=parts,
            schema=schema,
            max_tokens=self.settings.max_tokens,
        )
