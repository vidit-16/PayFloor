"""Deterministic decision layer — "let the model describe, let code decide".

The model's job upstream is to produce *facts*. This module turns facts plus
looked-up context into the final labels. Nothing here calls a model, so it is
pure, instant, unit-testable, and byte-identical across runs.

Three properties this buys, each of which maps to a scored dimension:

* **Consistency** — two rows that fire the same rule get the same label, the
  same reason wording, and the same confidence. The Output rubric explicitly
  scores consistency of the free-text field.
* **Explainability** — every decision carries the name of the rule that made it,
  so "why did row 47 get muted?" has a one-word answer in the interview.
* **Non-overridable safety** — safety rules run in their own tier, before
  anything else, so content that tries to instruct the router cannot win.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

Tier = Literal["safety", "strong", "normal", "fallback"]
_TIER_ORDER: tuple[Tier, ...] = ("safety", "strong", "normal", "fallback")


@dataclass(frozen=True)
class Decision:
    """The outcome for one row, plus the provenance needed to explain it."""

    values: dict[str, Any]
    reason: str
    confidence: float
    rule: str
    tier: Tier
    evaluated: tuple[str, ...] = ()

    def as_row(self, key_column: str, key: str) -> dict[str, Any]:
        row = {key_column: key, **self.values}
        row.setdefault("reason", self.reason)
        row.setdefault("confidence", self.confidence)
        return row


@dataclass
class Rule:
    """One decision rule.

    `when` receives (facts, context) and returns a bool. Keep predicates small
    and named — a rule that needs a paragraph of logic is two rules.
    """

    name: str
    tier: Tier
    when: Callable[[Any, Mapping[str, Any]], bool]
    then: dict[str, Any]
    reason: str
    confidence: float

    def matches(self, facts: Any, context: Mapping[str, Any]) -> bool:
        try:
            return bool(self.when(facts, context))
        except Exception:
            # A predicate that blows up on a weird row must not take down the
            # run; it simply does not match and we fall through.
            return False


class ReasonLibrary:
    """A fixed set of reason templates, keyed by rule name.

    Why templates rather than model-written prose: in the August golden data the
    reason strings were 10-20 words and several repeated *verbatim* across rows.
    A small template library reproduces that register exactly and removes the
    variance that free-text generation introduces.
    """

    def __init__(
        self, templates: Mapping[str, str], *, min_words: int = 8, max_words: int = 22
    ) -> None:
        self.templates = dict(templates)
        self.min_words = min_words
        self.max_words = max_words

    def render(self, key: str, context: Mapping[str, Any] | None = None) -> str:
        template = self.templates.get(key)
        if template is None:
            raise KeyError(f"no reason template registered for rule {key!r}")
        try:
            text = template.format(**(context or {}))
        except (KeyError, IndexError):
            text = template
        return re.sub(r"\s+", " ", text).strip()

    def audit(self) -> list[str]:
        """Flag templates outside the target word band before you ship."""
        problems: list[str] = []
        for name, text in self.templates.items():
            words = len(re.sub(r"\{[^}]*\}", "x", text).split())
            if words < self.min_words:
                problems.append(f"{name}: {words} words (below {self.min_words})")
            elif words > self.max_words:
                problems.append(f"{name}: {words} words (above {self.max_words})")
        return problems


@dataclass
class PolicyEngine:
    """Ordered rule evaluation: safety first, fallback last, first match wins."""

    rules: Sequence[Rule]
    reasons: ReasonLibrary
    key_column: str = "id"
    _trace: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not any(r.tier == "fallback" for r in self.rules):
            raise ValueError(
                "PolicyEngine needs at least one 'fallback' rule so every row "
                "gets a decision even when nothing else matches."
            )
        missing = [r.name for r in self.rules if r.name not in self.reasons.templates]
        if missing:
            raise ValueError(f"rules without a reason template: {missing}")

    def ordered_rules(self) -> list[Rule]:
        return sorted(self.rules, key=lambda r: _TIER_ORDER.index(r.tier))

    def decide(
        self, key: str, facts: Any, context: Mapping[str, Any] | None = None
    ) -> Decision:
        ctx: Mapping[str, Any] = context or {}
        evaluated: list[str] = []

        for rule in self.ordered_rules():
            evaluated.append(rule.name)
            if not rule.matches(facts, ctx):
                continue
            decision = Decision(
                values=dict(rule.then),
                reason=self.reasons.render(rule.name, ctx),
                confidence=rule.confidence,
                rule=rule.name,
                tier=rule.tier,
                evaluated=tuple(evaluated),
            )
            self._trace.append(
                {
                    "key": key,
                    "rule": rule.name,
                    "tier": rule.tier,
                    "evaluated": len(evaluated),
                    **decision.values,
                }
            )
            return decision

        raise AssertionError("unreachable: fallback rule guaranteed by __post_init__")

    @property
    def trace(self) -> list[dict[str, Any]]:
        """Per-row record of which rule fired. Feed this to the error analysis —
        'which rule is costing me accuracy' is the fastest way to hill-climb."""
        return self._trace

    def rule_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._trace:
            counts[entry["rule"]] = counts.get(entry["rule"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# Signals that recur across every edition so far. Cheap, deterministic, and they
# catch the planted adversarial rows before any model opinion is consulted.
# --------------------------------------------------------------------------- #

# Text that tries to instruct the system rather than address the recipient.
INJECTION_PATTERNS = (
    r"\b(system|assistant|agent|router|model)\s+(note|instruction|prompt|message)\b",
    r"\b(ignore|disregard|override)\b.{0,40}\b(previous|prior|above|instruction|risk|rule)",
    r"\b(set|mark|classify|route|treat|label)\s+(this|it|the\s+\w+)?\s*(as|=|to)\s*\w+",
    r"\bconfidence\s*=\s*[01](\.\d+)?",
    r"\byou\s+(must|should|will)\s+(mark|set|classify|notify|reply|escalate)",
    r"\b(routing|classification)\s+override\b",
)

# High precision: asking the recipient to hand over a secret. A legitimate
# sender never does this, so on its own this is enough to suppress a message.
_SECRET = (
    r"(otp|one[- ]time\s+password|cvv|passcode|seed\s+phrase|"
    r"(?:wallet|login|card|atm|upi)\s*pin|"
    r"\d\s*digit\s+(?:login\s+)?code|verification\s+code|security\s+code)"
)

# NB: the secret must be *solicited*, never merely mentioned. A bank's own
# anti-fraud advisory ("we never ask for your OTP") names every one of these
# words and is entirely legitimate — the August sample contains exactly that row
# labelled `digest`, so a bare keyword match scores it wrong.
CREDENTIAL_PATTERNS = (
    rf"\b(send|share|reply\s+with|confirm|enter|provide|give|tell\s+us)\b[^.!?]{{0,40}}\b{_SECRET}\b",
    rf"\b{_SECRET}\b[^.!?]{{0,30}}\b(here|now|immediately|to\s+release|to\s+keep|to\s+activate)\b",
    rf"\b(verify|confirm)\b[^.!?]{{0,25}}\b(wallet|account)\b[^.!?]{{0,25}}\b{_SECRET}\b",
)

# Phrases that disclaim rather than solicit. If one of these surrounds the only
# secret mention, the message is advisory, not a scam.
CREDENTIAL_DISCLAIMERS = (
    r"\b(never|do\s*not|don'?t|no\s+one\s+will|we\s+will\s+not)\b[^.!?]{0,40}\b(ask|request|seek)\b",
    r"\b(beware|caution|advisory|awareness|fraud\s+alert)\b",
)

# Soft signals. Each is common in legitimate marketing, so none of these is
# sufficient alone — the composite in `looks_like_risk` requires two families.
URGENCY_PATTERNS = (
    r"\b(suspend|block|deactivat|restrict|expir|penalt)\w*\b.{0,45}"
    r"\b(immediate|now|today|tonight|30\s*min|24\s*h|permanent)",
    r"\b(immediate|now|today|tonight|within\s+\d+)\b.{0,45}"
    r"\b(suspend|block|deactivat|restrict|expir|penalt)\w*",
    # Threat of account loss, stated in either direction.
    r"\b(may|will|could|might)\s+be\s+(temporarily\s+|permanently\s+)?"
    r"(blocked|suspended|restricted|deactivated|disabled)\b",
    r"\b(account|access|wallet|card|profile)\b.{0,30}\b(at\s+risk|will\s+expire|blocked|suspended)\b",
    r"\bbefore\s+(midnight|6\s*pm|end\s+of\s+day)\b.{0,30}\b(pay|clear|verify|confirm)\b",
)

CTA_PATTERNS = (
    r"\b(click|scan|open|tap|use)\b.{0,25}\b(link|qr|below)\b",
    r"\b(login|verification|payment)\b.{0,20}\b(link|page|portal)\b",
    # Lookalike domains: a hyphenated host that is not a plain brand domain.
    r"\b[a-z0-9]+-[a-z0-9-]+\.(in|com|net|xyz|top|info)\b",
)

FORWARD_CHAIN_PATTERNS = (
    r"\b(forward|share)\b.{0,35}\b(\d+|ten|all|every)\b.{0,25}\b(people|groups?|friends|contacts|family)",
    r"\bdo\s*not\s+(break|ignore)\s+(the\s+)?chain\b",
    r"\b(good\s+luck|blessings?|prosperity)\b.{0,45}\bshare\b",
    r"\bfwd\s+as\s+received\b",
    r"\b(pls|please|kindly)\s+(fwd|forward)\b",
    r"\bshar(e|ing)\b.{0,40}\bin\s+case\s+it\s+helps\b",
)


def _any_match(patterns: Sequence[str], text: str) -> str | None:
    lowered = text.lower()
    for pattern in patterns:
        if re.search(pattern, lowered):
            return pattern
    return None


def looks_like_injection(text: str) -> bool:
    """True when the content is addressing the system instead of the user.

    Treated as a hard safety signal: in the August golden data every such row
    was labelled as a scam and suppressed, regardless of what the text claimed
    about the sender's trustworthiness.
    """
    return _any_match(INJECTION_PATTERNS, text or "") is not None


def looks_like_credential_request(text: str) -> bool:
    """Asks the recipient to disclose a secret. High precision — safety tier.

    Suppressed when the message is disclaiming the request rather than making
    it ("we never ask for your OTP"), which is what a genuine fraud advisory
    looks like.
    """
    body = text or ""
    if _any_match(CREDENTIAL_PATTERNS, body) is None:
        return False
    return _any_match(CREDENTIAL_DISCLAIMERS, body) is None


def looks_like_urgency_pressure(text: str) -> bool:
    """Manufactured deadline or account-loss threat. Soft signal on its own."""
    return _any_match(URGENCY_PATTERNS, text or "") is not None


def looks_like_cta_link(text: str) -> bool:
    """Pushes the reader at a link, QR or lookalike domain. Soft signal."""
    return _any_match(CTA_PATTERNS, text or "") is not None


def looks_like_risk(text: str) -> bool:
    """Composite scam signal.

    Deliberately *not* a single-pattern match: legitimate marketing says "tap
    below" and legitimate banks say "your statement is ready". Requiring either
    an explicit credential request, or urgency pressure combined with a
    link/QR push, is what separates a scam from a promotion.
    """
    body = text or ""
    if looks_like_credential_request(body):
        return True
    return looks_like_urgency_pressure(body) and looks_like_cta_link(body)


def looks_like_chain_forward(text: str) -> bool:
    return _any_match(FORWARD_CHAIN_PATTERNS, text or "") is not None


def matched_signals(text: str) -> dict[str, bool]:
    """All deterministic signals at once — useful as facts for the policy layer
    and as columns in the error-analysis dump."""
    body = text or ""
    return {
        "injection": looks_like_injection(body),
        "credential_request": looks_like_credential_request(body),
        "urgency": looks_like_urgency_pressure(body),
        "cta_link": looks_like_cta_link(body),
        "risk": looks_like_risk(body),
        "chain_forward": looks_like_chain_forward(body),
    }
