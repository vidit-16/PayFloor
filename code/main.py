"""CLI entry point.

    python main.py run      --dataset <dir> [--offline] [--limit N]
    python main.py score    --dataset <dir> [--offline]
    python main.py validate --dataset <dir> --predictions output.csv
    python main.py profile  --dataset <dir>

`run` produces `output.csv`, validates it against the output contract, and
writes `evaluation/evaluation_report.md`. `score` runs the same pipeline over
the labeled sample file and grades it, which is the loop you hill-climb on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from orchestrate.cache import CallCache
from orchestrate.config import load_settings
from orchestrate.context import ContextStore
from orchestrate.llm import Extractor, UsageLedger
from orchestrate.report import render_report, write_report
from orchestrate.runner import Runner
from orchestrate.score import load_csv, score_predictions
from pipelines import august2026 as pipeline


def _build(args) -> tuple:
    settings = load_settings(dataset_dir=args.dataset, output_path=getattr(args, "output", None))
    store = ContextStore(settings.dataset_dir)
    cache = CallCache(settings.cache_path)
    ledger = UsageLedger()
    extractor = None if args.offline else Extractor(settings, cache, ledger)
    engine = pipeline.build_engine()
    return settings, store, cache, ledger, extractor, engine


def cmd_run(args) -> int:
    settings, store, cache, ledger, extractor, engine = _build(args)
    spec = pipeline.SPEC

    problems = pipeline.REASONS.audit()
    if problems:
        print("reason templates outside the target word band:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)

    rows = load_csv(settings.dataset_dir / args.input)
    if args.limit:
        rows = rows[: args.limit]

    runner = Runner(
        key_column=spec.key_column,
        checkpoint_path=settings.checkpoint_dir / f"{Path(args.input).stem}.jsonl",
        max_concurrency=settings.max_concurrency,
        fallback=pipeline.fallback_row,
    )
    result = runner.run(rows, pipeline.make_processor(store, engine, extractor))

    report = spec.validate(
        [spec.coerce_row(r) for r in result.rows],
        [str(r[spec.key_column]) for r in rows],
    )
    print("\n--- output contract ---")
    print(report.render())
    if not report.ok:
        print("\nVALIDATION FAILED — fix before submitting.", file=sys.stderr)

    target = spec.write_csv(result.rows, settings.output_path)
    print(f"\nwrote {target}")

    media = store.get("images")
    write_report(
        Path(args.evaluation_dir) / "evaluation_report.md",
        render_report(
            settings=settings,
            ledger=ledger,
            cache_stats=cache.stats(),
            row_count=len(result.rows),
            media_count=len(media) if media else 0,
            rule_histogram=engine.rule_histogram(),
            failures=len(result.failures),
            notes="Offline mode: decisions came from deterministic signals only."
            if args.offline
            else "",
        ),
    )
    print(f"wrote {args.evaluation_dir}/evaluation_report.md")
    cache.close()
    return 0 if report.ok else 1


def cmd_score(args) -> int:
    settings, store, cache, ledger, extractor, engine = _build(args)
    spec = pipeline.SPEC

    gold = load_csv(settings.dataset_dir / args.sample)
    runner = Runner(
        key_column=spec.key_column,
        checkpoint_path=settings.checkpoint_dir / f"{Path(args.sample).stem}.jsonl",
        max_concurrency=settings.max_concurrency,
        fallback=pipeline.fallback_row,
        verbose=not args.quiet,
    )
    result = runner.run(gold, pipeline.make_processor(store, engine, extractor))

    report = score_predictions(
        spec,
        [spec.coerce_row(r) | {"rule": r.get("rule", "")} for r in result.rows],
        gold,
        context_columns=("message_text", "conversation_type"),
    )
    print("\n" + report.render())

    errors = report.write_errors(Path(args.evaluation_dir) / "errors.csv")
    print(f"\nwrote {errors} ({len(report.mismatches)} mismatched rows)")
    print("\nrules fired:", engine.rule_histogram())
    cache.close()
    return 0


def cmd_validate(args) -> int:
    settings = load_settings(dataset_dir=args.dataset)
    spec = pipeline.SPEC
    predictions = load_csv(args.predictions)
    expected = [str(r[spec.key_column]) for r in load_csv(settings.dataset_dir / args.input)]
    report = spec.validate(predictions, expected)
    print(report.render())
    return 0 if report.ok else 1


def cmd_profile(args) -> int:
    import subprocess

    return subprocess.call(
        [sys.executable, str(Path(__file__).parent / "tools" / "profile_dataset.py"),
         "--dataset", args.dataset]
    )


def cmd_doctor(args) -> int:
    """Preflight: are credentials present, valid, and funded? Run this first."""
    import os

    settings = load_settings(dataset_dir=args.dataset)
    key = os.environ.get("OPENAI_API_KEY", "")
    print("--- credentials ---")
    shown = f"{key[:8]}...{key[-4:]} (len {len(key)})" if key else "not set"
    print(f"  {'OPENAI_API_KEY':<20} {shown}")
    print(f"  {'OPENAI_BASE_URL':<20} {settings.base_url}")
    print(f"\n  provider label : {settings.provider}")
    print(f"  model          : {settings.model}")

    if not settings.api_key:
        print("\nFAIL: OPENAI_API_KEY is not set.")
        print("  Put it in scaffold/.env (copy .env.example) - that file is gitignored.")
        return 1

    print("\n--- auth check ---")
    try:
        from openai import OpenAI

        client = OpenAI(max_retries=1, timeout=30.0)
        ids = sorted(m.id for m in client.models.list())
        print(f"  OK - {len(ids)} models visible")
        if settings.model not in ids:
            print(f"  WARNING: configured model {settings.model!r} is not in the list.")
            print(f"  closest available: {[m for m in ids if m.startswith(settings.model[:6])][:5]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {type(exc).__name__}: {str(exc)[:200]}")
        return 1

    print("\n--- billing check ---")
    try:
        # Reasoning models spend the budget on hidden reasoning tokens before
        # emitting anything, so a 1-token ceiling 400s rather than replying.
        client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=64,
        )
        print("  OK - key is funded and the model is callable")
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        lowered = detail.lower()
        # The request was accepted and billed; it only ran out of output room.
        if "output limit" in lowered or "max_tokens" in lowered:
            print("  OK - request was accepted and billed (hit the output ceiling only)")
            print("\nAll checks passed. Safe to run the pipeline.")
            return 0
        print(f"  FAIL: {type(exc).__name__}: {detail[:220]}")
        if any(w in lowered for w in ("quota", "credit", "billing", "exceeded")):
            print("  -> key is valid but has no usable credit on it.")
        return 1

    print("\nAll checks passed. Safe to run the pipeline.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--evaluation-dir", default="evaluation")
    parser.add_argument("--offline", action="store_true",
                        help="use deterministic signals instead of model calls (no API key needed)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="produce output.csv for the full input")
    run.add_argument("--input", default="messages.csv")
    run.add_argument("--output")
    run.add_argument("--limit", type=int)
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="grade the pipeline against the labeled sample")
    score.add_argument("--sample", default="sample_messages.csv")
    score.add_argument("--quiet", action="store_true")
    score.set_defaults(func=cmd_score)

    validate = sub.add_parser("validate", help="check an existing predictions file")
    validate.add_argument("--predictions", required=True)
    validate.add_argument("--input", default="messages.csv")
    validate.set_defaults(func=cmd_validate)

    profile = sub.add_parser("profile", help="profile the dataset")
    profile.set_defaults(func=cmd_profile)

    doctor = sub.add_parser("doctor", help="check credentials are present, valid and funded")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
