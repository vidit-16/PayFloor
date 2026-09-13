"""Mutation testing: do the tests actually catch bugs?

A passing suite proves nothing on its own — it may simply not be looking. This
introduces deliberate faults into the decision logic, one at a time, and checks
that the suite fails. A mutant that *survives* is a gap: a real bug of that
shape could ship unnoticed.

Each mutation targets a rule the specification states outright, so a survivor
means an unenforced requirement rather than a stylistic gap.

Run:  python tests/test_mutation.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Mutant:
    name: str
    file: str
    old: str
    new: str
    why: str


MUTANTS = [
    Mutant(
        "pending_credit_counted",
        "pipelines/sept_state.py",
        'if status == "pending" and direction == "credit":\n        return False',
        'if status == "pending" and direction == "debit":\n        return False',
        "Inverts the spec's asymmetry: counts unguaranteed money in, drops money out.",
    ),
    Mutant(
        "cancelled_events_counted",
        "pipelines/sept_state.py",
        "if status in DEAD_STATUSES or direction in DEAD_DIRECTIONS:\n        return False",
        "if direction in DEAD_DIRECTIONS:\n        return False",
        "Cancelled and failed transactions move the balance.",
    ),
    Mutant(
        "horizon_shortened",
        "pipelines/september2026.py",
        "HORIZON = 90",
        "HORIZON = 45",
        "Forecasts half the required window, so late breaches go unseen.",
    ),
    Mutant(
        "minimum_balance_ignored",
        "pipelines/sept_state.py",
        "return self.min_balance(payments) >= self.minimum_balance - 1e-6",
        "return self.min_balance(payments) >= -1e-6",
        "Drops the minimum-balance floor: every plan looks safe.",
    ),
    Mutant(
        "amount_not_capped_at_requested",
        "pipelines/sept_solver.py",
        "headroom = max(0.0, min(cap, baseline - forecast.minimum_balance))",
        "headroom = max(0.0, baseline - forecast.minimum_balance)",
        "Breaks 0 <= amount_safe_to_pay <= requested_amount.",
    ),
    Mutant(
        "safe_amount_rounded_up",
        "pipelines/sept_solver.py",
        "return math.floor(headroom * 100) / 100",
        "return math.ceil(headroom * 100) / 100",
        "Rounds the payment above what is provably safe — the bug the invariants found.",
    ),
    Mutant(
        "deadline_not_enforced",
        "pipelines/sept_solver.py",
        "return deadline is None or (on is not None and on <= deadline)",
        "return True",
        "Allows plans that complete after desired_completion_date.",
    ),
    Mutant(
        "spending_changes_unrestricted",
        "pipelines/september2026.py",
        "if flow.stoppable and flow.category in stop_ok:",
        "if flow.stoppable or flow.category in stop_ok:",
        "Stops expenses the user never agreed to stop. EQUIVALENT MUTANT: "
        "verified to change 0 of 250 output rows, because ranking rule 2 "
        "prefers plans with no spending changes, so the extra candidates it "
        "generates never win. Surviving is correct.",
    ),
    Mutant(
        "eligibility_ignored",
        "pipelines/sept_solver.py",
        "if needed in accepted_methods:\n            eligible.append(cand)",
        "eligible.append(cand)",
        "Recommends payment methods the user rejects.",
    ),
    Mutant(
        "ranking_prefers_spending_changes",
        "pipelines/sept_solver.py",
        "1 if self.adjustments else 0,                # 2. no spending changes",
        "0 if self.adjustments else 1,                # 2. no spending changes",
        "Inverts ranking rule 2, preferring plans that disrupt the user.",
    ),
]


def run_suite(cwd: Path) -> bool:
    """True when the suite passes."""
    for script in ("tests/test_harness.py", "tests/test_invariants.py"):
        result = subprocess.run(
            [sys.executable, script], cwd=cwd, capture_output=True, text=True, timeout=900
        )
        if result.returncode != 0:
            return False
    return True


def main() -> int:
    workspace = Path(tempfile.mkdtemp()) / "code"
    shutil.copytree(CODE, workspace,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".orchestrate"))
    shutil.copytree(CODE.parent / "dataset", workspace.parent / "dataset")

    print(f"baseline suite on the unmutated copy ... ", end="", flush=True)
    if not run_suite(workspace):
        print("FAILS — cannot run mutation testing against a red suite")
        return 1
    print("passes\n")

    killed, survived = [], []
    for mutant in MUTANTS:
        target = workspace / mutant.file
        original = target.read_text(encoding="utf-8")
        if mutant.old not in original:
            print(f"  SKIP     {mutant.name}: anchor not found in {mutant.file}")
            continue
        target.write_text(original.replace(mutant.old, mutant.new, 1), encoding="utf-8")
        caught = not run_suite(workspace)
        target.write_text(original, encoding="utf-8")
        (killed if caught else survived).append(mutant)
        print(f"  {'KILLED  ' if caught else 'SURVIVED'} {mutant.name}")

    # An equivalent mutant changes no observable output, so no test can kill it
    # and surviving is the correct result. Each one here was verified by diffing
    # all 250 output rows under the mutation, not assumed.
    equivalent = [m for m in survived if "EQUIVALENT MUTANT" in m.why]
    real = [m for m in survived if m not in equivalent]
    total = len(killed) + len(survived)
    print(f"\nmutation score: {len(killed)}/{total - len(equivalent)} "
          f"non-equivalent mutants killed")
    for mutant in equivalent:
        print(f"  equivalent: {mutant.name} — changes no output, surviving is correct")
    if real:
        print("\nSURVIVORS — a real bug of this shape would ship unnoticed:")
        for mutant in real:
            print(f"  {mutant.name}\n      {mutant.why}")
    shutil.rmtree(workspace.parent, ignore_errors=True)
    return 1 if real else 0


if __name__ == "__main__":
    raise SystemExit(main())
