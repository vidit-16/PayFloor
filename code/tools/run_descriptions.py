"""Classify every distinct financial-event description once.

The description carries semantics the structured columns do not: "Final employer
payroll" means the income stops, "Payroll before leave" explains a gap, "Card
authorization" is superseded by its settlement. Grouping purely on category
silently discards this, which is what made the forecast wrong for any user whose
circumstances changed. 164 distinct strings -> 164 cached calls.
"""
import csv, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from orchestrate.cache import CallCache
from orchestrate.config import load_settings
from orchestrate.llm import Extractor, TextPart, UsageLedger


class DescriptionFacts(BaseModel):
    continuity: str = Field(description=(
        "One of: continues (a routine flow expected to keep repeating), "
        "final (explicitly the last occurrence - employment ending, a closing or "
        "final payment, a plan being terminated), "
        "resumes (a flow restarting after a pause), "
        "one_off (a single non-repeating event), "
        "superseded (a provisional record replaced by a settled one, e.g. an "
        "authorization or a reversal)."))
    is_commitment: bool = Field(description=(
        "True for a regular obligation or income the user can rely on (rent, "
        "salary, loan instalment, subscription). False for discretionary or "
        "incidental spending."))
    note: str = Field(description="Short clause explaining the classification.")


SYSTEM = """\
You classify a single bank-transaction description for a cash-flow forecaster.

Decide only what the wording implies about whether this flow will keep occurring.
Wording like "final", "last", "closing", "ends" means it will not repeat.
Wording about leave, pauses or returning implies an interruption, not an end.
An "authorization", "pending" or "reversed" record is superseded by its settled
counterpart. Do not infer anything the words do not support.\
"""

settings = load_settings(dataset_dir="../dataset")
cache = CallCache(settings.cache_path); ledger = UsageLedger()
ex = Extractor(settings, cache, ledger)
rows = list(csv.DictReader(open("../dataset/financial_events.csv", newline="", encoding="utf-8")))
descs = sorted({r["description"].strip() for r in rows if r["description"].strip()})
print(f"distinct descriptions: {len(descs)}")

out, errs = {}, []
def work(d):
    return d, ex.extract(namespace="description_facts", system=SYSTEM,
                         parts=[TextPart(f"Transaction description: {d}")],
                         schema=DescriptionFacts)

t0 = time.time()
with ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
    for fut in as_completed([pool.submit(work, d) for d in descs]):
        try:
            d, r = fut.result(); out[d] = r.model_dump()
        except Exception as e:
            errs.append(str(e))
Path("extracted").mkdir(exist_ok=True)
Path("extracted/descriptions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"classified {len(out)}/{len(descs)} errors={len(errs)} in {time.time()-t0:.0f}s")
import collections
print("continuity:", dict(collections.Counter(v["continuity"] for v in out.values())))
print("\n--- the decisive ones ---")
for d, v in sorted(out.items()):
    if v["continuity"] in ("final", "resumes", "superseded"):
        print(f"  [{v['continuity']:<11}] {d}")
print(json.dumps(ledger.summary(settings), indent=1))
