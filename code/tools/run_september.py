"""Run the Buy or Wait? pipeline: score on samples, or write output.csv."""
import argparse, csv, sys, collections, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipelines.september2026 import SPEC, Dataset, solve_request, fallback_row

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="../dataset")
ap.add_argument("--mode", choices=["score", "run"], default="score")
ap.add_argument("--out", default="../dataset/output.csv")
a = ap.parse_args()

ds = Dataset(a.dataset)
rows = ds.samples if a.mode == "score" else ds.requests
preds, failures = [], []
for r in rows:
    try:
        preds.append(solve_request(r, ds))
    except Exception as e:
        failures.append((r.get("request_id"), f"{type(e).__name__}: {e}"))
        preds.append(fallback_row(r, e))

print(f"rows={len(preds)}  failures={len(failures)}")
for rid, err in failures[:5]: print("  FAIL", rid, err)

if a.mode == "run":
    rep = SPEC.validate([SPEC.coerce_row(p) for p in preds], [r["request_id"] for r in rows])
    print("\n--- output contract ---"); print(rep.render())
    SPEC.write_csv(preds, a.out); print(f"\nwrote {a.out}")
    for col in ("affordability_status", "recommended_payment_method"):
        print(f"{col}: {dict(collections.Counter(p[col] for p in preds))}")
else:
    gold = {r["request_id"]: r for r in ds.samples}
    cols = [c.name for c in SPEC.columns if c.kind != "key"]
    hits = collections.Counter(); errs = []
    for p in preds:
        g = gold[p["request_id"]]
        for c in cols:
            gv, pv = str(g.get(c, "")).strip(), str(p.get(c, "")).strip()
            if c == "amount_safe_to_pay":
                try:
                    ok = abs(float(gv) - float(pv)) <= max(0.01, abs(float(gv)) * 0.01)
                except ValueError: ok = False
                errs.append(abs(float(gv) - float(pv)) / max(float(gv), 1) * 100)
            else:
                ok = gv == pv
            hits[c] += ok
    n = len(preds)
    print(f"\n{'column':<34}{'exact':>10}")
    print("-" * 46)
    for c in cols: print(f"{c:<34}{hits[c]}/{n:<4} {hits[c]/n:>6.0%}")
    print("-" * 46)
    print(f"{'MEAN':<34}{statistics.mean(hits[c]/n for c in cols):>10.1%}")
    print(f"amount_safe median err: {statistics.median(errs):.1f}%")
