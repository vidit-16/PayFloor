import argparse, csv, sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrate.cache import CallCache
from orchestrate.config import load_settings
from orchestrate.llm import Extractor, UsageLedger
from pipelines.sept_extract import extract_message, extract_image

D = Path("../dataset")
L = lambda f: list(csv.DictReader(open(D / f, newline="", encoding="utf-8")))
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default="extracted",
                 help="directory for the extraction JSON (use a separate one to A/B models)")
_args = _ap.parse_args()
OUT = Path(_args.out)

settings = load_settings(dataset_dir="../dataset")
cache = CallCache(settings.cache_path); ledger = UsageLedger()
ex = Extractor(settings, cache, ledger)

msgs = L("messages.csv"); imgs = L("images.csv")
media = D / "media" / "images"
out_m, out_i, errs = {}, {}, []
t0 = time.time()

with ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
    fm = {pool.submit(extract_message, m, ex): m["message_id"] for m in msgs}
    fi = {pool.submit(extract_image, i, media, ex): i["image_id"] for i in imgs}
    for fut in as_completed(list(fm) + list(fi)):
        key = fm.get(fut) or fi.get(fut)
        try:
            r = fut.result()
            if r is None: continue
            (out_m if key in fm.values() else out_i)[key] = r.model_dump()
        except Exception as e:
            errs.append((key, f"{type(e).__name__}: {e}"))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "messages.json").write_text(json.dumps(out_m, indent=1), encoding="utf-8")
(OUT / "images.json").write_text(json.dumps(out_i, indent=1), encoding="utf-8")
print(f"messages={len(out_m)}/{len(msgs)}  images={len(out_i)}/{len(imgs)}  errors={len(errs)}  {time.time()-t0:.0f}s")
for k, e in errs[:5]: print("  ERR", k, e)
print(json.dumps(ledger.summary(settings), indent=1))
