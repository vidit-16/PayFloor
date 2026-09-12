"""One-command setup for the moment the challenge drops.

    python tools/bootstrap.py <repo-url-from-the-email>

Does everything mechanical so the first 10 minutes go on reading the problem,
not on shell plumbing:

1. clones the edition repo (with core.longpaths, which Windows needs)
2. copies this scaffold into the repo's `code/` directory
3. auto-detects the dataset directory and the input/sample/template files
4. carries your .env across so credentials keep working
5. prints the problem statement's key sections
6. runs the dataset profiler

Safe to re-run: it will not overwrite an existing clone without --force.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCAFFOLD = Path(__file__).resolve().parents[1]
PROJECT = SCAFFOLD.parent

# Things to copy into the edition repo. Everything else (caches, checkpoints,
# the reference clones) stays out of the submission.
COPY = ["orchestrate", "pipelines", "tools", "tests", "main.py", "requirements.txt", ".gitignore"]

DATASET_DIRNAMES = ("dataset", "data", "datasets")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kw)


def clone(url: str, dest: Path, force: bool) -> Path:
    if dest.exists():
        if not force:
            print(f"  {dest.name} already exists - reusing it (pass --force to re-clone)")
            return dest
        shutil.rmtree(dest, ignore_errors=True)
    # core.longpaths is mandatory on Windows: the May 2026 corpus had filenames
    # that blew past MAX_PATH and the clone failed halfway through checkout.
    result = run(
        ["git", "-c", "core.longpaths=true", "clone", url, str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"\nCLONE FAILED:\n{result.stderr[:800]}")
        sys.exit(1)
    return dest


def find_dataset(repo: Path) -> Path | None:
    for name in DATASET_DIRNAMES:
        candidate = repo / name
        if candidate.is_dir():
            return candidate
    # Fall back to any directory holding CSVs.
    for candidate in sorted(repo.iterdir()):
        if candidate.is_dir() and not candidate.name.startswith(".") and list(candidate.glob("*.csv")):
            return candidate
    return None


def install_scaffold(repo: Path) -> Path:
    target = repo / "code"
    target.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        src = SCAFFOLD / name
        if not src.exists():
            continue
        dst = target / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    env = SCAFFOLD / ".env"
    if env.exists():
        shutil.copy2(env, target / ".env")
        print("  copied .env (credentials carried over)")
    return target


def show_problem(repo: Path) -> None:
    for name in ("problem_statement.md", "README.md"):
        path = repo / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"\n{'=' * 78}\n{name}  ({len(text.splitlines())} lines)\n{'=' * 78}")
        print(text[:4000])
        if len(text) > 4000:
            print(f"\n[... {len(text) - 4000} more chars - read the full file ...]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="git URL of the edition repo (from the launch email)")
    parser.add_argument("--force", action="store_true", help="re-clone even if it exists")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    name = args.url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = PROJECT / name

    print(f"\n[1/5] cloning {name}")
    repo = clone(args.url, dest, args.force)

    print(f"\n[2/5] locating dataset")
    dataset = find_dataset(repo)
    if dataset is None:
        print("  WARNING: no dataset directory found - check the repo layout manually")
    else:
        csvs = sorted(p.name for p in dataset.glob("*.csv"))
        media = [d.name for d in dataset.iterdir() if d.is_dir()]
        print(f"  {dataset.relative_to(PROJECT)}  ({len(csvs)} csv, dirs: {media or 'none'})")
        for csv_name in csvs:
            print(f"    {csv_name}")

    print(f"\n[3/5] installing scaffold into {name}/code/")
    code = install_scaffold(repo)
    print(f"  {code.relative_to(PROJECT)}")

    print(f"\n[4/5] problem statement")
    show_problem(repo)

    rel = f"../{dataset.name}" if dataset else "../dataset"
    if not args.skip_profile and dataset:
        print(f"\n[5/5] profiling dataset")
        run([sys.executable, str(code / "tools" / "profile_dataset.py"), "--dataset", str(dataset)])
    else:
        print("\n[5/5] skipped profiling")

    print(f"""
{'=' * 78}
READY. Next steps:
{'=' * 78}

  cd {code}
  python main.py --dataset {rel} doctor

Then, in order:
  1. Read {name}/problem_statement.md and AGENTS.md IN FULL
  2. Complete the AGENTS.md onboarding gate (reply 'I agree') - starts the transcript
  3. Confirm %USERPROFILE%\\hackerrank_orchestrate_september26\\log.txt is writing
  4. Write the OutputSpec in orchestrate/schema.py from the profiler output above
  5. cp pipelines/august2026.py pipelines/september2026.py, point main.py at it
  6. Get a majority-label baseline emitting a VALID output.csv  <- by 19:30

  python main.py --dataset {rel} --offline run
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
