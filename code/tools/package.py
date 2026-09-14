"""Build the submission code.zip and verify it before it ships.

The spec requires excluding virtualenvs, node_modules, build artifacts, the
data corpus and dataset/. This adds the non-negotiable one: no credentials.
Every file is inspected for secret-shaped content before the archive is sealed,
because a leaked key in a submitted zip cannot be un-shipped.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
OUT = CODE.parent / "code.zip"

EXCLUDE_DIRS = {".orchestrate", "__pycache__", ".venv", "venv", "node_modules",
                ".pytest_cache", ".mypy_cache", ".git", "dataset", "data"}
EXCLUDE_FILES = {".env", "code.zip"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".sqlite3", ".log"}

# Secret shapes. Checked against every text file that goes in.
SECRET = re.compile(
    r"sk-[A-Za-z0-9_\-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{12,}",
    re.IGNORECASE,
)


def included() -> list[Path]:
    out = []
    for path in sorted(CODE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(CODE)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.name in EXCLUDE_FILES or path.suffix in EXCLUDE_SUFFIX:
            continue
        out.append(path)
    return out


def scan(files: list[Path]) -> list[str]:
    findings = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in SECRET.finditer(text):
            snippet = match.group(0)[:24]
            findings.append(f"{path.relative_to(CODE)}: {snippet}...")
    return findings


def main() -> int:
    files = included()
    print(f"files to include: {len(files)}")

    findings = scan(files)
    if findings:
        print("\nABORTING - secret-shaped content found:")
        for f in findings:
            print(f"  {f}")
        return 1
    print("secret scan: clean")

    # The design documents live in docs/ at the repository root, outside code/.
    required = {"main.py", "README.md", "requirements.txt",
                "evaluation/usage_report.md"}
    present = {str(p.relative_to(CODE)).replace("\\", "/") for p in files}
    missing = required - present
    if missing:
        print(f"\nABORTING - required files missing: {sorted(missing)}")
        return 1
    print("required files: all present")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, Path("code") / path.relative_to(CODE))

    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
        bad = [n for n in names if Path(n).name in EXCLUDE_FILES
               or any(p in EXCLUDE_DIRS for p in Path(n).parts)]
        assert not bad, f"excluded content leaked into the archive: {bad}"
    size = OUT.stat().st_size
    print(f"\nwrote {OUT} ({size / 1024:.0f} KB, {len(names)} entries)")
    print("verified: no .env, no caches, no dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
