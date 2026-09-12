"""Context assembly: the part that actually wins rows.

Every edition ships one input file plus a pile of context (other CSVs, a
markdown corpus, media manifests). The August test set contained six message
texts repeated *verbatim* across thirteen rows with different users — identical
words, different correct label. A pipeline that reads only the input row loses
those by construction. So context lookup is a first-class stage here, built
deterministically and cached in memory, never delegated to the model.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

Row = dict[str, str]


@dataclass
class Table:
    """One CSV, with lazily-built single-column indexes."""

    name: str
    rows: list[Row]
    _indexes: dict[str, dict[str, list[Row]]] = field(default_factory=dict, repr=False)

    @property
    def columns(self) -> list[str]:
        return list(self.rows[0].keys()) if self.rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def index(self, column: str) -> dict[str, list[Row]]:
        if column not in self._indexes:
            bucket: dict[str, list[Row]] = defaultdict(list)
            for row in self.rows:
                bucket[str(row.get(column, "")).strip()].append(row)
            self._indexes[column] = dict(bucket)
        return self._indexes[column]

    def find(self, column: str, value: Any) -> list[Row]:
        return self.index(column).get(str(value).strip(), [])

    def find_one(self, column: str, value: Any) -> Row | None:
        hits = self.find(column, value)
        return hits[0] if hits else None

    def where(self, **equals: Any) -> list[Row]:
        """Multi-column equality filter, driven off whichever index is cheapest."""
        if not equals:
            return list(self.rows)
        pivot, pivot_value = next(iter(equals.items()))
        candidates = self.find(pivot, pivot_value)
        rest = {k: str(v).strip() for k, v in equals.items() if k != pivot}
        if not rest:
            return candidates
        return [
            row
            for row in candidates
            if all(str(row.get(k, "")).strip() == v for k, v in rest.items())
        ]

    def distinct(self, column: str) -> Counter:
        return Counter(str(row.get(column, "")).strip() for row in self.rows)


class ContextStore:
    """Every CSV in the dataset directory, loaded once and indexed on demand."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self._tables: dict[str, Table] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(self.dataset_dir.glob("*.csv")):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = [
                    {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                    for row in csv.DictReader(handle)
                ]
            self._tables[path.stem] = Table(path.stem, rows)

    @property
    def names(self) -> list[str]:
        return sorted(self._tables)

    def table(self, name: str) -> Table:
        if name not in self._tables:
            raise KeyError(f"no table {name!r}; available: {self.names}")
        return self._tables[name]

    def get(self, name: str) -> Table | None:
        return self._tables.get(name)

    def lookup(self, table: str, column: str, value: Any) -> Row | None:
        found = self.get(table)
        return found.find_one(column, value) if found else None

    def summary(self) -> list[str]:
        return [
            f"{name:32s} rows={len(t):5d}  cols={len(t.columns):2d}  {', '.join(t.columns[:6])}"
            + (" ..." if len(t.columns) > 6 else "")
            for name, t in sorted(self._tables.items())
        ]


# --------------------------------------------------------------------------- #
# Text retrieval, for editions that ship a document corpus (May 2026 shipped 774
# markdown files). BM25 with no dependencies: deterministic, instant, and good
# enough that reaching for a vector DB in a 24h sprint is rarely the right call.
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


@dataclass
class Document:
    doc_id: str
    path: Path
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)


class BM25Index:
    """Classic BM25. Build once over the corpus, then `search()` per row."""

    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._freqs: list[Counter] = []
        self._df: Counter = Counter()
        self._lengths: list[int] = []

        for doc in self.documents:
            if not doc.tokens:
                doc.tokens = tokenize(doc.text)
            counts = Counter(doc.tokens)
            self._freqs.append(counts)
            self._lengths.append(len(doc.tokens))
            self._df.update(counts.keys())

        self.n = len(self.documents)
        self.avg_len = (sum(self._lengths) / self.n) if self.n else 0.0
        self._idf = {
            term: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    @classmethod
    def from_directory(
        cls, root: str | Path, pattern: str = "**/*.md", max_chars: int = 20_000
    ) -> "BM25Index":
        docs: list[Document] = []
        base = Path(root)
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            except OSError:
                continue
            docs.append(
                Document(doc_id=str(path.relative_to(base)).replace("\\", "/"), path=path, text=text)
            )
        return cls(docs)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        terms = tokenize(query)
        if not terms or not self.n:
            return []
        scores: list[tuple[Document, float]] = []
        for i, doc in enumerate(self.documents):
            freqs = self._freqs[i]
            length = self._lengths[i] or 1
            score = 0.0
            for term in terms:
                tf = freqs.get(term)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((doc, score))
        scores.sort(key=lambda pair: (-pair[1], pair[0].doc_id))
        return scores[:k]

    def __len__(self) -> int:
        return self.n


def find_duplicate_values(rows: Iterable[Row], column: str) -> dict[str, list[Row]]:
    """Group rows sharing an identical value in `column`.

    Run this on the input file at T+0. Any group with more than one member is a
    minimal pair: the same text needing different labels from context alone.
    """
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        value = (row.get(column) or "").strip()
        if value:
            groups[value].append(row)
    return {value: group for value, group in groups.items() if len(group) > 1}
