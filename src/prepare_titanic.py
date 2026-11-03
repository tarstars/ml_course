"""Prepare the Titanic data asset for the classification metrics lab.

Command line (completed in a later task):

    uv run python src/prepare_titanic.py --train train.csv --test test.csv \
        --titanic3 titanic3.csv --out <output.js>

The Kaggle test file has no labels; they are recovered from the public
titanic3 dataset: https://hbiostat.org/data/repo/titanic3.csv
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

KNOWN_TITLES = {"Mr", "Mrs", "Miss", "Master"}
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def normalize(text: str) -> str:
    """Drop double quotes, collapse whitespace, strip and lower-case."""
    return re.sub(r"\s+", " ", text.replace('"', "")).strip().lower()


def title_of(name: str) -> str:
    """Return Mr, Mrs, Miss, Master or Rare from the token after the first comma."""
    match = re.search(r",\s*([^.]*)\.", name)
    token = match.group(1).strip() if match else ""
    token = TITLE_ALIASES.get(token, token)
    return token if token in KNOWN_TITLES else "Rare"


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def match_key(row: dict, kaggle: bool) -> tuple[str, str, str, str]:
    """Key (name, ticket, pclass, sex) for matching Kaggle rows to titanic3 rows."""
    name, ticket, pclass, sex = (
        ("Name", "Ticket", "Pclass", "Sex") if kaggle else ("name", "ticket", "pclass", "sex")
    )
    return normalize(row[name]), normalize(row[ticket]), row[pclass].strip(), normalize(row[sex])


def _lookups(titanic3: list[dict]):
    by_key = {match_key(row, kaggle=False): int(row["survived"]) for row in titanic3}
    name_counts = Counter(normalize(row["name"]) for row in titanic3)
    by_name = {normalize(row["name"]): int(row["survived"])
               for row in titanic3 if name_counts[normalize(row["name"])] == 1}
    return by_key, by_name


def _label_of(row: dict, by_key: dict, by_name: dict) -> int:
    key = match_key(row, kaggle=True)
    if key in by_key:
        return by_key[key]
    if key[0] in by_name:
        return by_name[key[0]]
    raise ValueError(f"unmatched: {row['Name']}")


def recover_labels(rows: list[dict], titanic3: list[dict]) -> list[int]:
    """Survival labels for Kaggle rows: full key first, then unique normalized name."""
    by_key, by_name = _lookups(titanic3)
    return [_label_of(row, by_key, by_name) for row in rows]


def verify_train_labels(rows: list[dict], titanic3: list[dict]) -> None:
    """Raise ValueError naming the first passenger whose recovered label differs from Survived."""
    by_key, by_name = _lookups(titanic3)
    for row in rows:
        if _label_of(row, by_key, by_name) != int(row["Survived"]):
            raise ValueError(f"label disagrees with Survived: {row['Name']}")


def main(argv: list[str]) -> int:
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
