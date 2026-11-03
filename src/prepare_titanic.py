"""Prepare the Titanic data asset for the classification metrics lab.

Command line (completed in a later task):

    uv run python src/prepare_titanic.py --train train.csv --test test.csv \
        --titanic3 titanic3.csv --out <output.js>

The Kaggle test file has no labels; they are recovered from the public
titanic3 dataset: https://hbiostat.org/data/repo/titanic3.csv
"""
from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
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


def _group(gid, en, ru, columns, kind, default_on, note_en, note_ru):
    return {"id": gid, "label": {"en": en, "ru": ru}, "columns": columns, "kind": kind,
            "default_on": default_on, "note": {"en": note_en, "ru": note_ru}}


GROUPS = [
    _group("pclass", "Passenger class", "Класс", ["pclass_1", "pclass_2"], "binary", True,
           "Two indicators; third class is the reference.", "Два индикатора; третий класс — базовый."),
    _group("sex", "Sex", "Пол", ["female"], "binary", True,
           "One indicator for female; male is the reference.", "Один индикатор для женщин; мужчины — базовая категория."),
    _group("age", "Age", "Возраст", ["age"], "standardized", True,
           "Age in years, missing values filled with the median age of the passenger's title, then standardized.",
           "Возраст в годах; пропуски заполнены медианой по обращению, затем стандартизация."),
    _group("age_missing", "Age was filled", "Возраст указан", ["age_missing"], "binary", True,
           "Equals 1 when the age was missing in the source data.", "Равен 1, если возраст в исходных данных отсутствовал."),
    _group("sibsp", "Siblings and spouses", "Братья, сёстры, супруги", ["sibsp"], "standardized", True,
           "Number of siblings and spouses aboard, standardized.", "Число братьев, сестёр и супругов на борту, стандартизовано."),
    _group("parch", "Parents and children", "Родители и дети", ["parch"], "standardized", True,
           "Number of parents and children aboard, standardized.", "Число родителей и детей на борту, стандартизовано."),
    _group("fare", "Fare", "Стоимость билета", ["log_fare"], "standardized", True,
           "Natural log of one plus the fare, standardized.", "Натуральный логарифм от единицы плюс стоимость билета, стандартизовано."),
    _group("fare_missing", "Fare was filled", "Стоимость указана", ["fare_missing"], "binary", False,
           "Equals 1 when the fare was missing; it never happens in the training data.",
           "Равен 1, если стоимость билета отсутствовала; в обучающих данных такого нет."),
    _group("embarked", "Port of embarkation", "Порт посадки", ["embarked_c", "embarked_q"], "binary", True,
           "Two indicators for Cherbourg and Queenstown; Southampton is the reference.",
           "Два индикатора: Шербур и Квинстаун; Саутгемптон — базовый порт."),
    _group("embarked_missing", "Port was filled", "Порт указан", ["embarked_missing"], "binary", False,
           "Equals 1 when the port was missing; only two training passengers have it.",
           "Равен 1, если порт отсутствовал; таких пассажиров в обучении только двое."),
    _group("title", "Title from name", "Обращение в имени", ["title_mrs", "title_miss", "title_master", "title_rare"],
           "binary", True, "Four indicators; Mr is the reference.", "Четыре индикатора; обращение Mr — базовое."),
    _group("family_size", "Family size", "Размер семьи", ["family_size"], "standardized", False,
           "Siblings and spouses plus parents and children plus the passenger, standardized.",
           "Братья, сёстры, супруги плюс родители и дети плюс сам пассажир, стандартизовано."),
    _group("is_alone", "Travelling alone", "Путешествует один", ["is_alone"], "binary", False,
           "Equals 1 when family size is 1.", "Равен 1, если размер семьи равен 1."),
    _group("cabin_known", "Cabin was filled", "Каюта указана", ["cabin_known"], "binary", True,
           "Equals 1 when a cabin number is recorded.", "Равен 1, если номер каюты указан."),
]
COLUMNS = [column for group in GROUPS for column in group["columns"]]
STANDARDIZED = ("age", "sibsp", "parch", "log_fare", "family_size")


@dataclass
class TrainingStats:
    age_by_title: dict[str, float]
    fare_by_class: dict[str, float]
    embarked_mode: str
    scaling: dict[str, tuple[float, float]]  # column -> (mean, population std)


def _imputed(row: dict, stats: TrainingStats) -> tuple[float, float, str]:
    age = float(row["Age"]) if row["Age"] != "" else stats.age_by_title[title_of(row["Name"])]
    fare = float(row["Fare"]) if row["Fare"] != "" else stats.fare_by_class[row["Pclass"]]
    return age, fare, row["Embarked"] or stats.embarked_mode


def raw_features(row: dict, stats: TrainingStats) -> dict[str, float]:
    """The 19 unscaled feature values of one Kaggle row, with training-set imputation."""
    age, fare, port = _imputed(row, stats)
    title = title_of(row["Name"])
    sibsp, parch = int(row["SibSp"]), int(row["Parch"])
    family_size = sibsp + parch + 1
    return {
        "pclass_1": int(row["Pclass"] == "1"), "pclass_2": int(row["Pclass"] == "2"),
        "female": int(row["Sex"] == "female"),
        "age": age, "age_missing": int(row["Age"] == ""),
        "sibsp": sibsp, "parch": parch,
        "log_fare": math.log1p(fare), "fare_missing": int(row["Fare"] == ""),
        "embarked_c": int(port == "C"), "embarked_q": int(port == "Q"),
        "embarked_missing": int(row["Embarked"] == ""),
        "title_mrs": int(title == "Mrs"), "title_miss": int(title == "Miss"),
        "title_master": int(title == "Master"), "title_rare": int(title == "Rare"),
        "family_size": family_size, "is_alone": int(family_size == 1),
        "cabin_known": int(row["Cabin"] != ""),
    }


def training_stats(rows: list[dict]) -> TrainingStats:
    """Medians, port mode and standardization constants from the training rows only."""
    ages: dict[str, list[float]] = {}
    fares: dict[str, list[float]] = {}
    for row in rows:
        if row["Age"] != "":
            ages.setdefault(title_of(row["Name"]), []).append(float(row["Age"]))
        if row["Fare"] != "":
            fares.setdefault(row["Pclass"], []).append(float(row["Fare"]))
    ports = Counter(row["Embarked"] for row in rows if row["Embarked"] != "")
    stats = TrainingStats(
        age_by_title={title: statistics.median(values) for title, values in ages.items()},
        fare_by_class={pclass: statistics.median(values) for pclass, values in fares.items()},
        embarked_mode=ports.most_common(1)[0][0],
        scaling={},
    )
    features = [raw_features(row, stats) for row in rows]
    for column in STANDARDIZED:
        values = [f[column] for f in features]
        stats.scaling[column] = (statistics.fmean(values), statistics.pstdev(values))
    return stats


def design_row(row: dict, stats: TrainingStats) -> list[float]:
    """Model inputs in COLUMNS order: standardized columns scaled, rounded to 4 decimals."""
    raw = raw_features(row, stats)
    out = []
    for column in COLUMNS:
        value = raw[column]
        if column in STANDARDIZED:
            mean, std = stats.scaling[column]
            value = (value - mean) / std
        out.append(round(value, 4))
    return out


def main(argv: list[str]) -> int:
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
