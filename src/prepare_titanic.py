"""Prepare the Titanic data asset for the classification metrics lab.

Command line:

    uv run python src/prepare_titanic.py --train train.csv --test test.csv \
        --titanic3 titanic3.csv --out <output.js>

The Kaggle test file has no labels; they are recovered from the public
titanic3 dataset: https://hbiostat.org/data/repo/titanic3.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
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


def _objective(X: list[list[float]], y: list[int], lam: float, params: list[float]) -> float:
    """Sum of logistic losses plus (lam / 2) * ||w||^2; params = weights followed by the intercept."""
    *w, b = params
    total = 0.0
    for row, label in zip(X, y):
        z = sum(wj * xj for wj, xj in zip(w, row)) + b
        margin = z if label else -z
        total += max(0.0, -margin) + math.log1p(math.exp(-abs(z)))
    return total + 0.5 * lam * sum(wj * wj for wj in w)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise RuntimeError("singular Hessian")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            if factor:
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def fit_logistic_l2(X: list[list[float]], y: list[int], lam: float) -> tuple[list[float], float]:
    """L2-regularized logistic regression by Newton's method; the intercept is not penalized."""
    p = len(X[0])
    params = [0.0] * (p + 1)
    penalty = [lam] * p + [0.0]
    current = _objective(X, y, lam, params)
    for _ in range(100):
        grad = [penalty[j] * params[j] for j in range(p + 1)]
        hess = [[0.0] * (p + 1) for _ in range(p + 1)]
        for row, label in zip(X, y):
            xi = row + [1.0]
            prob = _sigmoid(sum(params[j] * xi[j] for j in range(p + 1)))
            weight = prob * (1.0 - prob)
            for j in range(p + 1):
                grad[j] += (prob - label) * xi[j]
                if xi[j]:
                    for k in range(p + 1):
                        hess[j][k] += weight * xi[j] * xi[k]
        for j in range(p + 1):
            hess[j][j] += penalty[j]
        step = _solve(hess, grad)
        scale = 1.0
        candidate, value = params, current
        for _ in range(60):
            candidate = [params[j] - scale * step[j] for j in range(p + 1)]
            value = _objective(X, y, lam, candidate)
            if value < current:
                break
            scale /= 2.0
        else:
            candidate, value = params, current  # no decrease possible: at the optimum
        change = current - value
        params, current = candidate, value
        if change < 1e-10:
            return params[:p], params[p]
    raise RuntimeError("Newton's method did not converge in 100 iterations")


def log_loss(probabilities: list[float], y: list[int]) -> float:
    total = 0.0
    for prob, label in zip(probabilities, y):
        prob = min(max(prob, 1e-15), 1.0 - 1e-15)
        z = math.log(prob / (1.0 - prob))
        margin = z if label else -z
        total += max(0.0, -margin) + math.log1p(math.exp(-abs(z)))
    return total / len(y)


def auc(scores: list[float], y: list[int]) -> float:
    """Rank statistic (Mann-Whitney U) with average ranks for tied scores."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    positives = sum(y)
    negatives = len(y) - positives
    rank_sum = sum(rank for rank, label in zip(ranks, y) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def confusion(scores: list[float], y: list[int], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for score, label in zip(scores, y):
        positive = score >= threshold
        key = ("tp" if positive else "fn") if label else ("fp" if positive else "tn")
        counts[key] += 1
    return counts


def _clean(value: float, digits: int = 6) -> float:
    return round(value, digits) + 0.0  # + 0.0 turns -0.0 into 0.0


def _evaluate(X, y, weights, intercept) -> dict:
    probs = [_sigmoid(sum(w * x for w, x in zip(weights, row)) + intercept) for row in X]
    return {"log_loss": _clean(log_loss(probs, y)), "auc": _clean(auc(probs, y)),
            "confusion": confusion(probs, y, 0.5)}


def build_dataset(train_rows: list[dict], test_rows: list[dict], titanic3_rows: list[dict]) -> dict:
    """Assemble the ML_TITANIC object, including the reference fit on the default-on groups."""
    if (len(train_rows), len(test_rows)) != (891, 418):
        raise ValueError(f"unexpected row counts {len(train_rows)}/{len(test_rows)}")
    verify_train_labels(train_rows, titanic3_rows)
    y_train = [int(row["Survived"]) for row in train_rows]
    y_test = recover_labels(test_rows, titanic3_rows)
    if (sum(y_train), sum(y_test)) != (342, 158):
        raise ValueError(f"unexpected positives {sum(y_train)}/{sum(y_test)}")
    stats = training_stats(train_rows)
    X_train = [design_row(row, stats) for row in train_rows]
    X_test = [design_row(row, stats) for row in test_rows]
    if not all(math.isfinite(v) for row in X_train + X_test for v in row):
        raise ValueError("non-finite value in the design matrix")

    groups = [g["id"] for g in GROUPS if g["default_on"]]
    wanted = {c for g in GROUPS if g["default_on"] for c in g["columns"]}
    index = [i for i, c in enumerate(COLUMNS) if c in wanted]
    columns = [COLUMNS[i] for i in index]
    sub_train = [[row[i] for i in index] for row in X_train]
    sub_test = [[row[i] for i in index] for row in X_test]
    weights, intercept = fit_logistic_l2(sub_train, y_train, 1.0)
    weights = [_clean(w) for w in weights]
    intercept = _clean(intercept)
    return {
        "schema": 1,
        "columns": COLUMNS,
        "groups": GROUPS,
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test},
        "meta": {
            "source": [
                "Kaggle Titanic train.csv and test.csv",
                "Vanderbilt titanic3.csv (https://hbiostat.org/data/repo/titanic3.csv), "
                "originally from Encyclopedia Titanica",
            ],
            "prepared_at": date.today().isoformat(),
            "counts": {"train": len(y_train), "test": len(y_test),
                       "train_positive": sum(y_train), "test_positive": sum(y_test)},
            "imputation": {"age_by_title": stats.age_by_title, "fare_by_class": stats.fare_by_class,
                           "embarked_mode": stats.embarked_mode},
            "scaling": {c: {"mean": stats.scaling[c][0], "std": stats.scaling[c][1]} for c in STANDARDIZED},
        },
        "reference": {
            "groups": groups, "penalty": "l2", "lambda": 1, "columns": columns,
            "weights": weights, "intercept": intercept,
            "train": _evaluate(sub_train, y_train, weights, intercept),
            "test": _evaluate(sub_test, y_test, weights, intercept),
        },
    }


def render_js(dataset: dict) -> str:
    payload = json.dumps(dataset, separators=(",", ":"), ensure_ascii=False)
    header = f"/* Generated by src/prepare_titanic.py on {dataset['meta']['prepared_at']}. Do not edit by hand. */"
    return (header + "\n(function (root) { const data = " + payload + "; "
            'if (typeof module !== "undefined" && module.exports) module.exports = data; '
            "else root.ML_TITANIC = data; })"
            '(typeof window !== "undefined" ? window : globalThis);\n')


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Prepare the Titanic data asset.")
    for flag in ("--train", "--test", "--titanic3", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    args = parser.parse_args(argv)
    data = build_dataset(read_rows(args.train.expanduser()), read_rows(args.test.expanduser()),
                         read_rows(args.titanic3.expanduser()))
    meta = data["meta"]
    print("counts:", meta["counts"])
    print("imputation:", json.dumps(meta["imputation"]))
    print("scaling:", json.dumps(meta["scaling"]))
    ref = data["reference"]
    print(f"reference AUC: train {ref['train']['auc']:.4f}, test {ref['test']['auc']:.4f}")
    out = args.out.expanduser()
    out.write_text(render_js(data), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
