"""Tests for the Titanic data-preparation script."""
from __future__ import annotations

import math
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_titanic import (  # noqa: E402
    COLUMNS, GROUPS, TrainingStats, auc, build_dataset, confusion, design_row, fit_logistic_l2,
    log_loss, normalize, raw_features, read_rows, recover_labels, render_js, title_of,
    training_stats, verify_train_labels,
)

TRAIN = Path(os.environ.get("TITANIC_TRAIN", "~/prj/kaggle/titanic/data/raw/train.csv")).expanduser()
TEST = Path(os.environ.get("TITANIC_TEST", "~/prj/kaggle/titanic/data/raw/test.csv")).expanduser()
TITANIC3 = Path(os.environ.get("TITANIC3", "~/prj/kaggle/titanic/data/raw/titanic3.csv")).expanduser()
REAL_FILES_EXIST = all(p.is_file() for p in (TRAIN, TEST, TITANIC3))


class TitleTests(unittest.TestCase):
    def test_title_of(self):
        cases = {"Braund, Mr. Owen Harris": "Mr",
                 "Cumings, Mrs. John Bradley (Florence Briggs Thayer)": "Mrs",
                 "Sagesser, Mlle. Emma": "Miss", "Reynaldo, Ms. Encarnacion": "Miss",
                 "Aubart, Mme. Leontine Pauline": "Mrs", "Palsson, Master. Gosta Leonard": "Master",
                 "Minahan, Dr. William Edward": "Rare",
                 "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)": "Rare"}
        for name, title in cases.items():
            self.assertEqual(title_of(name), title)


class NormalizeTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize('  "Kelly,   Mr. James" '), "kelly, mr. james")


class LabelTests(unittest.TestCase):
    T3 = [{"name": "Kelly, Mr. James", "ticket": "330911", "pclass": "3", "sex": "male", "survived": "0"},
          {"name": "Kelly, Mr. James", "ticket": "363592", "pclass": "3", "sex": "male", "survived": "1"},
          {"name": "Allen, Miss. Elisabeth Walton", "ticket": "24160", "pclass": "1", "sex": "female", "survived": "1"}]

    def test_full_key_disambiguates_duplicate_names(self):
        rows = [{"Name": "Kelly, Mr. James", "Ticket": "363592", "Pclass": "3", "Sex": "male"}]
        self.assertEqual(recover_labels(rows, self.T3), [1])

    def test_unique_name_fallback(self):
        rows = [{"Name": '"Allen, Miss. Elisabeth Walton"', "Ticket": "unknown", "Pclass": "1", "Sex": "female"}]
        self.assertEqual(recover_labels(rows, self.T3), [1])

    def test_unmatched_raises(self):
        with self.assertRaises(ValueError):
            recover_labels([{"Name": "Nobody, Mr. X", "Ticket": "1", "Pclass": "1", "Sex": "male"}], self.T3)

    def test_verify_train_labels_raises_on_disagreement(self):
        rows = [{"Name": "Allen, Miss. Elisabeth Walton", "Ticket": "24160", "Pclass": "1", "Sex": "female", "Survived": "0"}]
        with self.assertRaises(ValueError):
            verify_train_labels(rows, self.T3)


@unittest.skipUnless(REAL_FILES_EXIST, "raw Titanic files not available")
class RealLabelTests(unittest.TestCase):
    def test_all_rows_match(self):
        train, test, t3 = read_rows(TRAIN), read_rows(TEST), read_rows(TITANIC3)
        verify_train_labels(train, t3)  # no exception: 891/891 agree
        labels = recover_labels(test, t3)
        self.assertEqual((len(labels), sum(labels)), (418, 158))


ROW = {"PassengerId": "1", "Pclass": "3", "Name": "Braund, Mr. Owen Harris", "Sex": "male", "Age": "",
       "SibSp": "1", "Parch": "0", "Ticket": "A/5 21171", "Fare": "7.25", "Cabin": "", "Embarked": ""}
STATS = TrainingStats(age_by_title={"Mr": 30.0, "Mrs": 35.0, "Miss": 21.0, "Master": 3.5, "Rare": 48.5},
                      fare_by_class={"1": 60.2875, "2": 14.25, "3": 8.05}, embarked_mode="S",
                      scaling={"age": (30.0, 10.0), "sibsp": (0.5, 1.0), "parch": (0.5, 1.0),
                               "log_fare": (3.0, 1.0), "family_size": (2.0, 2.0)})


class FeatureTests(unittest.TestCase):
    def test_columns_and_groups(self):
        self.assertEqual(len(COLUMNS), 19)
        self.assertEqual(COLUMNS[:3], ["pclass_1", "pclass_2", "female"])
        self.assertEqual([g["id"] for g in GROUPS][:3], ["pclass", "sex", "age"])
        self.assertEqual(sum(len(g["columns"]) for g in GROUPS), 19)
        self.assertEqual({g["id"] for g in GROUPS if not g["default_on"]},
                         {"fare_missing", "embarked_missing", "family_size", "is_alone"})

    def test_raw_features_imputes_and_flags(self):
        raw = raw_features(ROW, STATS)
        self.assertEqual((raw["age"], raw["age_missing"]), (30.0, 1))
        self.assertEqual((raw["embarked_c"], raw["embarked_q"], raw["embarked_missing"]), (0, 0, 1))
        self.assertEqual((raw["family_size"], raw["is_alone"], raw["cabin_known"], raw["female"]), (2, 0, 0, 0))
        self.assertAlmostEqual(raw["log_fare"], math.log1p(7.25))

    def test_design_row_standardizes(self):
        row = design_row(ROW, STATS)
        self.assertEqual(len(row), 19)
        self.assertEqual(row[COLUMNS.index("age")], 0.0)                       # (30 - 30) / 10
        self.assertEqual(row[COLUMNS.index("sibsp")], 0.5)                     # (1 - 0.5) / 1
        self.assertEqual(row[COLUMNS.index("log_fare")], round(math.log1p(7.25) - 3.0, 4))

    def test_training_stats_synthetic(self):
        rows = [dict(ROW, Age="20", Embarked="C", Fare="10"), dict(ROW, Age="40", Embarked="C", Fare="30"),
                dict(ROW, Age="", Embarked="S", Fare="20")]
        stats = training_stats(rows)
        self.assertEqual(stats.age_by_title["Mr"], 30.0)
        self.assertEqual(stats.fare_by_class["3"], 20.0)
        self.assertEqual(stats.embarked_mode, "C")
        mean, std = stats.scaling["age"]                                       # imputed ages 20, 40, 30
        self.assertEqual((mean, round(std, 4)), (30.0, 8.165))


@unittest.skipUnless(REAL_FILES_EXIST, "raw Titanic files not available")
class RealStatsTests(unittest.TestCase):
    def test_training_statistics(self):
        stats = training_stats(read_rows(TRAIN))
        self.assertEqual(stats.age_by_title, {"Mr": 30.0, "Mrs": 35.0, "Miss": 21.0, "Master": 3.5, "Rare": 48.5})
        self.assertEqual(stats.embarked_mode, "S")
        self.assertAlmostEqual(stats.fare_by_class["1"], 60.2875, places=4)
        self.assertEqual((stats.fare_by_class["2"], stats.fare_by_class["3"]), (14.25, 8.05))
        for column, expected in {"age": (29.3924, 13.2609), "sibsp": (0.523, 1.1021), "parch": (0.3816, 0.8056),
                                 "log_fare": (2.9622, 0.9685), "family_size": (1.9046, 1.6126)}.items():
            for actual, want in zip(stats.scaling[column], expected):
                self.assertAlmostEqual(actual, want, places=3)

    def test_counts(self):
        train, test = read_rows(TRAIN), read_rows(TEST)
        stats = training_stats(train)

        def count(rows, column):
            return sum(raw_features(r, stats)[column] for r in rows)
        self.assertEqual([count(train, c) for c in ("age_missing", "cabin_known", "embarked_missing", "fare_missing", "is_alone")], [177, 204, 2, 0, 537])
        self.assertEqual([count(test, c) for c in ("age_missing", "cabin_known", "embarked_missing", "fare_missing", "is_alone")], [86, 91, 0, 1, 253])
        self.assertEqual(Counter(title_of(r["Name"]) for r in train), {"Mr": 517, "Miss": 185, "Mrs": 126, "Master": 40, "Rare": 23})
        self.assertEqual(Counter(title_of(r["Name"]) for r in test), {"Mr": 240, "Miss": 79, "Mrs": 72, "Master": 21, "Rare": 6})


class FitTests(unittest.TestCase):
    def test_single_binary_feature_closed_form(self):
        X = [[0.0]] * 10 + [[1.0]] * 10
        y = [1] * 3 + [0] * 7 + [1] * 7 + [0] * 3
        w, b = fit_logistic_l2(X, y, 0.0)
        self.assertAlmostEqual(b, math.log(3 / 7), places=6)
        self.assertAlmostEqual(w[0], 2 * math.log(7 / 3), places=6)

    def test_strong_penalty_shrinks_weight_not_intercept(self):
        X = [[0.0]] * 10 + [[1.0]] * 10
        y = [1] * 3 + [0] * 7 + [1] * 7 + [0] * 3
        w, b = fit_logistic_l2(X, y, 1e6)
        self.assertAlmostEqual(w[0], 0.0, places=4)
        self.assertAlmostEqual(b, 0.0, places=4)  # logit of the 50 % base rate

    def test_metrics(self):
        scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05]
        y = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
        self.assertEqual(confusion(scores, y, 0.5), {"tp": 3, "fn": 2, "fp": 2, "tn": 3})
        self.assertAlmostEqual(auc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1]), 0.875)
        self.assertAlmostEqual(log_loss([0.5, 0.5], [0, 1]), math.log(2))


@unittest.skipUnless(REAL_FILES_EXIST, "raw Titanic files not available")
class DatasetTests(unittest.TestCase):
    def test_build_dataset_and_render(self):
        data = build_dataset(read_rows(TRAIN), read_rows(TEST), read_rows(TITANIC3))
        self.assertEqual((len(data["train"]["X"]), sum(data["train"]["y"])), (891, 342))
        self.assertEqual((len(data["test"]["X"]), sum(data["test"]["y"])), (418, 158))
        self.assertTrue(all(len(row) == 19 and all(math.isfinite(v) for v in row) for row in data["train"]["X"] + data["test"]["X"]))
        self.assertEqual(len(data["reference"]["columns"]), 15)
        self.assertGreater(data["reference"]["train"]["auc"], 0.85)
        self.assertGreater(data["reference"]["test"]["auc"], 0.80)
        self.assertEqual(sum(data["reference"]["train"]["confusion"].values()), 891)
        text = render_js(data)
        self.assertTrue(text.startswith("/* Generated by src/prepare_titanic.py"))
        self.assertIn('root.ML_TITANIC = data', text)
        self.assertLess(len(text.encode()), 300_000)


if __name__ == "__main__":
    unittest.main()
