"""Tests for the Titanic data-preparation script."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_titanic import normalize, read_rows, recover_labels, title_of, verify_train_labels  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
