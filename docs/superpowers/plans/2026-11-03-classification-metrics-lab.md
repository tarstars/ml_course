# Classification Metrics Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second interactive lab to the ML course site that fits logistic regression on the Titanic data in the browser and explains binary classification metrics (score distributions, threshold, confusion matrix, derived metrics, ROC/AUC) with a train/test switch.

**Architecture:** A standard-library Python script turns the Kaggle Titanic files plus the Vanderbilt `titanic3.csv` into one committed JS data asset (`titanic-data.js`). Three browser modules follow the regression lab's pattern: `classification-math.js` (pure numerics, Node-testable), `classification-plots.js` (pure SVG string builders, Node-testable), `classification-lab.js` (state, markup, DOM wiring, exposing `html/mount/unmount`). The site shell, router, translations and the Playwright smoke check are extended the same way the regression lab extended them.

**Tech Stack:** Vanilla ES2020 JavaScript (no bundler, no runtime deps), SVG, CSS with the site tokens; Python 3.12 standard library for data prep and `unittest`; Node 18 + `node:assert` for numeric checks; Playwright (already in the uv dev group) for browser checks; `uv` for the build.

**Spec:** `docs/superpowers/specs/2026-11-03-classification-metrics-lab-design.md`

## Global Constraints

- Build in a clean `git worktree` of `main` (the repo root in the main working copy holds ~200 untracked files that `build.py`'s validation rejects). Stage commits by explicit path. Never relax the uncataloged-file check.
- Build and checks: `uv sync --frozen`, `uv run python src/build.py`, serve with `uv run python -m http.server 8000 --bind 127.0.0.1 --directory dist`, then `uv run python src/check_site.py --url http://127.0.0.1:8000/`. Chromium for Playwright is already installed under `~/.cache/ms-playwright`.
- `src/prepare_titanic.py` uses the **standard library only**. No new entries in `pyproject.toml` / `uv.lock`.
- Raw inputs (not committed): `~/prj/kaggle/titanic/data/raw/train.csv`, `test.csv`, `titanic3.csv`. Never run recursive searches over `~/prj`.
- Browser code: no network requests at runtime, no new vendored libraries, Node 18 compatible (`check_*.cjs` must not use Node 20+ APIs). Every new module uses the regression lab's UMD-style wrapper: `module.exports` under Node, `window.<Name>` in the browser.
- Numbers in `titanic-data.js` are rounded to 4 decimals; the stored `reference` fit is computed on the rounded matrix.
- Objective (spec §5.3): `Σᵢ log(1 + exp(−sᵢ zᵢ)) + λ·P(w)`, `P = ½‖w‖²` (L2) or `‖w‖₁` (L1), λ = 0 for "none", intercept never penalized, `sᵢ = ±1`. Convergence: objective change < 1e-8 or 100 iterations (L1: 200 outer iterations). "Predicted positive" means `score ≥ threshold`.
- Standardization uses the **population** standard deviation (ddof = 0), training rows only.
- Naming: CSS classes and element ids prefixed `cl-`; root section `id="classification-lab"`; route `#classification`; nav label RU "Лаборатория классификации" / EN "Classification lab"; eyebrow "INTERACTIVE LAB / 02" / "ИНТЕРАКТИВНАЯ ЛАБОРАТОРИЯ / 02". In-lab strings use the inline `w(en, ru)` pattern; only the nav label goes into `i18n.js`.
- Lecture view shares the regression lab's localStorage keys `ml-regression-presentation` and `ml-regression-text-scale` and toggles the same `body.lr-presenting` class (the chrome-hiding rules already live in `regression-lab.css`); the lab root gets `cl-presentation`.
- Default state: groups 1–7, 9, 11, 14 on (15 columns); penalty L2; λ = 1; axis probability; threshold 0.5; split train; lecture view on unless the stored preference says otherwise.
- Colours: survived (positive) curve `#234b3b`, died (negative) curve `#817199`, dashed. Region/cell tints: TP `#cfe3d3`, TN `#e4dfec`, FP `#f0d9c0`, FN `#e9c9b8`.
- Commit messages: one imperative sentence like the repo's history, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Do not push.

## Review Focus

1. **All feature switches off.** Intercept-only model: every passenger gets the same score, AUC must be exactly 0.5, the ROC the diagonal, metrics defined, no exception. Test: Task 4 (`fit` with no active columns) and Task 9 (browser).
2. **Nobody (or everybody) predicted positive.** Intercept-only model at threshold 0.5 → no predicted survivors → precision and F1 undefined; at 0.3 everybody is predicted positive → NPV undefined. The UI shows "—", not NaN. Test: Task 5 (`metrics` nulls) and Task 9 (browser).
3. **Axis toggle must keep the cut.** Switch to logit, move the slider, switch back: the probability equals σ(logit) to 3 decimals. Test: Task 8 (browser).
4. **Column constant in train but not in test** (`fare_missing`: 0 training rows, 1 test row). The column is dropped with weight exactly 0 and the test row is scored as if the column did not exist. Test: Task 4.
5. **Perfect separation under "none"** (`embarked_missing` on: both such passengers survived). The page must stay responsive, weights stay finite, status says separation. Test: Task 4 (`separated` flag) and Task 7 (browser).

---

### Task 1: Prep script — parsing, titles, normalization, label recovery

**Files:**
- Create: `src/prepare_titanic.py`
- Create: `src/test_prepare_titanic.py`

**Interfaces:**
- Produces (in `prepare_titanic.py`):
  - `normalize(text: str) -> str` — remove double quotes, collapse whitespace runs to one space, strip, lower-case.
  - `title_of(name: str) -> str` — token between the first comma and the next period; `Mlle`/`Ms` → `Miss`, `Mme` → `Mrs`; keep `Mr`, `Mrs`, `Miss`, `Master`; else `Rare`.
  - `read_rows(path: Path) -> list[dict[str, str]]` — `csv.DictReader`, utf-8.
  - `match_key(row: dict, kaggle: bool) -> tuple[str, str, str, str]` — `(normalize(name), normalize(ticket), pclass.strip(), normalize(sex))`; Kaggle columns `Name, Ticket, Pclass, Sex`; titanic3 columns `name, ticket, pclass, sex`.
  - `recover_labels(rows: list[dict], titanic3: list[dict]) -> list[int]` — full key first, then unique normalized name; raises `ValueError("unmatched: <name>")`.
  - `verify_train_labels(rows: list[dict], titanic3: list[dict]) -> None` — raises `ValueError` on any disagreement with `Survived`.
- Test file bootstraps `sys.path.insert(0, str(Path(__file__).resolve().parent))` and reads optional real paths from env `TITANIC_TRAIN`, `TITANIC_TEST`, `TITANIC3` (defaults: `~/prj/kaggle/titanic/data/raw/{train,test,titanic3}.csv`), skipping real-file tests with `unittest.skipUnless` when absent.

- [ ] **Step 1: Write the failing tests**

```python
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
        verify_train_labels(train, t3)                      # no exception: 891/891 agree
        labels = recover_labels(test, t3)
        self.assertEqual((len(labels), sum(labels)), (418, 158))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd <worktree> && uv run python -m unittest src/test_prepare_titanic.py -v`
Expected: FAIL / ERROR with `ImportError` or `NameError` for `title_of`.

- [ ] **Step 3: Implement the five functions in `src/prepare_titanic.py`**

Module docstring: purpose, the command line from spec §4.2, the titanic3 URL. Add `if __name__ == "__main__": raise SystemExit(main(sys.argv[1:]))` with a placeholder `main` that returns 2 (completed in Task 3).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest src/test_prepare_titanic.py -v`
Expected: all PASS, including `RealLabelTests.test_all_rows_match`.

- [ ] **Step 5: Commit**

```bash
git add src/prepare_titanic.py src/test_prepare_titanic.py
git commit -m "Add Titanic label recovery and title parsing for the classification lab"
```

---

### Task 2: Prep script — training statistics, feature groups, design rows

**Files:**
- Modify: `src/prepare_titanic.py`
- Modify: `src/test_prepare_titanic.py`

**Interfaces:**
- Produces:
  - `COLUMNS: list[str]` — exactly, in order: `pclass_1, pclass_2, female, age, age_missing, sibsp, parch, log_fare, fare_missing, embarked_c, embarked_q, embarked_missing, title_mrs, title_miss, title_master, title_rare, family_size, is_alone, cabin_known` (19).
  - `GROUPS: list[dict]` — 14 entries in spec §4.3 order; each `{"id", "label": {"en", "ru"}, "columns": [...], "kind": "binary"|"standardized", "default_on": bool, "note": {"en", "ru"}}`; ids: `pclass, sex, age, age_missing, sibsp, parch, fare, fare_missing, embarked, embarked_missing, title, family_size, is_alone, cabin_known`. `default_on` false for `fare_missing, embarked_missing, family_size, is_alone`.
  - `STANDARDIZED: tuple[str, ...] = ("age", "sibsp", "parch", "log_fare", "family_size")`.
  - `@dataclass TrainingStats`: `age_by_title: dict[str, float]`, `fare_by_class: dict[str, float]`, `embarked_mode: str`, `scaling: dict[str, tuple[float, float]]` (mean, population std per standardized column).
  - `training_stats(rows: list[dict]) -> TrainingStats` — medians via `statistics.median`; scaling computed on the imputed, unscaled values.
  - `raw_features(row: dict, stats: TrainingStats) -> dict[str, float]` — the 19 unscaled values (imputed Age, `log_fare = log1p(fare)`, flags from the original empty strings, `family_size = sibsp + parch + 1`, `is_alone = family_size == 1`).
  - `design_row(row: dict, stats: TrainingStats) -> list[float]` — `raw_features` with standardized columns transformed `(v − mean) / std`, in `COLUMNS` order, rounded to 4 decimals.

- [ ] **Step 1: Write the failing tests**

```python
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
        self.assertEqual(row[COLUMNS.index("age")], 0.0)                       # (30 − 30) / 10
        self.assertEqual(row[COLUMNS.index("sibsp")], 0.5)                     # (1 − 0.5) / 1
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
        def count(rows, column): return sum(raw_features(r, stats)[column] for r in rows)
        self.assertEqual([count(train, c) for c in ("age_missing", "cabin_known", "embarked_missing", "fare_missing", "is_alone")], [177, 204, 2, 0, 537])
        self.assertEqual([count(test, c) for c in ("age_missing", "cabin_known", "embarked_missing", "fare_missing", "is_alone")], [86, 91, 0, 1, 253])
        self.assertEqual(Counter(title_of(r["Name"]) for r in train), {"Mr": 517, "Miss": 185, "Mrs": 126, "Master": 40, "Rare": 23})
        self.assertEqual(Counter(title_of(r["Name"]) for r in test), {"Mr": 240, "Miss": 79, "Mrs": 72, "Master": 21, "Rare": 6})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest src/test_prepare_titanic.py -v`
Expected: the new tests ERROR with `ImportError` for `COLUMNS` / `TrainingStats`.

- [ ] **Step 3: Implement `COLUMNS`, `GROUPS`, `STANDARDIZED`, `TrainingStats`, `training_stats`, `raw_features`, `design_row`**

Group labels (en / ru): Passenger class / Класс; Sex / Пол; Age / Возраст; Age was filled / Возраст указан; Siblings and spouses / Братья, сёстры, супруги; Parents and children / Родители и дети; Fare / Стоимость билета; Fare was filled / Стоимость указана; Port of embarkation / Порт посадки; Port was filled / Порт указан; Title from name / Обращение в имени; Family size / Размер семьи; Travelling alone / Путешествует один; Cabin was filled / Каюта указана. Notes: one sentence each on what the columns encode and the reference category (e.g. "Two indicators; third class is the reference." / "Два индикатора; третий класс — базовый.").

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest src/test_prepare_titanic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prepare_titanic.py src/test_prepare_titanic.py
git commit -m "Build Titanic feature groups with leakage-free training statistics"
```

---

### Task 3: Prep script — reference fit, metrics, JS writer, CLI; generate the data asset

**Files:**
- Modify: `src/prepare_titanic.py`
- Modify: `src/test_prepare_titanic.py`
- Create (generated): `src/static/titanic-data.js`

**Interfaces:**
- Produces:
  - `fit_logistic_l2(X: list[list[float]], y: list[int], lam: float) -> tuple[list[float], float]` — Newton's method with Gaussian elimination (partial pivoting) on the (p+1)×(p+1) system, objective per Global Constraints, returns `(weights, intercept)`; stops when objective change < 1e-10 or after 100 iterations; raises `RuntimeError` if not converged.
  - `log_loss(probabilities: list[float], y: list[int]) -> float` (mean, numerically stable via `log1p(exp(−|z|))` form); `auc(scores: list[float], y: list[int]) -> float` (rank statistic with average ranks for ties); `confusion(scores, y, threshold) -> dict` with keys `tp, fn, fp, tn` (`score ≥ threshold` is positive).
  - `build_dataset(train_rows, test_rows, titanic3_rows) -> dict` — the full `ML_TITANIC` object of spec §4.4 with `schema: 1`, `columns`, `groups`, `train {X, y}`, `test {X, y}`, `meta` (`source` strings, `prepared_at` ISO date, `counts`, `imputation`, `scaling`), and `reference` = `{"groups": [default-on ids], "penalty": "l2", "lambda": 1, "columns": [15 names], "weights": [...], "intercept": b, "train": {"log_loss", "auc", "confusion"}, "test": {...}}` with confusion at probability 0.5. Fit uses the rounded design rows. Aborts (`ValueError`) on NaN or on counts other than 891/342 and 418/158.
  - `render_js(dataset: dict) -> str` — header comment `/* Generated by src/prepare_titanic.py on <date>. Do not edit by hand. */`, then the UMD wrapper: `(function (root) { const data = <json>; if (typeof module !== "undefined" && module.exports) module.exports = data; else root.ML_TITANIC = data; })(typeof window !== "undefined" ? window : globalThis);` with `json.dumps(..., separators=(",", ":"))`.
  - `main(argv: list[str]) -> int` — argparse `--train --test --titanic3 --out`; prints the summary (row counts, positives, imputation values, scaling, reference AUCs); writes the file; returns 0.

- [ ] **Step 1: Write the failing tests**

```python
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
        self.assertAlmostEqual(b, 0.0, places=4)                     # logit of the 50 % base rate
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest src/test_prepare_titanic.py -v`
Expected: ERROR with `ImportError` for `fit_logistic_l2`.

- [ ] **Step 3: Implement the solver, metrics, `build_dataset`, `render_js`, `main`**

Newton details the tests do not fix: gradient `Xᵀ(p − y) + λw` (0 for the intercept component), Hessian `XᵀWX + λI` with `W = diag(p(1−p))`, halve the step while the objective does not decrease.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest src/test_prepare_titanic.py -v`
Expected: all PASS.

- [ ] **Step 5: Generate the asset and check it loads under Node**

Run:
```bash
uv run python src/prepare_titanic.py --train ~/prj/kaggle/titanic/data/raw/train.csv --test ~/prj/kaggle/titanic/data/raw/test.csv --titanic3 ~/prj/kaggle/titanic/data/raw/titanic3.csv --out src/static/titanic-data.js
node -e 'const d = require("./src/static/titanic-data.js"); console.log(d.train.X.length, d.test.X.length, d.columns.length, d.reference.columns.length)'
```
Expected: summary printed; Node prints `891 418 19 15`; file size under 300 KB (`ls -l src/static/titanic-data.js`).

- [ ] **Step 6: Commit**

```bash
git add src/prepare_titanic.py src/test_prepare_titanic.py src/static/titanic-data.js
git commit -m "Generate the Titanic data asset with a reference logistic fit"
```

---

### Task 4: `classification-math.js` — solver and predictions

**Files:**
- Create: `src/static/classification-math.js`
- Create: `src/check_classification.cjs`

**Interfaces:**
- Consumes: `src/static/titanic-data.js` (Task 3) via `require` in the check.
- Produces (`window.MLClassificationMath` / `module.exports`):
  - `sigmoid(z: number) -> number`, `logit(p: number) -> number`.
  - `solveSymmetric(H: number[][], g: number[]) -> { x: number[], rank: number }` — cyclic Jacobi eigendecomposition of the symmetric matrix; eigenvalues ≤ `1e-6 × max|eigenvalue|` are treated as zero (minimum-norm solution). The threshold is 1e-6, not machine precision, because the shipped matrix is rounded to 4 decimals: `family_size` is then collinear with `sibsp` and `parch` only up to ~5e-8 relative, and it must still be detected.
  - `fit(X: number[][], y: (0|1)[], options: { penalty: "none"|"l1"|"l2", lambda: number, columns?: number[], initial?: { weights, intercept } }) -> { weights: number[], intercept: number, iterations: number, converged: boolean, objective: number, separated: boolean, constant: number[], collinear: boolean, rank: number }` — `weights.length === X[0].length`, zeros for columns not in `columns` and for constant columns (`constant` lists their indices); `columns` defaults to all; `collinear` is true when the unpenalized Hessian is rank-deficient (per `solveSymmetric`) over the active non-constant columns; `separated` is true when penalty is "none" and, after fitting, `max(|weights|, |intercept|) > 10` (with standardized and 0/1 columns an honest coefficient never gets near 10, odds ratio 22 000; on separable data Newton stalls around 17–18 once the loss contribution drops below the convergence tolerance). Newton with step halving for none/L2; for L1 an IRLS outer loop with coordinate descent + soft-thresholding on the weighted quadratic approximation, intercept updated unpenalized, stop when the objective change < 1e-8 (cap 200 outer iterations). Never throws on numeric trouble; always returns finite numbers.
  - `scores(X: number[][], weights: number[], intercept: number) -> number[]` — logits.
  - `probabilities(logits: number[]) -> number[]`.

- [ ] **Step 1: Write the failing checks**

`src/check_classification.cjs` in the style of `check_regression.cjs` (`node:assert/strict`, `close(actual, expected, tol)` helper, `console.log("PASS: ...")` at the end):

```js
const M = require("./static/classification-math.js");
const D = require("./static/titanic-data.js");
// closed form: single binary feature, 3/10 vs 7/10 positives
const X1 = [...Array(10).fill([0]), ...Array(10).fill([1])];
const y1 = [1,1,1,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,0,0,0];
const single = M.fit(X1, y1, { penalty: "none", lambda: 0 });
close(single.intercept, Math.log(3 / 7), 1e-6); close(single.weights[0], 2 * Math.log(7 / 3), 1e-6);
assert.equal(single.converged, true); assert.equal(single.separated, false);
// intercept only
const none = M.fit(X1, y1, { penalty: "l2", lambda: 1, columns: [] });
close(none.intercept, 0, 1e-9); assert.deepEqual(none.weights, [0]);
// gradient optimality (L2) and KKT (L1) on a deterministic 60×4 dataset built from sin/cos of the row index
const gradient = (X, y, w, b, lambda) => { /* Xᵀ(p − y) + λw, intercept component without penalty */ };
const ridge = M.fit(X2, y2, { penalty: "l2", lambda: 0.5 });
assert.ok(Math.max(...gradient(X2, y2, ridge.weights, ridge.intercept, 0.5).map(Math.abs)) < 1e-7);
const lasso = M.fit(X2, y2, { penalty: "l1", lambda: 2 });
const g = gradient(X2, y2, lasso.weights, lasso.intercept, 0);
lasso.weights.forEach((w, j) => w !== 0 ? assert.ok(Math.abs(g[j] + 2 * Math.sign(w)) < 1e-5) : assert.ok(Math.abs(g[j]) <= 2 + 1e-5));
assert.ok(Math.abs(g[g.length - 1]) < 1e-7);                              // intercept gradient
const flat = M.fit(X2, y2, { penalty: "l1", lambda: 1e4 });
assert.ok(flat.weights.every((w) => w === 0)); close(flat.intercept, M.logit(y2.reduce((a, b) => a + b) / y2.length), 1e-6);
// separation
const sep = M.fit([[0], [1], [0], [1]], [0, 1, 0, 1], { penalty: "none", lambda: 0 });
assert.equal(sep.separated, true); assert.ok(sep.weights.every(Number.isFinite) && Number.isFinite(sep.intercept));
assert.equal(single.separated, false);                                   // honest large-ish coefficients are not flagged
assert.equal(M.fit([[0], [1], [0], [1]], [0, 1, 0, 1], { penalty: "l2", lambda: 1 }).separated, false);
// constant column dropped; a test row with a nonzero value there is scored as if the column did not exist
const Xc = X2.map((row) => [...row, 0]);
const constant = M.fit(Xc, y2, { penalty: "l2", lambda: 0.5 });
assert.deepEqual(constant.constant, [4]); assert.equal(constant.weights[4], 0);
close(M.scores([[...X2[0], 1]], constant.weights, constant.intercept)[0], M.scores([X2[0]], ridge.weights, ridge.intercept)[0], 1e-9);
// collinear: a, b, a + b under "none" → flag set, predictions equal to the a, b fit
const Xab = X2.map((r) => [r[0], r[1]]), Xabc = X2.map((r) => [r[0], r[1], r[0] + r[1]]);
const ab = M.fit(Xab, y2, { penalty: "none", lambda: 0 }), abc = M.fit(Xabc, y2, { penalty: "none", lambda: 0 });
assert.equal(abc.collinear, true); assert.equal(ab.collinear, false);
M.scores(Xabc, abc.weights, abc.intercept).forEach((s, i) => close(s, M.scores(Xab, ab.weights, ab.intercept)[i], 1e-6));
// reference reproduction on the shipped data
const active = D.reference.columns.map((name) => D.columns.indexOf(name));
const start = performance.now();
const ref = M.fit(D.train.X, D.train.y, { penalty: "l2", lambda: 1, columns: active });
const elapsed = performance.now() - start;
active.forEach((j, k) => close(ref.weights[j], D.reference.weights[k], 1e-4));
close(ref.intercept, D.reference.intercept, 1e-4);
assert.ok(elapsed < 100, `fit took ${elapsed} ms`);
```

- [ ] **Step 2: Run the checks to verify they fail**

Run: `node src/check_classification.cjs`
Expected: `Error: Cannot find module './static/classification-math.js'`.

- [ ] **Step 3: Implement `src/static/classification-math.js` (solver part)**

File header comment like `regression-math.js`. Stable loss term: `log(1 + exp(−s z)) = max(0, −s z) + log1p(exp(−|z|))`.

- [ ] **Step 4: Run the checks to verify they pass**

Run: `node src/check_classification.cjs`
Expected: `PASS: ...` and the fit time printed (< 100 ms; target < 50 ms).

- [ ] **Step 5: Commit**

```bash
git add src/static/classification-math.js src/check_classification.cjs
git commit -m "Add the browser logistic regression solver with L1 and L2 penalties"
```

---

### Task 5: `classification-math.js` — confusion, metrics, ROC, losses, histogram, KDE

**Files:**
- Modify: `src/static/classification-math.js`
- Modify: `src/check_classification.cjs`

**Interfaces:**
- Produces (same module):
  - `confusion(scores: number[], y: (0|1)[], threshold: number) -> { tp, fn, fp, tn }` — positive iff `score >= threshold`.
  - `metrics(c: { tp, fn, fp, tn }) -> { accuracy, precision, recall, specificity, fpr, fnr, f1, balancedAccuracy, mcc, youdenJ, npv }` — `null` wherever a denominator is 0.
  - `roc(scores: number[], y: (0|1)[]) -> { points: { fpr, tpr, threshold }[], auc: number }` — points from `(0, 0)` to `(1, 1)`, one step per distinct score (descending), `auc` by the trapezoid rule (equals the tie-averaged rank statistic).
  - `logLoss(probabilities: number[], y: (0|1)[]) -> number` (mean, clamped to [1e-15, 1 − 1e-15]); `brier(probabilities, y) -> number`.
  - `histogram(values: number[], edges: number[]) -> number[]` — `edges.length − 1` counts; values equal to the last edge fall in the last bin; values outside are ignored.
  - `silverman(values: number[]) -> number` — `0.9 · min(sd, IQR / 1.34) · n^(−1/5)`, falling back to `sd · n^(−1/5)` when IQR is 0 and to 1e-3 when both are 0.
  - `kde(values: number[], grid: number[], bandwidth: number, bounds?: [number, number]) -> number[]` — Gaussian kernel density at each grid point; with `bounds`, reflect each value at both bounds.

- [ ] **Step 1: Write the failing checks** (append to `check_classification.cjs`)

```js
const s10 = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05], y10 = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0];
assert.deepEqual(M.confusion(s10, y10, 0.5), { tp: 3, fn: 2, fp: 2, tn: 3 });
const m = M.metrics({ tp: 3, fn: 2, fp: 2, tn: 3 });
["accuracy", "precision", "recall", "specificity", "f1", "balancedAccuracy", "npv"].forEach((k) => close(m[k], 0.6, 1e-12));
close(m.fpr, 0.4, 1e-12); close(m.fnr, 0.4, 1e-12); close(m.mcc, 0.2, 1e-12); close(m.youdenJ, 0.2, 1e-12);
const nothing = M.metrics({ tp: 0, fn: 5, fp: 0, tn: 5 });
assert.equal(nothing.precision, null); assert.equal(nothing.f1, null); close(nothing.specificity, 1, 1e-12);
assert.deepEqual(M.confusion(s10, y10, 0.95), { tp: 0, fn: 5, fp: 0, tn: 5 });
close(M.roc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1]).auc, 0.875, 1e-12);
close(M.roc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]).auc, 1, 1e-12);
close(M.roc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]).auc, 0, 1e-12);
close(M.roc([3, 3, 3, 3], [0, 1, 0, 1]).auc, 0.5, 1e-12);                   // intercept-only model
const curve = M.roc(s10, y10).points;
assert.deepEqual([curve[0].fpr, curve[0].tpr, curve.at(-1).fpr, curve.at(-1).tpr], [0, 0, 1, 1]);
curve.slice(1).forEach((p, i) => assert.ok(p.fpr >= curve[i].fpr && p.tpr >= curve[i].tpr));
close(M.logLoss([0.5, 0.5], [0, 1]), Math.log(2), 1e-12); close(M.brier([0.5, 1], [0, 1]), 0.125, 1e-12);
assert.deepEqual(M.histogram([0, 0.5, 1, 1, 2.5], [0, 1, 2]), [2, 2]);           // 2.5 ignored, 1 goes to the second bin
const grid = Array.from({ length: 401 }, (_, i) => -10 + i * 0.05);
const dens = M.kde([-1, 0, 1, 0.5, -0.5], grid, M.silverman([-1, 0, 1, 0.5, -0.5]));
close(dens.reduce((a, b) => a + b) * 0.05, 1, 1e-2);
const unit = Array.from({ length: 201 }, (_, i) => i / 200);
const near0 = M.kde([0.01, 0.02, 0.05, 0.03, 0.04], unit, 0.05, [0, 1]);
close(near0.reduce((a, b) => a + b) * 0.005 - (near0[0] + near0.at(-1)) * 0.0025, 1, 2e-2);    // mass stays inside [0, 1]
```

- [ ] **Step 2: Run the checks to verify they fail**

Run: `node src/check_classification.cjs`
Expected: `TypeError: M.confusion is not a function`.

- [ ] **Step 3: Implement the functions**

- [ ] **Step 4: Run the checks to verify they pass**

Run: `node src/check_classification.cjs`
Expected: `PASS: ...`.

- [ ] **Step 5: Commit**

```bash
git add src/static/classification-math.js src/check_classification.cjs
git commit -m "Add confusion, metrics, ROC and density helpers for the classification lab"
```

---

### Task 6: `classification-plots.js` — SVG builders for the score distribution and the ROC

**Files:**
- Create: `src/static/classification-plots.js`
- Modify: `src/check_classification.cjs`

**Interfaces:**
- Consumes: `MLClassificationMath.histogram`, `kde`, `silverman` (Task 5).
- Produces (`window.MLClassificationPlots` / `module.exports`):
  - `DISTRIBUTION = { width: 960, height: 420, box: { left: 64, right: 940, top: 24, bottom: 360 } }`, `ROC = { width: 480, height: 480, box: { left: 56, right: 456, top: 24, bottom: 424 } }`.
  - `scale(range: [lo, hi], box) -> { px: (v) => number, value: (px) => number }` — linear map of `range` onto `[box.left, box.right]`.
  - `distributionPlot(o: { axis: "logit"|"probability", range: [lo, hi], bins: 40, positives: number[], negatives: number[], threshold: number, fontSize: number, labels: { positive, negative, x, y, empty } }) -> string` — SVG inner markup (no `<svg>` element): grid and tick labels; per class a histogram as `<rect class="cl-bar cl-positive|cl-negative">` (one per non-empty bin); per class a KDE curve `<path class="cl-curve cl-positive|cl-negative">` evaluated on 200 grid points, scaled to counts per bin (`density × n × binWidth`, bounds `[0, 1]` on the probability axis); four filled regions `<path class="cl-region cl-tp|cl-fn|cl-fp|cl-tn">` from the curves split at the threshold; threshold line `<line id="cl-threshold-line">` and a handle `<g id="cl-threshold-handle" tabindex="0">` at `px(threshold)`; y axis ticks in counts. If both classes are empty, a centred `labels.empty` text.
  - `rocPlot(o: { primary: { fpr, tpr }[], secondary: { fpr, tpr }[] | null, operating: { fpr, tpr }, fontSize: number, labels: { x, y } }) -> string` — diagonal `<line class="cl-diagonal">`, `<path class="cl-roc cl-primary">`, optional `<path class="cl-roc cl-secondary">` (dashed via CSS), `<circle id="cl-operating-point">` at the operating point.

- [ ] **Step 1: Write the failing checks** (append)

```js
const P = require("./static/classification-plots.js");
const sx = P.scale([-6, 6], P.DISTRIBUTION.box);
close(sx.px(0), (P.DISTRIBUTION.box.left + P.DISTRIBUTION.box.right) / 2, 1e-9); close(sx.value(sx.px(2.5)), 2.5, 1e-9);
const svg = P.distributionPlot({ axis: "logit", range: [-6, 6], bins: 40, positives: [1, 2, 3, 2.5], negatives: [-1, -2, -3, 0.5], threshold: 0,
  fontSize: 16, labels: { positive: "survived", negative: "died", x: "logit", y: "passengers", empty: "no data" } });
assert.equal((svg.match(/class="cl-curve cl-positive"/g) || []).length, 1);
assert.equal((svg.match(/class="cl-curve cl-negative"/g) || []).length, 1);
["cl-tp", "cl-fn", "cl-fp", "cl-tn"].forEach((k) => assert.ok(svg.includes(`class="cl-region ${k}"`), k));
assert.ok(svg.includes(`id="cl-threshold-line" x1="${sx.px(0)}"`));
assert.ok((svg.match(/class="cl-bar cl-positive"/g) || []).length >= 3);
assert.ok(P.distributionPlot({ axis: "probability", range: [0, 1], bins: 40, positives: [], negatives: [], threshold: 0.5, fontSize: 16,
  labels: { positive: "", negative: "", x: "", y: "", empty: "no data" } }).includes("no data"));
const rocSvg = P.rocPlot({ primary: [{ fpr: 0, tpr: 0 }, { fpr: 0.2, tpr: 0.8 }, { fpr: 1, tpr: 1 }], secondary: null, operating: { fpr: 0.2, tpr: 0.8 }, fontSize: 16, labels: { x: "FPR", y: "TPR" } });
const rx = P.scale([0, 1], P.ROC.box);
assert.ok(rocSvg.includes(`id="cl-operating-point" cx="${rx.px(0.2)}"`));
assert.ok(!rocSvg.includes("cl-secondary"));
```

- [ ] **Step 2: Run the checks to verify they fail**

Run: `node src/check_classification.cjs`
Expected: `Cannot find module './static/classification-plots.js'`.

- [ ] **Step 3: Implement `src/static/classification-plots.js`**

Region construction: sample each curve on the grid, split the polyline at the threshold x (interpolating a point exactly at the threshold), close each half down to the baseline. Y scale: `[0, 1.08 × max(histogram counts, curve counts)]`, at least 1.

- [ ] **Step 4: Run the checks to verify they pass**

Run: `node src/check_classification.cjs`
Expected: `PASS: ...`.

- [ ] **Step 5: Commit**

```bash
git add src/static/classification-plots.js src/check_classification.cjs
git commit -m "Add SVG builders for the score distribution and ROC plots"
```

---

### Task 7: Lab shell, model card, site wiring, first browser check

**Files:**
- Create: `src/static/classification-lab.js`
- Create: `src/static/classification-lab.css`
- Modify: `src/index.html:14-17` (stylesheet and scripts), `src/index.html:55-57` (nav link)
- Modify: `src/static/app.js:319` (unmount), `:330` (title map), `:339` (view table), `:363-364` (mount)
- Modify: `src/static/i18n.js:2` (nav label)
- Modify: `src/check_site.py` (new `check_classification`, called right after `check_regression` at line 406)
- Modify: `README.md:12` (feature bullet; full README update is Task 11)

**Interfaces:**
- Consumes: `ML_TITANIC` (Task 3), `MLClassificationMath` (Tasks 4–5), `MLClassificationPlots` (Task 6).
- Produces: `window.MLClassificationLab = { html(language) -> string, mount(root: Element, language) -> void, unmount() -> void }`. `html` returns `<section id="classification-lab" class="cl-lab" aria-labelledby="cl-title">…</section>`. Internal state object `{ on: Set<string> (group ids), penalty, lambda, axis, threshold (probability), split, presentation, textScale }`; a `refit()` that calls `fit` with `columns` = indices of the active groups' columns and caches scores for train and test; `render()` that updates every panel; threshold moves must not refit.
- Element ids and hooks (used by Tasks 8–10 and the checks): `#cl-title`; `#cl-presentation` checkbox; `#cl-text-size` select (1 / 1.25 / 1.5); `#cl-groups` containing one `<label>` per group with `<input type="checkbox" data-cl-group="<id>">`; `#cl-penalty` select (`none`, `l2`, `l1`); `#cl-lambda-range` (log10, min −3, max 2, step 0.1) and `#cl-lambda` number (min 0.001, max 100), disabled when penalty is none; `#cl-coefficients` with one `.cl-coefficient` row per active column (`data-cl-column="<name>"`, `<strong>` value); `#cl-status` (`aria-live="polite"`); root attributes `data-cl-columns="<active column count>"` and `data-cl-iterations`.
- Placeholders for later tasks: empty `<section class="cl-card" id="cl-scores-card">`, `#cl-confusion-card`, `#cl-roc-card` so the page structure exists.

- [ ] **Step 1: Write the failing browser check** (in `check_site.py`, after `check_regression`)

```python
def check_classification(page, base: str, screenshots: Path):
    page.goto(base + "#classification?lang=en")
    lab = page.locator("#classification-lab")
    expect(page.locator("#cl-title")).to_have_text("Classification metrics")
    expect(page.locator('[data-nav="classification"]')).to_have_attribute("aria-current", "page")
    expect(page.locator("#cl-groups input")).to_have_count(14)
    expect(page.locator("#cl-groups input:checked")).to_have_count(10)
    expect(lab).to_have_attribute("data-cl-columns", "15")
    expect(page.locator("#cl-coefficients .cl-coefficient")).to_have_count(15)
    expect(page.locator("#cl-penalty")).to_have_value("l2")
    expect(page.locator("#cl-lambda")).to_have_value("1")
    female = page.locator('#cl-coefficients [data-cl-column="female"] strong')
    before = float(female.text_content())
    page.locator('[data-cl-group="title"]').uncheck()
    expect(lab).to_have_attribute("data-cl-columns", "11")
    assert abs(float(female.text_content()) - before) > 0.05
    page.locator('[data-cl-group="title"]').check()
    page.locator("#cl-lambda").fill("-5")
    page.locator("#cl-lambda").press("Tab")
    expect(page.locator("#cl-lambda")).to_have_value("1")           # invalid input is ignored
    page.locator("#cl-penalty").select_option("none")
    expect(page.locator("#cl-lambda")).to_be_disabled()
    page.locator('[data-cl-group="embarked_missing"]').check()
    expect(page.locator("#cl-status")).to_contain_text("separation")
    page.locator('[data-cl-group="embarked_missing"]').uncheck()
    page.locator('[data-cl-group="family_size"]').check()
    expect(page.locator("#cl-status")).to_contain_text("collinear")
    page.locator('[data-cl-group="family_size"]').uncheck()
    page.locator("#cl-penalty").select_option("l2")
    page.locator('[data-lang="ru"]').click()
    expect(page.locator("#cl-title")).to_have_text("Метрики классификации")
    page.locator('[data-lang="en"]').click()
    page.screenshot(path=str(screenshots / "classification-en.png"), full_page=True)
    print("PASS: classification lab loads, feature switches refit, invalid λ ignored, separation and collinearity reported, RU/EN")
```

Also add `"classification"` to the mobile route tuple at line 410.

- [ ] **Step 2: Build and run the check to verify it fails**

Run (worktree): `uv run python src/build.py && (uv run python -m http.server 8000 --bind 127.0.0.1 --directory dist & echo $! > /tmp/cl-server.pid) && sleep 1 && uv run python src/check_site.py --url http://127.0.0.1:8000/`
Expected: the regression checks pass, then `check_classification` fails on `#cl-title` (route renders "Entry not found").

- [ ] **Step 3: Wire the site**

`index.html`: add `<link rel="stylesheet" href="assets/classification-lab.css" />` after line 14 and `<script defer src="assets/titanic-data.js">`, `classification-math.js`, `classification-plots.js`, `classification-lab.js` after line 17 (before `app.js`); add the nav link `<a href="#classification" data-nav="classification"><span class="nav-icon">◑</span> Лаборатория классификации</a>` after the regression link. `app.js`: `window.MLClassificationLab.unmount();` next to line 319; `classification: "Лаборатория классификации",` in the title map; `classification: () => window.MLClassificationLab.html(language),` in the view table; `if (state.view === "classification") window.MLClassificationLab.mount($("#classification-lab"), language);` next to line 364. `i18n.js`: `"Лаборатория классификации": "Classification lab",` after line 2.

- [ ] **Step 4: Implement `html`, `mount`, `unmount` with the model card**

Heading: title "Classification metrics" / "Метрики классификации"; description "Switch features on and off, pick a penalty, then move the threshold and watch every metric respond." / "Включайте признаки, выбирайте штраф, двигайте порог и смотрите, как отвечают метрики."; flow strip `01 Model → 02 Scores → 03 Confusion matrix → 04 ROC`. Each group toggle is a `<label>` carrying the group's note (Task 2) as its `title` and in a `.cl-hint` span. Status texts: constant `"<columns> constant in the training set: weight fixed at 0"`; separation `"Perfect separation on <columns>: without a penalty the weights grow without bound. Add L2."`; collinear `"<columns> are collinear: without a penalty their coefficients are not unique; predictions and metrics still are."` (plus RU versions). `mount` installs listeners, performs the first `refit()` + `render()`, applies presentation (body class `lr-presenting`, root `cl-presentation`, CSS variable `--cl-text-scale`), and sets `teardown` to remove listeners and the body class; `unmount` calls `teardown`.

CSS: `.cl-lab`, `.cl-card` (same border/radius/background as `.lr-card`), `.cl-groups` wrapping grid of toggles, `.cl-coefficients` rows with a proportional bar, status styling, `--cl-text-scale: 1` on `.cl-lab`.

- [ ] **Step 5: Rebuild and run the checks**

Run: `uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/ && node src/check_classification.cjs`
Expected: `PASS: classification lab loads, …`, the existing PASS lines, no JavaScript errors.

- [ ] **Step 6: Commit**

```bash
git add src/static/classification-lab.js src/static/classification-lab.css src/index.html src/static/app.js src/static/i18n.js src/check_site.py README.md
git commit -m "Add the classification lab shell with feature switches and penalties"
```

---

### Task 8: Scores card — train/test switch, axis toggle, distribution plot, threshold

**Files:**
- Modify: `src/static/classification-lab.js`, `src/static/classification-lab.css`, `src/check_site.py`

**Interfaces:**
- Consumes: `distributionPlot`, `scale`, `DISTRIBUTION` (Task 6); cached train/test scores from Task 7's `refit()`.
- Produces hooks: `#cl-split` group with `<button data-cl-split="train|test" aria-pressed>` labelled "Train · 891" / "Test · 418" ("Обучение · 891" / "Тест · 418"); `#cl-axis` group with `<button data-cl-axis="logit|probability" aria-pressed>`; `<svg id="cl-plot" viewBox="0 0 960 420" role="img" tabindex="0">`; `#cl-threshold-range` and `#cl-threshold` number (probability: min 0.005, max 0.995, step 0.005; logit: min −6, max 6, step 0.05); `#cl-readout` text `p ≥ 0.50 (logit ≥ 0.00): <k> predicted survivors of <n>`; root attribute `data-cl-split`. Keyboard: arrows on the plot handle move the threshold by one step. Dragging the handle or clicking the plot sets the threshold at the pointer. Logit axis range: training scores min…max padded 5 %, clipped to [−8, 8].

- [ ] **Step 1: Extend the browser check**

```python
    expect(page.locator('[data-cl-axis="probability"]')).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#cl-threshold")).to_have_value("0.5")
    expect(page.locator("#cl-plot .cl-curve")).to_have_count(2)
    expect(page.locator("#cl-plot .cl-region")).to_have_count(4)
    expect(page.locator("#cl-readout")).to_contain_text("of 891")
    page.locator('[data-cl-axis="logit"]').click()
    expect(page.locator("#cl-threshold")).to_have_value("0")
    page.locator("#cl-threshold").fill("1.5")
    page.locator("#cl-threshold").press("Tab")
    page.locator('[data-cl-axis="probability"]').click()
    assert abs(float(page.locator("#cl-threshold").input_value()) - 0.8176) < 0.004   # σ(1.5), quantized to the 0.005 slider step
    page.locator('[data-cl-split="test"]').click()
    expect(lab).to_have_attribute("data-cl-split", "test")
    expect(page.locator("#cl-readout")).to_contain_text("of 418")
    page.locator('[data-cl-split="train"]').click()
    plot = page.locator("#cl-plot"); box = plot.bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    moved = float(page.locator("#cl-threshold").input_value())
    assert 0.3 < moved < 0.7, moved                                      # the plot's centre is p ≈ 0.475, not the 0.818 set before
    page.locator("#cl-threshold").fill("0.5"); page.locator("#cl-threshold").press("Tab")
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/`
Expected: fails at `[data-cl-axis="probability"]` (missing).

- [ ] **Step 3: Implement the scores card**

Threshold stored as probability `state.threshold`; the slider shows `logit(threshold)` when the axis is logit. `distributionPlot` receives the selected split's scores and the threshold already converted to the displayed axis (logits, or `probabilities(logits)` and `threshold`). Threshold moves re-render the plot regions, line and handle (rebuild the SVG string; the KDE/histogram inputs are cached per split and axis), the readout, and (Task 9) the confusion/metrics/operating point. Use `requestAnimationFrame` coalescing for pointer drags.

- [ ] **Step 4: Rebuild and run the checks**

Run: `uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/`
Expected: PASS lines, no JavaScript errors.

- [ ] **Step 5: Commit**

```bash
git add src/static/classification-lab.js src/static/classification-lab.css src/check_site.py
git commit -m "Add the score distribution with axis toggle, threshold slider and train/test switch"
```

---

### Task 9: Confusion matrix, metrics table, ROC card

**Files:**
- Modify: `src/static/classification-lab.js`, `src/static/classification-lab.css`, `src/check_site.py`

**Interfaces:**
- Consumes: `confusion`, `metrics`, `roc`, `logLoss`, `brier` (Task 5), `rocPlot`, `ROC` (Task 6).
- Produces hooks: `#cl-confusion` table with cells `#cl-tp`, `#cl-fn`, `#cl-fp`, `#cl-tn` (classes `cl-tp` … for tints) and margin totals; `#cl-metrics` table, one `<tr data-cl-metric="<key>">` per metric with `<td class="cl-formula">`, `<td class="cl-primary">` (selected split), `<td class="cl-secondary">` (other split), keys in this order: `accuracy, precision, recall, specificity, fpr, fnr, f1, balancedAccuracy, mcc, youdenJ, npv, auc, logLoss, brier`; undefined values render `—`; `<svg id="cl-roc" viewBox="0 0 480 480" role="img">`; `#cl-auc` badge `AUC train 0.86 · test 0.84` (RU: `AUC обучение … · тест …`). ROC curves for both splits are cached at refit; the operating point updates on threshold moves.

- [ ] **Step 1: Extend the browser check**

```python
    cells = [int(page.locator(f"#cl-{k}").text_content()) for k in ("tp", "fn", "fp", "tn")]
    assert sum(cells) == 891, cells
    ref = page.evaluate("ML_TITANIC.reference.train.confusion")
    assert cells == [ref["tp"], ref["fn"], ref["fp"], ref["tn"]], (cells, ref)
    auc_text = page.locator("#cl-auc").text_content()
    assert f"{page.evaluate('ML_TITANIC.reference.train.auc'):.2f}" in auc_text, auc_text
    assert f"{page.evaluate('ML_TITANIC.reference.test.auc'):.2f}" in auc_text, auc_text
    expect(page.locator("#cl-metrics tr[data-cl-metric]")).to_have_count(14)
    point_before = page.locator("#cl-operating-point").get_attribute("cx")
    page.locator("#cl-threshold").fill("0.9"); page.locator("#cl-threshold").press("Tab")
    assert page.locator("#cl-operating-point").get_attribute("cx") != point_before
    assert sum(int(page.locator(f"#cl-{k}").text_content()) for k in ("tp", "fn", "fp", "tn")) == 891
    page.locator("#cl-threshold").fill("0.5"); page.locator("#cl-threshold").press("Tab")
    page.locator('[data-cl-split="test"]').click()
    assert sum(int(page.locator(f"#cl-{k}").text_content()) for k in ("tp", "fn", "fp", "tn")) == 418
    page.locator('[data-cl-split="train"]').click()
    for group in page.locator("#cl-groups input:checked").all(): group.uncheck()          # intercept-only: every p = 342/891 ≈ 0.384
    expect(lab).to_have_attribute("data-cl-columns", "0")
    expect(page.locator('[data-cl-metric="auc"] .cl-primary')).to_have_text("0.500")
    expect(page.locator("#cl-tp")).to_have_text("0")                                      # nobody reaches p ≥ 0.5
    expect(page.locator('[data-cl-metric="precision"] .cl-primary')).to_have_text("—")
    expect(page.locator('[data-cl-metric="f1"] .cl-primary')).to_have_text("—")
    page.locator("#cl-threshold").fill("0.3"); page.locator("#cl-threshold").press("Tab")  # everybody predicted positive
    expect(page.locator('[data-cl-metric="npv"] .cl-primary')).to_have_text("—")
    page.goto(base + "#classification?lang=en")                                          # restore defaults
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/`
Expected: fails at `#cl-tp` (missing).

- [ ] **Step 3: Implement the confusion and ROC cards**

Confusion counts come from `confusion(logits, y, logit(state.threshold))` on the cached logits of each split. Formulas column (plain text): `(TP + TN) / n`, `TP / (TP + FP)`, `TP / (TP + FN)`, `TN / (TN + FP)`, `FP / (FP + TN)`, `FN / (FN + TP)`, `2·P·R / (P + R)`, `(TPR + TNR) / 2`, `(TP·TN − FP·FN) / √(…)`, `TPR − FPR`, `TN / (TN + FN)`, `∫ TPR d(FPR)`, `−mean(y log p + (1−y) log(1−p))`, `mean((p − y)²)`. One-line explanations in RU/EN per metric. Numbers to 3 decimals. The ROC card's paragraph: "Every threshold gives one (FPR, TPR) pair; sweeping it traces the curve. AUC is the area under it: the probability that a random survivor scores above a random non-survivor." (and RU).

- [ ] **Step 4: Rebuild and run the checks**

Run: `uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/`
Expected: PASS, no JavaScript errors.

- [ ] **Step 5: Commit**

```bash
git add src/static/classification-lab.js src/static/classification-lab.css src/check_site.py
git commit -m "Add the confusion matrix, metric table and ROC curve to the classification lab"
```

---

### Task 10: Lecture view, text size, mobile layout, screenshots

**Files:**
- Modify: `src/static/classification-lab.js`, `src/static/classification-lab.css`, `src/check_site.py` (`check_lecture_view` and the mobile block after line 426)

**Interfaces:**
- Consumes: shared preference keys and `body.lr-presenting` rules (Global Constraints).
- Produces: `.cl-presentation` rules: metric primaries, confusion cells and coefficient values at `calc(32px * var(--cl-text-scale))`, plot fonts via `--cl-plot-font` (`26px` presenting, `20px` otherwise, times the scale), cards keep full width; at ≤ 720 px the groups grid becomes two columns, the confusion matrix and the metrics table stack, SVGs keep `aspect-ratio` and `width: 100%`.

- [ ] **Step 1: Extend the checks**

In `check_lecture_view`, after the regression assertions:
```python
    page.goto(base + "#classification?lang=en")
    expect(page.locator("#cl-presentation")).to_be_checked()
    expect(page.locator("body")).to_have_class("lr-presenting")
    expect(page.locator("#sidebar")).not_to_be_visible()
    assert page.locator('[data-cl-metric="accuracy"] .cl-primary').evaluate("e => parseFloat(getComputedStyle(e).fontSize)") >= 32
    assert page.locator("#cl-plot text").first.evaluate("e => parseFloat(getComputedStyle(e).fontSize) * e.getScreenCTM().a") >= 25
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.select_option("#cl-text-size", "1.5")
    assert page.locator('[data-cl-metric="accuracy"] .cl-primary').evaluate("e => parseFloat(getComputedStyle(e).fontSize)") >= 48
    page.screenshot(path=str(screenshots / "lecture-classification-1080p.png"))
    page.locator("#cl-presentation").uncheck()
    expect(page.locator("#sidebar")).to_be_visible()
```
In the mobile block (per width), after the regression lines:
```python
            page.goto(base + "#classification?lang=en")
            expect(page.locator("#cl-plot .cl-curve")).to_have_count(2)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, "classification")
            page.screenshot(path=str(screenshots / f"classification-mobile-{width}.png"), full_page=True, animations="disabled")
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/`
Expected: fails on the 32 px font assertion or a mobile overflow.

- [ ] **Step 3: Implement presentation and responsive CSS, and the text-size handling**

- [ ] **Step 4: Rebuild and run the checks; look at the screenshots**

Run: `uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/`
Expected: PASS. Open `/tmp/ml-course-screenshots/classification-en.png`, `lecture-classification-1080p.png`, `classification-mobile-320.png` and confirm: four tinted regions visible, the threshold handle on the line, the ROC dot on the curve, nothing clipped.

- [ ] **Step 5: Commit**

```bash
git add src/static/classification-lab.js src/static/classification-lab.css src/check_site.py
git commit -m "Add lecture view and mobile layout to the classification lab"
```

---

### Task 11: README and final verification

**Files:**
- Modify: `README.md` (feature bullet at line 12, structure tree after line 42, Checks section after line 108, Provenance section)

- [ ] **Step 1: Update the README**

Add: feature bullet "**Classification lab:** logistic regression on the Titanic passengers fitted in the browser, with feature switches, L1/L2 penalties, a logit/probability score distribution with a threshold slider, the confusion matrix and twelve metrics, ROC/AUC, and a train/test switch. Open `#classification?lang=en`."; tree entries for `prepare_titanic.py`, `test_prepare_titanic.py`, `check_classification.cjs`, `titanic-data.js`, `classification-math.js`, `classification-plots.js`, `classification-lab.js`, `classification-lab.css`; in Checks: `uv run python -m unittest src/test_prepare_titanic.py` and `node src/check_classification.cjs`; a "Classification lab" paragraph stating the objective from Global Constraints, the scikit-learn equivalence (`LogisticRegression(penalty="l2", C=1/λ, fit_intercept=True, solver="lbfgs", tol=1e-8)` reproduces the L2 coefficients to about 1e-4 on the committed matrix; L1 ↔ `penalty="l1", solver="saga"` up to tolerance; under "none" with collinear columns only predictions are unique), the prep command, and the provenance sentence (Kaggle inputs not redistributed; test labels recovered from Vanderbilt's `titanic3.csv`, originally Encyclopedia Titanica; `titanic-data.js` carries derived numeric columns and labels only).

- [ ] **Step 2: Run everything once more from the worktree**

Run:
```bash
uv run python -m unittest src/test_prepare_titanic.py -v
node src/check_regression.cjs && node src/check_classification.cjs
uv run python src/build.py && uv run python src/check_site.py --url http://127.0.0.1:8000/
git status --porcelain
```
Expected: all PASS; `git status` shows only `README.md` modified. Then stop the server: `kill $(cat /tmp/cl-server.pid)`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document the classification lab, its data preparation and checks"
```

- [ ] **Step 4: Hand back**

Report the worktree path and branch, the list of commits, the screenshot paths, and that pushing to `main` (which deploys) awaits the author's word.
