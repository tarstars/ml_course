# Classification metrics lab — design

Date: 2026-11-03. Status: draft for approval.
Site: https://tarstars.github.io/ml_course/ (repo `tarstars/ml_course`, GitHub Pages, static, no backend).
Predecessor: the regression lab (`src/static/regression-lab.js`, route `#regression`), which sets the conventions this lab follows.

## 1. Intent

A second interactive lab for the ML course site that explains **binary classification quality metrics** using **logistic regression on the Titanic data**. It is shown on a projector during lectures and used by students at home. Everything runs in the browser.

The teaching story, in the order the page is read:

1. **Model.** Which passenger features are switched on, and how strongly the fit is regularized (none / L2 / L1).
2. **Scores and threshold.** Every passenger gets a score (logit or probability). The two classes (survived / did not) form two distributions on that axis. A threshold cuts the axis into "predicted survived" and "predicted died".
3. **Confusion matrix and metrics.** The cut produces TP / FN / FP / TN, and from them precision, recall, F1 and the rest. Moving the threshold trades one metric for another.
4. **ROC and AUC.** Sweeping the threshold through all values traces the ROC curve; AUC summarizes it independent of any threshold. The current threshold is a dot on the curve.

A **train / test switch** shows every panel for the training set or for an independent test set, so the gap between fitting and evaluation is visible without leaving the page.

## 2. Goals, non-goals, anti-goals

**Goals**

- Every panel reacts immediately to: feature switches, penalty type and strength, logit/probability axis toggle, threshold slider, train/test switch.
- Numbers students can verify in a notebook with scikit-learn on the same prepared matrix (see §5.3).
- Bilingual RU/EN, presentation mode with larger text, keyboard-operable, usable at phone width. Same quality bar as the regression lab.
- Data preparation is a reproducible script whose output is committed, so the site build stays unchanged.

**Non-goals (v1)**

- No feature-engineering tutorial, no model comparison across algorithms, no hyperparameter search, no Kaggle submission file.
- No precision–recall curve, no calibration plot, no cost-sensitive decision panel (reasonable follow-ups; keep the code open to them).
- No in-browser editing of the data.

**Anti-goals (failure even if it "works")**

- Panels that disagree with each other or lag behind the slider.
- Numbers that differ from scikit-learn on the same matrix beyond solver tolerance.
- Wrong or dubious test-set labels.
- A page that stutters on a lecture laptop. Threshold moves must stay well under one frame; a refit under 50 ms.

## 3. Decisions made with the author (theirs)

- **Layout C, "lecture scroll":** everything full width, stacked in teaching order (model → scores and threshold → confusion matrix and metrics → ROC). Chosen from three mockups; the mockup file is `.superpowers/brainstorm/3437280-1793693314/content/layout.html`, option C.
- **Training set** = Kaggle `train.csv`, 891 passengers, 342 survived. **Not split**; the author considers 891 too small to split.
- **Test set** = Kaggle `test.csv`, 418 passengers, with labels recovered from the public Vanderbilt `titanic3.csv` (1309 passengers, from Encyclopedia Titanica, https://hbiostat.org/data/repo/titanic3.csv). Verified on 2026-11-03: all 418 test rows match by name + ticket + class + sex; all 891 train rows match and their labels agree 891/891; recovered test labels: 158 survived (37.8%).
- **Features:** raw columns, engineered features, and a "was this field filled?" indicator for every raw field that has gaps. One switch per feature group.
- **Regularization:** none / L2 / L1 with a strength slider, as in the regression lab.
- **Score axis:** toggle between logit and probability; threshold slider; two class curves in different colours.
- **Metrics:** confusion matrix and its components, precision, recall, F1 "and others"; ROC with AUC; all reacting to the threshold.

## 4. Step one: data preparation

### 4.1 Inputs (not committed)

- `train.csv`, `test.csv` from Kaggle (locally at `~/prj/kaggle/titanic/data/raw/`).
- `titanic3.csv` from Vanderbilt, stored next to them at `~/prj/kaggle/titanic/data/raw/titanic3.csv` (116 752 bytes, sha256 starts `db6df9666818c69a`; the script takes a local path and documents the URL).

Kaggle's files are not redistributed in the repo. The committed output contains only derived numeric columns and labels, no names or tickets.

### 4.2 Script

`src/prepare_titanic.py`, **standard library only** (csv, json, re, statistics, math), run as

```
uv run python src/prepare_titanic.py --train <train.csv> --test <test.csv> --titanic3 <titanic3.csv> --out src/static/titanic-data.js
```

It must be leakage-free: every statistic used to transform rows (medians, means, standard deviations, modes) is computed on the training rows only and then applied to both sets. Test labels are only ever used to write the `y` vector of the test set.

Label recovery, exactly the rule verified on 2026-11-03: normalize a string by removing double quotes, collapsing runs of whitespace to one space, stripping, and lower-casing. Key = (normalized Name, normalized Ticket, Pclass as a string, normalized Sex); titanic3's columns are `name`, `ticket`, `pclass`, `sex`. If the key is absent, fall back to the normalized name alone when it is unique in titanic3 (two names are duplicated there: "Connolly, Miss. Kate" and "Kelly, Mr. James"; both resolve through the full key). With this rule all 891 train rows and all 418 test rows match. The script aborts if any test row is unmatched or any train label disagrees with titanic3.

### 4.3 Feature groups (one switch each)

| # | Group (switch label) | Columns produced | Kind | Default |
|---|---|---|---|---|
| 1 | Passenger class | `pclass_1`, `pclass_2` (reference: 3rd) | binary | on |
| 2 | Sex | `female` | binary | on |
| 3 | Age | `age` (imputed, standardized) | standardized | on |
| 4 | Age was filled | `age_missing` | binary | on |
| 5 | Siblings / spouses | `sibsp` | standardized | on |
| 6 | Parents / children | `parch` | standardized | on |
| 7 | Fare | `log_fare` = log(1 + fare), standardized | standardized | on |
| 8 | Fare was filled | `fare_missing` | binary | off |
| 9 | Port of embarkation | `embarked_c`, `embarked_q` (reference: S) | binary | on |
| 10 | Port was filled | `embarked_missing` | binary | off |
| 11 | Title from name | `title_mrs`, `title_miss`, `title_master`, `title_rare` (reference: Mr) | binary | on |
| 12 | Family size | `family_size` = sibsp + parch + 1, standardized | standardized | off |
| 13 | Travelling alone | `is_alone` | binary | off |
| 14 | Cabin was filled | `cabin_known` | binary | on |

Rules:

- **Title** is the token between the comma and the first period in Name. Map Mlle, Ms → Miss; Mme → Mrs; keep Mr, Mrs, Miss, Master; everything else → Rare.
- **Age imputation:** median age of the passenger's title group, computed on training rows; `age_missing` records whether the original value was absent.
- **Fare imputation** (one test row): median training fare of the passenger's class. **Embarked imputation** (two training rows): training mode (S).
- **Standardized** columns use training mean and standard deviation. **Binary** columns stay 0/1 so coefficients read as plain logit shifts ("female adds +2.4 to the logit").
- Groups 8 and 10 are constant or nearly constant in training (0 and 2 rows). They are kept because the author asked for a filled-or-not feature per field, default off, and the app must handle them gracefully (§5.3). The two Embarked-missing passengers both survived, which gives perfect separation under no penalty; this is a teaching moment, not a bug, and the UI says so.
- Group 12 is an exact affine combination of groups 5 and 6 plus the intercept, so with all three on and no penalty the design matrix is rank-deficient. It is therefore **off by default** (my call), and the app handles the rank-deficient case gracefully when a user switches it on (§5.3). Group 13 is different: `is_alone` is an indicator of `family_size == 1`, not an affine combination of other columns, so the fit with it has full rank even under no penalty; it is off by default only to keep the default feature set small.

### 4.4 Output: `src/static/titanic-data.js`

Defines `window.ML_TITANIC` (a JS asset, not a fetched JSON, so it loads synchronously and gets the build's content versioning). Rounded to 4 decimals. Expected size under 300 KB. Shape:

```
{
  schema: 1,
  columns: ["pclass_1", "pclass_2", "female", ...],          // order of X columns
  groups: [{ id, label: {en, ru}, columns: [...], kind, default_on, note: {en, ru} }, ...],
  train: { X: [[...], ...], y: [0|1, ...] },                   // 891 rows
  test:  { X: [[...], ...], y: [0|1, ...] },                   // 418 rows
  meta: { source, prepared_at, counts: {train, test, train_positive, test_positive},
          imputation: { age_by_title: {...}, fare_by_class: {...}, embarked_mode }, scaling: {column: {mean, std}} },
  reference: { ... }                                           // see §5.6
}
```

The script prints a summary (row counts, positives, imputation values, column means) and aborts on any NaN or on unexpected counts (train 891 / 342 positive; test 418 / 158 positive).

## 5. Step two: the application

### 5.1 Placement and wiring

- Route `#classification`; sidebar link after the regression lab: "Лаборатория классификации" / "Classification lab" (nav strings go into `i18n.js`; everything inside the lab uses the regression lab's inline `w(en, ru)` pattern).
- `src/static/classification-lab.js` exposes `window.MLClassificationLab = { html(language), mount(root, language), unmount() }`, registered in `app.js` exactly like `MLRegressionLab` (view table, title map, unmount on render, mount after translate).
- `src/static/classification-math.js` exposes `window.MLClassificationMath` with pure functions: fit, predict, confusion/metrics at a threshold, ROC + AUC, log loss, Brier, histogram/KDE. No DOM. Loadable from Node for the numeric check.
- `src/static/classification-lab.css`, class prefix `cl-`, built on the site tokens (`--paper`, `--ink`, `--green`, `--accent`, `--line`, `--muted`).
- `index.html` gains the three script/style tags; `build.py` needs no change (it copies `static/` and versions `assets/*.js|css`).
- Eyebrow "INTERACTIVE LAB / 02". Heading, description, "all in your browser" badge, lecture-view checkbox and text-size select mirror the regression lab and **share its localStorage keys** (`ml-regression-presentation`, `ml-regression-text-scale`) so one setting serves both labs. Because the regression lab defaults that key to on, this lab also opens in lecture view by default.
- **Lecture view on this page means** (my call): hide the site chrome the same way the regression lab does (`body.lr-presenting` rules, or a shared class), scale text and plot fonts by the chosen factor, and enlarge the plots. It does **not** introduce a second arrangement; the author chose the stacked layout knowing the slider and the ROC may not share one projector screen. One layout, one set of overflow checks.

### 5.2 Page structure (layout C)

Full-width cards, top to bottom:

1. **Model card.** Feature switches as a wrapping row of labelled toggles (a short note per group on hover/focus); penalty segmented control none / L2 / L1; logarithmic strength slider with numeric input (range 10⁻³ … 10², default λ = 1 under L2, as in the regression lab); a compact **coefficients readout** (column name, value, small bar; L1 zeros shown as exactly 0); a one-line **status**: number of active columns, iterations, and any warning (constant column dropped, separation detected).
2. **Scores card.** Train / test segmented control ("Train · 891", "Test · 418"); logit / probability axis toggle; the distribution plot (§5.4); directly under it the threshold slider with numeric input; a readout line "p ≥ 0.50 (logit ≥ 0.00): 303 predicted survivors of 891".
3. **Confusion card.** 2×2 matrix (rows actual survived / died, columns predicted survived / died) with counts and margins, cells tinted in the same four colours as the plot regions; next to it the metrics table (§5.5).
4. **ROC card.** ROC plot with the diagonal, the selected split's curve solid, the other split's curve faint and dashed, the current operating point (FPR, TPR) as a dot that follows the slider; badge "AUC train 0.86 · test 0.84". One-paragraph explanation.

On narrow screens the cards simply stack; nothing may scroll horizontally at 360 px and up (the existing mobile check applies).

### 5.3 Fitting

- Logistic regression with intercept on the **training rows only**, columns = union of switched-on groups.
- Objective (matches scikit-learn's parameterization with C = 1/λ, intercept unpenalized):
  `Σᵢ log(1 + exp(−sᵢ zᵢ)) + λ·P(w)`, where `P = ½‖w‖²` for L2, `‖w‖₁` for L1, and λ = 0 for "none". `sᵢ = ±1`, `zᵢ = w·xᵢ + b`.
- Solver: Newton / IRLS for none and L2; for L1 a proximal Newton or glmnet-style coordinate descent, builder's choice. Convergence: objective change below 1e-8 or 100 iterations. Budget: under 50 ms for 891 × 20 on a laptop.
- **Constant columns** in the training rows (zero variance, e.g. `fare_missing`) are excluded from the fit, reported with weight 0 and a status note.
- **Separation** (unpenalized objective keeps decreasing while ‖w‖ grows): stop at the iteration cap, show the status note "perfect separation on …: without a penalty the weights grow without bound; add L2", keep the page responsive. Never throw.
- **Collinear columns** (e.g. family size with sibsp and parch): with a penalty the objective is strictly convex and nothing special happens. Under "none" the Newton step must use a minimum-norm solve (SVD pseudo-inverse, as the regression lab does for singular systems) and the status note says "sibsp, parch and family size are collinear: without a penalty their coefficients are not unique; predictions and metrics still are". The README's scikit-learn claim is stated for predictions and metrics in that case, not coefficients.
- Scikit-learn equivalence to document in the README: `LogisticRegression(penalty="l2", C=1/λ, fit_intercept=True, solver="lbfgs", tol=1e-8)` on the committed matrix reproduces the L2 coefficients to about 1e-4; L1 corresponds to `penalty="l1", solver="saga"` up to solver tolerance.

### 5.4 Distribution plot

- SVG. Horizontal axis: logit (range = training scores' min…max padded by 5 %, clipped to [−8, 8]) or probability [0, 1]. Vertical axis: count per bin.
- For each class: a **histogram** (shared bins, say 40 across the axis) drawn faintly, and a **smooth curve** (Gaussian KDE, Silverman bandwidth, scaled to counts per bin so curve and bars agree; reflected at 0 and 1 on the probability axis). Curves are the primary mark; bars give honesty about sample size.
- Both marks are scaled to **class counts**, not normalized densities, so class imbalance is visible and the four tinted regions correspond to TP / FN / FP / TN by area.
- Threshold: a vertical line with a draggable handle (pointer events on the SVG as an alternative to the slider). Regions: under the survived curve, right of the line = TP, left = FN; under the died curve, right = FP, left = TN. Tints match the confusion matrix cells.
- Colours: survived uses the site green (`--green`), died uses the regression lab's purple (`#817199`) or brass (`--accent`); the pair must remain distinguishable in grayscale (different dash pattern on the curves). Builder picks within the palette.
- Switching the axis keeps the **same cut**: the threshold is stored as a probability; the slider's range and step follow the displayed axis (probability 0.005 steps; logit −6…6 in 0.05 steps) and is repositioned on toggle. Default 0.5 / 0.
- The plot shows the **selected split** (train or test) scored by the model fitted on train.

### 5.5 Metrics

At the current threshold, for the selected split, with the other split's value shown small beside it so the gap is visible without toggling:

accuracy; precision (PPV); recall (TPR, sensitivity); specificity (TNR); false positive rate; false negative rate; F1; balanced accuracy; Matthews correlation coefficient; Youden's J (TPR − FPR); negative predictive value.

Threshold-free, same two-column form: ROC AUC; log loss (mean); Brier score.

Each row shows the formula in symbols (e.g. `TP / (TP + FP)`) and a one-line RU/EN explanation. Undefined values (0/0) render as "—" with a hint, never NaN. Values update within the same frame as the slider.

### 5.6 ROC and AUC

- Points for every distinct score; FPR on x, TPR on y; AUC by the trapezoid rule, which equals the rank (Mann–Whitney) statistic with ties averaged.
- Recomputed on refit or split change (O(n log n)); the operating point on slider moves is O(1) from the current confusion counts.

### 5.7 Reference values and checks

- The prep script stores in `reference` the training fit for the default configuration (default-on groups, L2, λ = 1): coefficients, intercept, train/test log loss and AUC, and the confusion matrix at 0.5. The builder decides how the script computes them (a pure-Python Newton solver is enough; scikit-learn may be used as a dev-only tool to cross-check but is not a dependency of the build).
- `src/check_classification.cjs` (Node, like `check_regression.cjs`) loads the math module and the data asset and asserts the browser solver reproduces `reference` to 1e-4, plus unit cases: metrics on a hand-made 10-row example, AUC with ties, constant-column handling, separation handling.
- `check_site.py` gains `check_classification`: open `#classification?lang=en`; title; default metrics match `reference`; toggling a feature changes the coefficients and metrics; dragging the slider keeps TP+FN+FP+TN equal to the split size and moves the ROC dot; train/test switch changes the counts to 418; RU/EN switch; mobile widths without horizontal scroll; screenshots.

### 5.8 Documentation and delivery

- README: add the lab to the feature list and the structure tree; document the prep command, data provenance (Kaggle for the inputs, Vanderbilt titanic3 for the test labels, Encyclopedia Titanica as the original source) and the scikit-learn equivalence.
- Add `.superpowers/` to `.gitignore`.
- Work lands on `main` in small commits. Pushing (which deploys via the Pages workflow) happens when the author says so.
- **Where to build and verify** (my call): the repo root doubles as the author's download folder (about 200 untracked files), and `build.py`'s validation rejects uncataloged files, so the working copy cannot be built as is. Build, serve and run the checks from a clean `git worktree` of `main`, and stage commits by explicit path. Do not relax the uncataloged-file check.
- Local Node is v18; `check_classification.cjs` must not rely on Node 20+ APIs.

## 6. Constraints

- Static site, no backend, no runtime network requests, no new runtime dependencies for the build; vendored assets only.
- Keep the regression lab untouched except for shared wiring (`app.js`, `index.html`, `i18n.js`).
- Accessibility parity with the regression lab: native range inputs, `aria-live` on metric readouts, focusable SVGs with text alternatives, visible focus.
- Do not walk the FUSE mounts under `~/prj` (`arcadia`, `arc00`–`arc02`); the Kaggle folder `~/prj/kaggle` is local.

## 7. Calls made after the builder check

A fresh reviewer read this document as a builder and asked four things; none needed the author, and they are settled above as follows:

1. Family size is collinear with sibsp + parch under no penalty → default off, min-norm Newton step and a status note when switched on (§4.3, §5.3). Travelling alone (`is_alone`, an indicator of `family_size == 1`) is not collinear; it is default off only to keep the default feature set small.
2. Lecture view = hidden chrome plus text scaling, no second arrangement (§5.1).
3. Exact label-recovery rule and the location of `titanic3.csv` are now written down (§4.1, §4.2).
4. Build and check from a clean worktree; Node 18 (§5.8).

## 8. Left to the builder

- L1 solver choice; KDE bandwidth details; SVG geometry; exact palette within the site tokens; wording of explanations; internal module structure beyond the public APIs in §5.1; how `reference` is computed; whether the coefficients readout is a table or bars.
