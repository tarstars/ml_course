/* Analytic fixtures and the shipped reference fit for the browser's logistic regression. */
const assert = require("node:assert/strict");
const M = require("./static/classification-math.js");
const D = require("./static/titanic-data.js");
const close = (actual, expected, tolerance = 1e-7) => {
  assert.ok(Math.abs(actual - expected) <= tolerance * (1 + Math.abs(expected)), `${actual} != ${expected}`);
};

// Closed form: a single binary feature, 3/10 positives at x = 0 and 7/10 at x = 1.
const X1 = [...Array(10).fill([0]), ...Array(10).fill([1])];
const y1 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0];
const single = M.fit(X1, y1, { penalty: "none", lambda: 0 });
close(single.intercept, Math.log(3 / 7), 1e-6);
close(single.weights[0], 2 * Math.log(7 / 3), 1e-6);
assert.equal(single.converged, true);
assert.equal(single.separated, false);

// Intercept only: the logit of the positive rate (10/20 here).
const none = M.fit(X1, y1, { penalty: "l2", lambda: 1, columns: [] });
close(none.intercept, 0, 1e-9);
assert.deepEqual(none.weights, [0]);

// A deterministic, non-separable 60×4 dataset (28 positives); the label depends on three columns plus pseudo-noise.
const X2 = Array.from({ length: 60 }, (_, i) => [Math.sin(i), Math.cos(i / 2), Math.sin(i / 3), ((i % 7) - 3) / 3]);
const y2 = X2.map((x, i) => +(x[0] + 0.5 * x[1] + 0.25 * x[3] + 0.8 * Math.sin(13 * i) > 0.2));
// Xᵀ(p − y) + λw for the weights, Σ(p − y) (no penalty) for the intercept as the last component.
const gradient = (X, y, w, b, lambda) => {
  const g = Array(w.length + 1).fill(0);
  X.forEach((row, i) => {
    const residual = M.sigmoid(b + row.reduce((sum, x, j) => sum + x * w[j], 0)) - y[i];
    row.forEach((x, j) => { g[j] += residual * x; });
    g[w.length] += residual;
  });
  return g.map((v, j) => (j < w.length ? v + lambda * w[j] : v));
};
const ridge = M.fit(X2, y2, { penalty: "l2", lambda: 0.5 });
assert.ok(Math.max(...gradient(X2, y2, ridge.weights, ridge.intercept, 0.5).map(Math.abs)) < 1e-7);
const lasso = M.fit(X2, y2, { penalty: "l1", lambda: 2 });
assert.ok(lasso.weights.some((w) => w === 0) && lasso.weights.some((w) => w !== 0), `${lasso.weights}`);
const g = gradient(X2, y2, lasso.weights, lasso.intercept, 0);
lasso.weights.forEach((w, j) => (w !== 0
  ? assert.ok(Math.abs(g[j] + 2 * Math.sign(w)) < 1e-5)
  : assert.ok(Math.abs(g[j]) <= 2 + 1e-5)));
assert.ok(Math.abs(g[g.length - 1]) < 1e-7);
const flat = M.fit(X2, y2, { penalty: "l1", lambda: 1e4 });
assert.ok(flat.weights.every((w) => w === 0));
close(flat.intercept, M.logit(y2.reduce((a, b) => a + b) / y2.length), 1e-6);

// Separation: finite weights and a flag under "none", nothing special with a penalty.
const sep = M.fit([[0], [1], [0], [1]], [0, 1, 0, 1], { penalty: "none", lambda: 0 });
assert.equal(sep.separated, true);
assert.ok(sep.weights.every(Number.isFinite) && Number.isFinite(sep.intercept));
assert.equal(M.fit([[0], [1], [0], [1]], [0, 1, 0, 1], { penalty: "l2", lambda: 1 }).separated, false);

// A constant column is dropped; a test row with a nonzero value there is scored as if it did not exist.
const Xc = X2.map((row) => [...row, 0]);
const constant = M.fit(Xc, y2, { penalty: "l2", lambda: 0.5 });
assert.deepEqual(constant.constant, [4]);
assert.equal(constant.weights[4], 0);
close(M.scores([[...X2[0], 1]], constant.weights, constant.intercept)[0],
  M.scores([X2[0]], ridge.weights, ridge.intercept)[0], 1e-9);

// Collinear a, b, a + b under "none": the flag is set and predictions equal those of the a, b fit.
const Xab = X2.map((r) => [r[0], r[1]]);
const Xabc = X2.map((r) => [r[0], r[1], r[0] + r[1]]);
const ab = M.fit(Xab, y2, { penalty: "none", lambda: 0 });
const abc = M.fit(Xabc, y2, { penalty: "none", lambda: 0 });
assert.equal(abc.collinear, true);
assert.equal(ab.collinear, false);
const abScores = M.scores(Xab, ab.weights, ab.intercept);
M.scores(Xabc, abc.weights, abc.intercept).forEach((s, i) => close(s, abScores[i], 1e-6));
// solveSymmetric: minimum-norm solution, eigenvalues at or below 1e-6 of the largest count as zero.
const singular = M.solveSymmetric([[1, 1], [1, 1]], [2, 2]);
assert.equal(singular.rank, 1);
singular.x.forEach((x) => close(x, 1));
assert.deepEqual(M.solveSymmetric([[1, 0], [0, 1e-7]], [1, 1]), { x: [1, 0], rank: 1 });
assert.equal(M.solveSymmetric([[1, 0], [0, 1e-5]], [1, 1]).rank, 2);
close(M.probabilities([0])[0], 0.5);
close(M.sigmoid(-800), 0);
close(M.sigmoid(800), 1);
close(M.logit(M.sigmoid(1.5)), 1.5);

// Reference reproduction on the shipped data (default columns, L2, λ = 1).
const active = D.reference.columns.map((name) => D.columns.indexOf(name));
const start = performance.now();
const ref = M.fit(D.train.X, D.train.y, { penalty: "l2", lambda: 1, columns: active });
const elapsed = performance.now() - start;
active.forEach((j, k) => close(ref.weights[j], D.reference.weights[k], 1e-4));
close(ref.intercept, D.reference.intercept, 1e-4);
assert.ok(elapsed < 100, `fit took ${elapsed} ms`);
console.log(`reference fit: ${D.train.X.length} rows, ${active.length} columns, ${ref.iterations} iterations, ${elapsed.toFixed(1)} ms`);

// The shipped edge cases the lab reports: fare_missing constant, family_size collinear, embarked_missing separated.
const column = (name) => D.columns.indexOf(name);
const all = M.fit(D.train.X, D.train.y, { penalty: "l2", lambda: 1 });
assert.deepEqual(all.constant, [column("fare_missing")]);
const family = M.fit(D.train.X, D.train.y, { penalty: "none", lambda: 0, columns: [...active, column("family_size")] });
assert.equal(family.collinear, true);
assert.equal(family.rank, active.length + 1);
const embarked = M.fit(D.train.X, D.train.y, { penalty: "none", lambda: 0, columns: [...active, column("embarked_missing")] });
assert.equal(embarked.separated, true);
assert.equal(embarked.collinear, false);
assert.ok(embarked.weights.every(Number.isFinite) && Number.isFinite(embarked.intercept));
// L1 at the weakest penalty with the collinear columns still converges within the outer cap.
assert.equal(M.fit(D.train.X, D.train.y, { penalty: "l1", lambda: 1e-3, columns: [...active, column("family_size")] }).converged, true);
// Confusion, metrics, ROC, losses, histogram and density helpers.
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
const P = require("./static/classification-plots.js");
const sx = P.scale([-6, 6], P.DISTRIBUTION.box);
close(sx.px(0), (P.DISTRIBUTION.box.left + P.DISTRIBUTION.box.right) / 2, 1e-9); close(sx.value(sx.px(2.5)), 2.5, 1e-9);
const svg = P.distributionPlot({ axis: "logit", range: [-6, 6], bins: 40, positives: [1, 2, 3, 2.5], negatives: [-1, -2, -3, 0.5], threshold: 0,
  fontSize: 16, labels: { positive: "survived", negative: "died", x: "logit", y: "passengers", empty: "no data" } });
assert.equal((svg.match(/class="cl-curve cl-positive"/g) || []).length, 1);
assert.equal((svg.match(/class="cl-curve cl-negative"/g) || []).length, 1);
["cl-tp", "cl-fn", "cl-fp", "cl-tn"].forEach((k) => assert.ok(svg.includes(`class="cl-region ${k}"`), k));
assert.ok(svg.includes(`id="cl-threshold-line" class="cl-threshold" x1="${sx.px(0)}"`));
assert.ok((svg.match(/class="cl-bar cl-positive"/g) || []).length >= 3);
assert.ok(P.distributionPlot({ axis: "probability", range: [0, 1], bins: 40, positives: [], negatives: [], threshold: 0.5, fontSize: 16,
  labels: { positive: "", negative: "", x: "", y: "", empty: "no data" } }).includes("no data"));
const rocSvg = P.rocPlot({ primary: [{ fpr: 0, tpr: 0 }, { fpr: 0.2, tpr: 0.8 }, { fpr: 1, tpr: 1 }], secondary: null, operating: { fpr: 0.2, tpr: 0.8 }, fontSize: 16, labels: { x: "FPR", y: "TPR" } });
const rx = P.scale([0, 1], P.ROC.box);
assert.ok(rocSvg.includes(`id="cl-operating-point" class="cl-operating" cx="${rx.px(0.2)}"`));
assert.ok(!rocSvg.includes("cl-secondary"));
console.log("PASS: closed-form fit, intercept only, L2 optimality, L1 KKT and exact zeros, separation, constant and collinear columns, minimum-norm solve, reference reproduction, shipped edge cases, confusion matrix, metrics with null denominators, ROC and AUC, log loss and Brier, histogram, Silverman bandwidth and reflected KDE, distribution and ROC SVG builders");
