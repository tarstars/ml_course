/* Analytic fixtures for the browser's actual numerical implementation. */
const assert = require("node:assert/strict");
const { fit, features, predict } = require("./static/regression-math.js");
const close = (actual, expected, tolerance = 1e-7) => {
  assert.ok(Math.abs(actual - expected) <= tolerance * (1 + Math.abs(expected)), `${actual} != ${expected}`);
};
const coefficients = (result, expected, tolerance) => {
  assert.equal(result.theta.length, expected.length);
  result.theta.forEach((x, i) => close(x, expected[i], tolerance));
};
const line = [-1, -0.5, 0, 0.5, 1].map((x) => ({ x, y: 0.3 + 0.8 * x }));
const ordinary = fit(line, "polynomial", 1);
coefficients(ordinary, [0.3, 0.8]);
close(ordinary.mse, 0);
close(ordinary.determinant, 12.5);
assert.equal(ordinary.rank, 2);
close(predict(0.25, ordinary.theta, "polynomial", 1), 0.5);

// Under-determined and repeated-x fits must select the minimum-norm solution.
coefficients(fit([{ x: 1, y: 1 }], "polynomial", 2), [1 / 3, 1 / 3, 1 / 3]);
const repeated = fit([{ x: 0, y: 0 }, { x: 0, y: 1 }], "polynomial", 3);
coefficients(repeated, [0.5, 0, 0, 0]);
assert.equal(repeated.rank, 1);
assert.equal(repeated.determinant, 0);
assert.equal(repeated.condition, Infinity);
const under = fit([{ x: -1, y: 1 }, { x: 1, y: 1 }], "polynomial", 4);
coefficients(under, [1 / 3, 0, 1 / 3, 0, 1 / 3]);

// With centered x and mean(x²) = 0.5, ridge and lasso have analytic solutions.
coefficients(fit(line, "polynomial", 1, "l2", 0.25), [0.3, 0.8 * 0.5 / 0.75]);
const sparse = fit(line, "polynomial", 1, "l1", 0.1);
coefficients(sparse, [0.3, 0.6]);
assert.equal(sparse.converged, true);
coefficients(fit(line, "polynomial", 1, "l1", 10), [0.3, 0]);
coefficients(fit(line, "polynomial", 1, "l2", 100), [0.3, 0.4 / 100.5]);
for (const kind of ["none", "l1", "l2"]) {
  coefficients(fit(line, "polynomial", 1, kind, 0), ordinary.theta);
  const original = fit(line, "polynomial", 1, kind, 0.1);
  const duplicated = fit([...line, ...line], "polynomial", 1, kind, 0.1);
  coefficients(duplicated, original.theta);
  close(duplicated.objective, original.objective);
  coefficients(fit([{ x: 0.2, y: 0.7 }, { x: -0.6, y: 0.7 }], "polynomial", 0, kind, 10), [0.7]);
}
const ridge = fit(line, "polynomial", 1, "l2", 0.25);
close(ridge.system[0][0], ridge.gram[0][0]);
close(ridge.system[1][1] - ridge.gram[1][1], line.length * 0.25);

const fourierTheta = [0.1, 0.6, -0.2, 0.15, 0.05];
const wave = Array.from({ length: 30 }, (_, i) => {
  const x = -1 + 2 * i / 30;
  return { x, y: predict(x, fourierTheta, "fourier", 2) };
});
coefficients(fit(wave, "fourier", 2), fourierTheta);
assert.equal(features(0.2, "fourier", 10).length, 21);
const empty = fit([], "polynomial", 3);
assert.equal(empty.theta, null);
assert.equal(empty.mse, null);

// High-degree power features exercise stability without forming an inverse.
const high = Array.from({ length: 80 }, (_, i) => {
  const x = -1 + 2 * i / 79;
  return { x, y: 0.15 - 0.4 * x + 0.6 * x * x };
});
const highFit = fit(high, "polynomial", 20);
assert.equal(highFit.rank, 21);
assert.ok(highFit.mse < 1e-18, `high-degree MSE ${highFit.mse}`);
assert.ok(highFit.theta.every(Number.isFinite));
for (const kind of ["none", "l1", "l2"]) {
  const dense = Array.from({ length: 200 }, (_, i) => ({ x: -1 + 2 * i / 199, y: Math.sin(i * 0.1) }));
  const start = performance.now();
  const fitted = fit(dense, "polynomial", 20, kind, 0.01);
  assert.ok(fitted.theta.every(Number.isFinite));
  console.log(`${kind}: 200 observations, 21 coefficients, ${(performance.now() - start).toFixed(1)} ms`);
}
console.log("PASS: analytic OLS/ridge/lasso, minimum-norm singular fits, Fourier, zero penalty, unpenalized intercept, dataset duplication, high degree, empty data");
