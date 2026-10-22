/* Small dense regression problems, with no runtime dependencies. */
(function (root) {
  "use strict";
  const zeros = (n) => Array(n).fill(0);
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);

  function features(x, basis, complexity) {
    if (basis === "fourier") {
      const row = [1];
      for (let k = 1; k <= complexity; k++)
        row.push(Math.sin(Math.PI * k * x), Math.cos(Math.PI * k * x));
      return row;
    }
    const row = [1];
    for (let j = 1; j <= complexity; j++) row.push(row[j - 1] * x);
    return row;
  }

  // One-sided Jacobi SVD: orthogonalize columns of A directly, avoiding
  // the squared condition number of an eigendecomposition of AᵀA.
  function decompose(A, p) {
    const columns = Array.from({ length: p }, (_, j) => A.map((row) => row[j]));
    const vectors = Array.from({ length: p }, (_, j) =>
      Array.from({ length: p }, (_, i) => +(i === j)),
    );
    for (let sweep = 0; sweep < 100; sweep++) {
      let changed = false;
      for (let j = 0; j < p; j++) {
        for (let k = j + 1; k < p; k++) {
          const a = dot(columns[j], columns[j]);
          const b = dot(columns[k], columns[k]);
          const cross = dot(columns[j], columns[k]);
          if (!a || !b || Math.abs(cross) <= 1e-14 * Math.sqrt(a) * Math.sqrt(b)) continue;
          const tau = (b - a) / (2 * cross);
          const tangent = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.hypot(1, tau));
          const c = 1 / Math.hypot(1, tangent), s = c * tangent;
          for (const pair of [columns, vectors]) {
            for (let i = 0; i < pair[j].length; i++) {
              const old = pair[j][i];
              pair[j][i] = c * old - s * pair[k][i];
              pair[k][i] = s * old + c * pair[k][i];
            }
          }
          changed = true;
        }
      }
      if (!changed) break;
    }
    const singular = columns.map((column) => Math.sqrt(dot(column, column)));
    const largest = Math.max(0, ...singular);
    const cutoff = largest * Math.max(A.length, p) * Number.EPSILON * 10;
    const retained = singular.filter((s) => s > cutoff);
    const rank = Math.min(A.length, retained.length);
    return {
      rank,
      condition: rank < p ? Infinity : largest / Math.min(...retained),
      determinant: rank < p ? 0 : Math.exp(singular.reduce((s, x) => s + 2 * Math.log(x), 0)),
      solve(y) {
        const theta = zeros(p);
        for (let j = 0; j < p; j++) {
          if (singular[j] <= cutoff) continue;
          const weight = dot(columns[j], y) / (singular[j] * singular[j]);
          for (let i = 0; i < p; i++) theta[i] += vectors[j][i] * weight;
        }
        return theta;
      },
    };
  }

  function lasso(X, Y, lambda, initial) {
    const n = X.length, p = X[0].length;
    const theta = initial?.length === p && initial.every(Number.isFinite) ? [...initial] : zeros(p);
    const columns = Array.from({ length: p }, (_, j) => X.map((r) => r[j]));
    const norms = columns.map((c) => dot(c, c) / n);
    const residual = Y.map((y, i) => y - dot(X[i], theta));
    const tolerance = 1e-8 * (1 + Math.max(...columns.map((c) => Math.abs(dot(c, Y) / n))));
    let violation = Infinity, iterations = 0;
    for (; iterations < 3000; iterations++) {
      for (let j = 0; j < p; j++) {
        const previous = theta[j];
        const rho = dot(columns[j], residual) / n + norms[j] * previous;
        theta[j] = !norms[j] ? 0 : j === 0 ? rho / norms[j]
          : Math.sign(rho) * Math.max(0, Math.abs(rho) - lambda) / norms[j];
        const delta = theta[j] - previous;
        if (delta) for (let i = 0; i < n; i++) residual[i] -= delta * X[i][j];
      }
      // Check the optimality conditions, including zero coefficients.
      violation = 0;
      for (let j = 0; j < p; j++) {
        const gradient = dot(columns[j], residual) / n;
        violation = Math.max(violation, j === 0 ? Math.abs(gradient)
          : theta[j] === 0 ? Math.max(0, Math.abs(gradient) - lambda)
            : Math.abs(gradient - lambda * Math.sign(theta[j])));
      }
      if (violation <= tolerance) break;
    }
    return { theta, converged: violation <= tolerance, iterations: Math.min(iterations + 1, 3000) };
  }

  function fit(points, basis = "polynomial", complexity = 3, regularization = "none", lambda = 0, initial) {
    const p = basis === "fourier" ? 2 * complexity + 1 : complexity + 1;
    const X = points.map(({ x }) => features(x, basis, complexity));
    const Y = points.map(({ y }) => y);
    const gram = Array.from({ length: p }, () => zeros(p));
    const xty = zeros(p);
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < p; j++) {
        xty[j] += X[i][j] * Y[i];
        for (let k = 0; k < p; k++) gram[j][k] += X[i][j] * X[i][k];
      }
    }
    const result = { X, Y, gram, xty, p, n: points.length, theta: null, rank: 0,
      determinant: 0, condition: Infinity, mse: null, penalty: 0, objective: null, converged: true };
    if (!points.length) return result;
    const svd = decompose(X, p);
    Object.assign(result, { rank: svd.rank, determinant: svd.determinant, condition: svd.condition });
    if (regularization === "l1" && lambda > 0) {
      Object.assign(result, lasso(X, Y, lambda, initial));
    } else if (regularization === "l2" && lambda > 0) {
      const augmented = X.map((r) => [...r]);
      const target = [...Y];
      for (let j = 1; j < p; j++) {
        const row = zeros(p);
        row[j] = Math.sqrt(points.length * lambda);
        augmented.push(row);
        target.push(0);
      }
      result.theta = decompose(augmented, p).solve(target);
    } else result.theta = svd.solve(Y);
    result.system = gram.map((row, j) => row.map((v, k) =>
      v + (j === k && j > 0 && regularization === "l2" ? points.length * lambda : 0)));
    result.predictions = X.map((row) => dot(row, result.theta));
    result.mse = Y.reduce((sum, y, i) => sum + (y - result.predictions[i]) ** 2, 0) / points.length;
    const weights = result.theta.slice(1);
    result.penalty = regularization === "l1" ? lambda * weights.reduce((s, x) => s + Math.abs(x), 0)
      : regularization === "l2" ? lambda / 2 * dot(weights, weights) : 0;
    result.objective = result.mse / 2 + result.penalty;
    return result;
  }

  const api = { features, fit, predict: (x, theta, basis, complexity) => dot(features(x, basis, complexity), theta) };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MLRegressionMath = api;
})(typeof window !== "undefined" ? window : globalThis);
