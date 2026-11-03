/* Logistic regression with optional L1/L2 penalties, solved in the browser with no runtime dependencies. */
(function (root) {
  "use strict";
  // Eigenvalues at or below this fraction of the largest count as zero: 1e-6 for the column rank
  // (see solveSymmetric); 1e-12 for solver steps, which must follow separation as p(1 − p) → 0.
  const rankTolerance = 1e-6, stepTolerance = 1e-12;
  const zeros = (n) => Array(n).fill(0);
  const dot = (a, b) => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
  };

  const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  const logit = (p) => Math.log(p / (1 - p));
  // log(1 + exp(−s z)) with s = ±1 for the label, without overflow for large |z|.
  const loss = (z, label) => Math.max(0, label ? -z : z) + Math.log1p(Math.exp(-Math.abs(z)));

  // (first, second) ← (c·first − s·second, s·first + c·second), elementwise.
  function rotate(first, second, c, s) {
    for (let k = 0; k < first.length; k++) {
      const value = first[k];
      first[k] = c * value - s * second[k];
      second[k] = s * value + c * second[k];
    }
  }

  // Cyclic Jacobi eigendecomposition H = Vᵀ diag(values) V of a symmetric matrix; the rows of
  // `vectors` are the eigenvectors.
  function eigenSymmetric(H) {
    const n = H.length;
    const a = H.map((row) => [...row]);
    const vectors = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = 0, total = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          total += a[i][j] * a[i][j];
          if (i !== j) off += a[i][j] * a[i][j];
        }
      }
      if (!(off > 1e-30 * total)) break; // off-diagonal part below 1e-15 of the norm
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          const app = a[p][p], aqq = a[q][q], apq = a[p][q];
          if (apq === 0) continue;
          const theta = (aqq - app) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          // Rows p and q of JᵀAJ off the 2×2 block come from rotating the rows; symmetry gives the columns.
          rotate(a[p], a[q], c, s);
          for (let k = 0; k < n; k++) {
            a[k][p] = a[p][k];
            a[k][q] = a[q][k];
          }
          a[p][p] = app - t * apq;
          a[q][q] = aqq + t * apq;
          a[p][q] = a[q][p] = 0;
          rotate(vectors[p], vectors[q], c, s);
        }
      }
    }
    return { values: a.map((row, i) => row[i]), vectors };
  }

  // Indices of the eigenvalues above tolerance × the largest magnitude; the rest count as zero.
  function retained(values, tolerance) {
    const cutoff = tolerance * Math.max(0, ...values.map(Math.abs));
    return [...values.keys()].filter((k) => values[k] > cutoff);
  }

  // Minimum-norm solution of H x = g for a symmetric H: x = Σ v (v·g) / value over retained eigenvalues.
  function solveSymmetric(H, g, tolerance = rankTolerance) {
    const { values, vectors } = eigenSymmetric(H);
    const kept = retained(values, tolerance);
    const x = zeros(g.length);
    for (const k of kept) {
      const weight = dot(vectors[k], g) / values[k];
      vectors[k].forEach((component, i) => { x[i] += component * weight; });
    }
    return { x, rank: kept.length };
  }

  // Σ log(1 + exp(−s z)) + λ·P(w); the last component of theta is the unpenalized intercept.
  function objective(rows, y, theta, lambda, penalty) {
    const w = theta.slice(0, -1);
    const P = penalty === "l1" ? w.reduce((sum, v) => sum + Math.abs(v), 0) : dot(w, w) / 2;
    return rows.reduce((sum, row, i) => sum + loss(dot(row, theta), y[i]), 0) + lambda * P;
  }

  // Gradient Σ (p − y) x and Hessian Σ p(1 − p) x xᵀ of the unpenalized loss.
  function derivatives(rows, y, theta) {
    const m = theta.length;
    const gradient = zeros(m), hessian = Array.from({ length: m }, () => zeros(m));
    rows.forEach((row, i) => {
      const z = dot(row, theta);
      const residual = y[i] ? -sigmoid(-z) : sigmoid(z);
      const weight = sigmoid(z) * sigmoid(-z);
      for (let j = 0; j < m; j++) {
        if (row[j] === 0) continue;
        gradient[j] += residual * row[j];
        const scaled = weight * row[j], line = hessian[j];
        for (let k = j; k < m; k++) line[k] += scaled * row[k];
      }
    });
    for (let j = 0; j < m; j++) for (let k = 0; k < j; k++) hessian[j][k] = hessian[k][j];
    return { gradient, hessian };
  }

  // Step halving along the direction until the objective does not increase; null after 60 halvings.
  function lineSearch(theta, direction, value, evaluate) {
    if (!direction.every(Number.isFinite)) return null;
    for (let t = 1, k = 0; k <= 60; k++, t /= 2) {
      const candidate = theta.map((v, j) => v + t * direction[j]);
      const next = evaluate(candidate);
      if (next <= value) return { theta: candidate, value: next };
    }
    return null;
  }

  // Newton's method for the unpenalized or L2 objective.
  function newton(rows, y, lambda, start) {
    const m = start.length;
    const evaluate = (theta) => objective(rows, y, theta, lambda, "l2");
    let theta = start, value = evaluate(theta), iterations = 0, converged = false;
    while (iterations < 100 && !converged) {
      iterations++;
      const { gradient, hessian } = derivatives(rows, y, theta);
      for (let j = 0; j < m - 1; j++) {
        gradient[j] += lambda * theta[j];
        hessian[j][j] += lambda;
      }
      const step = solveSymmetric(hessian, gradient, stepTolerance).x;
      const next = lineSearch(theta, step.map((v) => -v), value, evaluate);
      if (!next) break;
      converged = value - next.value < 1e-8;
      ({ theta, value } = next);
    }
    return { theta, value, iterations, converged };
  }

  // Exact steps for the quadratic model on the support of u with the signs held fixed, where the model is
  // smooth. A step that would change the sign of a weight stops where it reaches zero; that weight is then
  // dropped and the step repeated on the smaller support. Steps that do not decrease the model are skipped.
  function polish(H, slope, u, lambda, move) {
    const m = u.length;
    for (;;) {
      const support = [...u.keys()].filter((j) => j === m - 1 || u[j] !== 0);
      const pull = support.map((j) => slope[j] + (j === m - 1 ? 0 : lambda * Math.sign(u[j])));
      const block = support.map((j) => support.map((k) => H[j][k]));
      const step = solveSymmetric(block, pull, stepTolerance).x.map((v) => -v);
      let fraction = 1, blocking = -1;
      support.forEach((j, a) => {
        if (j < m - 1 && u[j] * (u[j] + step[a]) < 0 && -u[j] / step[a] < fraction) {
          fraction = -u[j] / step[a];
          blocking = j;
        }
      });
      const change = fraction * dot(step, pull) + fraction * fraction * dot(step, block.map((row) => dot(row, step))) / 2;
      if (!(change < 0)) return;
      support.forEach((j, a) => move(j, j === blocking ? -u[j] : fraction * step[a]));
      if (blocking < 0) return;
    }
  }

  // Minimizes gᵀ(u − θ) + ½(u − θ)ᵀH(u − θ) + λ‖u without its last component‖₁ by cyclic coordinate
  // descent with soft-thresholding; the last (intercept) coordinate is updated without penalty. Every
  // tenth sweep a polish step finishes the job on the support, which plain sweeps do slowly when
  // columns are strongly correlated.
  function coordinateDescent(g, H, theta, lambda) {
    const m = theta.length, u = theta.slice(), slope = g.slice();
    const move = (j, delta) => {
      u[j] += delta;
      for (let k = 0; k < m; k++) slope[k] += delta * H[k][j];
    };
    for (let sweep = 0; sweep < 1000; sweep++) {
      let largest = 0;
      for (let j = 0; j < m; j++) {
        const h = H[j][j];
        if (!(h > 0)) continue;
        const a = h * u[j] - slope[j];
        const delta = (j === m - 1 ? a : Math.max(0, a - lambda) + Math.min(0, a + lambda)) / h - u[j];
        if (delta === 0) continue;
        move(j, delta);
        largest = Math.max(largest, h * delta * delta);
      }
      if (largest < 1e-24) break;
      if (sweep % 10 === 9) polish(H, slope, u, lambda, move);
    }
    return u;
  }

  // L1: IRLS outer loop on the quadratic approximation at the current point, solved by coordinate descent.
  function proximalNewton(rows, y, lambda, start) {
    const evaluate = (theta) => objective(rows, y, theta, lambda, "l1");
    let theta = start, value = evaluate(theta), iterations = 0, converged = false;
    while (iterations < 200 && !converged) {
      iterations++;
      const { gradient, hessian } = derivatives(rows, y, theta);
      const target = coordinateDescent(gradient, hessian, theta, lambda);
      const next = lineSearch(theta, target.map((v, j) => v - theta[j]), value, evaluate);
      if (!next) break;
      converged = value - next.value < 1e-8;
      ({ theta, value } = next);
    }
    return { theta, value, iterations, converged };
  }

  function fit(X, y, options) {
    const { penalty, initial } = options;
    const p = (X[0] || []).length;
    const lambda = penalty !== "none" && Number.isFinite(options.lambda) && options.lambda > 0 ? options.lambda : 0;
    const columns = options.columns || [...Array(p).keys()];
    const constant = columns.filter((j) => X.every((row) => row[j] === X[0][j]));
    const active = columns.filter((j) => !constant.includes(j));
    const m = active.length + 1;
    const rows = X.map((row) => [...active.map((j) => row[j]), 1]);
    const warm = initial && [...active.map((j) => initial.weights?.[j]), initial.intercept];
    const start = warm && warm.every(Number.isFinite) ? warm : zeros(m);
    // Rank of the unpenalized Hessian at the origin, X̃ᵀX̃ / 4: collinearity of the columns themselves,
    // not the near-singularity that separation causes later in the iterations.
    const origin = eigenSymmetric(derivatives(rows, y, zeros(m)).hessian);
    const kept = retained(origin.values, rankTolerance);
    let solved;
    if (penalty === "l1") solved = proximalNewton(rows, y, lambda, start);
    else if (lambda === 0 && kept.length < m) {
      // Minimum-norm coefficients: Newton in the column space, θ = Σ βₖ vₖ over the retained eigenvectors.
      const basis = kept.map((k) => origin.vectors[k]);
      const project = (row) => basis.map((v) => dot(v, row));
      const reduced = newton(rows.map(project), y, 0, project(start));
      const theta = zeros(m).map((_, i) => basis.reduce((sum, v, k) => sum + v[i] * reduced.theta[k], 0));
      solved = { ...reduced, theta };
    } else solved = newton(rows, y, lambda, start);
    const weights = zeros(p);
    active.forEach((j, k) => { weights[j] = solved.theta[k]; });
    const intercept = solved.theta[active.length];
    return {
      weights,
      intercept,
      iterations: solved.iterations,
      converged: solved.converged,
      objective: solved.value,
      separated: penalty === "none" && Math.max(Math.abs(intercept), ...weights.map(Math.abs)) > 10,
      constant,
      collinear: penalty === "none" && kept.length < m,
      rank: kept.length,
    };
  }

  const scores = (X, weights, intercept) => X.map((row) => intercept + dot(row, weights));
  const probabilities = (logits) => logits.map(sigmoid);

  function confusion(scoreList, y, threshold) {
    const c = { tp: 0, fn: 0, fp: 0, tn: 0 };
    scoreList.forEach((s, i) => {
      const positive = s >= threshold;
      if (y[i] === 1) c[positive ? "tp" : "fn"] += 1;
      else c[positive ? "fp" : "tn"] += 1;
    });
    return c;
  }

  const ratio = (num, den) => (den === 0 ? null : num / den);

  function metrics(c) {
    const { tp, fn, fp, tn } = c;
    const precision = ratio(tp, tp + fp);
    const recall = ratio(tp, tp + fn);
    const specificity = ratio(tn, tn + fp);
    const fpr = ratio(fp, fp + tn);
    const fnr = ratio(fn, fn + tp);
    const both = (a, b, f) => (a === null || b === null ? null : f(a, b));
    const f1 = both(precision, recall, (p, r) => (p + r === 0 ? null : (2 * p * r) / (p + r)));
    const root = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    return {
      accuracy: ratio(tp + tn, tp + fn + fp + tn),
      precision,
      recall,
      specificity,
      fpr,
      fnr,
      f1,
      balancedAccuracy: both(recall, specificity, (r, s) => (r + s) / 2),
      mcc: root === 0 ? null : (tp * tn - fp * fn) / root,
      youdenJ: both(recall, fpr, (r, f) => r - f),
      npv: ratio(tn, tn + fn),
    };
  }

  function roc(scoreList, y) {
    const positives = y.reduce((a, v) => a + (v === 1 ? 1 : 0), 0);
    const negatives = y.length - positives;
    const start = { fpr: 0, tpr: 0, threshold: Infinity };
    if (positives === 0 || negatives === 0) return { points: [start, { fpr: 1, tpr: 1, threshold: -Infinity }], auc: 0.5 };
    const order = scoreList.map((s, i) => i).sort((a, b) => scoreList[b] - scoreList[a]);
    const points = [start];
    let tp = 0, fp = 0;
    for (let k = 0; k < order.length; k++) {
      const i = order[k];
      if (y[i] === 1) tp += 1; else fp += 1;
      if (k + 1 === order.length || scoreList[order[k + 1]] !== scoreList[i]) {
        points.push({ fpr: fp / negatives, tpr: tp / positives, threshold: scoreList[i] });
      }
    }
    let auc = 0;
    for (let k = 1; k < points.length; k++) auc += ((points[k].fpr - points[k - 1].fpr) * (points[k].tpr + points[k - 1].tpr)) / 2;
    return { points, auc };
  }

  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const logLoss = (p, y) => mean(p.map((v, i) => {
    const q = Math.min(1 - 1e-15, Math.max(1e-15, v));
    return -(y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q));
  }));
  const brier = (p, y) => mean(p.map((v, i) => (v - y[i]) ** 2));

  function histogram(values, edges) {
    const bins = edges.length - 1;
    const counts = zeros(bins);
    const last = edges[bins];
    values.forEach((v) => {
      if (!(v >= edges[0] && v <= last)) return;
      if (v === last) { counts[bins - 1] += 1; return; }
      let lo = 0, hi = bins - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (edges[mid] <= v) lo = mid; else hi = mid - 1;
      }
      counts[lo] += 1;
    });
    return counts;
  }

  function silverman(values) {
    const n = values.length;
    if (n < 2) return 1e-3;
    const mu = mean(values);
    const sd = Math.sqrt(values.reduce((a, v) => a + (v - mu) ** 2, 0) / n);
    const sorted = [...values].sort((a, b) => a - b);
    const quantile = (q) => {
      const pos = q * (n - 1), lo = Math.floor(pos);
      return sorted[lo] + (pos - lo) * ((sorted[Math.min(lo + 1, n - 1)]) - sorted[lo]);
    };
    const iqr = quantile(0.75) - quantile(0.25);
    const scale = Math.pow(n, -0.2);
    const h = iqr === 0 ? sd * scale : 0.9 * Math.min(sd, iqr / 1.34) * scale;
    return h > 0 ? h : 1e-3;
  }

  function kde(values, grid, bandwidth, bounds) {
    if (values.length === 0) return grid.map(() => 0);
    const points = bounds ? values.flatMap((v) => [v, 2 * bounds[0] - v, 2 * bounds[1] - v]) : values;
    const norm = 1 / (values.length * bandwidth * Math.sqrt(2 * Math.PI));
    return grid.map((x) => norm * points.reduce((sum, v) => sum + Math.exp(-0.5 * ((x - v) / bandwidth) ** 2), 0));
  }

  const api = { sigmoid, logit, solveSymmetric, fit, scores, probabilities, confusion, metrics, roc, logLoss, brier, histogram, silverman, kde };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MLClassificationMath = api;
})(typeof window !== "undefined" ? window : globalThis);
