/* SVG inner-markup builders for the score distribution and the ROC plot; pure strings, no DOM. */
(function (root) {
  "use strict";
  const MathApi = typeof module !== "undefined" && module.exports ? require("./classification-math.js") : root.MLClassificationMath;
  const DISTRIBUTION = { width: 960, height: 420 };
  const ROC = { width: 480, height: 480 };
  const gridPoints = 200;
  // Width budget per character, in em: monospace tick digits are 0.6 em; titles are budgeted the same.
  const EM = 0.6;
  // The plot area inside the viewBox. Each margin holds what is drawn in it, so it grows with the font:
  // left the widest y tick label (3 characters on the distribution, "0.25" on the ROC), bottom the x tick
  // labels and the x title, top the y title. At fontSize 16 these are the original fixed boxes.
  function plotBox(kind, fontSize) {
    const roc = kind === "roc", { width, height } = roc ? ROC : DISTRIBUTION;
    return {
      left: Math.max(roc ? 56 : 64, (roc ? 4 : 3) * EM * fontSize + 12),
      right: width - (roc ? 24 : 20),
      top: Math.max(24, 1.5 * fontSize),
      bottom: height - Math.max(roc ? 56 : 60, 3.4 * fontSize),
    };
  }
  DISTRIBUTION.box = plotBox("distribution", 16);
  ROC.box = plotBox("roc", 16);

  const scale = (range, box) => {
    const [lo, hi] = range;
    return {
      px: (v) => box.left + ((v - lo) / (hi - lo)) * (box.right - box.left),
      value: (px) => lo + ((px - box.left) / (box.right - box.left)) * (hi - lo),
    };
  };
  // Vertical scale: larger values sit higher on screen.
  const scaleY = (range, box) => {
    const [lo, hi] = range;
    return {
      px: (v) => box.bottom - ((v - lo) / (hi - lo)) * (box.bottom - box.top),
      value: (px) => lo + ((box.bottom - px) / (box.bottom - box.top)) * (hi - lo),
    };
  };
  const escapeXml = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");
  const text = (x, y, label, fontSize, cls, anchor) =>
    `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}" font-size="${fontSize}">${escapeXml(label)}</text>`;
  const linePath = (points) => points.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" ");
  const closedPath = (points, baseline) =>
    points.length ? `${linePath(points)} L${points.at(-1)[0]} ${baseline} L${points[0][0]} ${baseline} Z` : "";
  const label = (v) => String(Math.round(v * 1e6) / 1e6);
  // Label every k-th tick counted from 0, with the smallest k that keeps labels `need` apart.
  const every = (step, spacing, need) => {
    const k = Math.max(1, Math.ceil(need / spacing - 1e-9));
    return (v) => Math.abs(Math.round(v / step)) % k === 0;
  };
  // Axis titles: x centred under the plot area, shifted to stay inside; y at the top-left corner of the
  // plot area. A title's font shrinks below fontSize only when the text would not fit the viewBox.
  function axisTitles(out, size, box, fs, lab) {
    const fit = (title, room) => Math.min(fs, room / (EM * Math.max(1, String(title).length)));
    const xf = fit(lab.x, size.width - 8), half = (EM * String(lab.x).length * xf) / 2;
    const x = Math.min(size.width - 4 - half, Math.max(4 + half, (box.left + box.right) / 2));
    out.push(text(x, box.bottom + fs * 2.8, lab.x, xf, "cl-axis-title", "middle"));
    out.push(text(box.left, box.top - fs * 0.55, lab.y, fit(lab.y, size.width - box.left - 4), "cl-axis-title", "start"));
  }

  // Gridline values: about `count` "nice" steps (1, 2, 5 × 10^k) from 0 up to `max`.
  function niceTicks(max, count) {
    const raw = max / count, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [];
    for (let v = 0; v <= max + 1e-9; v += step) ticks.push(Math.round(v / step) * step);
    return ticks;
  }
  function xTicks(axis, range) {
    const step = axis === "probability" ? 0.1 : 1, ticks = [];
    for (let i = Math.ceil(range[0] / step - 1e-9); i * step <= range[1] + 1e-9; i++) ticks.push(Math.round(i * step * 1e6) / 1e6);
    return ticks;
  }

  // Split a sampled curve at x = t, interpolating a point exactly at t; returns [left, right].
  function splitAt(points, t) {
    const left = [], right = [];
    points.forEach(([x, y], i) => {
      if (x <= t) left.push([x, y]);
      if (x >= t) right.push([x, y]);
      const next = points[i + 1];
      if (next && x < t && next[0] > t) {
        const yt = y + ((t - x) / (next[0] - x)) * (next[1] - y);
        left.push([t, yt]);
        right.push([t, yt]);
      }
    });
    return [left, right];
  }

  // The expensive part of the distribution plot: per-class histogram counts and KDE curves scaled to
  // counts per bin. It depends only on the scores, the axis and the range, so a caller can compute it
  // once and pass it back as `layers` while only the threshold moves.
  function distributionLayers(o) {
    const [lo, hi] = o.range, binWidth = (hi - lo) / o.bins;
    const edges = Array.from({ length: o.bins + 1 }, (_, i) => lo + i * binWidth);
    const grid = Array.from({ length: gridPoints }, (_, i) => lo + (i / (gridPoints - 1)) * (hi - lo));
    const classes = ["positive", "negative"].map((name) => {
      const values = name === "positive" ? o.positives : o.negatives;
      const counts = values.length ? MathApi.histogram(values, edges) : Array(o.bins).fill(0);
      const density = values.length
        ? MathApi.kde(values, grid, MathApi.silverman(values), o.axis === "probability" ? o.range : undefined)
        : Array(gridPoints).fill(0);
      return { name, counts, curve: density.map((d) => d * values.length * binWidth), total: Math.max(0, ...counts) };
    });
    return { empty: !o.positives.length && !o.negatives.length, edges, grid, classes };
  }

  function distributionPlot(o) {
    const fs = o.fontSize, lab = o.labels, box = plotBox("distribution", fs);
    const layers = o.layers || distributionLayers(o);
    const sx = scale(o.range, box);
    const out = [];
    const axes = (yMax) => {
      const sy = scaleY([0, yMax], box), yTicks = niceTicks(yMax, 6), xs = xTicks(o.axis, o.range);
      const yStep = yTicks[1] - yTicks[0], xStep = o.axis === "probability" ? 0.1 : 1;
      const yShown = every(yStep, sy.px(0) - sy.px(yStep), 1.4 * fs);
      const xShown = every(xStep, sx.px(xStep) - sx.px(0), (EM * Math.max(1, ...xs.map((v) => label(v).length)) + 0.5) * fs);
      yTicks.forEach((v) => {
        const y = sy.px(v);
        out.push(`<line class="cl-grid" x1="${box.left}" x2="${box.right}" y1="${y}" y2="${y}"/>`);
        if (yShown(v)) out.push(text(box.left - 8, y + fs * 0.35, label(v), fs, "cl-tick", "end"));
      });
      xs.forEach((v) => {
        const x = sx.px(v);
        out.push(`<line class="cl-grid" x1="${x}" x2="${x}" y1="${box.top}" y2="${box.bottom}"/>`);
        if (xShown(v)) out.push(text(x, box.bottom + fs * 1.3, label(v), fs, "cl-tick", "middle"));
      });
      axisTitles(out, DISTRIBUTION, box, fs, lab);
      return sy;
    };
    if (layers.empty) {
      axes(1);
      out.push(text((box.left + box.right) / 2, (box.top + box.bottom) / 2, lab.empty, fs, "cl-empty", "middle"));
      return out.join("\n");
    }

    const { edges, grid, classes } = layers;
    const yMax = Math.max(1, 1.08 * Math.max(...classes.flatMap((c) => [...c.counts, ...c.curve])));
    const sy = axes(yMax);
    const baseline = box.bottom;
    const points = (c) => c.curve.map((v, i) => [sx.px(grid[i]), sy.px(v)]);
    const [pos, neg] = classes;

    const regions = [[pos, "tp", "fn"], [neg, "fp", "tn"]].map(([c, rightKey, leftKey]) => {
      const [left, right] = splitAt(points(c), sx.px(o.threshold));
      return `<path class="cl-region cl-${rightKey}" d="${closedPath(right, baseline)}"/>\n<path class="cl-region cl-${leftKey}" d="${closedPath(left, baseline)}"/>`;
    });
    out.push(...regions);

    // Taller class first so both stay visible where the bars overlap.
    const bars = (c) => c.counts.map((count, i) => count
      ? `<rect class="cl-bar cl-${c.name}" x="${sx.px(edges[i])}" y="${sy.px(count)}" width="${sx.px(edges[i + 1]) - sx.px(edges[i])}" height="${baseline - sy.px(count)}"/>`
      : "").filter(Boolean);
    const byHeight = pos.total >= neg.total ? [pos, neg] : [neg, pos];
    byHeight.forEach((c) => out.push(...bars(c)));
    classes.forEach((c) => out.push(`<path class="cl-curve cl-${c.name}" d="${linePath(points(c))}"/>`));

    const tx = sx.px(o.threshold);
    out.push(`<line id="cl-threshold-line" x1="${tx}" x2="${tx}" y1="${box.top}" y2="${box.bottom}" class="cl-threshold"/>`);
    out.push(`<g id="cl-threshold-handle" class="cl-handle" tabindex="0" transform="translate(${tx}, ${box.top})"><circle r="9"/></g>`);
    return out.join("\n");
  }

  function rocPlot(o) {
    const fs = o.fontSize, lab = o.labels, box = plotBox("roc", fs);
    const sx = scale([0, 1], box), sy = scaleY([0, 1], box);
    const yShown = every(0.25, (box.bottom - box.top) / 4, 1.4 * fs), xShown = every(0.25, (box.right - box.left) / 4, (EM * 4 + 0.5) * fs);
    const out = [];
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      out.push(`<line class="cl-grid" x1="${box.left}" x2="${box.right}" y1="${sy.px(v)}" y2="${sy.px(v)}"/>`);
      if (yShown(v)) out.push(text(box.left - 8, sy.px(v) + fs * 0.35, label(v), fs, "cl-tick", "end"));
      out.push(`<line class="cl-grid" x1="${sx.px(v)}" x2="${sx.px(v)}" y1="${box.top}" y2="${box.bottom}"/>`);
      if (xShown(v)) out.push(text(sx.px(v), box.bottom + fs * 1.3, label(v), fs, "cl-tick", "middle"));
    });
    axisTitles(out, ROC, box, fs, lab);
    out.push(`<line class="cl-diagonal" x1="${sx.px(0)}" y1="${sy.px(0)}" x2="${sx.px(1)}" y2="${sy.px(1)}"/>`);
    const curve = (pts) => linePath(pts.map((p) => [sx.px(p.fpr), sy.px(p.tpr)]));
    if (Array.isArray(o.secondary) && o.secondary.length) out.push(`<path class="cl-roc cl-secondary" d="${curve(o.secondary)}"/>`);
    out.push(`<path class="cl-roc cl-primary" d="${curve(o.primary)}"/>`);
    out.push(`<circle id="cl-operating-point" cx="${sx.px(o.operating.fpr)}" cy="${sy.px(o.operating.tpr)}" r="7" class="cl-operating"/>`);
    return out.join("\n");
  }

  const api = { DISTRIBUTION, ROC, plotBox, scale, distributionLayers, distributionPlot, rocPlot };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MLClassificationPlots = api;
})(typeof window !== "undefined" ? window : globalThis);
