/* SVG inner-markup builders for the score distribution and the ROC plot; pure strings, no DOM. */
(function (root) {
  "use strict";
  const MathApi = typeof module !== "undefined" && module.exports ? require("./classification-math.js") : root.MLClassificationMath;
  const DISTRIBUTION = { width: 960, height: 420, box: { left: 64, right: 940, top: 24, bottom: 360 } };
  const ROC = { width: 480, height: 480, box: { left: 56, right: 456, top: 24, bottom: 424 } };
  const gridPoints = 200;

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
  const text = (x, y, label, fontSize, cls, anchor, extra = "") =>
    `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}" font-size="${fontSize}"${extra}>${escapeXml(label)}</text>`;
  const linePath = (points) => points.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" ");
  const closedPath = (points, baseline) =>
    points.length ? `${linePath(points)} L${points.at(-1)[0]} ${baseline} L${points[0][0]} ${baseline} Z` : "";
  const label = (v) => String(Math.round(v * 1e6) / 1e6);

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

  function distributionPlot(o) {
    const { box } = DISTRIBUTION, fs = o.fontSize, lab = o.labels;
    const n = { positive: o.positives.length, negative: o.negatives.length };
    const sx = scale(o.range, box);
    const out = [];
    const axes = (yMax) => {
      const sy = scaleY([0, yMax], box);
      niceTicks(yMax, 6).forEach((v) => {
        const y = sy.px(v);
        out.push(`<line class="cl-grid" x1="${box.left}" x2="${box.right}" y1="${y}" y2="${y}"/>`);
        out.push(text(box.left - 8, y + fs * 0.35, label(v), fs, "cl-tick", "end"));
      });
      xTicks(o.axis, o.range).forEach((v) => {
        const x = sx.px(v);
        out.push(`<line class="cl-grid" x1="${x}" x2="${x}" y1="${box.top}" y2="${box.bottom}"/>`);
        out.push(text(x, box.bottom + fs * 1.3, label(v), fs, "cl-tick", "middle"));
      });
      out.push(text((box.left + box.right) / 2, box.bottom + fs * 2.8, lab.x, fs, "cl-axis-title", "middle"));
      const cy = (box.top + box.bottom) / 2;
      out.push(text(16, cy, lab.y, fs, "cl-axis-title", "middle", ` transform="rotate(-90 16 ${cy})"`));
      return sy;
    };
    if (!n.positive && !n.negative) {
      axes(1);
      out.push(text((box.left + box.right) / 2, (box.top + box.bottom) / 2, lab.empty, fs, "cl-empty", "middle"));
      return out.join("\n");
    }

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
    const { box } = ROC, fs = o.fontSize, lab = o.labels;
    const sx = scale([0, 1], box), sy = scaleY([0, 1], box);
    const out = [];
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      out.push(`<line class="cl-grid" x1="${box.left}" x2="${box.right}" y1="${sy.px(v)}" y2="${sy.px(v)}"/>`);
      out.push(text(box.left - 8, sy.px(v) + fs * 0.35, label(v), fs, "cl-tick", "end"));
      out.push(`<line class="cl-grid" x1="${sx.px(v)}" x2="${sx.px(v)}" y1="${box.top}" y2="${box.bottom}"/>`);
      out.push(text(sx.px(v), box.bottom + fs * 1.3, label(v), fs, "cl-tick", "middle"));
    });
    out.push(text((box.left + box.right) / 2, box.bottom + fs * 2.8, lab.x, fs, "cl-axis-title", "middle"));
    const cy = (box.top + box.bottom) / 2;
    out.push(text(16, cy, lab.y, fs, "cl-axis-title", "middle", ` transform="rotate(-90 16 ${cy})"`));
    out.push(`<line class="cl-diagonal" x1="${sx.px(0)}" y1="${sy.px(0)}" x2="${sx.px(1)}" y2="${sy.px(1)}"/>`);
    const curve = (pts) => linePath(pts.map((p) => [sx.px(p.fpr), sy.px(p.tpr)]));
    if (Array.isArray(o.secondary) && o.secondary.length) out.push(`<path class="cl-roc cl-secondary" d="${curve(o.secondary)}"/>`);
    out.push(`<path class="cl-roc cl-primary" d="${curve(o.primary)}"/>`);
    out.push(`<circle id="cl-operating-point" cx="${sx.px(o.operating.fpr)}" cy="${sy.px(o.operating.tpr)}" r="7" class="cl-operating"/>`);
    return out.join("\n");
  }

  const api = { DISTRIBUTION, ROC, scale, distributionPlot, rocPlot };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MLClassificationPlots = api;
})(typeof window !== "undefined" ? window : globalThis);
