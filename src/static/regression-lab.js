(function () {
  "use strict";
  const M = window.MLRegressionMath;
  const LIMIT = 200;
  const preference = (key, fallback) => {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  };
  const state = {
    points: [], basis: "polynomial", degrees: { polynomial: 3, fourier: 3 },
    regularization: "none", lambda: 0.01, tool: "points", uniform: false,
    residuals: true, truth: "sine", selected: -1, page: 0, undo: [], redo: [],
    presentation: preference("ml-regression-presentation", "true") !== "false",
    textScale: ["1", "1.25", "1.5"].includes(preference("ml-regression-text-scale", "1"))
      ? +preference("ml-regression-text-scale", "1") : 1,
    view: "theta",
  };
  let teardown = () => {};
  const esc = (value) => String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmt = (v) => !Number.isFinite(v) ? v === Infinity ? "∞" : "—"
    : v === 0 ? "0" : Math.abs(v) < 0.001 || Math.abs(v) >= 10000 ? v.toExponential(3) : Number(v.toPrecision(4)).toString();
  const truthValue = (name, x) => name === "sine" ? 0.8 * Math.sin(Math.PI * x)
    : name === "parabola" ? 1.7 * x * x - 0.65 : 0.75 * x + 0.15;
  function preset(name) {
    let seed = 42;
    const random = () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; };
    if (name === "singular") return [-0.65, -0.25, 0.25, 0.65].map((y) => ({ x: 0.25, y }));
    return Array.from({ length: 18 }, (_, i) => {
      const x = -0.95 + 1.9 * (i + 0.35 * (random() - 0.5)) / 17;
      return { x, y: truthValue(name, x) + 0.24 * (random() - 0.5) };
    });
  }
  state.points = preset("sine");
  const snapshot = () => ({ points: state.points.map((p) => ({ ...p })), truth: state.truth });
  function record(before) {
    if (JSON.stringify(before) === JSON.stringify(snapshot())) return;
    state.undo.push(before);
    if (state.undo.length > 50) state.undo.shift();
    state.redo = [];
  }

  function html(language) {
    const w = (en, ru) => language === "en" ? en : ru;
    return `<section id="regression-lab" class="lr-lab" aria-labelledby="lr-title">
      <div class="lr-heading"><div><p class="eyebrow">${w("INTERACTIVE LAB / 01", "ИНТЕРАКТИВНАЯ ЛАБОРАТОРИЯ / 01")}</p>
      <h1 id="lr-title">${w("Linear regression", "Линейная регрессия")}</h1>
      <p class="page-description">${w("Draw observations. Choose a basis. Watch the curve and the algebra change together.", "Нарисуйте наблюдения. Выберите базис. Следите, как меняются кривая и матрицы.")}</p></div><span class="lr-live"><span class="status-dot"></span> ${w("ALL IN YOUR BROWSER", "ВСЁ В ВАШЕМ БРАУЗЕРЕ")}</span><div class="lr-presentation-controls"><label><input id="lr-presentation" type="checkbox"> ${w("Lecture view", "Для лекции")}</label><label for="lr-text-size">${w("Text size", "Размер текста")}</label><select id="lr-text-size"><option value="1">100%</option><option value="1.25">125%</option><option value="1.5">150%</option></select></div></div>
      <div class="lr-flow" aria-label="${w("Learning steps", "Этапы моделирования")}"><span>01 ${w("Observations", "Наблюдения")}</span><b>→</b><span>02 ${w("Basis", "Базис")}</span><b>→</b><span>03 X, Y</span><b>→</b><span>04 θ</span><b>→</b><span>05 ŷ(x)</span></div>
      <section class="lr-controls lr-card" aria-label="${w("Model settings", "Настройки модели")}">
        <div class="lr-control"><label for="lr-basis">${w("Basis functions", "Базисные функции")}</label><select id="lr-basis"><option value="polynomial">${w("Polynomial", "Полиномиальный")}</option><option value="fourier">${w("Fourier", "Фурье")}</option></select><p id="lr-basis-note" class="lr-hint"></p></div>
        <div class="lr-control"><label for="lr-complexity" id="lr-complexity-label"></label><div class="lr-range-row"><input id="lr-complexity" type="range" min="0" max="20" step="1"><input id="lr-dimension" type="number" required min="0" max="20" step="1" aria-labelledby="lr-complexity-label"></div><p id="lr-parameter-count" class="lr-hint"></p></div>
        <div class="lr-control"><label for="lr-regularization">${w("Regularization", "Регуляризация")}</label><select id="lr-regularization"><option value="none">${w("None · least squares", "Нет · метод наименьших квадратов")}</option><option value="l1">L1 · Lasso</option><option value="l2">L2 · Ridge</option></select><p class="lr-hint">${w("The intercept θ₀ is never penalized.", "Свободный член θ₀ не штрафуется.")}</p></div>
        <div class="lr-control"><label for="lr-lambda">${w("Penalty strength", "Сила регуляризации")} λ</label><div class="lr-range-row"><input id="lr-lambda-range" type="range" min="-6" max="2" step="0.1" aria-label="${w("Logarithmic penalty strength", "Логарифмическая сила регуляризации")}"><input id="lr-lambda" type="number" required min="0" max="100" step="any"></div><button class="lr-text-button" data-lr-action="zero">${w("Set λ = 0", "Установить λ = 0")}</button></div>
        <div class="lr-model"><span>${w("MODEL", "МОДЕЛЬ")}</span><code id="lr-model"></code><small>${w("Linear in θ, even when the curve is nonlinear in x.", "Линейная по θ, даже если кривая нелинейна по x.")}</small></div>
      </section>
      <div class="lr-workbench" id="lr-panel-plot">
        <section class="lr-card lr-plot-card" aria-labelledby="lr-plot-title">
          <div class="lr-card-heading"><h2 id="lr-plot-title">${w("Your data, your curve", "Ваши данные и кривая")}</h2><span id="lr-point-count" class="lr-badge"></span></div>
          <div class="lr-toolbar"><div class="lr-tools" role="group" aria-label="${w("Plot tool", "Инструмент графика")}"><button data-lr-tool="points">↖ ${w("Points", "Точки")}</button><button data-lr-tool="draw">✎ ${w("Draw", "Рисовать")}</button><button data-lr-tool="erase">⌫ ${w("Erase", "Стереть")}</button></div><div class="lr-history"><button data-lr-action="undo" title="${w("Undo point edit (Ctrl/⌘ Z)", "Отменить правку точек (Ctrl/⌘ Z)")}" aria-label="${w("Undo", "Отменить")}">↶</button><button data-lr-action="redo" title="${w("Redo point edit (Ctrl/⌘ Shift Z)", "Повторить правку точек (Ctrl/⌘ Shift Z)")}" aria-label="${w("Redo", "Повторить")}">↷</button><button data-lr-action="clear">${w("Clear", "Очистить")}</button></div></div>
          <div class="lr-draw-options"><label><input id="lr-uniform" type="checkbox"> ${w("Uniform x-grid + replace painted interval", "Равномерная сетка x + замена на участке рисования")}</label><p id="lr-draw-note" class="lr-hint"></p></div>
          <svg id="lr-plot" viewBox="0 0 720 420" role="img" tabindex="0" aria-label="${w("Interactive regression plot. Edit coordinates in the table below as a keyboard alternative.", "Интерактивный график регрессии. Для ввода с клавиатуры используйте таблицу координат ниже.")}" aria-describedby="lr-plot-help"></svg>
          <p id="lr-plot-help" class="lr-plot-help"></p>
          <div class="lr-legend"><span><i class="lr-dot"></i>${w("Observations", "Наблюдения")}</span><span><i class="lr-line"></i>${w("Fitted curve", "Аппроксимация")}</span><span id="lr-truth-legend"><i class="lr-line lr-dashed"></i>${w("Known curve", "Исходная кривая")}</span><label><input id="lr-residuals" type="checkbox">${w("Residuals", "Остатки")}</label></div>
          <div class="lr-preset-row"><label for="lr-preset">${w("Try an example", "Попробуйте пример")}</label><select id="lr-preset"><option value="">${w("Custom observations", "Свои наблюдения")}</option><option value="sine">${w("Noisy sine", "Синус с шумом")}</option><option value="parabola">${w("Noisy parabola", "Парабола с шумом")}</option><option value="line">${w("Noisy line", "Прямая с шумом")}</option><option value="singular">${w("Repeated x · singular system", "Одинаковые x · вырожденная система")}</option></select></div>
          <div class="lr-metrics"><div><span>${w("TRAINING MSE", "MSE НА ДАННЫХ")}</span><strong id="lr-mse">—</strong></div><div><span>${w("PENALTY", "ШТРАФ")}</span><strong id="lr-penalty">—</strong></div><div><span>${w("OBJECTIVE", "ЦЕЛЕВАЯ ФУНКЦИЯ")}</span><strong id="lr-objective-value">—</strong></div></div>
        </section>
        <aside class="lr-card lr-inspector" aria-label="${w("Live model values", "Значения модели")}">
          <div class="lr-inspector-determinant" aria-live="polite" aria-atomic="true"><span>det(XᵀX)</span><strong id="lr-det">—</strong></div>
          <div class="lr-view-tabs" id="lr-view-tabs" role="tablist" aria-label="${w("Live model values", "Значения модели")}">${[["theta", "θ"], ["data", "X/Y"], ["gram", "XᵀX"]].map(([view, label]) => `<button id="lr-tab-${view}" role="tab" aria-controls="lr-panel-${view}" data-lr-view="${view}">${label}</button>`).join("")}</div>
          <div class="lr-inspector-body">
        <section class="lr-theta-card lr-inspector-panel" id="lr-panel-theta" role="tabpanel" aria-labelledby="lr-tab-theta"><div class="lr-card-heading"><h2 id="lr-theta-title">${w("Coefficients", "Коэффициенты")} θ</h2><span class="lr-badge" id="lr-theta-shape"></span></div><div id="lr-theta"></div><p class="lr-hint lr-zero-note">${w("L1 can set coefficients exactly to zero. L2 shrinks their magnitudes.", "L1 может обнулять коэффициенты. L2 уменьшает их величины.")}</p><div class="lr-equation"><span>${w("FITTED EQUATION", "УРАВНЕНИЕ МОДЕЛИ")}</span><code id="lr-equation"></code></div><div class="lr-loss"><span>${w("MINIMIZE", "МИНИМИЗИРУЕМ")}</span><code id="lr-loss"></code><p class="lr-hint">${w("Mean loss keeps λ comparable when the whole dataset is duplicated. Feature columns are used exactly as displayed, without standardization.", "Усреднение ошибки сохраняет смысл λ при дублировании всего набора. Признаки используются как в матрице, без стандартизации.")}</p></div></section>
      <section class="lr-data-card lr-inspector-panel" id="lr-panel-data" role="tabpanel" aria-labelledby="lr-tab-data"><div class="lr-card-heading"><div><p class="eyebrow">${w("LOOK INSIDE THE MODEL", "КАК УСТРОЕНА МОДЕЛЬ")}</p><h2 id="lr-data-title">${w("Matrices X and Y", "Матрицы X и Y")}</h2></div><div class="lr-pager"><button data-lr-action="previous" aria-label="${w("Previous observations", "Предыдущие наблюдения")}">←</button><span id="lr-page"></span><button data-lr-action="next" aria-label="${w("Next observations", "Следующие наблюдения")}">→</button></div></div>
        <p class="lr-hint">${w("Select a point or a matrix row to connect the picture with the numbers. All observations are used in the fit, across every page.", "Выберите точку или строку матрицы, чтобы увидеть связь. В расчёте участвуют все наблюдения со всех страниц.")}</p>
        <div class="lr-data-grid"><div class="lr-design"><h3>X <small id="lr-x-shape"></small></h3><p class="lr-hint">Xᵢⱼ = φⱼ(xᵢ)</p><div id="lr-x" class="lr-table-scroll" tabindex="0" aria-label="${w("Design matrix X", "Матрица признаков X")}"></div></div><div class="lr-target"><h3>Y <small id="lr-y-shape"></small></h3><p class="lr-hint">Yᵢ = yᵢ</p><div id="lr-y" class="lr-table-scroll"></div></div></div>
      </section>
      <section class="lr-gram-card lr-inspector-panel" id="lr-panel-gram" role="tabpanel" aria-labelledby="lr-tab-gram"><div class="lr-card-heading"><h2 id="lr-gram-title">${w("The normal equations", "Нормальные уравнения")}</h2><code>XᵀX θ = XᵀY</code></div><div class="lr-diagnostics" role="status" aria-live="polite"><span id="lr-rank"></span><span id="lr-condition"></span></div><div class="lr-gram-grid"><div><h3>XᵀX <small id="lr-gram-shape"></small></h3><div id="lr-gram" class="lr-table-scroll" tabindex="0" aria-label="${w("Gram matrix", "Матрица Грама")}"></div></div><details class="lr-crossproduct"><summary>XᵀY</summary><div id="lr-xty" class="lr-table-scroll"></div></details></div><details id="lr-ridge-panel"><summary>${w("L2: the regularized system", "L2: регуляризованная система")}</summary><p class="lr-hint">(XᵀX + nλD) θ = XᵀY · D = diag(0, 1, …, 1)</p><div id="lr-ridge" class="lr-table-scroll" tabindex="0" aria-label="${w("Regularized Gram matrix", "Регуляризованная матрица Грама")}"></div></details><details class="lr-math-notes"><summary>${w("How to read these values", "Как читать эти значения")}</summary><p id="lr-status" class="lr-status"></p><p class="lr-hint">${w("The determinant depends on scale. Rank and condition number are more useful for diagnosing an unstable fit. These diagnostics describe the original X, before regularization.", "Определитель зависит от масштаба. Ранг и число обусловленности лучше описывают устойчивость решения. Диагностика относится к исходной X, до регуляризации.")}</p><p class="lr-solver-note" id="lr-solver-note"></p></details></section>
          </div>
        </aside>
      </div>
      <details id="lr-observation-editor" class="lr-card lr-observation-editor" open><summary>${w("Edit observation coordinates", "Редактировать координаты наблюдений")}</summary><div class="lr-editor-content"><div class="lr-observations"><h3>${w("Observations", "Наблюдения")}</h3><div id="lr-coordinates" class="lr-table-scroll"></div><button data-lr-action="add" class="lr-add">+ ${w("Add observation", "Добавить наблюдение")}</button><p class="lr-hint">x ∈ [−1, 1], y ∈ [−1.5, 1.5]</p></div></div></details>
    </section>`;
  }

  function mount(root, language) {
    const w = (en, ru) => language === "en" ? en : ru;
    const $ = (selector) => root.querySelector(selector);
    const pageSize = () => state.presentation ? 5 : 8;
    const box = { left: 64, right: 700, top: 28, bottom: 370 };
    let plotWidth = 720, plotHeight = 420;
    const complexity = () => state.degrees[state.basis];
    const labels = () => state.basis === "polynomial"
      ? Array.from({ length: complexity() + 1 }, (_, j) => j === 0 ? "1" : j === 1 ? "x" : `x^${j}`)
      : ["1", ...Array.from({ length: complexity() }, (_, j) => [`sin(${j + 1}πx)`, `cos(${j + 1}πx)`]).flat()];
    let result, frame = 0, gesture = null, disposed = false, previousPresentation;
    const plot = $("#lr-plot");
    const px = (x) => box.left + (x + 1) / 2 * (box.right - box.left);
    const py = (y) => box.bottom - (y + 1.5) / 3 * (box.bottom - box.top);
    const xy = (event) => {
      const position = new DOMPoint(event.clientX, event.clientY).matrixTransform(plot.getScreenCTM().inverse());
      const sx = position.x, sy = position.y;
      return { x: Math.max(-1, Math.min(1, 2 * (sx - box.left) / (box.right - box.left) - 1)),
        y: Math.max(-1.5, Math.min(1.5, 3 * (box.bottom - sy) / (box.bottom - box.top) - 1.5)), sx, sy };
    };
    const near = (point) => {
      let index = -1, best = 15 ** 2;
      state.points.forEach((p, i) => {
        const d = (px(p.x) - point.sx) ** 2 + (py(p.y) - point.sy) ** 2;
        if (d < best) { best = d; index = i; }
      });
      return index;
    };
    function highlight(index, reveal = false) {
      state.selected = index;
      if (reveal && index >= 0 && state.page !== Math.floor(index / pageSize())) {
        state.page = Math.floor(index / pageSize());
        matrices();
      }
      root.querySelectorAll("[data-lr-row]").forEach((el) => el.classList.toggle("lr-selected", +el.dataset.lrRow === index));
      plot.querySelectorAll("[data-lr-point]").forEach((el) => el.classList.toggle("lr-selected", +el.dataset.lrPoint === index));
    }
    function path(fn) {
      let d = "", drawing = false;
      for (let i = 0; i <= 400; i++) {
        const x = -1 + i / 200, y = fn(x);
        if (!Number.isFinite(y)) { drawing = false; continue; }
        // Clamp far outside the view, preserving the clipped path without giant coordinates.
        d += `${drawing ? "L" : "M"}${px(x).toFixed(2)},${py(Math.max(-100, Math.min(100, y))).toFixed(2)} `;
        drawing = true;
      }
      return d;
    }
    function drawPlot() {
      let grid = "";
      const font = parseFloat(getComputedStyle(plot).getPropertyValue("--lr-plot-font")) || 20;
      const room = (box.right - box.left) / font;
      const xStep = room > 28 ? 0.25 : room > 12 ? 0.5 : room > 6 ? 1 : 2;
      const yStep = box.bottom - box.top > font * 10 ? 0.5 : 1;
      for (let x = -1; x <= 1.001; x += xStep) grid += `<line x1="${px(x)}" x2="${px(x)}" y1="${box.top}" y2="${box.bottom}" class="${Math.abs(x) < 0.001 ? "lr-axis" : "lr-gridline"}"/><text x="${px(x)}" y="${box.bottom + font * 1.5}" text-anchor="middle">${fmt(x)}</text>`;
      if (xStep === 2) grid += `<line x1="${px(0)}" x2="${px(0)}" y1="${box.top}" y2="${box.bottom}" class="lr-axis"/>`;
      for (let y = -1; y <= 1.001; y += yStep) grid += `<line x1="${box.left}" x2="${box.right}" y1="${py(y)}" y2="${py(y)}" class="${y === 0 ? "lr-axis" : "lr-gridline"}"/><text x="${box.left - 12}" y="${py(y) + font * 0.35}" text-anchor="end">${fmt(y)}</text>`;
      const residuals = result.theta && state.residuals ? state.points.map((p, i) => `<line class="lr-residual" x1="${px(p.x)}" x2="${px(p.x)}" y1="${py(p.y)}" y2="${py(Math.max(-100, Math.min(100, result.predictions[i])))}"/>`).join("") : "";
      const known = state.truth ? `<path class="lr-truth" d="${path((x) => truthValue(state.truth, x))}"/>` : "";
      const fit = result.theta ? `<path class="lr-fit" d="${path((x) => M.predict(x, result.theta, state.basis, complexity()))}"/>` : "";
      plot.innerHTML = `<defs><clipPath id="lr-clip"><rect x="${box.left}" y="${box.top}" width="${box.right - box.left}" height="${box.bottom - box.top}"/></clipPath></defs><g class="lr-ticks">${grid}<text x="${plotWidth - 12}" y="${plotHeight - 6}" text-anchor="end">x</text><text x="16" y="${font + 4}">y</text></g><g clip-path="url(#lr-clip)">${known}${residuals}${fit}</g><g>${state.points.map((p, i) => `<circle data-lr-point="${i}" class="lr-observation ${i === state.selected ? "lr-selected" : ""}" cx="${px(p.x)}" cy="${py(p.y)}" r="6"><title>${i + 1}: (${fmt(p.x)}, ${fmt(p.y)})</title></circle>`).join("")}</g>${!state.points.length ? `<text class="lr-empty-plot" x="${(box.left + box.right) / 2}" y="${(box.top + box.bottom) / 2}" text-anchor="middle">${w("Add points or draw.", "Добавьте точки или линию.")}</text>` : ""}`;
      $("#lr-truth-legend").hidden = !state.truth;
    }
    function table(values, headers, offset = null) {
      return `<table class="lr-matrix"><thead><tr>${offset !== null ? '<th scope="col">i</th>' : ""}${headers.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${values.map((row, i) => `<tr ${offset !== null ? `data-lr-row="${offset + i}" class="${offset + i === state.selected ? "lr-selected" : ""}"` : ""}>${offset !== null ? `<th scope="row">${offset + i + 1}</th>` : ""}${row.map((v) => `<td title="${esc(v)}">${fmt(v)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    }
    function matrices() {
      const n = state.points.length, p = result.p;
      const size = pageSize();
      state.page = Math.max(0, Math.min(state.page, Math.ceil(n / size) - 1));
      const from = state.page * size, to = Math.min(n, from + size);
      const empty = `<p class="lr-empty">${w("Add observations to populate this matrix.", "Добавьте наблюдения для заполнения матрицы.")}</p>`;
      $("#lr-x-shape").textContent = `${n} × ${p}`;
      $("#lr-y-shape").textContent = `${n} × 1`;
      $("#lr-page").textContent = n ? `${from + 1}–${to} / ${n}` : "0 / 0";
      $('[data-lr-action="previous"]').disabled = state.page === 0;
      $('[data-lr-action="next"]').disabled = to >= n;
      $("#lr-x").innerHTML = n ? table(result.X.slice(from, to), labels(), from) : empty;
      $("#lr-y").innerHTML = n ? table(result.Y.slice(from, to).map((y) => [y]), ["y"], from) : empty;
      $("#lr-coordinates").innerHTML = `<table class="lr-coordinate-table"><thead><tr><th>i</th><th>x</th><th>y</th><th><span class="sr-only">${w("Delete", "Удалить")}</span></th></tr></thead><tbody>${state.points.slice(from, to).map((p, j) => {
        const i = from + j;
        return `<tr data-lr-row="${i}" class="${i === state.selected ? "lr-selected" : ""}"><th scope="row">${i + 1}</th>${["x", "y"].map((axis) => `<td><input type="number" required step="any" min="${axis === "x" ? -1 : -1.5}" max="${axis === "x" ? 1 : 1.5}" value="${Number(p[axis].toPrecision(5))}" data-lr-coordinate="${axis}" data-index="${i}" aria-label="${w("Observation", "Наблюдение")} ${i + 1}, ${axis}"></td>`).join("")}<td><button data-lr-delete="${i}" aria-label="${w("Delete observation", "Удалить наблюдение")} ${i + 1}">×</button></td></tr>`;
      }).join("")}</tbody></table>`;
      $('[data-lr-action="add"]').disabled = n >= LIMIT;
    }
    function settings() {
      const degree = complexity(), fourier = state.basis === "fourier";
      $("#lr-basis").value = state.basis;
      $("#lr-complexity-label").textContent = w(fourier ? "Number of harmonics K" : "Polynomial degree d", fourier ? "Число гармоник K" : "Степень полинома d");
      for (const id of ["#lr-complexity", "#lr-dimension"]) {
        $(id).max = fourier ? 10 : 20;
        $(id).value = degree;
      }
      $("#lr-basis-note").textContent = fourier ? w("Period T = 2 · paired sine/cosine features", "Период T = 2 · пары синусов и косинусов") : w("Powers of x · includes a constant feature", "Степени x · со свободным членом");
      $("#lr-parameter-count").textContent = `${w("Coefficients", "Коэффициентов")}: p = ${fourier ? "2K + 1" : "d + 1"} = ${fourier ? 2 * degree + 1 : degree + 1}`;
      $("#lr-regularization").value = state.regularization;
      $("#lr-lambda").value = state.lambda > 0 && state.lambda < 0.001
        ? Number(state.lambda.toPrecision(4)).toExponential() : Number(state.lambda.toPrecision(6));
      $("#lr-lambda-range").value = Math.log10(state.lambda || 1e-6);
      for (const id of ["#lr-lambda", "#lr-lambda-range", '[data-lr-action="zero"]']) $(id).disabled = state.regularization === "none";
      $("#lr-model").textContent = fourier ? `ŷ = θ₀ + Σₖ₌₁…${degree} [θ₂ₖ₋₁ sin(kπx) + θ₂ₖ cos(kπx)]` : degree === 0 ? "ŷ = θ₀" : `ŷ = θ₀ + ${degree === 1 ? "θ₁x" : `θ₁x + … + θ${degree}x^${degree}`}`;
      if (fourier && degree === 0) $("#lr-model").textContent = "ŷ = θ₀";
      root.querySelectorAll("[data-lr-tool]").forEach((el) => el.setAttribute("aria-pressed", String(el.dataset.lrTool === state.tool)));
      $("#lr-uniform").checked = state.uniform;
      $("#lr-residuals").checked = state.residuals;
      $("#lr-draw-note").textContent = state.uniform
        ? w("Draw mode: sample every Δx = 0.04 and replace observations in the painted interval.", "Рисование: шаг Δx = 0.04, замена наблюдений на нарисованном участке.")
        : w("Draw mode: keep uneven x-spacing and append observations. Existing points stay in place.", "Рисование: неравномерная сетка x, новые наблюдения добавляются к существующим.");
      $("#lr-plot-help").textContent = state.tool === "points"
        ? w("Click to add · drag a point to move · arrows move the selected point · Delete removes it", "Нажмите для добавления · перетащите точку · стрелки двигают выбранную точку · Delete удаляет")
        : state.tool === "draw" ? w("Drag to draw observations · the grid option above controls sampling and replacement", "Проведите линию · переключатель выше управляет сеткой и заменой наблюдений")
          : w("Click or drag over points to erase them. Undo restores the whole gesture.", "Нажмите на точку или проведите по точкам для удаления. Отмена восстанавливает весь жест.");
      plot.dataset.tool = state.tool;
      $("#lr-preset").value = state.truth || "";
      if (state.points.length === 4 && state.points.every((p) => p.x === 0.25)) $("#lr-preset").value = "singular";
    }
    function resizePlot() {
      const rect = plot.getBoundingClientRect();
      if (!rect.width || !rect.height || disposed) return;
      plotWidth = rect.width;
      plotHeight = rect.height;
      const font = parseFloat(getComputedStyle(plot).getPropertyValue("--lr-plot-font")) || 20;
      Object.assign(box, { left: font * 2.9, right: plotWidth - font * 1.2,
        top: font * 1.2, bottom: plotHeight - font * 2.4 });
      plot.setAttribute("viewBox", `0 0 ${plotWidth} ${plotHeight}`);
      if (result) drawPlot();
    }
    function presentation() {
      document.body.classList.toggle("lr-presenting", state.presentation);
      root.classList.toggle("lr-presentation", state.presentation);
      root.style.setProperty("--lr-text-scale", state.textScale);
      root.style.setProperty("--lr-plot-font", `${(state.presentation ? 26 : 20) * state.textScale}px`);
      $("#lr-presentation").checked = state.presentation;
      $("#lr-text-size").value = String(state.textScale);
      for (const view of ["theta", "data", "gram"]) {
        const active = state.view === view;
        $(`#lr-panel-${view}`).hidden = !active;
        $(`#lr-tab-${view}`).setAttribute("aria-selected", String(active));
        $(`#lr-tab-${view}`).tabIndex = active ? 0 : -1;
      }
      if (previousPresentation !== state.presentation) {
        $("#lr-observation-editor").open = !state.presentation;
        previousPresentation = state.presentation;
      }
      try {
        localStorage.setItem("ml-regression-presentation", String(state.presentation));
        localStorage.setItem("ml-regression-text-scale", String(state.textScale));
      } catch { /* Presentation controls still work without storage. */ }
      resizePlot();
    }
    function refresh() {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      if (disposed) return;
      const scrollers = [...root.querySelectorAll(".lr-table-scroll, .lr-inspector-body")]
        .map((el) => ({ el, left: el.scrollLeft, top: el.scrollTop }));
      const start = performance.now();
      result = M.fit(state.points, state.basis, complexity(), state.regularization, state.lambda, result?.theta);
      root.dataset.fitMilliseconds = (performance.now() - start).toFixed(2);
      root.dataset.rank = result.rank;
      root.dataset.parameters = result.p;
      root.dataset.observations = result.n;
      $("#lr-point-count").textContent = `${result.n} / ${LIMIT} ${w("points", "точек")}`;
      $("#lr-mse").textContent = result.mse === null ? "—" : fmt(result.mse);
      $("#lr-penalty").textContent = result.theta ? fmt(result.penalty) : "—";
      $("#lr-objective-value").textContent = result.objective === null ? "—" : fmt(result.objective);
      $("#lr-theta-shape").textContent = `${result.p} × 1`;
      const basisLabels = labels(), max = Math.max(1e-12, ...(result.theta || []).map(Math.abs));
      $("#lr-theta").innerHTML = basisLabels.map((label, j) => {
        const value = result.theta?.[j];
        return `<div class="lr-coefficient ${value === 0 ? "lr-is-zero" : ""}"><div><span>θ<sub>${j}</sub> <small>· ${esc(label)}</small></span><strong title="${value ?? ""}">${value === undefined ? "—" : fmt(value)}</strong></div><div class="lr-bar-track"><i class="${value < 0 ? "lr-negative" : ""}" style="width:${Math.abs(value || 0) / max * 100}%"></i></div></div>`;
      }).join("");
      $("#lr-equation").textContent = result.theta ? "ŷ = " + result.theta.map((v, j) => `${j ? v < 0 ? " − " : " + " : v < 0 ? "−" : ""}${fmt(Math.abs(v))}${j ? `·${basisLabels[j]}` : ""}`).join("") : w("Add observations to fit a model.", "Добавьте наблюдения для построения модели.");
      $("#lr-loss").textContent = "‖Xθ − Y‖² / (2n)" + (state.regularization === "l1" ? " + λ Σⱼ₌₁ |θⱼ|" : state.regularization === "l2" ? " + (λ/2) Σⱼ₌₁ θⱼ²" : "");
      $("#lr-rank").textContent = `${w("rank(X)", "Ранг X")} = ${result.rank} / ${result.p}`;
      $("#lr-condition").textContent = `κ₂(X) = ${result.n ? fmt(result.condition) : "—"}`;
      $("#lr-det").textContent = result.n ? fmt(result.determinant) : "—";
      $(".lr-inspector-determinant").classList.toggle("lr-warning", result.n > 0 && result.rank < result.p);
      $("#lr-gram-shape").textContent = `${result.p} × ${result.p}`;
      $("#lr-gram").innerHTML = table(result.gram, basisLabels);
      $("#lr-xty").innerHTML = table(result.xty.map((v) => [v]), ["XᵀY"]);
      $("#lr-ridge-panel").hidden = state.regularization !== "l2";
      $("#lr-ridge").innerHTML = state.regularization === "l2" ? table(result.system || result.gram, basisLabels) : "";
      const deficient = result.rank < result.p;
      let status = !result.n ? w("Start with a point, a freehand stroke, or a preset.", "Начните с точки, линии от руки или готового примера.")
        : deficient ? w("X is rank deficient: unregularized coefficients are not unique. At λ = 0, the minimum-norm solution is shown.", "Ранг X неполный: коэффициенты без регуляризации не единственны. При λ = 0 показано решение минимальной нормы.")
          : result.condition > 1e6 ? w("Ill-conditioned features: small changes in observations can cause large changes in coefficients.", "Плохо обусловленные признаки: небольшие изменения данных могут сильно менять коэффициенты.")
            : w("X has full column rank. The unregularized least-squares coefficients are unique.", "X имеет полный столбцовый ранг. Коэффициенты МНК без регуляризации единственны.");
      if (!result.converged) status += w(" Lasso reached its iteration limit; the current approximation is shown.", " Lasso достиг предела итераций; показано текущее приближение.");
      if (result.n >= LIMIT) status += w(" Point limit reached; erase points to add more.", " Достигнут лимит точек; удалите часть, чтобы добавить новые.");
      $("#lr-status").textContent = status;
      $("#lr-status").classList.toggle("lr-warning", result.n > 0 && (deficient || result.condition > 1e6 || !result.converged));
      $("#lr-solver-note").textContent = state.regularization === "l1" && state.lambda > 0
        ? w(`L1 is solved by coordinate descent (${result.iterations || 0} sweeps). The normal equations above describe the unregularized problem.`, `L1: покоординатный спуск (${result.iterations || 0} проходов). Нормальные уравнения выше относятся к задаче без регуляризации.`)
        : w("Computed with a singular value decomposition of X (an augmented X for L2), without explicitly inverting XᵀX.", "Расчёт через сингулярное разложение X (расширенной X для L2), без явного обращения XᵀX.");
      $('[data-lr-action="undo"]').disabled = !state.undo.length;
      $('[data-lr-action="redo"]').disabled = !state.redo.length;
      $('[data-lr-action="clear"]').disabled = !result.n;
      matrices();
      drawPlot();
      for (const { el, left, top } of scrollers) { el.scrollLeft = left; el.scrollTop = top; }
      root.dataset.updateMilliseconds = (performance.now() - start).toFixed(2);
    }
    const schedule = () => { if (!frame) frame = requestAnimationFrame(refresh); };
    function changed(before) {
      state.truth = null;
      record(before);
      settings();
      refresh();
    }
    function paint(point) {
      if (!gesture) return;
      if (state.tool === "erase") {
        const index = near(point);
        if (index >= 0) { state.points.splice(index, 1); state.selected = -1; state.truth = null; }
      } else if (state.tool === "points") {
        if (gesture.index >= 0) { state.points[gesture.index] = { x: point.x, y: point.y }; state.truth = null; }
      } else {
        const last = gesture.stroke.at(-1);
        if (last && Math.hypot(point.sx - last.sx, point.sy - last.sy) < 3) return;
        gesture.stroke.push(point);
        if (!state.uniform) {
          if (state.points.length < LIMIT) state.points.push({ x: point.x, y: point.y });
        } else {
          const samples = new Map();
          const stroke = gesture.stroke;
          if (stroke.length === 1) {
            const k = Math.max(0, Math.min(50, Math.round((point.x + 1) / 0.04)));
            samples.set(k, { x: -1 + k * 0.04, y: point.y });
          }
          for (let i = 1; i < stroke.length; i++) {
            const a = stroke[i - 1], b = stroke[i];
            const first = Math.ceil((Math.min(a.x, b.x) + 1) / 0.04 - 1e-9);
            const last = Math.floor((Math.max(a.x, b.x) + 1) / 0.04 + 1e-9);
            for (let k = first; k <= last; k++) {
              const x = -1 + k * 0.04;
              samples.set(k, { x, y: b.x === a.x ? b.y : a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x) });
            }
          }
          const xs = [...stroke.map((p) => p.x), ...[...samples.values()].map((p) => p.x)];
          const lo = Math.min(...xs), hi = Math.max(...xs);
          const outside = gesture.before.points.filter((p) => p.x < lo - 1e-9 || p.x > hi + 1e-9);
          state.points = [...outside, ...[...samples.values()].sort((a, b) => a.x - b.x).slice(0, LIMIT - outside.length)];
        }
        state.truth = null;
        state.selected = state.view === "data" ? state.points.length - 1 : -1;
        if (state.selected >= 0) state.page = Math.floor(state.selected / pageSize());
      }
      schedule();
    }
    function finish() {
      if (!gesture) return;
      record(gesture.before);
      gesture = null;
      settings();
      if (frame) cancelAnimationFrame(frame);
      refresh();
    }
    plot.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || gesture) return;
      const point = xy(event);
      if (point.sx < box.left || point.sx > box.right || point.sy < box.top || point.sy > box.bottom) return;
      event.preventDefault();
      plot.focus({ preventScroll: true });
      plot.setPointerCapture(event.pointerId);
      gesture = { before: snapshot(), index: near(point), stroke: [], pointerId: event.pointerId };
      if (state.tool === "points") {
        if (gesture.index < 0 && state.points.length < LIMIT) {
          state.points.push({ x: point.x, y: point.y });
          gesture.index = state.points.length - 1;
          state.truth = null;
        }
        highlight(gesture.index, true);
        schedule();
      } else paint(point);
    });
    plot.addEventListener("pointermove", (event) => {
      if (gesture && event.pointerId === gesture.pointerId) paint(xy(event));
      else if (!gesture) highlight(near(xy(event)));
    });
    plot.addEventListener("pointerup", finish);
    plot.addEventListener("pointercancel", finish);
    plot.addEventListener("lostpointercapture", finish);
    root.addEventListener("pointerover", (event) => {
      const row = event.target.closest("[data-lr-row]");
      if (row && !gesture) highlight(+row.dataset.lrRow);
    });
    root.addEventListener("focusin", (event) => {
      const row = event.target.closest("[data-lr-row]");
      if (row) highlight(+row.dataset.lrRow);
    });
    root.addEventListener("click", (event) => {
      const view = event.target.closest("[data-lr-view]");
      if (view) {
        state.view = view.dataset.lrView;
        presentation();
        $(".lr-inspector-body").scrollTop = 0;
      }
      const tool = event.target.closest("[data-lr-tool]");
      if (tool) { state.tool = tool.dataset.lrTool; settings(); }
      const deletion = event.target.closest("[data-lr-delete]");
      if (deletion) {
        const before = snapshot();
        state.points.splice(+deletion.dataset.lrDelete, 1);
        state.selected = -1;
        changed(before);
      }
      const action = event.target.closest("[data-lr-action]")?.dataset.lrAction;
      if (!action) return;
      if (action === "undo" || action === "redo") {
        const source = state[action], target = state[action === "undo" ? "redo" : "undo"];
        if (!source.length) return;
        target.push(snapshot());
        Object.assign(state, source.pop(), { selected: -1 });
        settings(); refresh();
      } else if (action === "clear" || action === "add") {
        const before = snapshot();
        if (action === "clear") { state.points = []; state.page = 0; state.selected = -1; }
        else if (state.points.length < LIMIT) {
          state.points.push({ x: 0, y: 0 });
          state.selected = state.points.length - 1;
          state.page = Math.floor(state.selected / pageSize());
        }
        changed(before);
        if (action === "add") $(`[data-index="${state.selected}"][data-lr-coordinate="x"]`)?.focus();
      } else if (action === "zero") {
        state.lambda = 0; settings(); refresh();
      } else if (action === "previous" || action === "next") {
        state.page += action === "next" ? 1 : -1;
        matrices();
      }
    });
    root.addEventListener("input", (event) => {
      if (event.target.id === "lr-complexity") {
        state.degrees[state.basis] = +event.target.value;
        settings(); schedule();
      } else if (event.target.id === "lr-lambda-range") {
        state.lambda = 10 ** +event.target.value;
        settings(); schedule();
      }
    });
    root.addEventListener("change", (event) => {
      const el = event.target;
      if (el.id === "lr-presentation" || el.id === "lr-text-size") {
        const anchor = state.page * pageSize();
        if (el.id === "lr-presentation") state.presentation = el.checked;
        else state.textScale = +el.value;
        state.page = Math.floor(anchor / pageSize());
        presentation(); refresh(); return;
      }
      if (el.dataset.lrCoordinate) {
        if (!el.value || !el.checkValidity() || !Number.isFinite(+el.value)) { el.reportValidity(); return; }
        const before = snapshot();
        state.points[+el.dataset.index][el.dataset.lrCoordinate] = +el.value;
        changed(before); return;
      }
      if (el.id === "lr-basis") state.basis = el.value;
      else if (el.id === "lr-regularization") state.regularization = el.value;
      else if (el.id === "lr-dimension" || el.id === "lr-lambda") {
        if (!el.value || !el.checkValidity() || !Number.isFinite(+el.value)) { el.reportValidity(); return; }
        if (el.id === "lr-dimension") state.degrees[state.basis] = +el.value;
        else state.lambda = +el.value;
      } else if (el.id === "lr-uniform") state.uniform = el.checked;
      else if (el.id === "lr-residuals") state.residuals = el.checked;
      else if (el.id === "lr-preset" && el.value) {
        const before = snapshot();
        state.points = preset(el.value);
        state.truth = el.value === "singular" ? null : el.value;
        state.page = 0; state.selected = -1;
        record(before);
      } else return;
      settings(); refresh();
    });
    root.addEventListener("keydown", (event) => {
      if (event.target.matches("[data-lr-view]") && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const views = ["theta", "data", "gram"];
        const index = views.indexOf(state.view);
        state.view = views[event.key === "Home" ? 0 : event.key === "End" ? 2
          : (index + (event.key === "ArrowRight" ? 1 : 2)) % 3];
        presentation();
        $(".lr-inspector-body").scrollTop = 0;
        $(`#lr-tab-${state.view}`).focus(); return;
      }
      if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        $(`[data-lr-action="${event.shiftKey ? "redo" : "undo"}"]`).click();
      }
      if (event.target !== plot || state.selected < 0) return;
      const keys = ["Delete", "Backspace", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      const before = snapshot(), point = state.points[state.selected];
      if (event.key === "Delete" || event.key === "Backspace") {
        state.points.splice(state.selected, 1); state.selected = -1;
      } else {
        const step = event.shiftKey ? 0.1 : 0.01;
        point.x = Math.max(-1, Math.min(1, point.x + (event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0)));
        point.y = Math.max(-1.5, Math.min(1.5, point.y + (event.key === "ArrowUp" ? step : event.key === "ArrowDown" ? -step : 0)));
      }
      changed(before);
    });
    presentation(); settings(); refresh();
    const resizeObserver = new ResizeObserver(resizePlot);
    resizeObserver.observe(plot);
    teardown = () => {
      if (gesture) record(gesture.before);
      disposed = true;
      resizeObserver.disconnect();
      document.body.classList.remove("lr-presenting");
      if (frame) cancelAnimationFrame(frame);
    };
  }
  window.MLRegressionLab = { html, mount, unmount: () => teardown() };
})();
