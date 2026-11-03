(function () {
  "use strict";
  const M = window.MLClassificationMath;
  const D = window.ML_TITANIC;
  const P = window.MLClassificationPlots;
  const preference = (key, fallback) => {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  };
  // Lecture view shares the regression lab's keys, so one setting serves both labs.
  const preferences = () => ({
    presentation: preference("ml-regression-presentation", "true") !== "false",
    textScale: ["1", "1.25", "1.5"].includes(preference("ml-regression-text-scale", "1"))
      ? +preference("ml-regression-text-scale", "1") : 1,
  });
  const state = {
    on: new Set(D.groups.filter((group) => group.default_on).map((group) => group.id)),
    penalty: "l2", lambda: 1, axis: "probability", threshold: 0.5, split: "train",
    ...preferences(),
  };
  let teardown = () => {};
  const esc = (value) => String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmt = (value) => value === 0 ? "0" : value.toFixed(3);
  // Slider and number ranges on each displayed axis; the threshold itself is stored as a probability.
  const AXES = { probability: { min: 0.005, max: 0.995, step: 0.005 }, logit: { min: -6, max: 6, step: 0.05 } };
  const show = (value) => String(Number(value.toFixed(3)));
  // Indices into D.columns of every column in the switched-on groups, in D.columns order.
  const activeColumns = () => D.columns.flatMap((name, j) =>
    D.groups.some((group) => state.on.has(group.id) && group.columns.includes(name)) ? [j] : []);
  // Columns in a linear dependency with the others and the intercept. For the Gram matrix G of
  // [columns, 1], r − G⁺G r is the null-space part of a generic vector r: non-zero exactly on them.
  function collinearColumns(columns) {
    const m = columns.length + 1;
    const G = Array.from({ length: m }, () => Array(m).fill(0));
    for (const row of D.train.X) {
      const x = [...columns.map((j) => row[j]), 1];
      for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) G[a][b] += x[a] * x[b];
    }
    const r = G.map((_, a) => Math.sin(a + 1));
    const { x } = M.solveSymmetric(G, G.map((line) => line.reduce((sum, v, b) => sum + v * r[b], 0)));
    const residual = r.map((v, a) => Math.abs(v - x[a]));
    const largest = Math.max(...residual);
    return columns.filter((_, k) => residual[k] > 1e-3 * largest);
  }

  function html(language) {
    const w = (en, ru) => language === "en" ? en : ru;
    const groups = D.groups.map((group) => {
      const note = esc(w(group.note.en, group.note.ru));
      return `<label class="cl-group" title="${note}"><input type="checkbox" data-cl-group="${group.id}" aria-labelledby="cl-group-${group.id}" aria-describedby="cl-hint-${group.id}"><span><span class="cl-group-name" id="cl-group-${group.id}">${esc(w(group.label.en, group.label.ru))}</span><span class="cl-hint" id="cl-hint-${group.id}">${note}</span></span></label>`;
    }).join("");
    return `<section id="classification-lab" class="cl-lab" aria-labelledby="cl-title">
      <div class="cl-heading"><div><p class="eyebrow">${w("INTERACTIVE LAB / 02", "ИНТЕРАКТИВНАЯ ЛАБОРАТОРИЯ / 02")}</p>
      <h1 id="cl-title">${w("Classification metrics", "Метрики классификации")}</h1>
      <p class="page-description">${w("Switch features on and off, pick a penalty, then move the threshold and watch every metric respond.", "Включайте признаки, выбирайте штраф, двигайте порог и смотрите, как отвечают метрики.")}</p></div><div class="cl-presentation-controls"><label><input id="cl-presentation" type="checkbox"> ${w("Lecture view", "Для лекции")}</label><label for="cl-text-size">${w("Text size", "Размер текста")}</label><select id="cl-text-size"><option value="1">100%</option><option value="1.25">125%</option><option value="1.5">150%</option></select></div></div>
      <div class="cl-flow" aria-label="${w("Learning steps", "Этапы")}"><span>01 ${w("Model", "Модель")}</span><b>→</b><span>02 ${w("Scores", "Оценки")}</span><b>→</b><span>03 ${w("Confusion matrix", "Матрица ошибок")}</span><b>→</b><span>04 ROC</span></div>
      <section class="cl-card" id="cl-model-card" aria-labelledby="cl-model-title">
        <div class="cl-card-heading"><h2 id="cl-model-title">01 · ${w("Model", "Модель")}</h2><span class="cl-badge" id="cl-fit-summary"></span></div>
        <div class="cl-features" role="group" aria-labelledby="cl-features-title"><h3 id="cl-features-title">${w("Features", "Признаки")}</h3><div id="cl-groups" class="cl-groups">${groups}</div></div>
        <div class="cl-controls">
          <div class="cl-control"><label for="cl-penalty">${w("Penalty", "Штраф")}</label><select id="cl-penalty"><option value="none">${w("None · maximum likelihood", "Нет · максимум правдоподобия")}</option><option value="l2">L2 · Ridge</option><option value="l1">L1 · Lasso</option></select><p class="cl-hint">${w("The intercept b is never penalized.", "Свободный член b не штрафуется.")}</p></div>
          <div class="cl-control"><label for="cl-lambda">${w("Penalty strength", "Сила штрафа")} λ</label><div class="cl-range-row"><input id="cl-lambda-range" type="range" min="-3" max="2" step="0.1" aria-label="${w("Logarithmic penalty strength", "Логарифмическая сила штрафа")}"><input id="cl-lambda" type="number" required min="0.001" max="100" step="any"></div><p class="cl-hint">${w("From 0.001 to 100 on a logarithmic slider.", "От 0.001 до 100, ползунок логарифмический.")}</p></div>
        </div>
        <div class="cl-coefficients-block"><h3>${w("Coefficients", "Коэффициенты")} w</h3><p class="cl-hint">${w("A binary column shifts the logit by its weight; a standardized one, per standard deviation.", "Бинарный признак сдвигает логит на свой вес, стандартизованный — на вес за одно стандартное отклонение.")}</p><div id="cl-coefficients" class="cl-coefficients"></div><p id="cl-intercept" class="cl-intercept"></p></div>
        <p id="cl-status" class="cl-status" aria-live="polite"></p>
      </section>
      <section class="cl-card" id="cl-scores-card" aria-labelledby="cl-scores-title">
        <div class="cl-card-heading"><h2 id="cl-scores-title">02 · ${w("Scores and threshold", "Оценки и порог")}</h2></div>
        <div class="cl-toolbar">
          <div id="cl-split" class="cl-segmented" role="group" aria-label="${w("Data split", "Выборка")}"><button type="button" data-cl-split="train" aria-pressed="false">${w("Train", "Обучение")} · ${D.train.y.length}</button><button type="button" data-cl-split="test" aria-pressed="false">${w("Test", "Тест")} · ${D.test.y.length}</button></div>
          <div id="cl-axis" class="cl-segmented" role="group" aria-label="${w("Score axis", "Ось оценок")}"><button type="button" data-cl-axis="logit" aria-pressed="false">${w("Logit", "Логит")}</button><button type="button" data-cl-axis="probability" aria-pressed="false">${w("Probability", "Вероятность")}</button></div>
        </div>
        <svg id="cl-plot" viewBox="0 0 960 420" role="img" tabindex="0" aria-label="${w("Score distributions of passengers who survived and who died, cut by the threshold line. Drag, click or use the arrow keys to move the threshold.", "Распределения оценок выживших и погибших пассажиров, разрезанные линией порога. Перетащите, щёлкните или нажимайте стрелки, чтобы сдвинуть порог.")}"></svg>
        <div class="cl-legend"><span><i class="cl-swatch cl-positive"></i>${w("Survived", "Выжили")}</span><span><i class="cl-swatch cl-negative"></i>${w("Died", "Погибли")}</span><span class="cl-hint">${w("Drag the line or click the plot; ← → move it one step, with Shift ten.", "Перетащите линию или щёлкните по графику; ← → сдвигают на шаг, с Shift — на десять.")}</span></div>
        <div class="cl-threshold-row"><label for="cl-threshold">${w("Threshold", "Порог")}</label><div class="cl-range-row"><input id="cl-threshold-range" type="range" aria-label="${w("Threshold slider", "Ползунок порога")}"><input id="cl-threshold" type="number" required></div></div>
        <p id="cl-readout" class="cl-readout" aria-live="polite"></p>
      </section>
      <section class="cl-card" id="cl-confusion-card"></section>
      <section class="cl-card" id="cl-roc-card"></section>
    </section>`;
  }

  function mount(root, language) {
    const w = (en, ru) => language === "en" ? en : ru;
    const $ = (selector) => root.querySelector(selector);
    const controller = new AbortController();
    const listen = (type, handler, target = root) => target.addEventListener(type, handler, { signal: controller.signal });
    const plot = $("#cl-plot");
    let columns = [], model, logits, probabilities, logitRange, classes = null, frame = 0, thresholdFrame = 0, drag = null;
    function refit() {
      columns = activeColumns();
      model = M.fit(D.train.X, D.train.y, { penalty: state.penalty, lambda: state.lambda, columns });
      logits = {
        train: M.scores(D.train.X, model.weights, model.intercept),
        test: M.scores(D.test.X, model.weights, model.intercept),
      };
      probabilities = { train: M.probabilities(logits.train), test: M.probabilities(logits.test) };
      // The logit axis spans the training scores padded by 5 %, so switching the split keeps the axis.
      const lo = Math.min(...logits.train), hi = Math.max(...logits.train), pad = hi > lo ? 0.05 * (hi - lo) : 1;
      logitRange = [Math.max(-8, lo - pad), Math.min(8, hi + pad)];
      if (!(logitRange[1] > logitRange[0])) logitRange = [-8, 8];
      classes = null;
    }
    function settings() {
      root.querySelectorAll("[data-cl-group]").forEach((input) => { input.checked = state.on.has(input.dataset.clGroup); });
      $("#cl-penalty").value = state.penalty;
      $("#cl-lambda").value = String(state.lambda);
      $("#cl-lambda-range").value = Math.log10(state.lambda);
      for (const id of ["#cl-lambda", "#cl-lambda-range"]) $(id).disabled = state.penalty === "none";
    }
    function coefficients() {
      const largest = Math.max(1e-12, ...columns.map((j) => Math.abs(model.weights[j])));
      $("#cl-coefficients").innerHTML = columns.map((j) => {
        const name = D.columns[j], value = model.weights[j];
        return `<div class="cl-coefficient${value === 0 ? " cl-is-zero" : ""}" data-cl-column="${name}"><div><span>${name}</span><strong title="${value}">${fmt(value)}</strong></div><div class="cl-bar-track"><i class="${value < 0 ? "cl-negative" : ""}" style="width:${Math.abs(value) / largest * 100}%"></i></div></div>`;
      }).join("");
      $("#cl-intercept").innerHTML = `<span>${w("Intercept", "Свободный член")} b</span><strong title="${model.intercept}">${fmt(model.intercept)}</strong>`;
    }
    function status() {
      const names = (indices) => indices.map((j) => D.columns[j]).join(", ");
      const notes = [];
      if (model.constant.length) notes.push(w(`${names(model.constant)} constant in the training set: weight fixed at 0`, `${names(model.constant)} — константа в обучающей выборке: вес зафиксирован на 0`));
      // The solver flags separation when a weight exceeds 10 in absolute value; name those columns.
      if (model.separated) {
        const separating = names(columns.filter((j) => Math.abs(model.weights[j]) > 10));
        notes.push(w(`Perfect separation on ${separating}: without a penalty the weights grow without bound. Add L2.`, `Полное разделение по ${separating}: без штрафа веса растут неограниченно. Добавьте L2.`));
      }
      if (model.collinear) {
        const dependent = names(collinearColumns(columns.filter((j) => !model.constant.includes(j))));
        notes.push(w(`${dependent} are collinear: without a penalty their coefficients are not unique; predictions and metrics still are.`, `${dependent} коллинеарны: без штрафа их коэффициенты не единственны, а предсказания и метрики единственны.`));
      }
      $("#cl-status").textContent = notes.join(" · ");
      $("#cl-status").classList.toggle("cl-warning", model.separated || model.collinear);
    }
    const range = () => state.axis === "logit" ? logitRange : [0, 1];
    // The threshold on the displayed axis, and back to the stored probability.
    const shown = () => state.axis === "logit" ? M.logit(state.threshold) : state.threshold;
    const fromShown = (value) => state.axis === "logit" ? M.sigmoid(value) : value;
    // Round to the slider step and keep within the slider's range.
    const snap = (value) => {
      const { min, max, step } = AXES[state.axis];
      return Math.min(max, Math.max(min, Number((Math.round(value / step) * step).toFixed(3))));
    };
    const moveTo = (value) => { state.threshold = fromShown(snap(value)); };
    // The selected split's scores on the displayed axis, by class; kept until the next refit, axis or split change.
    function classScores() {
      if (!classes) {
        const values = state.axis === "logit" ? logits[state.split] : probabilities[state.split];
        const y = D[state.split].y;
        classes = { positives: values.filter((_, i) => y[i] === 1), negatives: values.filter((_, i) => y[i] === 0) };
      }
      return classes;
    }
    function distribution() {
      const [lo, hi] = range(), logit = state.axis === "logit";
      const focused = document.activeElement?.id === "cl-threshold-handle";
      plot.innerHTML = P.distributionPlot({
        axis: state.axis, range: [lo, hi], bins: 40, ...classScores(),
        threshold: Math.min(hi, Math.max(lo, shown())),
        fontSize: parseFloat(getComputedStyle(plot).getPropertyValue("--cl-plot-font")) || 16,
        labels: { positive: w("Survived", "Выжили"), negative: w("Died", "Погибли"), x: logit ? w("logit", "логит") : w("probability", "вероятность"), y: w("passengers", "пассажиры"), empty: w("No data", "Нет данных") },
      });
      // The new markup has a new handle: keep keyboard focus on it.
      if (focused) $("#cl-threshold-handle").focus({ preventScroll: true });
    }
    function readout() {
      const p = probabilities[state.split];
      const predicted = p.reduce((count, value) => count + (value >= state.threshold ? 1 : 0), 0);
      // Two decimals; the Russian readout uses a decimal comma.
      const number = (value) => {
        const text = (Math.abs(value) < 0.005 ? 0 : value).toFixed(2);
        return language === "en" ? text : text.replace(".", ",");
      };
      const t = number(state.threshold), z = number(M.logit(state.threshold));
      $("#cl-readout").textContent = w(`p ≥ ${t} (logit ≥ ${z}): ${predicted} predicted survivors of ${p.length}`, `p ≥ ${t} (логит ≥ ${z}): ${predicted} предсказанных выживших из ${p.length}`);
    }
    // Everything that follows the threshold, without refitting. Task 9: the confusion matrix, metrics and ROC operating point go here.
    function renderThreshold() {
      if (thresholdFrame) cancelAnimationFrame(thresholdFrame);
      thresholdFrame = 0;
      const value = show(snap(shown()));
      $("#cl-threshold-range").value = value;
      $("#cl-threshold").value = value;
      distribution();
      readout();
    }
    // Pointer drags and slider input render at most once per frame.
    const scheduleThreshold = () => { if (!thresholdFrame) thresholdFrame = requestAnimationFrame(renderThreshold); };
    // Split and axis switches and the inputs' ranges, then the threshold views.
    function scores() {
      root.dataset.clSplit = state.split;
      root.querySelectorAll("button[data-cl-split]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.clSplit === state.split)));
      root.querySelectorAll("button[data-cl-axis]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.clAxis === state.axis)));
      const { min, max, step } = AXES[state.axis];
      for (const id of ["#cl-threshold-range", "#cl-threshold"]) Object.assign($(id), { min, max, step });
      renderThreshold();
    }
    function render() {
      root.dataset.clColumns = columns.length;
      root.dataset.clIterations = model.iterations;
      $("#cl-fit-summary").textContent = `${w("columns", "столбцов")}: ${columns.length} · ${w("iterations", "итераций")}: ${model.iterations}`;
      settings();
      coefficients();
      status();
      scores();
    }
    function update() {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      refit();
      render();
    }
    function presentation() {
      document.body.classList.toggle("lr-presenting", state.presentation);
      root.classList.toggle("cl-presentation", state.presentation);
      root.style.setProperty("--cl-text-scale", state.textScale);
      $("#cl-presentation").checked = state.presentation;
      $("#cl-text-size").value = String(state.textScale);
    }
    // Pointer x in viewBox units, as a value on the displayed axis within the axis range.
    function pointerValue(event) {
      const [lo, hi] = range();
      const { x } = new DOMPoint(event.clientX, event.clientY).matrixTransform(plot.getScreenCTM().inverse());
      return Math.min(hi, Math.max(lo, P.scale([lo, hi], P.DISTRIBUTION.box).value(x)));
    }
    listen("pointerdown", (event) => {
      if (event.button !== 0 || drag !== null) return;
      event.preventDefault();
      plot.focus({ preventScroll: true });
      plot.setPointerCapture(event.pointerId);
      drag = event.pointerId;
      moveTo(pointerValue(event));
      renderThreshold();
    }, plot);
    listen("pointermove", (event) => {
      if (event.pointerId !== drag) return;
      moveTo(pointerValue(event));
      scheduleThreshold();
    }, plot);
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) listen(type, (event) => {
      if (event.pointerId !== drag) return;
      drag = null;
      if (thresholdFrame) renderThreshold();
    }, plot);
    listen("keydown", (event) => {
      if ((event.key !== "ArrowLeft" && event.key !== "ArrowRight") || !event.target.closest("#cl-plot")) return;
      event.preventDefault();
      const { step } = AXES[state.axis];
      moveTo(snap(shown()) + (event.key === "ArrowRight" ? step : -step) * (event.shiftKey ? 10 : 1));
      renderThreshold();
    });
    listen("click", (event) => {
      const split = event.target.closest("button[data-cl-split]"), axis = event.target.closest("button[data-cl-axis]");
      if (split) state.split = split.dataset.clSplit;
      else if (axis) state.axis = axis.dataset.clAxis;
      else return;
      classes = null;
      scores();
    });
    listen("input", (event) => {
      if (event.target.id === "cl-threshold-range") {
        state.threshold = fromShown(+event.target.value);
        scheduleThreshold();
        return;
      }
      if (event.target.id !== "cl-lambda-range") return;
      state.lambda = Number((10 ** +event.target.value).toPrecision(3));
      $("#cl-lambda").value = String(state.lambda);
      if (!frame) frame = requestAnimationFrame(update);
    });
    listen("change", (event) => {
      const el = event.target;
      if (el.id === "cl-presentation" || el.id === "cl-text-size") {
        if (el.id === "cl-presentation") state.presentation = el.checked;
        else state.textScale = +el.value;
        try {
          localStorage.setItem("ml-regression-presentation", String(state.presentation));
          localStorage.setItem("ml-regression-text-scale", String(state.textScale));
        } catch { /* Presentation controls still work without storage. */ }
        presentation(); return;
      }
      if (el.id === "cl-threshold") {
        // Out-of-range or malformed thresholds are ignored, like λ; others snap to the slider step.
        const value = +el.value, { min, max } = AXES[state.axis];
        if (el.value && Number.isFinite(value) && value >= min && value <= max) moveTo(value);
        renderThreshold(); return;
      }
      if (el.dataset.clGroup) {
        if (el.checked) state.on.add(el.dataset.clGroup);
        else state.on.delete(el.dataset.clGroup);
      } else if (el.id === "cl-penalty") state.penalty = el.value;
      else if (el.id === "cl-lambda") {
        // Out-of-range or malformed strengths are ignored: the field returns to the current λ.
        if (!el.value || !el.validity.valid || !Number.isFinite(+el.value)) { el.value = String(state.lambda); return; }
        state.lambda = +el.value;
      } else return;
      update();
    });
    Object.assign(state, preferences());
    presentation();
    update();
    teardown = () => {
      controller.abort();
      if (frame) cancelAnimationFrame(frame);
      if (thresholdFrame) cancelAnimationFrame(thresholdFrame);
      document.body.classList.remove("lr-presenting");
    };
  }
  window.MLClassificationLab = { html, mount, unmount: () => teardown() };
})();
