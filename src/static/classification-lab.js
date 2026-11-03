(function () {
  "use strict";
  const M = window.MLClassificationMath;
  const D = window.ML_TITANIC;
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
      <section class="cl-card" id="cl-scores-card"></section>
      <section class="cl-card" id="cl-confusion-card"></section>
      <section class="cl-card" id="cl-roc-card"></section>
    </section>`;
  }

  function mount(root, language) {
    const w = (en, ru) => language === "en" ? en : ru;
    const $ = (selector) => root.querySelector(selector);
    const controller = new AbortController();
    const listen = (type, handler) => root.addEventListener(type, handler, { signal: controller.signal });
    let columns = [], model, logits, frame = 0;
    function refit() {
      columns = activeColumns();
      model = M.fit(D.train.X, D.train.y, { penalty: state.penalty, lambda: state.lambda, columns });
      logits = {
        train: M.scores(D.train.X, model.weights, model.intercept),
        test: M.scores(D.test.X, model.weights, model.intercept),
      };
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
    function render() {
      root.dataset.clColumns = columns.length;
      root.dataset.clIterations = model.iterations;
      $("#cl-fit-summary").textContent = `${w("columns", "столбцов")}: ${columns.length} · ${w("iterations", "итераций")}: ${model.iterations}`;
      settings();
      coefficients();
      status();
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
    listen("input", (event) => {
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
      document.body.classList.remove("lr-presenting");
    };
  }
  window.MLClassificationLab = { html, mount, unmount: () => teardown() };
})();
