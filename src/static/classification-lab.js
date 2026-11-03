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

  const SPLITS = ["train", "test"];
  const other = (split) => split === "train" ? "test" : "train";
  // The metric table in display order: key, name, formula, one-line hint (EN, RU).
  const METRICS = [
    ["accuracy", ["Accuracy", "Доля верных ответов (accuracy)"], "(TP + TN) / n", ["Share of all passengers the model classifies correctly.", "Доля пассажиров, для которых предсказание верно."]],
    ["precision", ["Precision", "Точность (precision)"], "TP / (TP + FP)", ["Of the passengers predicted to survive, the share who did.", "Среди предсказанных выживших — доля действительно выживших."]],
    ["recall", ["Recall (TPR)", "Полнота (recall, TPR)"], "TP / (TP + FN)", ["Of the passengers who survived, the share the model finds.", "Среди выживших — доля, которую модель находит."]],
    ["specificity", ["Specificity (TNR)", "Специфичность (TNR)"], "TN / (TN + FP)", ["Of the passengers who died, the share predicted to die.", "Среди погибших — доля, для которых предсказана гибель."]],
    ["fpr", ["False positive rate (FPR)", "Доля ложных срабатываний (FPR)"], "FP / (FP + TN)", ["Of the passengers who died, the share wrongly predicted to survive.", "Среди погибших — доля, ошибочно предсказанных выжившими."]],
    ["fnr", ["False negative rate (FNR)", "Доля пропусков (FNR)"], "FN / (FN + TP)", ["Of the passengers who survived, the share the model misses.", "Среди выживших — доля, которую модель пропускает."]],
    ["f1", ["F1 score", "F1-мера"], "2·P·R / (P + R)", ["Harmonic mean of precision and recall; low if either is low.", "Гармоническое среднее точности и полноты; мало, если мала хотя бы одна."]],
    ["balancedAccuracy", ["Balanced accuracy", "Сбалансированная доля верных ответов"], "(TPR + TNR) / 2", ["Mean of recall and specificity, so the larger class does not dominate.", "Среднее полноты и специфичности: больший класс не перевешивает."]],
    ["mcc", ["Matthews correlation (MCC)", "Корреляция Мэтьюса (MCC)"], "(TP·TN − FP·FN) / √(…)", ["Correlation between predicted and actual classes, from −1 to 1.", "Корреляция предсказанных и настоящих классов, от −1 до 1."]],
    ["youdenJ", ["Youden's J", "Индекс Юдена (J)"], "TPR − FPR", ["Height of the operating point above the ROC diagonal.", "Высота рабочей точки над диагональю ROC."]],
    ["npv", ["Negative predictive value (NPV)", "Точность отрицательных ответов (NPV)"], "TN / (TN + FN)", ["Of the passengers predicted to die, the share who did.", "Среди предсказанных погибших — доля действительно погибших."]],
    ["auc", ["ROC AUC", "ROC AUC"], "∫ TPR d(FPR)", ["Chance that a random survivor scores above a random passenger who died.", "Вероятность, что случайный выживший получит оценку выше случайного погибшего."]],
    ["logLoss", ["Log loss", "Логистические потери (log loss)"], "−mean(y log p + (1−y) log(1−p))", ["Average −log of the probability given to the true class; lower is better.", "Среднее −log вероятности, отданной настоящему классу; чем меньше, тем лучше."]],
    ["brier", ["Brier score", "Оценка Брайера (Brier score)"], "mean((p − y)²)", ["Mean squared gap between the probability and the 0/1 outcome; lower is better.", "Средний квадрат разницы между вероятностью и исходом 0/1; чем меньше, тем лучше."]],
  ].map(([key, name, formula, hint]) => ({ key, name, formula, hint, free: ["auc", "logLoss", "brier"].includes(key) }));
  const metricText = (value) => value === null ? "—" : value.toFixed(3);
  // The selected split's (FPR, TPR) at the current threshold; a rate with an empty denominator is drawn at 0.
  const operatingPoint = ({ tp, fn, fp, tn }) => ({ fpr: fp + tn ? fp / (fp + tn) : 0, tpr: tp + fn ? tp / (tp + fn) : 0 });

  // Card 03: the confusion matrix beside the metric table. The structure is static; mount fills in the numbers.
  function confusionMarkup(w) {
    const cell = (key) => `<td id="cl-${key}" class="cl-cell cl-${key}" data-cl-label="${key.toUpperCase()}"></td>`;
    const rows = METRICS.map(({ key, name, formula, hint, free }) => {
      const tip = esc(w(...hint));
      const tag = free ? `<span class="cl-tag">${w("threshold-free", "не зависит от порога")}</span>` : "";
      return `<tr data-cl-metric="${key}"><th scope="row" title="${tip}"><span class="cl-metric-name">${esc(w(...name))}</span>${tag}<span class="cl-hint">${tip}</span></th><td class="cl-formula">${esc(formula)}</td><td class="cl-primary"></td><td class="cl-secondary"></td></tr>`;
    }).join("");
    return `<section class="cl-card" id="cl-confusion-card" aria-labelledby="cl-confusion-title">
        <div class="cl-card-heading"><h2 id="cl-confusion-title">03 · ${w("Confusion matrix and metrics", "Матрица ошибок и метрики")}</h2></div>
        <div class="cl-confusion-layout">
          <div class="cl-table-scroll cl-matrix-block"><table id="cl-confusion" class="cl-confusion"><caption id="cl-confusion-caption"></caption>
            <thead><tr><td></td><th scope="col">${w("Predicted survived", "Предсказано: выжил")}</th><th scope="col">${w("Predicted died", "Предсказано: погиб")}</th><th scope="col">${w("Total", "Всего")}</th></tr></thead>
            <tbody><tr><th scope="row">${w("Actually survived", "На самом деле: выжил")}</th>${cell("tp")}${cell("fn")}<td id="cl-actual-positive" class="cl-total"></td></tr>
            <tr><th scope="row">${w("Actually died", "На самом деле: погиб")}</th>${cell("fp")}${cell("tn")}<td id="cl-actual-negative" class="cl-total"></td></tr></tbody>
            <tfoot><tr><th scope="row">${w("Total", "Всего")}</th><td id="cl-predicted-positive" class="cl-total"></td><td id="cl-predicted-negative" class="cl-total"></td><td id="cl-n" class="cl-total"></td></tr></tfoot>
          </table></div>
          <div class="cl-table-scroll cl-metrics-block"><table id="cl-metrics" class="cl-metrics">
            <thead><tr><th scope="col">${w("Metric", "Метрика")}</th><th scope="col">${w("Formula", "Формула")}</th><th scope="col" id="cl-metrics-primary" class="cl-value-heading"></th><th scope="col" id="cl-metrics-secondary" class="cl-value-heading"></th></tr></thead>
            <tbody>${rows}</tbody>
          </table></div>
        </div>
      </section>`;
  }
  // Card 04: the ROC plot beside its explanation, the AUC badge and the legend.
  function rocMarkup(w) {
    return `<section class="cl-card" id="cl-roc-card" aria-labelledby="cl-roc-title">
        <div class="cl-card-heading"><h2 id="cl-roc-title">04 · ${w("ROC curve", "ROC-кривая")}</h2></div>
        <div class="cl-roc-layout">
          <svg id="cl-roc" viewBox="0 0 480 480" role="img" aria-label="${w("ROC curves of the selected split (solid) and the other split (dashed), with a dot at the current threshold.", "ROC-кривые выбранной выборки (сплошная) и другой выборки (пунктир); точка — текущий порог.")}"></svg>
          <div class="cl-roc-text"><span id="cl-auc" class="cl-badge"></span>
            <p>${w("Every threshold gives one (FPR, TPR) pair; sweeping it traces the curve. AUC is the area under it: the probability that a random survivor scores above a random non-survivor.", "Каждый порог даёт одну пару (FPR, TPR); если пройти порогом по всем значениям, получится кривая. AUC — площадь под ней: вероятность того, что случайный выживший получит оценку выше случайного погибшего.")}</p>
            <div class="cl-legend cl-roc-legend"><span><i class="cl-swatch"></i><span id="cl-roc-primary"></span></span><span><i class="cl-swatch cl-secondary"></i><span id="cl-roc-secondary"></span></span><span><i class="cl-dot"></i>${w("Current threshold", "Текущий порог")}</span></div>
          </div>
        </div>
      </section>`;
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
      ${confusionMarkup(w)}
      ${rocMarkup(w)}
    </section>`;
  }

  function mount(root, language) {
    const w = (en, ru) => language === "en" ? en : ru;
    const $ = (selector) => root.querySelector(selector);
    const controller = new AbortController();
    const listen = (type, handler, target = root) => target.addEventListener(type, handler, { signal: controller.signal });
    const plot = $("#cl-plot"), rocSvg = $("#cl-roc");
    // The metric table's value cells, updated in place on every threshold move.
    const metricCells = METRICS.map(({ key }) => {
      const row = $(`[data-cl-metric="${key}"]`);
      return { key, primary: row.querySelector(".cl-primary"), secondary: row.querySelector(".cl-secondary") };
    });
    const splitName = (split) => split === "train" ? w("Train", "Обучение") : w("Test", "Тест");
    // Two decimals; the Russian text uses a decimal comma.
    const number = (value) => {
      const text = (Math.abs(value) < 0.005 ? 0 : value).toFixed(2);
      return language === "en" ? text : text.replace(".", ",");
    };
    let columns = [], model, logits, probabilities, logitRange, fixed, classes = null, frame = 0, thresholdFrame = 0, drag = null;
    // --cl-plot-font is a size on screen, but the plots draw in viewBox units: divide it by each SVG's scale,
    // capped at the 16 units their fixed margins leave room for. Measured on presentation changes and resizes.
    const fonts = new Map();
    function measureFonts() {
      let changed = false;
      for (const svg of [plot, rocSvg]) {
        const { width, height } = svg.getBoundingClientRect(), box = svg.viewBox.baseVal;
        const px = parseFloat(getComputedStyle(svg).getPropertyValue("--cl-plot-font")) || 16;
        const scale = Math.min(width / box.width, height / box.height);
        const font = Math.min(16, scale > 0 ? px / scale : 16);
        if (font !== fonts.get(svg)) { fonts.set(svg, font); changed = true; }
      }
      return changed;
    }
    function refit() {
      columns = activeColumns();
      model = M.fit(D.train.X, D.train.y, { penalty: state.penalty, lambda: state.lambda, columns });
      logits = {
        train: M.scores(D.train.X, model.weights, model.intercept),
        test: M.scores(D.test.X, model.weights, model.intercept),
      };
      probabilities = { train: M.probabilities(logits.train), test: M.probabilities(logits.test) };
      // What does not depend on the threshold, per split: the ROC curve with its AUC, log loss and Brier score.
      fixed = Object.fromEntries(SPLITS.map((split) => [split, {
        roc: M.roc(logits[split], D[split].y),
        logLoss: M.logLoss(probabilities[split], D[split].y),
        brier: M.brier(probabilities[split], D[split].y),
      }]));
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
    // The selected split's scores on the displayed axis by class, with their histograms and KDE curves;
    // kept until the next refit, axis or split change, so a threshold move only redraws.
    function classScores() {
      if (!classes) {
        const values = state.axis === "logit" ? logits[state.split] : probabilities[state.split];
        const y = D[state.split].y;
        const positives = values.filter((_, i) => y[i] === 1), negatives = values.filter((_, i) => y[i] === 0);
        classes = { positives, negatives, layers: P.distributionLayers({ axis: state.axis, range: range(), bins: 40, positives, negatives }) };
      }
      return classes;
    }
    function distribution() {
      const [lo, hi] = range(), logit = state.axis === "logit";
      const focused = document.activeElement?.id === "cl-threshold-handle";
      plot.innerHTML = P.distributionPlot({
        axis: state.axis, range: [lo, hi], bins: 40, ...classScores(),
        threshold: Math.min(hi, Math.max(lo, shown())),
        fontSize: fonts.get(plot),
        labels: { positive: w("Survived", "Выжили"), negative: w("Died", "Погибли"), x: logit ? w("logit", "логит") : w("probability", "вероятность"), y: w("passengers", "пассажиры"), empty: w("No data", "Нет данных") },
      });
      // The new markup has a new handle: keep keyboard focus on it.
      if (focused) $("#cl-threshold-handle").focus({ preventScroll: true });
    }
    function readout() {
      const p = probabilities[state.split];
      const predicted = p.reduce((count, value) => count + (value >= state.threshold ? 1 : 0), 0);
      const t = number(state.threshold), z = number(M.logit(state.threshold));
      $("#cl-readout").textContent = w(`p ≥ ${t} (logit ≥ ${z}): ${predicted} predicted survivors of ${p.length}`, `p ≥ ${t} (логит ≥ ${z}): ${predicted} предсказанных выживших из ${p.length}`);
    }
    // Confusion counts and every metric for both splits at the current threshold: O(n) per split.
    function outcomes() {
      const cut = M.logit(state.threshold);
      return Object.fromEntries(SPLITS.map((split) => {
        const counts = M.confusion(logits[split], D[split].y, cut), { roc, logLoss, brier } = fixed[split];
        return [split, { counts, values: { ...M.metrics(counts), auc: roc.auc, logLoss, brier } }];
      }));
    }
    function confusionCard({ tp, fn, fp, tn }) {
      const cells = { "#cl-tp": tp, "#cl-fn": fn, "#cl-fp": fp, "#cl-tn": tn, "#cl-actual-positive": tp + fn, "#cl-actual-negative": fp + tn,
        "#cl-predicted-positive": tp + fp, "#cl-predicted-negative": fn + tn, "#cl-n": `n = ${tp + fn + fp + tn}` };
      for (const [id, value] of Object.entries(cells)) $(id).textContent = value;
      const t = number(state.threshold);
      $("#cl-confusion-caption").textContent = state.split === "train"
        ? w(`Train split · threshold p ≥ ${t}`, `Обучающая выборка · порог p ≥ ${t}`)
        : w(`Test split · threshold p ≥ ${t}`, `Тестовая выборка · порог p ≥ ${t}`);
    }
    // The primary column is always the selected split.
    function metricsCard(outcome) {
      const primary = outcome[state.split].values, secondary = outcome[other(state.split)].values;
      $("#cl-metrics-primary").textContent = splitName(state.split);
      $("#cl-metrics-secondary").textContent = splitName(other(state.split));
      for (const cell of metricCells) {
        cell.primary.textContent = metricText(primary[cell.key]);
        cell.secondary.textContent = metricText(secondary[cell.key]);
      }
    }
    function rocCard(counts) {
      rocSvg.innerHTML = P.rocPlot({
        primary: fixed[state.split].roc.points, secondary: fixed[other(state.split)].roc.points, operating: operatingPoint(counts),
        fontSize: fonts.get(rocSvg),
        labels: { x: w("False positive rate", "Доля ложных срабатываний"), y: w("True positive rate", "Доля верных срабатываний") },
      });
      $("#cl-roc-primary").textContent = splitName(state.split);
      $("#cl-roc-secondary").textContent = splitName(other(state.split));
      const auc = (split) => fixed[split].roc.auc.toFixed(2);
      $("#cl-auc").textContent = w(`AUC train ${auc("train")} · test ${auc("test")}`, `AUC обучение ${auc("train")} · тест ${auc("test")}`);
    }
    // Everything that follows the threshold, without refitting.
    function renderThreshold() {
      if (thresholdFrame) cancelAnimationFrame(thresholdFrame);
      thresholdFrame = 0;
      const value = show(snap(shown()));
      $("#cl-threshold-range").value = value;
      $("#cl-threshold").value = value;
      distribution();
      readout();
      const outcome = outcomes();
      confusionCard(outcome[state.split].counts);
      metricsCard(outcome);
      rocCard(outcome[state.split].counts);
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
      root.style.setProperty("--cl-plot-font", `${(state.presentation ? 26 : 20) * state.textScale}px`);
      $("#cl-presentation").checked = state.presentation;
      $("#cl-text-size").value = String(state.textScale);
      measureFonts();
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
        presentation();
        renderThreshold(); return;
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
    // A new plot width changes the font in viewBox units: redraw the plots.
    const resized = new ResizeObserver(() => { if (measureFonts()) scheduleThreshold(); });
    resized.observe(plot);
    resized.observe(rocSvg);
    teardown = () => {
      controller.abort();
      resized.disconnect();
      if (frame) cancelAnimationFrame(frame);
      if (thresholdFrame) cancelAnimationFrame(thresholdFrame);
      document.body.classList.remove("lr-presenting");
    };
  }
  window.MLClassificationLab = { html, mount, unmount: () => teardown() };
})();
