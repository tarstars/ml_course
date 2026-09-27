"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const e = escapeHTML;
const KIND = {
  ebook: "Книга",
  pdf: "PDF",
  notebook: "Ноутбук",
  image: "Изображение",
  video: "Запись",
  link: "Ссылка",
  code: "Код",
  archive: "Архив",
  slides: "Презентация",
  data: "Данные",
  document: "Документ",
};
const SYMBOL = {
  ebook: "EPUB",
  pdf: "PDF",
  notebook: "{ }",
  image: "▧",
  video: "▷",
  link: "↗",
  code: "</>",
  archive: "ZIP",
  slides: "PPT",
  data: "≡",
  document: "¶",
};
let rawCourse, course, byClass, byResource, byModule;
let language = window.ML_GET_LANGUAGE();
const t = (text) => (language === "en" ? window.ML_UI_EN[text] || text : text);
const tagLabel = (tag) =>
  language === "en" ? course.topic_translations.en[tag] || tag : tag;
function localizeCatalog() {
  course = structuredClone(rawCourse);
  for (const field of ["classes", "modules", "resources"]) {
    for (const record of course[field])
      if (language === "en") Object.assign(record, record.translations.en);
  }
  byClass = Object.fromEntries(course.classes.map((c) => [c.id, c]));
  byResource = Object.fromEntries(course.resources.map((r) => [r.id, r]));
  byModule = Object.fromEntries(course.modules.map((m) => [m.id, m]));
  document.documentElement.lang = language;
  document
    .querySelectorAll("[data-lang]")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.lang === language),
      ),
    );
  $("#module-nav").innerHTML = course.modules
    .map(
      (m, i) =>
        `<a href="${route("timeline", { module: m.id })}" data-module="${m.id}"><span class="module-number">0${i + 1}</span>${e(m.short)}</a>`,
    )
    .join("");
}
let state = {
  view: "timeline",
  id: "",
  q: "",
  module: "",
  period: "",
  tag: "",
  kind: "",
  sort: "",
};
let searchTimer;
let previewTrigger;
const normalize = (value) =>
  String(value).toLocaleLowerCase("ru").replace(/ё/g, "е").normalize("NFKC");
const contains = (text, query) =>
  normalize(query)
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => normalize(text).includes(word));
const plural = (n, one, few, many) =>
  `${n} ${n % 10 === 1 && n % 100 !== 11 ? one : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14) ? few : many}`;
const classCount = (n) =>
  language === "en"
    ? `${n} ${n === 1 ? "class" : "classes"}`
    : plural(n, "занятие", "занятия", "занятий");
const resourceCount = (n) =>
  language === "en"
    ? `${n} ${n === 1 ? "resource" : "resources"}`
    : plural(n, "материал", "материала", "материалов");
const size = (bytes) =>
  bytes >= 1e6
    ? `${(bytes / 1e6).toFixed(1)} ${language === "en" ? "MB" : "МБ"}`
    : `${Math.max(1, Math.round(bytes / 1e3))} ${language === "en" ? "KB" : "КБ"}`;
function dateText(c) {
  return c.date
    ? new Intl.DateTimeFormat(language === "en" ? "en-GB" : "ru", {
        day: "numeric",
        month: "long",
        year: "numeric",
        timeZone: "UTC",
      }).format(new Date(c.date))
    : c.date_label || "Дата не указана";
}
function route(view = state.view, values = {}) {
  const next = { view, ...values };
  const params = new URLSearchParams();
  params.set("lang", language);
  for (const key of ["q", "module", "period", "tag", "kind", "sort"])
    if (next[key]) params.set(key, next[key]);
  return `#${view}${next.id ? "/" + encodeURIComponent(next.id) : ""}${params.size ? "?" + params : ""}`;
}
function readRoute() {
  const [path, query] = (location.hash.slice(1) || "timeline").split("?");
  const [view, id] = path.split("/");
  const params = new URLSearchParams(query || "");
  const requestedLanguage = params.get("lang");
  if (
    ["en", "ru"].includes(requestedLanguage) &&
    requestedLanguage !== language
  ) {
    language = requestedLanguage;
    window.ML_SAVE_LANGUAGE(language);
    localizeCatalog();
  }
  state = { view, id: id || "" };
  for (const key of ["q", "module", "period", "tag", "kind", "sort"])
    state[key] = params.get(key) || "";
}
function setFilter(key, value) {
  state[key] = value;
  history.replaceState(null, "", route(state.view, state));
  render(true);
}
function tagLink(tag) {
  return `<a class="tag" href="${e(route("timeline", { tag }))}">${e(tagLabel(tag))}</a>`;
}
function moduleBadge(c) {
  return `<span class="module-badge ${e(c.module)}"><span></span>${e(byModule[c.module].short)}</span>`;
}
function heading(eyebrow, title, text, extra = "") {
  return `<section class="page-heading"><div><p class="eyebrow">${eyebrow}</p><h1>${title}</h1><p class="page-description">${text}</p></div>${extra}</section>`;
}
function hero() {
  return `<section class="hero"><div class="hero-copy"><div class="hero-kicker"><span class="tiny-line"></span> УЧЕБНЫЙ АРХИВ / 2024 — 2025</div><h1>Машинное обучение.<br><em>Шаг за шагом.</em></h1><p>Лекции, практика и заметки — в одном месте.<br>Выберите занятие или найдите нужную тему.</p><div class="hero-stats"><span><strong>${course.classes.length}</strong> занятий и блоков</span><span><strong>${course.modules.length}</strong> раздела</span><span><strong>${course.resources.length}</strong> ${resourceCount(course.resources.length).replace(/^\d+ /, "")}</span></div></div><div class="hero-art" aria-hidden="true"><div class="art-label">ОТ ДАННЫХ К МОДЕЛИ <span>FIG. 01</span></div><svg viewBox="0 0 300 190" fill="none"><g stroke="#648873" stroke-width="1" opacity=".45"><path d="M38 38L134 53M38 38L134 96M38 38L134 139M38 77L134 53M38 77L134 96M38 77L134 139M38 116L134 53M38 116L134 96M38 116L134 139M38 155L134 53M38 155L134 96M38 155L134 139M134 53L240 77M134 53L240 116M134 96L240 77M134 96L240 116M134 139L240 77M134 139L240 116"/></g><g stroke="#86a991" stroke-width="1.5" fill="#234b3b"><circle cx="38" cy="38" r="7"/><circle cx="38" cy="77" r="7"/><circle cx="38" cy="116" r="7"/><circle cx="38" cy="155" r="7"/><circle cx="134" cy="53" r="11"/><circle cx="134" cy="96" r="11"/><circle cx="134" cy="139" r="11"/><circle cx="240" cy="77" r="8"/><circle cx="240" cy="116" r="8"/></g><path d="M38 77L134 96L240 77" stroke="#d8b37a" stroke-width="1.5"/><circle cx="134" cy="96" r="5" fill="#d8b37a"/><circle cx="240" cy="77" r="3" fill="#d8b37a"/><g fill="#a9c0af" font-family="monospace" font-size="10"><text x="18" y="181">input</text><text x="114" y="181">learn</text><text x="223" y="181">predict</text></g></svg><div class="art-formula">ŷ = f(x; θ)<span>учиться → понимать → применять</span></div></div></section>`;
}
function searchBar(placeholder, withKinds = false) {
  const options = course.modules
    .map(
      (m) =>
        `<option value="${e(m.id)}" ${state.module === m.id ? "selected" : ""}>${e(m.short)}</option>`,
    )
    .join("");
  const periods = `<option value="">Оба семестра</option><option value="autumn" ${state.period === "autumn" ? "selected" : ""}>Осень 2024</option><option value="spring" ${state.period === "spring" ? "selected" : ""}>Весна 2025</option>`;
  let active = "";
  if (state.tag)
    active += `<button class="active-filter" data-filter="tag" data-value="" aria-label="Убрать фильтр по теме ${e(state.tag)}">${e(tagLabel(state.tag))} ×</button>`;
  const kinds = withKinds
    ? `<div class="kind-filters" role="group" aria-label="Формат материалов"><button data-filter="kind" data-value="" class="filter-pill ${!state.kind ? "selected" : ""}" aria-pressed="${!state.kind}">Все форматы</button>${Object.entries(
        KIND,
      )
        .map(
          ([key, label]) =>
            `<button class="filter-pill ${state.kind === key ? "selected" : ""}" data-filter="kind" data-value="${key}" aria-pressed="${state.kind === key}">${label}</button>`,
        )
        .join("")}</div>`
    : "";
  return `<div class="filter-area"><div class="filter-row"><label class="search-box"><span aria-hidden="true">⌕</span><input id="search" type="search" autocomplete="off" placeholder="${placeholder}" aria-label="Поиск по курсу" value="${e(state.q)}"><kbd>/</kbd></label><label class="sr-only" for="module-select">Раздел курса</label><select id="module-select" data-select="module"><option value="">Все разделы</option>${options}</select>${!withKinds ? `<label class="sr-only" for="period-select">Семестр</label><select id="period-select" data-select="period">${periods}</select>` : ""}</div>${kinds}${active ? `<div class="active-filters">${active}<a href="${route(state.view)}" class="text-link">Сбросить фильтры</a></div>` : ""}</div>`;
}
function matchingClasses() {
  let list = course.classes.filter(
    (c) =>
      (!state.module || c.module === state.module) &&
      (!state.period || c.period === state.period) &&
      (!state.tag || c.tags.includes(state.tag)) &&
      contains(
        [
          c.title,
          c.summary,
          ...c.topics,
          ...c.tags,
          ...c.tags.map(tagLabel),
          ...c.practice,
          c.notes_text,
          ...c.resource_ids.map(
            (id) => `${byResource[id].title} ${byResource[id].filename || ""}`,
          ),
        ].join(" "),
        state.q,
      ),
  );
  if (state.sort === "desc") list = [...list].reverse();
  return list;
}
function empty(message) {
  return `<div class="empty-state"><span>⌕</span><h2>Ничего не найдено</h2><p>${message}</p><a class="button" href="${route(state.view)}">Сбросить фильтры</a></div>`;
}
function classCard(c) {
  const resources = c.resource_ids.map((id) => byResource[id]);
  const types = [...new Set(resources.map((r) => r.kind))];
  const index = String(course.classes.indexOf(c) + 1).padStart(2, "0");
  return `<article class="timeline-item"><div class="date-column"><span class="timeline-dot"></span>${c.date ? `<span class="date-day">${Number(c.date.slice(8))}</span><span>${new Intl.DateTimeFormat(language === "en" ? "en-GB" : "ru", { month: "short", timeZone: "UTC" }).format(new Date(c.date))}</span><span class="date-year">${c.date.slice(0, 4)}</span>` : `<span class="date-unknown">${e(c.date_label || "Без даты")}</span>`}</div><div class="class-card"><div class="card-meta">${moduleBadge(c)}<span>${e(c.kind)} <span class="meta-divider">/</span> ${index}</span></div><h3><a href="${route("class", { id: c.id })}">${e(c.title)} <span class="card-arrow">↗</span></a></h3><p>${e(c.summary)}</p><div class="tags">${c.tags.slice(0, 4).map(tagLink).join("")}</div><div class="card-bottom"><div class="resource-summary">${types
    .slice(0, 4)
    .map(
      (t) =>
        `<span title="${KIND[t]}"><b class="mini-icon ${t}">${e(SYMBOL[t])}</b>${KIND[t]}</span>`,
    )
    .join(
      "",
    )}</div><a href="${route("class", { id: c.id })}" class="material-count">${resourceCount(resources.length)} <span>→</span></a></div></div></article>`;
}
function timeline() {
  const classes = matchingClasses();
  let lastGroup = "";
  const content = classes
    .map((c) => {
      const group =
        c.period === "autumn"
          ? "Осенний семестр · 2024"
          : "Весенний семестр · 2025";
      const divider =
        group !== lastGroup
          ? `<h2 class="timeline-group"><span>${group}</span><span class="group-line"></span></h2>`
          : "";
      lastGroup = group;
      return divider + classCard(c);
    })
    .join("");
  return `${hero()}<section class="content-section"><div class="section-heading"><div><p class="eyebrow">ПРОГРАММА КУРСА</p><h2>Хронология занятий</h2></div><span class="quiet">Все материалы под рукой</span></div>${searchBar("Найти тему, занятие или материал…")}<div class="results-toolbar"><span>${classCount(classes.length)}${classes.length !== course.classes.length ? ` ${language === "en" ? "of" : "из"} ${course.classes.length}` : ""}</span><label>Порядок <select id="sort-select" data-select="sort"><option value="">С начала курса</option><option value="desc" ${state.sort === "desc" ? "selected" : ""}>С конца курса</option></select></label></div><div class="timeline">${classes.length ? content : empty("Попробуйте другой запрос или выберите другой раздел.")}</div><p class="date-footnote">Даты взяты из программы и конспектов. Если дата занятия неизвестна, это отмечено в карточке.</p></section>`;
}
function resourceCard(r) {
  const previewable = ["image", "pdf"].includes(r.kind) && !r.external;
  const open = previewable
    ? `<button class="resource-title" data-preview="${r.id}">${e(r.title)}</button>`
    : `<a class="resource-title" href="${e(r.url + (!r.external && ["notebook", "document"].includes(r.kind) ? "?lang=" + language : ""))}" ${r.external ? 'target="_blank" rel="noopener noreferrer"' : r.kind === "ebook" ? "download" : ""}>${e(r.title)}</a>`;
  const metadata = [
    r.format || t(KIND[r.kind]),
    r.pages ? `${r.pages} ${language === "en" ? "pp." : "стр."}` : "",
    r.bytes ? size(r.bytes) : r.external ? new URL(r.url).hostname : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const image = r.thumbnail
    ? `<button class="resource-thumbnail" data-preview="${r.id}" aria-label="Просмотреть: ${e(r.title)}"><img src="${e(r.thumbnail)}" alt="" loading="lazy"></button>`
    : "";
  return `<article class="resource-card ${r.thumbnail ? "has-thumbnail" : ""}">${image}<div class="resource-content"><div class="resource-type"><span class="resource-icon ${r.kind}">${e(SYMBOL[r.kind])}</span><span>${e(metadata)}</span>${r.external ? '<span class="external-arrow" aria-hidden="true">↗</span>' : ""}</div>${open}${r.description ? `<p class="resource-description">${e(r.description)}</p>` : ""}<div class="resource-footer">${r.class_ids.length ? `<a href="${route("class", { id: r.class_ids[0] })}" class="related-class">${e(byClass[r.class_ids[0]].title)}</a>` : '<span class="related-class">Общие материалы</span>'}${r.download_url ? `<a class="download-link" href="${e(r.download_url)}" download title="Скачать оригинал" aria-label="Скачать: ${e(r.title)}">↓</a>` : ""}</div></div></article>`;
}
function library() {
  const resources = course.resources.filter(
    (r) =>
      (!state.kind || r.kind === state.kind) &&
      (!state.module ||
        r.class_ids.some((id) => byClass[id].module === state.module)) &&
      contains(
        [
          r.title,
          r.description,
          r.filename || "",
          KIND[r.kind],
          ...r.class_ids.flatMap((id) => [
            byClass[id].title,
            ...byClass[id].tags,
            ...byClass[id].tags.map(tagLabel),
          ]),
        ].join(" "),
        state.q,
      ),
  );
  return `${heading("КОНСПЕКТЫ / КОД / ЗАПИСИ", "Библиотека материалов", "Все файлы и ссылки курса. Читайте конспекты, открывайте ноутбуки и возвращайтесь к записям.")}<section>${searchBar("Поиск по материалам и именам файлов…", true)}<div class="results-toolbar"><span>${resourceCount(resources.length)}</span><span>Оригиналы доступны для скачивания</span></div><div class="resource-grid">${resources.length ? resources.map(resourceCard).join("") : empty("Попробуйте другой запрос или формат материала.")}</div></section>`;
}
function topics() {
  const tags = [...new Set(course.classes.flatMap((c) => c.tags))].sort(
    (a, b) => tagLabel(a).localeCompare(tagLabel(b), language),
  );
  const filtered = tags.filter(
    (tag) =>
      contains(`${tag} ${tagLabel(tag)}`, state.q) &&
      course.classes.some(
        (c) =>
          c.tags.includes(tag) &&
          (!state.module || c.module === state.module) &&
          (!state.period || c.period === state.period),
      ),
  );
  const cards = filtered
    .map((tag) => {
      const classes = course.classes.filter(
        (c) =>
          c.tags.includes(tag) &&
          (!state.module || c.module === state.module) &&
          (!state.period || c.period === state.period),
      );
      return `<article class="topic-card"><div class="topic-top"><span class="topic-symbol">#</span><span class="quiet">${classCount(classes.length)}</span></div><h2><a href="${e(route("timeline", { tag, module: state.module, period: state.period }))}">${e(tagLabel(tag))} <span>↗</span></a></h2><ul>${classes.map((c) => `<li><a href="${route("class", { id: c.id })}">${e(c.title)}</a><span>${c.date ? c.date.slice(5).split("-").reverse().join(".") : "—"}</span></li>`).join("")}</ul></article>`;
    })
    .join("");
  return `${heading("КАРТА ЗНАНИЙ", "Темы курса", "Одна тема может встречаться в нескольких занятиях. Здесь видны все связи.")}<section>${searchBar("Найти тему: Байес, PyTorch, RNN…")}<div class="results-toolbar"><span>${language === "en" ? `${filtered.length} topics` : plural(filtered.length, "тема", "темы", "тем")}</span></div><div class="topic-grid">${cards || empty("Попробуйте другую тему или раздел.")}</div></section>`;
}
function classDetail() {
  const c = byClass[state.id];
  if (!c) return notFound();
  const index = course.classes.indexOf(c);
  const previous = course.classes[index - 1],
    next = course.classes[index + 1];
  const resources = c.resource_ids.map((id) => byResource[id]);
  return `<a class="back-link" href="#timeline">← К хронологии</a><section class="detail-heading"><div class="detail-meta">${moduleBadge(c)}<span>${e(dateText(c))}</span><span>${e(c.kind)}</span></div><h1>${e(c.title)}</h1><p class="page-description">${e(c.summary)}</p><div class="tags">${c.tags.map(tagLink).join("")}</div></section><div class="detail-layout"><div class="detail-main"><section class="detail-panel"><p class="eyebrow">ПЛАН ЗАНЯТИЯ</p><h2>Что разбираем</h2>${c.topics.length ? `<ol class="topic-list">${c.topics.map((t, i) => `<li><span>${String(i + 1).padStart(2, "0")}</span><p>${e(t)}</p></li>`).join("")}</ol>` : '<p class="quiet">Подробная программа не записана в исходных материалах.</p>'}${c.practice.length ? `<div class="practice"><h3>Практика и вопросы</h3><ul>${c.practice.map((p) => `<li>${e(p)}</li>`).join("")}</ul></div>` : ""}</section></div><aside class="detail-aside"><div class="lesson-summary"><p class="eyebrow">О ЗАНЯТИИ</p><dl><div><dt>Раздел</dt><dd>${e(byModule[c.module].short)}</dd></div><div><dt>Семестр</dt><dd>${c.period === "autumn" ? "Осень 2024" : "Весна 2025"}</dd></div><div><dt>В архиве</dt><dd>${resourceCount(resources.length)}</dd></div></dl><a href="#class-materials" class="button" data-scroll-materials>К материалам ↓</a></div>${c.note ? `<div class="source-note"><span>ⓘ</span><h3>Об этой записи</h3><p>${e(c.note)}</p></div>` : ""}</aside></div><section id="class-materials" class="class-materials"><div class="section-heading"><h2>Материалы занятия</h2><span class="quiet">${resourceCount(resources.length)}</span></div>${resources.length ? `<div class="resource-grid">${resources.map(resourceCard).join("")}</div>` : '<div class="source-note"><p>Отдельных файлов нет. Программа сохранена в исходных заметках ниже.</p></div>'}</section>${c.notes_html ? `<details class="original-notes"><summary><span>Из исходных заметок</span><span>Читать программу и вопросы ＋</span></summary><div class="prose">${c.notes_html}</div><a class="text-link" href="readers/source-notes.html">Открыть весь исходный документ ↗</a></details>` : ""}<nav class="lesson-navigation" aria-label="Соседние занятия">${previous ? `<a href="${route("class", { id: previous.id })}"><span>← ПРЕДЫДУЩАЯ ЗАПИСЬ</span><strong>${e(previous.title)}</strong></a>` : "<span></span>"}${next ? `<a href="${route("class", { id: next.id })}"><span>СЛЕДУЮЩАЯ ЗАПИСЬ →</span><strong>${e(next.title)}</strong></a>` : ""}</nav>`;
}
function notFound() {
  return `${heading("404", "Запись не найдена", "Возможно, ссылка устарела или в ней есть опечатка.")}<a class="button" href="#timeline">Открыть хронологию</a>`;
}
function render(preserveFocus = false) {
  if (!course) return;
  const activeId = document.activeElement.id;
  const selection =
    document.activeElement instanceof HTMLInputElement
      ? document.activeElement.selectionStart
      : null;
  const title =
    {
      timeline: "Хронология",
      topics: "Темы курса",
      library: "Библиотека",
      class: byClass[state.id]?.title || "Занятие",
    }[state.view] || "Не найдено";
  $("#app").innerHTML = (
    { timeline, topics, library, class: classDetail }[state.view] || notFound
  )();
  $("#breadcrumb-current").textContent = title;
  document.title = `${t(title)} · Machine Learning`;
  document.querySelectorAll("[data-nav]").forEach((link) => {
    const current =
      link.dataset.nav === (state.view === "class" ? "timeline" : state.view);
    link.classList.toggle("active", current);
    if (current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document
    .querySelectorAll("[data-module]")
    .forEach((link) =>
      link.classList.toggle("active", link.dataset.module === state.module),
    );
  if (preserveFocus && activeId) {
    const el = document.getElementById(activeId);
    el?.focus({ preventScroll: true });
    if (el instanceof HTMLInputElement && selection !== null)
      el.setSelectionRange(selection, selection);
  }
  window.ML_TRANSLATE(document.body, language);
  $("#announcement").textContent =
    $(".results-toolbar")?.textContent || t(title);
}
function openPreview(id, trigger) {
  const r = byResource[id];
  if (!r) return;
  previewTrigger = trigger;
  $("#preview-title").textContent = r.title;
  $("#preview-format").textContent = [t(KIND[r.kind]), r.bytes && size(r.bytes)]
    .filter(Boolean)
    .join(" / ");
  $("#preview-open").href = r.url;
  $("#preview-download").href = r.download_url;
  $("#preview-body").innerHTML =
    r.kind === "pdf"
      ? `<iframe title="${e(r.title)}" src="${e(r.url)}#view=FitH"></iframe><p class="pdf-fallback">Если PDF не отображается, <a href="${e(r.url)}" target="_blank" rel="noopener noreferrer">откройте его в новой вкладке</a>.</p>`
      : `<img src="${e(r.url)}" alt="${e(r.title)}">`;
  window.ML_TRANSLATE($("#preview-dialog"), language);
  $("#preview-dialog").showModal();
  document.body.classList.add("modal-open");
}
document.addEventListener("input", (event) => {
  if (event.target.id !== "search") return;
  clearTimeout(searchTimer);
  const value = event.target.value;
  searchTimer = setTimeout(() => setFilter("q", value), 150);
});
document.addEventListener("change", (event) => {
  if (event.target.dataset.select)
    setFilter(event.target.dataset.select, event.target.value);
});
document.addEventListener("click", (event) => {
  const languageButton = event.target.closest("[data-lang]");
  if (languageButton) {
    clearTimeout(searchTimer);
    language = languageButton.dataset.lang;
    window.ML_SAVE_LANGUAGE(language);
    localizeCatalog();
    history.replaceState(null, "", route(state.view, state));
    render();
  }
  const filter = event.target.closest("[data-filter]");
  if (filter) setFilter(filter.dataset.filter, filter.dataset.value);
  const preview = event.target.closest("[data-preview]");
  if (preview) openPreview(preview.dataset.preview, preview);
  if (event.target.closest("[data-close-preview]"))
    $("#preview-dialog").close();
  if (event.target.closest("[data-scroll-materials]")) {
    event.preventDefault();
    $("#class-materials").scrollIntoView({
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
    });
  }
  if (event.target.closest(".skip-link")) {
    event.preventDefault();
    $("#main").focus();
  }
  if (event.target.closest(".mobile-menu")) {
    const open = !document.body.classList.contains("menu-open");
    document.body.classList.toggle("menu-open", open);
    $(".mobile-menu").setAttribute("aria-expanded", String(open));
  }
  if (event.target.closest("#sidebar a")) {
    document.body.classList.remove("menu-open");
    $(".mobile-menu").setAttribute("aria-expanded", "false");
  }
});
$("#preview-dialog").addEventListener("click", (event) => {
  if (event.target === $("#preview-dialog")) $("#preview-dialog").close();
});
$("#preview-dialog").addEventListener("close", () => {
  $("#preview-body").replaceChildren();
  document.body.classList.remove("modal-open");
  previewTrigger?.focus();
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "/" &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.altKey &&
    !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) &&
    !$("#preview-dialog").open
  ) {
    const input = $("#search");
    if (input) {
      event.preventDefault();
      input.focus();
    }
  }
  if (event.key === "Escape") {
    document.body.classList.remove("menu-open");
    $(".mobile-menu").setAttribute("aria-expanded", "false");
  }
});
window.addEventListener("hashchange", () => {
  clearTimeout(searchTimer);
  readRoute();
  render();
  window.scrollTo(0, 0);
  $("#main").focus({ preventScroll: true });
});
async function start() {
  try {
    const response = await fetch("data/course.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    rawCourse = await response.json();
    localizeCatalog();
    byClass = Object.fromEntries(course.classes.map((c) => [c.id, c]));
    byResource = Object.fromEntries(course.resources.map((r) => [r.id, r]));
    byModule = Object.fromEntries(course.modules.map((m) => [m.id, m]));
    $("#class-count").textContent = course.classes.length;
    $("#resource-count").textContent = course.resources.length;
    $("#module-nav").innerHTML = course.modules
      .map(
        (m, i) =>
          `<a href="${route("timeline", { module: m.id })}" data-module="${m.id}"><span class="module-number">0${i + 1}</span>${e(m.short)}</a>`,
      )
      .join("");
    readRoute();
    render();
  } catch (error) {
    $("#app").innerHTML =
      '<div class="empty-state"><h1>Не удалось открыть архив</h1><p>Обновите страницу или откройте исходные заметки.</p><a class="button" href="readers/source-notes.html">Исходные заметки</a></div>';
    window.ML_TRANSLATE(document.body, language);
    console.error(error);
  }
}
start();
