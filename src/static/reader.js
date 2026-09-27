(() => {
  const requested = new URLSearchParams(location.search).get("lang");
  let language = ["ru", "en"].includes(requested)
    ? requested
    : window.ML_GET_LANGUAGE();
  function render() {
    document.documentElement.lang = language;
    window.ML_TRANSLATE(document.body, language);
    const title =
      document.body.dataset[language === "en" ? "titleEn" : "titleRu"];
    document.getElementById("reader-title").textContent = title;
    document.title = `${title} · Machine Learning`;
    document
      .querySelectorAll("[data-lang]")
      .forEach((button) =>
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.lang === language),
        ),
      );
    document.querySelector(".reader-header>a").href =
      `../#library?lang=${language}`;
  }
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-lang]");
    if (!button) return;
    language = button.dataset.lang;
    window.ML_SAVE_LANGUAGE(language);
    history.replaceState(null, "", `?lang=${language}`);
    render();
  });
  render();
})();
