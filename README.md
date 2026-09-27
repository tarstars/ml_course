# Machine Learning course archive

A bilingual Russian / English website for the 2024–2025 course, published at
**https://tarstars.github.io/ml_course/**.

- **Timeline:** dated classes and clearly labeled undated study blocks.
- **Class pages:** topics, practice questions, resources, and excerpts from the original syllabus.
- **Topic index:** connections between topics and classes.
- **Library:** searchable PDFs, photos, notebooks, recordings, links, archives, datasets, and downloadable EPUB books.
- **Readers:** PDF/image previews, browser-compatible HEIC copies, notebook code and saved outputs, and locally rendered mathematical formulas. Notebooks are never executed by the build or the site.
- **Languages:** translated navigation and catalog content, with a persistent RU/EN switch and shareable language-specific links. Original attached documents retain their original language.

## Local use

Install [uv](https://docs.astral.sh/uv/), then run from the project root:

```bash
uv sync --frozen
uv run python src/build.py
uv run python -m http.server 8000 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:8000. A web server is needed; opening `index.html` as a local file will not load the catalog.

## Project structure

```text
src/
  build.py                 # Static build, validation, previews, notebook rendering
  check_site.py            # Browser and generated-link smoke checks
  index.html               # Site shell and accessible navigation
  content/
    course.json            # Russian catalog and relationships
    en.json                # English catalog translations, keyed by stable IDs
  static/
    app.js                 # Routes, filtering, search, previews
    i18n.js                # UI translations and language preferences
    reader.js              # Language switching in document readers
    style.css              # Responsive layouts
    vendor/mathjax/        # Pinned MathJax 3.2.2 and its license
.github/workflows/pages.yml # Build and Pages deployment on main
pyproject.toml / uv.lock    # Reproducible Python environment
dist/                      # Generated output; ignored by version control
```

The original course files remain in the project root, with their names preserved. The build copies only cataloged files into `dist/materials/`; it does not publish the source tree or environment. Archive members are also published individually. `notes.md` remains the original record.

## Editing the catalog

The catalog separates **modules**, **classes**, and **resources**. IDs are stable URL identifiers; a resource can be related to several classes through `class_ids`.

| Entity | Main fields |
| --- | --- |
| Module | `id`, `title`, `short`, `description` |
| Class | `id`, `date`, `period`, `module`, `title`, `summary`, `topics`, `tags`, `practice`, `note`, `source_ranges` |
| Resource | `id`, `title`, `description`, `class_ids`, and one source: `path`, `url`, or `archive` + `member` |

1. Put a new file in the project, then add its resource entry to `src/content/course.json`.
2. Add or update the relevant class, referring to its module ID. Use ISO `YYYY-MM-DD` dates, or `null` when unknown. `period` is `autumn` or `spring`; catalog order defines the timeline and the placement of undated blocks.
3. Add English display fields under the same ID in `src/content/en.json`. Topic translations use the Russian tag as the key.
4. Build locally and push to `main` to publish.

Example local resource:

```json
{
  "id": "new-class-slides",
  "title": "Слайды нового занятия",
  "description": "Презентация занятия.",
  "path": "new-slides.pdf",
  "class_ids": ["new-class"]
}
```

English entry in `en.json` → `resources`:

```json
"new-class-slides": {
  "title": "New class slides",
  "description": "The class presentation."
}
```

External resources use `url` and a `kind`: `link`, `video`, or `code`. Local formats are detected from the file extension. A resource with an empty `class_ids` array appears in the general library. Archive entries name the original ZIP and its exact `member` path. Keep the complete archive available for notebooks that depend on adjacent data files.

`source_ranges` contains inclusive, one-based line ranges into `notes.md`. Update these if the original document is edited. The source excerpts are explicitly labeled as originals; the class overview is translated separately.

The build rejects duplicate IDs, missing files, invalid dates, unknown class/module references, missing translations, and new root-level material files without catalog entries. This catches files added without a library entry.

## Checks

The build itself validates the catalog and reads every local resource. To run the browser checks against an already built and served site:

```bash
uv sync --frozen
uv run playwright install chromium
uv run python src/check_site.py --url http://127.0.0.1:8000/
```

Use `--browser /usr/bin/google-chrome` to use an existing Chrome installation. Checks cover both languages, search/filtering, navigation, modal previews, notebook rendering, mobile overflow, and all generated local resource links. External course links are preserved from the notes and are not downloaded or checked for availability by the build.

## GitHub Pages

The workflow follows GitHub's [custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages). The repository's **Settings → Pages → Source** must be **GitHub Actions**. Changes pushed to `main` rebuild and publish `dist/`. All asset URLs are relative, so the `/ml_course/` project path works without a special base URL.

## Adding a backend later

The browser fetches one versioned catalog from `data/course.json`. The generated contract includes stable entity IDs, translated fields, resource URLs, and many-to-many class/resource relationships. A backend can initially serve the same shape from an API, then add authenticated editing and file storage without changing the browsing model. `src/content` is the present source of truth; generated resource sizes, thumbnails, and reader URLs are added by the build.

## Provenance

Dates and relationships were reconstructed from `notes.md`, document contents, and filenames. Uncertain associations are labeled. In particular, the November 18 optimizer notes and November 28 syllabus remain separate entries, and March 4 on the interpretation slides is not asserted as a class date. Some filenames contain later import/photo dates, which are not used as class dates.

Original materials retain their authorship. MathJax's license is included alongside its vendored bundle.
