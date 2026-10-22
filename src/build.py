"""Build the course archive. Never executes notebook cells or fetches course links."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import zipfile
from urllib.parse import quote, unquote, urlsplit

import bleach
from markdown_it import MarkdownIt
from PIL import Image, ImageOps
import pillow_heif
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DIST = ROOT / "dist"
MD = MarkdownIt("commonmark", {"html": True, "breaks": True}).enable("table")
pillow_heif.register_heif_opener()


def markdown(text: str) -> str:
    # Preserve TeX before Markdown can interpret its underscores and backslashes.
    equations: list[str] = []

    def keep(match: re.Match) -> str:
        raw = match.group()
        block = raw.startswith("$$")
        math = raw[2:-2] if block else raw[1:-1]
        equations.append(f'<span class="math">{html.escape(("\\[" if block else "\\(") + math + ("\\]" if block else "\\)"))}</span>')
        return f"MATHPLACEHOLDER{len(equations) - 1}END"

    text = re.sub(r"\$\$[\s\S]+?\$\$|(?<![\\\w])\$(?!\s)[^$\n]+?\$", keep, text)
    rendered = MD.render(text)
    rendered = bleach.clean(
        rendered,
        tags={"p", "br", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "em", "strong", "s", "a", "img", "pre", "code", "blockquote", "hr", "table", "thead", "tbody", "tr", "th", "td", "sup", "sub", "details", "summary"},
        attributes={"a": ["href", "title"], "img": ["src", "alt", "title", "width", "height"], "ol": ["start"]},
        protocols={"http", "https", "mailto", "attachment"}, strip=True,
    )
    for i, equation in enumerate(equations):
        rendered = rendered.replace(f"MATHPLACEHOLDER{i}END", equation)
    return rendered


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".html":
        def version_asset(match: re.Match) -> str:
            url = match["url"]
            asset = (path.parent / unquote(url)).resolve()
            if not asset.is_relative_to((DIST / "assets").resolve()) or not asset.is_file():
                raise ValueError(f"Missing or unsafe asset: {url}")
            version = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
            return f'{match["attribute"]}="{url}?v={version}"'

        # A deployment must not combine new HTML with a cached older router.
        content = re.sub(
            r'(?P<attribute>\b(?:src|href))="(?P<url>(?:\.\./)?assets/[^"?#]+\.(?:js|css))"',
            version_asset,
            content,
        )
    path.write_text(content, encoding="utf-8")


def safe_source(name: str) -> Path:
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing or unsafe source path: {name}")
    return path


def validate(data: dict):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported catalog schema")
    for field in ("classes", "modules", "resources"):
        ids = [r["id"] for r in data[field]]
        if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9-]+", i) for i in ids):
            raise ValueError(f"Invalid or duplicate IDs in {field}")
    modules = {m["id"] for m in data["modules"]}
    classes = {c["id"] for c in data["classes"]}
    from datetime import date
    for c in data["classes"]:
        if c["module"] not in modules:
            raise ValueError(f"Unknown module: {c['module']}")
        if c["date"]:
            date.fromisoformat(c["date"])
    for r in data["resources"]:
        if not set(r["class_ids"]).issubset(classes):
            raise ValueError(f"Unknown class in {r['id']}")
        if sum(key in r for key in ("path", "archive", "url")) != 1:
            raise ValueError(f"Resource must have one source: {r['id']}")
        if "url" in r and urlsplit(r["url"]).scheme not in {"http", "https"}:
            raise ValueError(f"Unsafe URL: {r['id']}")
    # Keep new files visible: fail the build if a root course file isn't cataloged.
    types = {".pdf", ".pptx", ".heic", ".jpg", ".jpeg", ".png", ".ipynb", ".zip", ".json", ".epub"}
    originals = {p.name for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in types}
    cataloged = {r["path"] for r in data["resources"] if "path" in r}
    if originals - cataloged:
        raise ValueError(f"Uncataloged materials: {sorted(originals - cataloged)}")


def reader_page(resource: dict, body: str) -> str:
    title = html.escape(resource["title"])
    download = html.escape("../" + resource["download_url"], quote=True)
    return f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · ML course</title><link rel="icon" href="../assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="../assets/style.css">
<script src="../assets/math-config.js"></script><script defer src="../assets/vendor/mathjax/tex-svg.js"></script><script defer src="../assets/i18n.js"></script><script defer src="../assets/reader.js"></script></head>
<body class="reader-body" data-title-ru="{html.escape(resource['title'], quote=True)}" data-title-en="{html.escape(resource['translations']['en']['title'], quote=True)}"><header class="reader-header"><a href="../#library">← Библиотека курса</a><div class="topbar-right"><a class="button small" href="{download}" download>Скачать оригинал ↓</a><div class="language-switch" role="group" aria-label="Language / Язык"><button data-lang="ru">RU</button><button data-lang="en">EN</button></div></div></header>
<main class="reader"><p class="eyebrow">MACHINE LEARNING / МАТЕРИАЛЫ</p><h1 id="reader-title">{title}</h1>
<p class="reader-note">{'Сохранённый ноутбук: код и результаты из оригинала. Ячейки не запускаются в браузере.' if resource['kind'] == 'notebook' else 'Исходный документ курса.'}</p>{body}</main></body></html>'''


def notebook(resource: dict, file: Path) -> str:
    nb = json.loads(file.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            rendered = markdown(source)
            # Notebook attachments are limited to raster image MIME types.
            for name, formats in cell.get("attachments", {}).items():
                for mime in ("image/png", "image/jpeg"):
                    if mime in formats:
                        value = "".join(formats[mime])
                        base64.b64decode(value, validate=True)
                        rendered = rendered.replace(html.escape("attachment:" + name, quote=True), f"data:{mime};base64,{value}")
                        break
            cells.append(f'<section class="notebook-markdown prose">{rendered}</section>')
        elif cell["cell_type"] == "code":
            count = cell.get("execution_count")
            code = f'<div class="cell-label">In [{count if count is not None else " "}]</div><pre><code>{html.escape(source)}</code></pre>'
            outputs = []
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" in data:
                    value = "".join(data["image/png"])
                    # Decode and re-encode so only valid raster data enters generated HTML.
                    raster = base64.b64decode(value)
                    Image.open(io.BytesIO(raster)).verify()
                    value = base64.b64encode(raster).decode("ascii")
                    outputs.append(f'<img class="cell-image" alt="Сохранённый результат ячейки {index + 1}" loading="lazy" src="data:image/png;base64,{value}">')
                elif "text/html" in data:
                    outputs.append('<div class="output-table">' + markdown("".join(data["text/html"])) + '</div>')
                else:
                    text = output.get("text", data.get("text/plain", []))
                    if output.get("output_type") == "error":
                        text = f"{output.get('ename', '')}: {output.get('evalue', '')}"
                    if text:
                        outputs.append(f'<pre class="cell-output">{html.escape("".join(text))}</pre>')
            cells.append(f'<section class="notebook-cell">{code}{"".join(outputs)}</section>')
    return "\n".join(cells)


def process_resource(r: dict):
    if "url" in r:
        r["external"] = True
        return
    if "archive" in r:
        archive = safe_source(r["archive"])
        with zipfile.ZipFile(archive) as z:
            payload = z.read(r["member"])
        name = Path(r["member"]).name
        destination = DIST / "materials" / "unpacked" / r["id"] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        r["filename"] = r["member"]
    else:
        file = safe_source(r["path"])
        destination = DIST / "materials" / file.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
        r["filename"] = file.name
    r["download_url"] = quote(destination.relative_to(DIST).as_posix())
    r["url"] = r["download_url"]
    r["bytes"] = destination.stat().st_size
    ext = destination.suffix.lower()
    r["format"] = ext[1:].upper()
    r["kind"] = {".epub": "ebook", ".pdf": "pdf", ".ipynb": "notebook", ".jpg": "image", ".png": "image", ".jpeg": "image", ".heic": "image", ".zip": "archive", ".pptx": "slides", ".md": "document"}.get(ext, "data")
    preview = DIST / "previews" / (r["id"] + ".webp")
    preview.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".pdf":
        with pymupdf.open(destination) as doc:
            r["pages"] = len(doc)
            pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(0.7, 0.7), alpha=False)
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            im.thumbnail((560, 440))
            im.save(preview, quality=78)
    elif r["kind"] == "image":
        with Image.open(destination) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
            if ext == ".heic":
                browser_image = DIST / "previews" / (r["id"] + ".jpg")
                im.save(browser_image, quality=92)
                r["url"] = browser_image.relative_to(DIST).as_posix()
            im.thumbnail((560, 440))
            im.save(preview, quality=80)
    if preview.exists():
        r["thumbnail"] = preview.relative_to(DIST).as_posix()
    if ext in {".ipynb", ".md"}:
        body = notebook(r, destination) if ext == ".ipynb" else '<div class="prose">' + markdown(destination.read_text(encoding="utf-8")) + '</div>'
        r["url"] = f"readers/{r['id']}.html"
        write(DIST / r["url"], reader_page(r, body))


def build():
    data = json.loads((SRC / "content/course.json").read_text(encoding="utf-8"))
    validate(data)
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    shutil.copytree(SRC / "static", DIST / "assets")
    data = copy.deepcopy(data)
    english = json.loads((SRC / "content/en.json").read_text(encoding="utf-8"))
    translated_fields = {
        "classes": {"title", "summary", "topics", "practice", "note"},
        "modules": {"title", "short", "description"},
        "resources": {"title", "description"},
    }
    for field in ("classes", "modules", "resources"):
        for record in data[field]:
            translation = english[field][record["id"]]
            missing = translated_fields[field] - translation.keys()
            if missing:
                raise ValueError(f"Missing English fields in {record['id']}: {sorted(missing)}")
            record["translations"] = {"en": translation}
    if {tag for c in data["classes"] for tag in c["tags"]} - english["topics"].keys():
        raise ValueError("Missing English topic translations")
    data["topic_translations"] = {"en": english["topics"]}
    notes = (ROOT / "notes.md").read_text(encoding="utf-8").splitlines()
    for c in data["classes"]:
        excerpts = []
        for start, end in c["source_ranges"]:
            if not 1 <= start <= end <= len(notes):
                raise ValueError(f"Invalid note range: {c['id']}")
            excerpts.append("\n".join(notes[start - 1:end]))
        c["notes_html"] = markdown("\n\n---\n\n".join(excerpts))
        c["notes_text"] = "\n".join(excerpts)
        c["resource_ids"] = [r["id"] for r in data["resources"] if c["id"] in r["class_ids"]]
    for r in data["resources"]:
        process_resource(r)
    write(DIST / "data/course.json", json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    template = (SRC / "index.html").read_text(encoding="utf-8")
    fallback = '<h1>Machine Learning · материалы курса</h1>'
    for c in data["classes"]:
        fallback += f'<section><h2>{html.escape(c["title"])}</h2><p>{html.escape(c["summary"])}</p><ul>'
        fallback += ''.join(f'<li>{html.escape(t)}</li>' for t in c['topics']) + '</ul></section>'
    fallback += '<h2>Все материалы</h2><ul>' + ''.join(f'<li><a href="{html.escape(r["url"], quote=True)}">{html.escape(r["title"])}</a></li>' for r in data['resources']) + '</ul>'
    write(DIST / "index.html", template.replace("<!-- FALLBACK -->", fallback))
    write(DIST / ".nojekyll", "")
    print(f"Built {len(data['classes'])} classes and {len(data['resources'])} resources → {DIST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
