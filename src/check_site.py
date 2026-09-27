"""Smoke-check generated files and the course's main browser journeys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def check_files():
    dist = ROOT / "dist"
    catalog = json.loads((dist / "data/course.json").read_text())
    for resource in catalog["resources"]:
        for key in ("url", "download_url", "thumbnail"):
            url = resource.get(key)
            if not url or urlsplit(url).scheme:
                continue
            target = (dist / unquote(urlsplit(url).path)).resolve()
            assert target.is_relative_to(dist), (resource["id"], url)
            assert target.is_file(), (resource["id"], url)
    for section in ("classes", "modules", "resources"):
        for record in catalog[section]:
            assert record["translations"]["en"]["title"], record["id"]
    print(f"PASS: every local resource link and all {len(catalog['resources'])} resource translations")
    return catalog


def check_browser(base: str, executable: str | None, screenshots: Path):
    errors = []
    local_failures = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, locale="en-GB")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: local_failures.append(f"{response.status} {response.url}") if response.status >= 400 and response.url.startswith(base) else None)
        page.goto(base + "#timeline?lang=en")
        expect(page.locator(".class-card")).to_have_count(18)
        expect(page.locator("html")).to_have_attribute("lang", "en")
        expect(page.locator(".hero h1")).to_contain_text("One step at a time")
        screenshots.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshots / "timeline-en.png"))
        page.locator('[data-lang="ru"]').click()
        expect(page.locator(".hero h1")).to_contain_text("Шаг за шагом")
        expect(page.locator("html")).to_have_attribute("lang", "ru")
        page.locator('[data-lang="en"]').click()
        page.locator("#search").fill("Bayes")
        expect(page.locator(".class-card")).to_have_count(1)
        expect(page.locator(".class-card h3")).to_contain_text("Bayes")
        page.locator(".class-card h3 a").click()
        expect(page.locator(".detail-heading h1")).to_have_text("Bayes, hypotheses and embeddings")
        expect(page.locator(".topic-list li")).to_have_count(6)
        page.locator('[data-preview="bayes-notes"]').first.click()
        expect(page.locator("#preview-dialog")).to_be_visible()
        expect(page.locator("#preview-body iframe")).to_have_count(1)
        page.keyboard.press("Escape")
        expect(page.locator("#preview-dialog")).not_to_be_visible()
        page.locator('[data-preview="bayes-visual"]').first.click()
        expect(page.locator("#preview-body img")).to_be_visible()
        page.locator("[data-close-preview]").click()
        page.goto(base + "#timeline?lang=en&module=advanced&period=spring")
        expect(page.locator(".class-card")).to_have_count(6)
        page.goto(base + "#topics?lang=en")
        expect(page.locator(".topic-card")).to_have_count(22)
        page.locator("#search").fill("Derivatives")
        expect(page.locator(".topic-card")).to_have_count(1)
        page.locator(".topic-card h2 a").click()
        expect(page.locator(".class-card")).to_have_count(3)
        page.goto(base + "#library?lang=en&kind=notebook")
        expect(page.locator(".resource-card")).to_have_count(19)
        page.locator("#search").fill("transformer")
        expect(page.locator(".resource-card")).to_have_count(2)
        page.locator(".resource-title").first.click()
        expect(page.locator("#reader-title")).to_contain_text("Transformer")
        expect(page.locator(".notebook-cell").first).to_be_visible()
        page.screenshot(path=str(screenshots / "notebook-en.png"))
        page.locator('[data-lang="ru"]').click()
        expect(page.locator(".reader-note")).to_contain_text("Сохранённый ноутбук")
        page.goto(base + "#library?lang=en&kind=image")
        page.screenshot(path=str(screenshots / "library-en.png"))
        page.goto(base + "#timeline?lang=en&q=__no_results_expected__")
        expect(page.locator(".empty-state")).to_be_visible()
        page.goto(base + "#class/not-a-class?lang=en")
        expect(page.locator("#app h1")).to_have_text("Entry not found")
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            for lang in ("ru", "en"):
                for route in ("timeline", "library", "topics", "class/bayes"):
                    page.goto(base + f"#{route}?lang={lang}")
                    expect(page.locator("#app h1")).to_be_visible()
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, lang, route)
            page.goto(base + "#timeline?lang=en")
            page.locator(".mobile-menu").click()
            expect(page.locator(".mobile-menu")).to_have_attribute("aria-expanded", "true")
            page.locator('[data-nav="library"]').click()
            expect(page.locator(".mobile-menu")).to_have_attribute("aria-expanded", "false")
            page.goto(base + "#class/trees?lang=en")
            page.screenshot(path=str(screenshots / f"class-mobile-{width}.png"), full_page=True, animations="disabled")
        assert not errors, errors
        assert not local_failures, local_failures
        browser.close()
    print("PASS: RU/EN, search, filters, topics, class navigation, PDF/image previews, notebooks, empty/404 states, mobile layouts; no JavaScript or local HTTP errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/")
    parser.add_argument("--browser", default=None)
    parser.add_argument("--screenshots", type=Path, default=Path("/tmp/ml-course-screenshots"))
    args = parser.parse_args()
    check_files()
    check_browser(args.url.rstrip("/") + "/", args.browser, args.screenshots)
