"""Smoke-check generated files and the course's main browser journeys."""
from __future__ import annotations

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        url = attributes.get("src") if tag == "script" else attributes.get("href") if tag == "link" else None
        if url and "assets/" in url and urlsplit(url).path.endswith((".js", ".css")):
            self.urls.append(url)


def check_files():
    dist = ROOT / "dist"
    for page in [dist / "index.html", *(dist / "readers").glob("*.html")]:
        parser = AssetLinks()
        parser.feed(page.read_text())
        assert parser.urls, page
        for url in parser.urls:
            parts = urlsplit(url)
            asset = (page.parent / unquote(parts.path)).resolve()
            assert asset.is_relative_to(dist / "assets"), (page, url)
            expected = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
            assert parse_qs(parts.query).get("v") == [expected], (page, url)
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
    print(f"PASS: content-versioned JS/CSS on every page, every local resource link and all {len(catalog['resources'])} resource translations")
    return catalog


def check_regression(page, base: str, screenshots: Path):
    page.goto(base + "#regression?lang=en")
    lab = page.locator("#regression-lab")
    expect(page.locator("#lr-title")).to_have_text("Linear regression")
    expect(page.locator('[data-nav="regression"]')).to_have_attribute("aria-current", "page")
    expect(page.locator("#lr-uniform")).not_to_be_checked()
    expect(page.locator("#lr-lambda")).to_be_disabled()
    expect(page.locator("#lr-theta .lr-coefficient")).to_have_count(4)
    page.locator("#lr-tab-data").click()
    page.locator('[data-lr-action="next"]').click()
    expect(page.locator("#lr-page")).to_have_text("9–16 / 18")
    page.locator('[data-lr-action="previous"]').click()
    page.locator("#lr-x tbody tr").first.hover()
    expect(page.locator('#lr-plot [data-lr-point="0"]')).to_have_class("lr-observation lr-selected")

    def count():
        return page.locator("#lr-plot .lr-observation").count()

    def coordinates(x, y):
        plot = page.locator("#lr-plot")
        plot.scroll_into_view_if_needed()
        point = plot.evaluate("""(el, [x, y]) => {
            const r = el.querySelector('clipPath rect');
            const point = new DOMPoint(+r.getAttribute('x') + (x + 1) / 2 * +r.getAttribute('width'),
                +r.getAttribute('y') + (1.5 - y) / 3 * +r.getAttribute('height')).matrixTransform(el.getScreenCTM());
            return [point.x, point.y];
        }""", [x, y])
        return tuple(point)

    def stroke(points):
        page.mouse.move(*coordinates(*points[0]))
        page.mouse.down()
        for point in points[1:]:
            page.mouse.move(*coordinates(*point))
        page.mouse.up()

    def plot_xs():
        return page.locator("#lr-plot").evaluate("""el => {
            const r = el.querySelector('clipPath rect');
            return [...el.querySelectorAll('.lr-observation')].map(p =>
                2 * (+p.getAttribute('cx') - +r.getAttribute('x')) / +r.getAttribute('width') - 1);
        }""")

    page.locator('[data-lr-action="clear"]').click()
    expect(lab).to_have_attribute("data-observations", "0")
    expect(page.locator("#lr-mse")).to_have_text("—")
    page.locator("#lr-dimension").fill("1")
    page.locator("#lr-dimension").press("Tab")
    for point in [(-0.5, -0.25), (0.5, 0.75)]:
        page.mouse.click(*coordinates(*point))
    expect(lab).to_have_attribute("data-observations", "2")
    coefficients = page.locator("#lr-theta strong").all_text_contents()
    assert abs(float(coefficients[0]) - 0.25) < 1e-5, coefficients
    assert abs(float(coefficients[1]) - 1) < 1e-5, coefficients
    expect(page.locator("#lr-x tbody tr").first.locator("td").nth(1)).to_have_text("-0.5")
    expect(page.locator("#lr-y tbody tr").first.locator("td")).to_have_text("-0.25")
    stroke([(-0.5, -0.25), (-0.4, -0.1)])
    assert abs(float(page.locator('[data-lr-coordinate="x"][data-index="0"]').input_value()) + 0.4) < 1e-6
    page.locator('[data-lr-action="undo"]').click()
    assert abs(float(page.locator('[data-lr-coordinate="x"][data-index="0"]').input_value()) + 0.5) < 1e-6
    page.locator('[data-lr-coordinate="y"][data-index="0"]').fill("-0.2")
    page.locator('[data-lr-coordinate="y"][data-index="0"]').press("Tab")
    expect(page.locator("#lr-y tbody tr").first.locator("td")).to_have_text("-0.2")
    page.locator('[data-lr-delete="0"]').click()
    expect(lab).to_have_attribute("data-observations", "1")
    page.locator('[data-lr-action="undo"]').click()
    expect(lab).to_have_attribute("data-observations", "2")
    page.mouse.click(*coordinates(-0.5, -0.2))
    page.keyboard.press("ArrowUp")
    assert abs(float(page.locator('[data-lr-coordinate="y"][data-index="0"]').input_value()) + 0.19) < 1e-6
    page.keyboard.press("Delete")
    expect(lab).to_have_attribute("data-observations", "1")
    page.locator('[data-lr-action="undo"]').click()
    page.locator('[data-lr-tool="erase"]').click()
    page.mouse.click(*coordinates(0.5, 0.75))
    expect(lab).to_have_attribute("data-observations", "1")
    page.locator('[data-lr-action="undo"]').click()
    expect(lab).to_have_attribute("data-observations", "2")
    page.locator('[data-lr-tool="points"]').click()

    page.locator('[data-lr-action="clear"]').click()
    page.mouse.click(*coordinates(-0.8, 0))
    page.locator('[data-lr-tool="draw"]').click()
    stroke([(-0.6, 0.1), (-0.47, 0.3), (-0.11, -0.2), (0.28, 0.5), (0.65, 0.1)])
    uneven_count = count()
    assert uneven_count >= 6, uneven_count
    assert any(abs(x + 0.8) < 1e-6 for x in plot_xs())
    assert any(abs((x + 1) / 0.04 - round((x + 1) / 0.04)) > 0.01 for x in plot_xs())
    page.locator('[data-lr-action="undo"]').click()
    expect(lab).to_have_attribute("data-observations", "1")
    page.locator('[data-lr-action="redo"]').click()
    expect(lab).to_have_attribute("data-observations", str(uneven_count))
    outside = [x for x in plot_xs() if x < -0.35 or x > 0.35]
    page.locator("#lr-uniform").check()
    stroke([(-0.35, 0.4), (0.1, 0.6), (0.35, 0.2)])
    inside = [x for x in plot_xs() if -0.35 <= x <= 0.35]
    assert len(inside) >= 15, inside
    assert all(abs((x + 1) / 0.04 - round((x + 1) / 0.04)) < 1e-6 for x in inside), inside
    assert all(any(abs(x - old) < 1e-6 for x in plot_xs()) for old in outside)

    page.locator("#lr-preset").select_option("line")
    page.locator("#lr-regularization").select_option("l2")
    expect(page.locator("#lr-lambda")).to_be_enabled()
    page.locator("#lr-tab-gram").click()
    gram = page.locator("#lr-gram").inner_text()
    det = page.locator("#lr-det").inner_text()
    page.locator("#lr-lambda").fill("0.8")
    page.locator("#lr-lambda").press("Tab")
    expect(page.locator("#lr-ridge-panel")).to_be_visible()
    assert page.locator("#lr-gram").inner_text() == gram
    assert page.locator("#lr-det").inner_text() == det
    assert abs(float(page.locator("#lr-theta strong").nth(1).inner_text())) < 0.4
    page.locator("#lr-regularization").select_option("l1")
    expect(page.locator("#lr-theta strong").nth(1)).to_have_text("0")
    page.locator('[data-lr-action="zero"]').click()
    assert abs(float(page.locator("#lr-theta strong").nth(1).inner_text())) > 0.6
    page.locator("#lr-basis").select_option("fourier")
    page.locator("#lr-dimension").fill("2")
    page.locator("#lr-dimension").press("Tab")
    expect(page.locator("#lr-theta .lr-coefficient")).to_have_count(5)
    expect(page.locator("#lr-x thead")).to_contain_text("sin(1πx)")
    n = count()
    page.locator('[data-lang="ru"]').click()
    expect(page.locator("#lr-title")).to_have_text("Линейная регрессия")
    expect(page.locator("#lr-uniform")).to_be_checked()
    expect(page.locator("#lr-basis")).to_have_value("fourier")
    assert count() == n
    page.locator('[data-lang="en"]').click()
    page.locator('[data-nav="topics"]').click()
    page.locator('[data-nav="regression"]').click()
    expect(page.locator("#lr-dimension")).to_have_value("2")
    assert count() == n
    page.locator("#lr-preset").select_option("singular")
    expect(page.locator("#lr-status")).to_contain_text("rank deficient")
    expect(page.locator("#lr-det")).to_have_text("0")
    assert "NaN" not in page.locator("#lr-theta").inner_text()
    page.locator("#lr-regularization").select_option("none")
    page.locator("#lr-basis").select_option("polynomial")
    page.locator("#lr-dimension").fill("3")
    page.locator("#lr-dimension").press("Tab")
    page.locator("#lr-preset").select_option("sine")
    page.locator("#lr-uniform").uncheck()
    page.locator("#lr-tab-theta").click()
    page.screenshot(path=str(screenshots / "regression-en.png"), full_page=True)
    timings = page.evaluate("""() => ['none', 'l1', 'l2'].map(kind => {
      const points = Array.from({length: 200}, (_, i) => ({x: -1 + i * 2 / 199, y: Math.sin(i / 10)}));
      const start = performance.now();
      const result = MLRegressionMath.fit(points, 'polynomial', 20, kind, 0.01);
      if (!result.theta.every(Number.isFinite)) throw Error('Non-finite dense fit');
      return {kind, milliseconds: performance.now() - start};
    })""")
    page.locator('[data-lr-action="clear"]').click()
    page.locator('[data-lr-tool="draw"]').click()
    page.mouse.move(*coordinates(-0.95, -1.1))
    page.mouse.down()
    page.locator("#lr-plot").evaluate("""el => {
      const r = el.querySelector('clipPath rect');
      for (let i = 0; i < 240; i++) {
        const x = -0.9 + (i % 20) * 0.09, y = -1 + Math.floor(i / 20) * 0.17;
        const point = new DOMPoint(+r.getAttribute('x') + (x + 1) / 2 * +r.getAttribute('width'),
          +r.getAttribute('y') + (1.5 - y) / 3 * +r.getAttribute('height')).matrixTransform(el.getScreenCTM());
        el.dispatchEvent(new PointerEvent('pointermove', {bubbles: true, pointerId: 1, buttons: 1,
          clientX: point.x, clientY: point.y}));
      }
    }""")
    page.mouse.up()
    expect(lab).to_have_attribute("data-observations", "200")
    expect(page.locator('[data-lr-action="add"]')).to_be_disabled()
    expect(page.locator("#lr-status")).to_contain_text("Point limit reached")
    page.locator('[data-lr-action="undo"]').click()
    expect(lab).to_have_attribute("data-observations", "0")
    page.locator("#lr-preset").select_option("sine")
    print(f"PASS: regression editing, undo/redo, uneven drawing, optional uniform replacement, matrices, penalties, singular systems, RU/EN state; dense-fit timings: {timings}")


def check_lecture_view(browser, base: str, screenshots: Path):
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(base + "#regression?lang=en")
    expect(page.locator("#lr-presentation")).to_be_checked()
    expect(page.locator("body")).to_have_class("lr-presenting")
    expect(page.locator("#sidebar")).not_to_be_visible()
    expect(page.locator("#lr-panel-plot")).to_be_visible()
    expect(page.locator("#lr-panel-theta")).to_be_visible()
    expect(page.locator("#lr-panel-data")).not_to_be_visible()
    assert page.locator("#lr-theta strong").first.evaluate("e => parseFloat(getComputedStyle(e).fontSize)") >= 32
    assert page.locator("#lr-plot .lr-ticks text").first.evaluate("e => parseFloat(getComputedStyle(e).fontSize) * e.getScreenCTM().a") >= 25
    assert page.locator(".lr-metrics").bounding_box()["y"] + page.locator(".lr-metrics").bounding_box()["height"] <= 1080
    graph_box = page.locator(".lr-plot-card").bounding_box()
    assert page.locator(".lr-inspector").bounding_box()["x"] >= graph_box["x"] + graph_box["width"]
    page.screenshot(path=str(screenshots / "lecture-plot-1080p.png"))
    page.locator("#lr-tab-data").click()
    expect(page.locator("#lr-page")).to_have_text("1–5 / 18")
    expect(page.locator("#lr-plot")).to_be_visible()
    expect(page.locator("#lr-det")).to_be_visible()
    assert page.locator("#lr-x td").first.evaluate("e => parseFloat(getComputedStyle(e).fontSize)") >= 32
    assert page.locator("#lr-y").evaluate("e => e.scrollWidth <= e.clientWidth"), "Y should fit at its full readable size"
    page.locator('[data-lr-action="next"]').click()
    expect(page.locator("#lr-page")).to_have_text("6–10 / 18")
    page.screenshot(path=str(screenshots / "lecture-matrices-1080p.png"))
    page.locator("#lr-tab-data").focus()
    page.keyboard.press("ArrowRight")
    expect(page.locator("#lr-tab-gram")).to_have_attribute("aria-selected", "true")
    expect(page.locator("#lr-plot")).to_be_visible()
    expect(page.locator("#lr-panel-gram")).to_be_visible()
    page.screenshot(path=str(screenshots / "lecture-gram-1080p.png"))

    def coordinates(x, y):
        plot = page.locator("#lr-plot")
        plot.scroll_into_view_if_needed()
        return plot.evaluate("""(el, [x, y]) => {
            const r = el.querySelector('clipPath rect');
            const point = new DOMPoint(+r.getAttribute('x') + (x + 1) / 2 * +r.getAttribute('width'),
                +r.getAttribute('y') + (1.5 - y) / 3 * +r.getAttribute('height')).matrixTransform(el.getScreenCTM());
            return [point.x, point.y];
        }""", [x, y])

    def start_drag(before, after):
        page.mouse.move(*coordinates(*before))
        page.mouse.down()
        page.mouse.move(*coordinates(*after))

    # Assert visible values while the pointer is still held down, before pointerup.
    page.locator('[data-lr-action="clear"]').click()
    page.locator("#lr-dimension").fill("1")
    page.locator("#lr-dimension").press("Tab")
    for point in ((-0.5, -0.25), (0.5, 0.75)):
        page.mouse.click(*coordinates(*point))
    expect(page.locator("#lr-det")).to_have_text("1")
    page.locator("#lr-tab-data").click()
    start_drag((-0.5, -0.25), (-0.25, -0.1))
    expect(page.locator("#lr-x tbody tr").first.locator("td").nth(1)).to_have_text("-0.25")
    expect(page.locator("#lr-y tbody tr").first.locator("td")).to_have_text("-0.1")
    expect(page.locator("#lr-det")).to_have_text("0.5625")
    expect(page.locator("#lr-tab-data")).to_have_attribute("aria-selected", "true")
    page.mouse.up()
    page.locator("#lr-tab-gram").click()
    start_drag((-0.25, -0.1), (0, 0))
    expect(page.locator("#lr-gram tbody tr").nth(1).locator("td").nth(1)).to_have_text("0.25")
    expect(page.locator("#lr-det")).to_have_text("0.25")
    page.mouse.up()
    page.locator("#lr-tab-theta").click()
    start_drag((0, 0), (-0.5, -0.25))
    expect(page.locator("#lr-theta strong").first).to_have_text("0.25")
    expect(page.locator("#lr-theta strong").nth(1)).to_have_text("1")
    expect(page.locator("#lr-det")).to_have_text("1")
    page.mouse.up()
    page.locator("#lr-dimension").fill("20")
    page.locator("#lr-dimension").press("Tab")
    page.locator("#lr-tab-gram").click()
    page.locator("#lr-gram").evaluate("e => e.scrollLeft = 160")
    page.mouse.click(*coordinates(0, 0.3))
    assert page.locator("#lr-gram").evaluate("e => e.scrollLeft") >= 159
    expect(page.locator("#lr-det")).to_have_text("0")
    page.locator(".lr-inspector-body").evaluate("e => e.scrollTop = 120")
    assert page.locator(".lr-inspector-body").evaluate("e => e.scrollTop") >= 119
    page.locator("#lr-tab-theta").click()
    assert page.locator(".lr-inspector-body").evaluate("e => e.scrollTop") == 0
    page.locator("#lr-tab-gram").click()
    assert page.locator(".lr-inspector-body").evaluate("e => e.scrollTop") == 0
    page.locator("#lr-text-size").select_option("1.5")
    assert page.locator("#lr-gram td").first.evaluate("e => parseFloat(getComputedStyle(e).fontSize)") >= 48
    page.locator('[data-lang="ru"]').click()
    expect(page.locator("#lr-text-size")).to_have_value("1.5")
    expect(page.locator("#lr-title")).to_have_text("Линейная регрессия")
    expect(page.locator("#lr-tab-gram")).to_have_attribute("aria-selected", "true")
    page.reload()
    expect(page.locator("#lr-presentation")).to_be_checked()
    expect(page.locator("#lr-text-size")).to_have_value("1.5")
    page.locator("#lr-text-size").select_option("1")
    for width, height in ((1280, 720), (390, 844), (320, 844)):
        page.set_viewport_size({"width": width, "height": height})
        for view in ("theta", "data", "gram"):
            page.locator("#lr-tab-" + view).click()
            expect(page.locator("#lr-plot")).to_be_visible()
            expect(page.locator("#lr-det")).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), ("lecture", width, view)
    page.locator("#lr-presentation").uncheck()
    expect(page.locator("body")).not_to_have_class("lr-presenting")
    expect(page.locator("#lr-panel-gram")).to_be_visible()
    expect(page.locator("#lr-panel-theta")).not_to_be_visible()
    expect(page.locator("#lr-plot")).to_be_visible()
    page.goto(base + "#topics?lang=en")
    expect(page.locator("#lr-presentation")).to_have_count(0)
    expect(page.locator("body")).not_to_have_class("lr-presenting")
    assert not errors, errors
    context.close()
    print("PASS: live theta/X/Y/Gram/determinant updates before pointerup, graph beside all tabs, preserved matrix scroll, 32–48 px values, 1080p/720p/mobile, language and size preferences")


def check_browser(base: str, executable: str | None, screenshots: Path):
    errors = []
    local_failures = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, locale="en-GB")
        context.add_init_script("localStorage.setItem('ml-regression-presentation', 'false')")
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
        check_regression(page, base, screenshots)
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            for lang in ("ru", "en"):
                for route in ("timeline", "library", "topics", "class/bayes", "regression"):
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
            page.goto(base + "#regression?lang=en")
            page.locator("#lr-dimension").fill("20")
            page.locator("#lr-dimension").press("Tab")
            expect(page.locator("#lr-theta .lr-coefficient")).to_have_count(21)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, "regression", "degree 20")
            page.screenshot(path=str(screenshots / f"regression-mobile-{width}.png"), full_page=True, animations="disabled")
        touch_context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        touch_page = touch_context.new_page()
        touch_page.on("pageerror", lambda error: errors.append(str(error)))
        touch_page.goto(base + "#regression?lang=en")
        touch_page.locator('[data-lr-action="clear"]').tap()
        touch_plot = touch_page.locator("#lr-plot")
        touch_plot.scroll_into_view_if_needed()
        bounds = touch_plot.bounding_box()
        touch_page.touchscreen.tap(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
        expect(touch_page.locator("#regression-lab")).to_have_attribute("data-observations", "1")
        touch_context.close()
        check_lecture_view(browser, base, screenshots)
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
