from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from .utils import ensure_dir


@dataclass(slots=True)
class BrowserBundle:
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page

    def close(self) -> None:
        try:
            self.context.close()
        finally:
            try:
                self.browser.close()
            finally:
                self.playwright.stop()


def has_any(page: Page, selectors: Iterable[str]) -> bool:
    for selector in selectors:
        try:
            if page.locator(selector).count() > 0:
                return True
        except Exception:
            continue
    return False


def click_first(page: Page, selectors: Iterable[str], timeout_ms: int = 2500) -> bool:
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if loc.count() == 0:
                continue
            loc.click(timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def fill_first(page: Page, selectors: Iterable[str], value: str, timeout_ms: int = 2500) -> bool:
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if loc.count() == 0:
                continue
            loc.fill(value, timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def save_failure_artifacts(page: Page, artifact_dir: Path, stem: str) -> tuple[Path, Path]:
    ensure_dir(artifact_dir)
    png = artifact_dir / f"{stem}.png"
    html = artifact_dir / f"{stem}.html"
    page.screenshot(path=str(png), full_page=True)
    html.write_text(page.content(), encoding="utf-8")
    return png, html


@contextmanager
def open_browser(*, headless: bool, storage_state_path: Path):
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=headless)
    context_args: dict[str, object] = {"accept_downloads": True}
    if storage_state_path.exists():
        context_args["storage_state"] = str(storage_state_path)
    context = browser.new_context(**context_args)
    page = context.new_page()
    bundle = BrowserBundle(playwright=pw, browser=browser, context=context, page=page)
    try:
        yield bundle
        ensure_dir(storage_state_path.parent)
        context.storage_state(path=str(storage_state_path))
    finally:
        bundle.close()
