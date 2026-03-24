from __future__ import annotations

import logging

from playwright.sync_api import Page

from . import selectors
from .parser import parse_header_lines, parse_speaker_blocks
from .utils import normalize_text


def _first_existing(page: Page, selector_group: list[str]):
    for selector in selector_group:
        loc = page.locator(selector).first
        if loc.count() > 0:
            return loc
    return None


def _open_row_in_results(results_page: Page, title: str | None, row_text: str) -> bool:
    candidates = [title or "", row_text.split("\n", 1)[0], row_text[:120]]
    candidates = [c.strip() for c in candidates if c and c.strip()]

    for _ in range(80):
        for text in candidates:
            label = results_page.locator("label", has_text=text).first
            if label.count() > 0:
                try:
                    label.click(timeout=2000)
                    results_page.wait_for_timeout(350)
                    return True
                except Exception:
                    pass
            container = results_page.locator("div", has_text=text).first
            if container.count() > 0:
                try:
                    container.click(timeout=2000)
                    results_page.wait_for_timeout(350)
                    return True
                except Exception:
                    pass
        results_page.mouse.wheel(0, 1200)
        results_page.wait_for_timeout(100)
    return False


def _extract_from_page(page: Page) -> dict:
    panel = _first_existing(page, selectors.ARTICLE_CONTAINER)
    if panel is None:
        return {
            "company_name": None,
            "title": None,
            "event_type": None,
            "transcript_date": None,
            "participants": [],
            "raw_text": "",
            "raw_html": None,
            "speaker_blocks": [],
            "subtitle": None,
        }

    try:
        raw_html = panel.inner_html(timeout=2000)
    except Exception:
        raw_html = None

    header_lines: list[str] = []
    head = _first_existing(panel, selectors.HEADER) if hasattr(panel, "locator") else None
    if head is not None:
        try:
            header_lines = [line for line in head.inner_text(timeout=1000).split("\n") if line.strip()]
        except Exception:
            header_lines = []
    meta = parse_header_lines(header_lines)

    participants: list[str] = []
    part = _first_existing(panel, selectors.PARTICIPANTS) if hasattr(panel, "locator") else None
    if part is not None:
        try:
            text = part.inner_text(timeout=1000)
            for token in text.replace("Event Participants", "").replace("Executives", "").split(","):
                token = token.strip()
                if len(token) > 1:
                    participants.append(token)
        except Exception:
            participants = []

    editor = _first_existing(panel, selectors.SLATE_EDITOR) if hasattr(panel, "locator") else None
    raw_blocks: list[dict] = []
    raw_text = ""
    if editor is not None:
        try:
            raw_blocks = editor.evaluate(
                """
                (el) => {
                  const blocks = [];
                  let current = null;
                  for (const node of Array.from(el.children)) {
                    const tag = (node.tagName || '').toLowerCase();
                    const text = (node.innerText || '').trim();
                    if (!text) continue;
                    if (tag === 'h3') {
                      if (current) blocks.push(current);
                      current = { speaker: text, role: null, text: '' };
                      continue;
                    }
                    if (!current) current = { speaker: null, role: null, text: '' };
                    current.text = current.text ? current.text + '\n\n' + text : text;
                  }
                  if (current) blocks.push(current);
                  return blocks;
                }
                """
            )
        except Exception:
            raw_blocks = []
        try:
            raw_text = normalize_text(editor.inner_text(timeout=1500))
        except Exception:
            raw_text = ""

    speaker_blocks = parse_speaker_blocks(raw_blocks)
    if not raw_text:
        try:
            raw_text = normalize_text(panel.inner_text(timeout=1500))
        except Exception:
            raw_text = ""

    return {
        "company_name": meta.get("company_name"),
        "title": meta.get("title"),
        "event_type": meta.get("event_type") or "Earnings Call",
        "transcript_date": meta.get("transcript_date"),
        "participants": sorted(set(participants)),
        "raw_text": raw_text,
        "raw_html": raw_html,
        "speaker_blocks": speaker_blocks,
        "subtitle": meta.get("subtitle"),
    }


def close_open_panel(page: Page, logger: logging.Logger) -> None:
    for selector in selectors.CLOSE_PANEL:
        loc = page.locator(selector).first
        if loc.count() == 0:
            continue
        try:
            loc.click(timeout=900)
            page.wait_for_timeout(150)
            return
        except Exception:
            continue
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(120)
    except Exception:
        logger.debug("Failed to close transcript panel", extra={"event": "close_panel"})


def fetch_transcript_via_browser(
    *,
    results_page: Page,
    detail_page: Page,
    title: str | None,
    row_text: str,
    source_url: str | None,
    use_detail_page: bool,
    logger: logging.Logger,
) -> dict:
    if use_detail_page and source_url:
        detail_page.goto(source_url, wait_until="domcontentloaded", timeout=45000)
        detail_page.wait_for_timeout(600)
        data = _extract_from_page(detail_page)
        data["source_url"] = source_url
        return data

    opened = _open_row_in_results(results_page, title=title, row_text=row_text)
    if not opened:
        raise RuntimeError("Could not open transcript row on results page")

    data = _extract_from_page(results_page)
    if not data.get("source_url"):
        data["source_url"] = source_url or results_page.url
    close_open_panel(results_page, logger)
    return data
