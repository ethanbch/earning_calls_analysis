from __future__ import annotations

import logging
from datetime import date
from urllib.parse import urljoin

from playwright.sync_api import Page

from . import selectors
from .browser import click_first, fill_first, has_any
from .models import ManifestRow
from .utils import generate_transcript_uid


def goto_transcripts_search(page: Page, logger: logging.Logger, *, validate_nav: bool = False) -> None:
    # Production path: repeatedly force the transcript URL because Koyfin can
    # bounce to the last viewed ticker route after login/session restore.
    for _ in range(4):
        page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded", timeout=45000)
        try:
            page.wait_for_url("**/search/transcripts**", timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(1200)
        if "/search/transcripts" in page.url.lower():
            return

        # Force SPA location in case internal router keeps overriding page.goto.
        page.evaluate(f"() => window.location.assign('{selectors.KOYFIN_TRANSCRIPTS_URL}')")
        page.wait_for_timeout(1000)
        if "/search/transcripts" in page.url.lower():
            return

    # Fallback nav-click path if direct forcing still fails.
    if click_first(page, selectors.ADVANCED_SEARCH_NAV, timeout_ms=2500):
        page.wait_for_timeout(300)
    click_first(page, selectors.TRANSCRIPTS_SEARCH_NAV, timeout_ms=2500)
    page.wait_for_timeout(1200)
    if "/search/transcripts" in page.url.lower():
        return

    if not validate_nav:
        raise RuntimeError(f"Could not load transcripts page directly: {page.url}")


def open_date_picker(page: Page) -> None:
    # The current Koyfin transcripts UI often requires a forced click on the
    # visible time-range control rather than the nearby label container.
    forced_selectors = [
        "div[class*='time-range__root']",
        "text=Mar 22 2006 - Mar 22 2026",
    ]
    for selector in forced_selectors:
        try:
            loc = page.locator(selector).last
            if loc.count() == 0:
                continue
            loc.click(force=True, timeout=2500)
            page.wait_for_timeout(300)
            return
        except Exception:
            continue

    if click_first(page, selectors.DATE_RANGE_BOX, timeout_ms=1500):
        page.wait_for_timeout(200)


def wait_for_results_refresh(page: Page) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=12000)
    except Exception:
        pass
    page.wait_for_timeout(900)


def set_date_range(page: Page, start_date: date, end_date: date, logger: logging.Logger) -> None:
    open_date_picker(page)
    s = start_date.strftime("%m/%d/%Y")
    e = end_date.strftime("%m/%d/%Y")

    begin_ok = fill_first(page, selectors.DATE_BEGIN_INPUT, s, timeout_ms=2500)
    end_ok = fill_first(page, selectors.DATE_END_INPUT, e, timeout_ms=2500)

    if not (begin_ok and end_ok):
        # Fallback: fill visible MM/DD/YYYY fields in one browser-side pass.
        filled = page.evaluate(
            """
            ({ start, end }) => {
              const isVisible = (el) => {
                const r = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return r.width > 0 && r.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
              };
              const inputs = Array.from(document.querySelectorAll("input[placeholder*='MM/DD/YYYY'], input[placeholder='MM/DD/YYYY']"))
                .filter(isVisible);
              if (inputs.length < 2) return false;
              const fire = (el) => {
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.dispatchEvent(new Event('blur', { bubbles: true }));
              };
              inputs[0].focus();
              inputs[0].value = start;
              fire(inputs[0]);
              inputs[1].focus();
              inputs[1].value = end;
              fire(inputs[1]);
              return true;
            }
            """,
            {"start": s, "end": e},
        )
        if not filled:
            raise RuntimeError("Could not locate begin/end date inputs")

    if not click_first(page, selectors.DATE_APPLY, timeout_ms=2500):
        applied = page.evaluate(
            """
            () => {
              const candidates = Array.from(document.querySelectorAll("button,label,span,div"));
              const target = candidates.find((el) => /apply dates|apply|search transcripts/i.test((el.textContent || '').trim()));
              if (!target) return false;
              target.click();
              return true;
            }
            """
        )
        if not applied:
            page.keyboard.press("Enter")

    wait_for_results_refresh(page)
    logger.info(
        "Date range applied",
        extra={"event": "date_range", "start": start_date.isoformat(), "end": end_date.isoformat()},
    )


def extract_rows_bulk(page: Page, result_page: int) -> list[dict]:
    # Single browser-side evaluation to avoid repeated locator roundtrips.
    rows = page.evaluate(
        r"""
        ({ resultPage }) => {
          const out = [];
          const seen = new Set();
          const normalize = (s) => (s || '').replace(/\s+/g, ' ').trim();

          const labels = Array.from(document.querySelectorAll('label'))
            .filter((el) => /earnings call/i.test(el.textContent || ''));

          const candidates = labels.length
            ? labels
            : Array.from(document.querySelectorAll('div,li,a,span'))
                .filter((el) => /earnings call/i.test(el.textContent || ''));

          candidates.forEach((node, idx) => {
            const row = node.closest('div,li,tr,article') || node;
            const title = normalize((node.textContent || '').split('\n')[0]);
            const rowText = normalize(row.textContent || node.textContent || '');
            const lower = rowText.toLowerCase();
            const isEarnings = lower.includes('earnings call');

            const linkEl = row.querySelector('a[href]') || node.closest('a[href]') || node.querySelector?.('a[href]');
            const href = linkEl ? linkEl.getAttribute('href') : null;
            const dataId = row.getAttribute('data-id') || node.getAttribute('data-id') || null;
            const domKey = row.id || node.id || dataId || null;
            const company = title.includes(',') ? title.split(',')[0].trim() : null;

            const key = [title, href || '', domKey || '', String(idx)].join('|');
            if (!title || seen.has(key)) return;
            seen.add(key);

            out.push({
              title,
              company_name: company,
              row_text: rowText,
              href,
              dom_key: domKey,
              data_id: dataId,
              is_earnings_call: isEarnings,
              result_page: resultPage,
              result_position: out.length,
            });
          });

          return out;
        }
        """,
        {"resultPage": result_page},
    )
    return rows or []


def filter_earnings_call_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        text = str(row.get("row_text") or "").lower()
        if row.get("is_earnings_call") or "earnings call" in text:
            out.append(row)
    return out


def build_manifest_for_page(rows: list[dict], *, window_start: str, window_end: str) -> list[ManifestRow]:
    manifests: list[ManifestRow] = []
    for row in rows:
        source_url = row.get("href")
        if isinstance(source_url, str) and source_url.startswith("/"):
            source_url = urljoin(selectors.KOYFIN_BASE_URL, source_url)
        uid = generate_transcript_uid(
            title=row.get("title"),
            company_name=row.get("company_name"),
            source_url=source_url,
            dom_key=row.get("dom_key"),
            result_page=int(row.get("result_page") or 1),
            result_position=int(row.get("result_position") or 0),
            window_start=window_start,
        )
        manifests.append(
            ManifestRow(
                transcript_uid=uid,
                window_start=window_start,
                window_end=window_end,
                result_page=int(row.get("result_page") or 1),
                result_position=int(row.get("result_position") or 0),
                title=(str(row.get("title") or "").strip() or None),
                company_name=(str(row.get("company_name") or "").strip() or None),
                row_text=str(row.get("row_text") or ""),
                source_url=(str(source_url).strip() if source_url else None),
                dom_key=(str(row.get("dom_key")).strip() if row.get("dom_key") else None),
                data_id=(str(row.get("data_id")).strip() if row.get("data_id") else None),
                is_earnings_call=bool(row.get("is_earnings_call", True)),
                metadata={},
            )
        )
    return manifests


def goto_next_results_page(page: Page, logger: logging.Logger) -> bool:
    for selector in selectors.NEXT_PAGE:
        loc = page.locator(selector).first
        if loc.count() == 0:
            continue
        try:
            if loc.is_disabled():
                continue
        except Exception:
            pass
        try:
            loc.click(timeout=2000)
            wait_for_results_refresh(page)
            logger.info("Moved to next results page", extra={"event": "next_page"})
            return True
        except Exception:
            continue
    return False
