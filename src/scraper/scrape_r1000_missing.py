"""Scrape missing Russell 1000 earnings call transcripts from the current Koyfin search view.

This script expects a valid Playwright storage state JSON created from an
already-authenticated Koyfin session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


RESULTS_SCROLLER = "div.news-virtual-list__newsVirtualList__container___a0EHh"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("r1000_missing")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_existing_keys(path: Path, logger: logging.Logger) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            company = norm(row.get("company") or row.get("company_name"))
            title = norm(row.get("title"))
            if company and title:
                keys.add((company, title))
    logger.info("Loaded existing corpus keys: %d", len(keys))
    return keys


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "scroll_round": 0,
        "seen_row_keys": [],
        "stored": 0,
        "skipped_existing": 0,
        "errors": 0,
        "last_visible_titles": [],
        "done": False,
    }


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def parse_transcript_date(raw: str | None) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%A, %B %d, %Y %I:%M %p",
        "%b %d '%y",
    ):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except Exception:
            continue
    return None


def target_path(raw_dir: Path, transcript_date: str | None, uid: str) -> Path:
    day = (transcript_date or utc_now_iso()[:10]).split("-")
    out_dir = ensure_dir(raw_dir / day[0] / day[1] / day[2])
    return out_dir / f"{uid}.json"


def persist_json(raw_dir: Path, payload: dict[str, Any]) -> bool:
    uid = payload["transcript_uid"]
    target = target_path(raw_dir, payload.get("transcript_date"), uid)
    if target.exists():
        return False
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return True


def goto_r1000_search(page, logger: logging.Logger) -> None:
    page.goto("https://app.koyfin.com", wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(2500)
    page.locator("text=Advanced Search").first.click(force=True)
    page.wait_for_timeout(700)
    page.locator("text=Transcripts Search").last.click(force=True)
    page.wait_for_timeout(3000)
    body = page.locator("body").inner_text()
    if "Russell 1000 (Large-Cap)" not in body:
        raise RuntimeError("Russell 1000 filter is not active in the saved search state")
    if "Showing most recent results of" not in body:
        raise RuntimeError("Could not confirm transcript search results are visible")
    logger.info("Loaded Russell 1000 transcript search surface")


def extract_visible_rows(page) -> list[dict[str, Any]]:
    rows = page.evaluate(
        r"""
        () => {
          const out = [];
          const seen = new Set();
          const normalize = (s) => (s || '').replace(/\s+/g, ' ').trim();
          const root = document.querySelector('div.news-virtual-list__newsVirtualList__container___a0EHh');
          if (!root) return out;
          const candidates = Array.from(root.querySelectorAll('div, label, a, span')).filter((el) => /earnings call/i.test(el.textContent || ''));
          for (const node of candidates) {
            const row = node.closest('div') || node;
            const text = normalize(row.innerText || node.innerText || '');
            if (!/earnings call/i.test(text)) continue;
            const lines = text.split(/\n+/).map(normalize).filter(Boolean);
            const title = lines.find((x) => /earnings call/i.test(x) && x.toLowerCase() !== 'earnings calls') || null;
            if (!title) continue;
            const key = title + '|' + text;
            if (seen.has(key)) continue;
            seen.add(key);
            const hrefEl = row.querySelector('a[href]') || node.closest('a[href]');
            out.push({
              title,
              company_name: title.includes(',') ? title.split(',')[0].trim() : null,
              row_text: text,
              href: hrefEl ? hrefEl.getAttribute('href') : null,
            });
          }
          return out;
        }
        """
    )
    return rows or []


def open_row_and_extract(page, row: dict[str, Any]) -> dict[str, Any]:
    title = row.get("title") or ""
    candidates = [title, (row.get("row_text") or "").split("\n", 1)[0], (row.get("row_text") or "")[:120]]
    candidates = [c.strip() for c in candidates if c and c.strip()]

    opened = False
    for _ in range(80):
        for text in candidates:
            for selector in ["label", "div"]:
                loc = page.locator(selector, has_text=text).first
                if loc.count() == 0:
                    continue
                try:
                    loc.click(timeout=1200)
                    page.wait_for_timeout(350)
                    opened = True
                    break
                except Exception:
                    continue
            if opened:
                break
        if opened:
            break
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(100)

    if not opened:
        raise RuntimeError("Could not open transcript row on results page")

    body_text = page.locator("body").inner_text()
    raw_text = body_text.strip()
    source_url = page.url
    transcript_date = None
    date_match = re.search(r"([A-Z][a-z]+\s+\d{1,2}\s+'\d{2})", body_text)
    if date_match:
        transcript_date = parse_transcript_date(date_match.group(1))

    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
    except Exception:
        pass

    return {
        "raw_text": raw_text,
        "source_url": source_url,
        "transcript_date": transcript_date,
        "company_name": row.get("company_name"),
        "title": row.get("title"),
        "event_type": "Earnings Call",
        "participants": [],
        "speaker_blocks": [],
        "raw_html": None,
        "subtitle": None,
    }


def make_payload(row: dict[str, Any], extracted: dict[str, Any], round_no: int, position: int) -> dict[str, Any]:
    title = extracted.get("title") or row.get("title")
    company = extracted.get("company_name") or row.get("company_name")
    transcript_date = extracted.get("transcript_date")
    raw_text = extracted.get("raw_text") or ""
    digest = sha256_text("|".join([norm(company), norm(title), transcript_date or "", raw_text[:500]]))[:24]
    uid = f"r1000_{digest}"
    return {
        "transcript_uid": uid,
        "source": "koyfin_r1000_missing",
        "fetch_timestamp_utc": utc_now_iso(),
        "transcript_date": transcript_date,
        "company_name": company,
        "title": title,
        "event_type": extracted.get("event_type") or "Earnings Call",
        "source_url": extracted.get("source_url"),
        "participants": list(extracted.get("participants") or []),
        "speaker_blocks": list(extracted.get("speaker_blocks") or []),
        "raw_text": raw_text,
        "raw_html": extracted.get("raw_html"),
        "window_start": None,
        "window_end": None,
        "result_page": round_no,
        "result_position": position,
        "sha256_raw_text": sha256_text(raw_text),
        "metadata": {
            "subtitle": extracted.get("subtitle"),
            "scroll_round": round_no,
            "row_title": row.get("title"),
            "row_text": row.get("row_text"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape missing Russell 1000 earnings call transcripts from Koyfin")
    parser.add_argument("--existing-jsonl", required=True, help="Existing consolidated raw JSONL corpus")
    parser.add_argument("--storage-state-path", required=True, help="Playwright storage state JSON for an authenticated Koyfin session")
    parser.add_argument("--raw-output-dir", default="raw/r1000_missing_raw", help="Where new transcript JSON files are written")
    parser.add_argument("--checkpoint-path", default="state/r1000_missing_resume.json", help="Resume checkpoint JSON")
    parser.add_argument("--log-file", default="logs/r1000_missing.log", help="Log file path")
    parser.add_argument("--max-stagnant-rounds", type=int, default=25, help="Stop after this many no-progress scroll rounds")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run browser headless")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(Path(args.log_file))
    existing_keys = load_existing_keys(Path(args.existing_jsonl), logger)
    checkpoint_path = Path(args.checkpoint_path)
    state = load_checkpoint(checkpoint_path)
    seen_row_keys = set(tuple(x) for x in state.get("seen_row_keys", []))
    raw_output_dir = Path(args.raw_output_dir)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(storage_state=str(Path(args.storage_state_path)))
        page = context.new_page()
        goto_r1000_search(page, logger)

        scroller = page.locator(RESULTS_SCROLLER).first
        if scroller.count() == 0:
            raise RuntimeError("Could not locate transcript results scroller")

        rounds = int(state.get("scroll_round", 0))
        if rounds > 0:
            logger.info("Fast-forwarding %d prior rounds", rounds)
            for _ in range(rounds):
                scroller.evaluate("(el) => { el.scrollTop = el.scrollTop + 1200; }")
                page.wait_for_timeout(150)

        stagnant_rounds = 0
        while stagnant_rounds < args.max_stagnant_rounds:
            rows = [r for r in extract_visible_rows(page) if r.get("title") and r.get("title") != "Earnings Calls"]
            visible_titles = [str(r.get("title")) for r in rows]

            new_rows = []
            for row in rows:
                key = (norm(row.get("company_name")), norm(row.get("title")))
                if key in seen_row_keys:
                    continue
                seen_row_keys.add(key)
                new_rows.append(row)

            if new_rows:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1

            for idx, row in enumerate(new_rows, start=1):
                key = (norm(row.get("company_name")), norm(row.get("title")))
                if key in existing_keys:
                    state["skipped_existing"] = int(state.get("skipped_existing", 0)) + 1
                    continue
                try:
                    extracted = open_row_and_extract(page, row)
                    payload = make_payload(row, extracted, rounds, idx)
                    if persist_json(raw_output_dir, payload):
                        state["stored"] = int(state.get("stored", 0)) + 1
                        logger.info("Stored missing transcript: %s", row.get("title"))
                except Exception as exc:
                    state["errors"] = int(state.get("errors", 0)) + 1
                    logger.warning("Fetch failed for %s: %s", row.get("title"), exc)

            state["scroll_round"] = rounds
            state["seen_row_keys"] = [list(x) for x in seen_row_keys]
            state["last_visible_titles"] = visible_titles[-10:]
            save_checkpoint(checkpoint_path, state)

            prev_last = visible_titles[-1] if visible_titles else None
            scroller.evaluate("(el) => { el.scrollTop = el.scrollTop + Math.max(1200, el.clientHeight - 40); }")
            page.wait_for_timeout(1200)
            rows_after = [r for r in extract_visible_rows(page) if r.get("title") and r.get("title") != "Earnings Calls"]
            last_after = rows_after[-1].get("title") if rows_after else None
            rounds += 1
            state["scroll_round"] = rounds
            if last_after == prev_last:
                stagnant_rounds += 1
            logger.info(
                "Progress | round=%d | visible=%d | stored=%d | skipped_existing=%d | errors=%d | stagnant=%d",
                rounds,
                len(rows_after),
                int(state.get("stored", 0)),
                int(state.get("skipped_existing", 0)),
                int(state.get("errors", 0)),
                stagnant_rounds,
            )

        state["done"] = True
        save_checkpoint(checkpoint_path, state)
        context.storage_state(path=str(Path(args.storage_state_path)))
        context.close()
        browser.close()
        logger.info("Finished missing-only Russell 1000 scrape loop")


if __name__ == "__main__":
    main()
