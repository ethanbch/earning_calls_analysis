from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str((Path(__file__).resolve().parent / 'src')))

from koyfin_fast.models import TranscriptRecord
from koyfin_fast.search_page import extract_rows_bulk, filter_earnings_call_rows
from koyfin_fast.storage import Storage
from koyfin_fast.retriever import fetch_transcript_via_browser_only
from koyfin_fast.utils import ensure_dir, sha256_text, utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / 'data' / 'state' / 'playwright_storage_state.json'
EXISTING_JSONL = ROOT / 'koyfin_transcripts_full_2006_2026.jsonl'
RAW_DIR = ROOT / 'data' / 'raw' / 'koyfin_r1000_missing'
PROCESSED_DIR = ROOT / 'data' / 'processed_r1000_missing'
CHECKPOINT_PATH = ROOT / 'data' / 'state' / 'r1000_missing_resume.json'
LOG_PATH = ROOT / 'data' / 'logs' / 'r1000_missing.log'
RESULTS_SCROLLER = "div.news-virtual-list__newsVirtualList__container___a0EHh"


def setup_logger() -> logging.Logger:
    ensure_dir(LOG_PATH.parent)
    logger = logging.getLogger('r1000_missing')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def norm(value: str | None) -> str:
    return re.sub(r'\s+', ' ', (value or '').strip().lower())


def load_existing_keys(logger: logging.Logger) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    with EXISTING_JSONL.open('r', encoding='utf-8', errors='ignore') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            company = norm(row.get('company') or row.get('company_name'))
            title = norm(row.get('title'))
            if company and title:
                keys.add((company, title))
    logger.info('Loaded existing corpus keys: %d', len(keys))
    return keys


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding='utf-8'))
    return {
        'scroll_round': 0,
        'seen_row_keys': [],
        'stored': 0,
        'skipped_existing': 0,
        'errors': 0,
        'last_visible_titles': [],
        'done': False,
    }


def save_checkpoint(state: dict[str, Any]) -> None:
    ensure_dir(CHECKPOINT_PATH.parent)
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2), encoding='utf-8')


def goto_r1000_search(page, logger: logging.Logger) -> None:
    page.goto('https://app.koyfin.com', wait_until='domcontentloaded', timeout=45000)
    page.wait_for_timeout(2500)
    page.locator('text=Advanced Search').first.click(force=True)
    page.wait_for_timeout(700)
    page.locator('text=Transcripts Search').last.click(force=True)
    page.wait_for_timeout(3000)
    body = page.locator('body').inner_text()
    if 'Russell 1000 (Large-Cap)' not in body:
        raise RuntimeError('Russell 1000 filter is not active in saved search state')
    if 'Showing most recent results of' not in body:
        raise RuntimeError('Could not confirm transcript search results are visible')
    logger.info('Loaded Russell 1000 transcript search surface')


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (norm(row.get('company_name')), norm(row.get('title')))


def make_record(payload: dict[str, Any], row: dict[str, Any], scroll_round: int, pos: int) -> TranscriptRecord:
    title = payload.get('title') or row.get('title')
    company = payload.get('company_name') or row.get('company_name')
    transcript_date = payload.get('transcript_date')
    raw_text = str(payload.get('raw_text') or '')
    digest = sha256_text('|'.join([norm(company), norm(title), transcript_date or '', raw_text[:500]]))[:24]
    uid = f'r1000_{digest}'
    return TranscriptRecord(
        transcript_uid=uid,
        source='koyfin_r1000_missing',
        fetch_timestamp_utc=utc_now_iso(),
        transcript_date=transcript_date,
        company_name=company,
        title=title,
        event_type=payload.get('event_type') or 'Earnings Call',
        source_url=payload.get('source_url'),
        participants=list(payload.get('participants') or []),
        speaker_blocks=list(payload.get('speaker_blocks') or []),
        raw_text=raw_text,
        raw_html=payload.get('raw_html'),
        window_start='2006-03-22',
        window_end='2026-03-22',
        result_page=scroll_round,
        result_position=pos,
        sha256_raw_text=sha256_text(raw_text),
        metadata={
            'subtitle': payload.get('subtitle'),
            'scroll_round': scroll_round,
            'row_title': row.get('title'),
            'row_text': row.get('row_text'),
            'company_name_row': row.get('company_name'),
        },
    )


def main() -> None:
    logger = setup_logger()
    existing_keys = load_existing_keys(logger)
    state = load_checkpoint()
    seen_row_keys = set(tuple(x) for x in state.get('seen_row_keys', []))
    storage = Storage(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, logger=logger)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()
        detail_page = context.new_page()

        goto_r1000_search(page, logger)
        scroller = page.locator(RESULTS_SCROLLER).first
        if scroller.count() == 0:
            raise RuntimeError('Could not locate results scroller')

        # Fast-forward scroll position on resume.
        rounds = int(state.get('scroll_round', 0))
        if rounds > 0:
            logger.info('Fast-forwarding scroll state for %d prior rounds', rounds)
            for _ in range(rounds):
                scroller.evaluate('(el) => { el.scrollTop = el.scrollTop + 1600; }')
                page.wait_for_timeout(150)

        stagnant_rounds = 0
        while stagnant_rounds < 25:
            rows = [r for r in filter_earnings_call_rows(extract_rows_bulk(page, 1)) if r.get('title') and r.get('title') != 'Earnings Calls']
            visible_titles = [str(r.get('title')) for r in rows]
            new_rows = []
            for r in rows:
                k = row_key(r)
                if k in seen_row_keys:
                    continue
                seen_row_keys.add(k)
                new_rows.append(r)

            if new_rows:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1

            for idx, row in enumerate(new_rows, start=1):
                k = row_key(row)
                if k in existing_keys:
                    state['skipped_existing'] = int(state.get('skipped_existing', 0)) + 1
                    continue
                try:
                    payload = fetch_transcript_via_browser_only(
                        type('Manifest', (), {
                            'title': row.get('title'),
                            'row_text': row.get('row_text'),
                            'source_url': row.get('href'),
                        })(),
                        results_page=page,
                        detail_page=detail_page,
                        logger=logger,
                    )
                    record = make_record(payload, row, rounds, idx)
                    persisted = storage.persist_raw_transcript(record)
                    if persisted.stored:
                        state['stored'] = int(state.get('stored', 0)) + 1
                        logger.info('Stored missing transcript: %s', row.get('title'))
                    else:
                        logger.info('Skipped persist: %s (%s)', row.get('title'), persisted.reason)
                except Exception as exc:
                    state['errors'] = int(state.get('errors', 0)) + 1
                    logger.warning('Fetch failed for %s: %s', row.get('title'), exc)

            state['scroll_round'] = rounds
            state['seen_row_keys'] = [list(x) for x in seen_row_keys]
            state['last_visible_titles'] = visible_titles[-10:]
            save_checkpoint(state)
            storage.export_parquets(type('StateShim', (), {
                'fetch_manifest_export_rows': staticmethod(lambda: []),
                'fetch_index_export_rows': staticmethod(lambda: []),
            })())

            prev_last = visible_titles[-1] if visible_titles else None
            scroller.evaluate('(el) => { el.scrollTop = el.scrollTop + Math.max(1200, el.clientHeight - 40); }')
            page.wait_for_timeout(1200)
            rows_after = [r for r in filter_earnings_call_rows(extract_rows_bulk(page, 1)) if r.get('title') and r.get('title') != 'Earnings Calls']
            last_after = rows_after[-1].get('title') if rows_after else None
            rounds += 1
            state['scroll_round'] = rounds
            if last_after == prev_last:
                stagnant_rounds += 1
            logger.info(
                'Progress | round=%d | visible=%d | stored=%d | skipped_existing=%d | errors=%d | stagnant=%d',
                rounds,
                len(rows_after),
                int(state.get('stored', 0)),
                int(state.get('skipped_existing', 0)),
                int(state.get('errors', 0)),
                stagnant_rounds,
            )

        state['done'] = True
        save_checkpoint(state)
        context.storage_state(path=str(STATE_PATH))
        context.close()
        browser.close()
        logger.info('Finished missing-only Russell 1000 scrape loop')


if __name__ == '__main__':
    main()
