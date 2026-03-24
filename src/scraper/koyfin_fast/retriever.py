from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from .models import ManifestRow
from .parser import extract_text_from_http_payload, validate_raw_text
from .transcript_page import fetch_transcript_via_browser
from .utils import normalize_text


def playwright_cookies_to_dict(cookies: list[dict[str, Any]]) -> dict[str, str]:
    return {str(c["name"]): str(c["value"]) for c in cookies if c.get("name") and c.get("value")}


def fetch_transcript_via_http(
    manifest: ManifestRow,
    *,
    cookie_jar: dict[str, str],
    headers: dict[str, str],
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    if not manifest.source_url:
        raise RuntimeError("Manifest has no source_url for HTTP retrieval")

    with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers, cookies=cookie_jar) as client:
        resp = client.get(manifest.source_url)
        resp.raise_for_status()
        raw_text, raw_html = extract_text_from_http_payload(resp.headers.get("content-type", ""), resp.text)
        return {
            "raw_text": normalize_text(raw_text),
            "raw_html": raw_html,
            "participants": [],
            "speaker_blocks": [],
            "company_name": manifest.company_name,
            "title": manifest.title,
            "event_type": "Earnings Call",
            "transcript_date": None,
            "subtitle": None,
            "source_url": manifest.source_url,
        }


def fetch_many_via_http(
    manifests: list[ManifestRow],
    *,
    cookie_jar: dict[str, str],
    headers: dict[str, str],
    workers: int,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not manifests:
        return out

    workers = max(1, workers)

    def _job(manifest: ManifestRow) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        try:
            payload = fetch_transcript_via_http(manifest, cookie_jar=cookie_jar, headers=headers)
            payload["_latency_s"] = max(0.0, time.perf_counter() - started)
            payload["_ok"] = True
            return manifest.transcript_uid, payload
        except Exception as exc:  # noqa: BLE001
            return manifest.transcript_uid, {"_ok": False, "_error": str(exc), "_latency_s": max(0.0, time.perf_counter() - started)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, m) for m in manifests]
        for fut in as_completed(futures):
            uid, payload = fut.result()
            out[uid] = payload

    logger.info("HTTP batch retrieval finished", extra={"event": "http_batch", "count": str(len(out))})
    return out


def fetch_transcript_via_browser_hybrid(
    manifest: ManifestRow,
    *,
    results_page,
    detail_page,
    logger: logging.Logger,
) -> dict[str, Any]:
    return fetch_transcript_via_browser(
        results_page=results_page,
        detail_page=detail_page,
        title=manifest.title,
        row_text=manifest.row_text,
        source_url=manifest.source_url,
        use_detail_page=True,
        logger=logger,
    )


def fetch_transcript_via_browser_only(
    manifest: ManifestRow,
    *,
    results_page,
    detail_page,
    logger: logging.Logger,
) -> dict[str, Any]:
    return fetch_transcript_via_browser(
        results_page=results_page,
        detail_page=detail_page,
        title=manifest.title,
        row_text=manifest.row_text,
        source_url=manifest.source_url,
        use_detail_page=False,
        logger=logger,
    )


def payload_is_valid(payload: dict[str, Any], min_text_length: int) -> dict[str, Any]:
    return validate_raw_text(payload.get("raw_text") or "", min_text_length)
