from __future__ import annotations

import re
from typing import Any

from .utils import html_to_text, maybe_parse_transcript_date, normalize_text

SPEAKER_LINE_RE = re.compile(
    r"^(?P<speaker>[A-Za-z][A-Za-z .,\-'&;/]+?)\s*(?P<role>Operator|Executive|Attendee)?\s*$",
    re.MULTILINE,
)


def validate_raw_text(raw_text: str, min_text_length: int) -> dict[str, Any]:
    text = normalize_text(raw_text)
    flags = {
        "non_empty": bool(text),
        "min_length_ok": len(text) >= min_text_length,
        "ui_junk": False,
    }
    junk = ["upgrade now", "trial access", "help center", "log in", "sign in"]
    lowered = text.lower()
    if any(marker in lowered for marker in junk):
        flags["ui_junk"] = True
    flags["ok"] = bool(flags["non_empty"] and flags["min_length_ok"] and not flags["ui_junk"])
    return flags


def extract_text_from_http_payload(content_type: str, body_text: str) -> tuple[str, str | None]:
    lowered = (content_type or "").lower()
    raw_html = None
    if "html" in lowered or body_text.lstrip().startswith("<"):
        raw_html = body_text
        raw_text = html_to_text(body_text)
        return raw_text, raw_html
    raw_text = normalize_text(body_text)
    return raw_text, raw_html


def parse_header_lines(lines: list[str]) -> dict[str, Any]:
    clean = [normalize_text(line) for line in lines if normalize_text(line)]
    company_name = clean[0] if clean else None
    title = clean[1] if len(clean) > 1 else (clean[0] if clean else None)
    subtitle = clean[2] if len(clean) > 2 else None
    event_type = "Earnings Call" if (title and "earnings call" in title.lower()) else None
    transcript_date = maybe_parse_transcript_date(subtitle or "") if subtitle else None
    return {
        "company_name": company_name,
        "title": title,
        "subtitle": subtitle,
        "event_type": event_type,
        "transcript_date": transcript_date,
    }


def parse_speaker_blocks(raw_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in raw_blocks:
        speaker = (block.get("speaker") or "").strip() or None
        text = normalize_text(str(block.get("text") or ""))
        if not text:
            continue
        role = (block.get("role") or "").strip() or None
        out.append({"speaker": speaker, "role": role, "text": text})
    return out
