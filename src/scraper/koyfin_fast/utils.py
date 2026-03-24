from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import DateWindow


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> str:
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", no_script, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", no_style)
    text = re.sub(r"\s+", " ", text)
    return normalize_text(text)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_transcript_uid(
    *,
    title: str | None,
    company_name: str | None,
    source_url: str | None,
    dom_key: str | None,
    result_page: int,
    result_position: int,
    window_start: str,
) -> str:
    parts = [
        (title or "").strip().lower(),
        (company_name or "").strip().lower(),
        (source_url or "").strip(),
        (dom_key or "").strip(),
        str(result_page),
        str(result_position),
        window_start,
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"koyfin_{digest}"


def generate_windows(start_date: date, end_date: date, window_days: int) -> list[DateWindow]:
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    windows: list[DateWindow] = []
    cursor = start_date
    while cursor <= end_date:
        end = min(end_date, cursor + timedelta(days=window_days - 1))
        windows.append(DateWindow(start=cursor, end=end))
        cursor = end + timedelta(days=1)
    return windows


def adapt_window_days(current_days: int, rows_seen: int, had_error: bool, min_days: int = 1, max_days: int = 30) -> int:
    if had_error or rows_seen > 700:
        return max(min_days, max(1, current_days // 2))
    if rows_seen < 80:
        return min(max_days, current_days + 1)
    return current_days


def maybe_parse_transcript_date(text: str) -> str | None:
    s = text.strip()
    for fmt in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%A, %B %d, %Y %I:%M %p",
        "%b %d '%y",
    ):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass
    m = re.search(r"([A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%A, %B %d, %Y %I:%M %p").date().isoformat()
        except Exception:
            return None
    return None
