"""Extract and standardise transcript metadata.

Derives: company, ticker, event_date, year, quarter, year_quarter, title.
Ticker is extracted from the title when not present in the raw data (Koyfin
transcripts do not include a ticker field).

DataFrame in → DataFrame out.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Mapping from company name → ticker built via yfinance from the 987 unique
# company names in the Koyfin JSONL.  Loaded once at import time.
_MAP_PATH = Path(__file__).parent / "company_ticker_map.json"
_COMPANY_TICKER_MAP: Dict[str, str] = (
    json.loads(_MAP_PATH.read_text(encoding="utf-8")) if _MAP_PATH.exists() else {}
)


def _parse_date_from_text(text: str) -> Optional[pd.Timestamp]:
    patterns = [
        r"\b(20\d{2}-\d{1,2}-\d{1,2})\b",
        r"\b(\d{1,2}/\d{1,2}/20\d{2})\b",
        r"\b([A-Za-z]+\s+\d{1,2},\s+20\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            dt = pd.to_datetime(m.group(1), errors="coerce")
            if pd.notna(dt):
                return dt
    return None


def _extract_quarter_from_title(title: str) -> Optional[int]:
    m = re.search(r"\bQ([1-4])\b", title, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _quarter_from_date(dt: Optional[pd.Timestamp]) -> Optional[int]:
    if dt is None or pd.isna(dt):
        return None
    return (dt.month - 1) // 3 + 1


def _extract_row(row: pd.Series) -> Dict[str, Any]:
    raw_text = str(row.get("raw_text", "") or "")
    header = "\n".join(raw_text.splitlines()[:20])
    title = row.get("title")
    company = row.get("company")
    ticker = row.get("ticker")

    # --- title ---
    if pd.isna(title) or not str(title).strip():
        first_line = next(
            (ln.strip() for ln in raw_text.splitlines() if ln.strip()), ""
        )
        title = first_line[:200] if first_line else None

    # --- company ---
    if pd.isna(company) or not str(company).strip():
        if isinstance(title, str) and "-" in title:
            company = title.split("-")[0].strip()

    # --- ticker ---
    if pd.isna(ticker) or not str(ticker).strip():
        # Try parenthesised ticker in header
        m = re.search(r"\(([A-Z]{1,6}(?:\.[A-Z])?)\)", header)
        if not m:
            m = re.search(
                r"\b(?:Ticker|Symbol)\s*[:\-]\s*([A-Z]{1,6}(?:\.[A-Z])?)\b",
                header,
                flags=re.IGNORECASE,
            )
        ticker = m.group(1) if m else None
    if (pd.isna(ticker) or not str(ticker).strip()) and company:
        ticker = _COMPANY_TICKER_MAP.get(str(company).strip())

    # --- event_date ---
    event_date = pd.to_datetime(row.get("derived_event_date"), errors="coerce")
    if pd.isna(event_date):
        event_date = pd.to_datetime(row.get("datetime"), errors="coerce")
    if pd.isna(event_date):
        event_date = pd.to_datetime(row.get("date"), errors="coerce")
    # Fallback: regex parse the raw date string (handles "Friday, January 3, 2020 11:00 AM"
    # from Koyfin parquet `subheader` column which pd.to_datetime can't parse directly).
    if pd.isna(event_date):
        for _field in ("date", "datetime", "derived_event_date"):
            _val = str(row.get(_field) or "")
            if _val.strip():
                _parsed = _parse_date_from_text(_val)
                if _parsed is not None:
                    event_date = _parsed
                    break
    if pd.isna(event_date):
        event_date = _parse_date_from_text(header)
    if pd.isna(event_date):
        event_date = pd.to_datetime(row.get("scraped_at"), errors="coerce")

    # --- year / quarter ---
    year = (
        int(event_date.year)
        if event_date is not None and pd.notna(event_date)
        else None
    )
    if year is None:
        y_m = re.search(r"\b(20\d{2})\b", header)
        year = int(y_m.group(1)) if y_m else None

    quarter = _extract_quarter_from_title(str(title or ""))
    if quarter is None:
        quarter = _extract_quarter_from_title(header)
    if quarter is None:
        quarter = _quarter_from_date(event_date)

    year_quarter = (
        f"{year}Q{quarter}" if year is not None and quarter is not None else None
    )

    return {
        "company": (
            str(company).strip() if company is not None and pd.notna(company) else None
        ),
        "ticker": (
            str(ticker).strip() if ticker is not None and pd.notna(ticker) else None
        ),
        "event_date": (
            event_date.strftime("%Y-%m-%d")
            if event_date is not None and pd.notna(event_date)
            else None
        ),
        "year": year,
        "quarter": quarter,
        "year_quarter": year_quarter,
        "title": str(title).strip() if title is not None and pd.notna(title) else None,
    }


def extract_metadata(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Populate / overwrite metadata columns from raw data heuristics."""
    df = df.copy()
    meta_rows = [_extract_row(row) for _, row in df.iterrows()]
    meta_df = pd.DataFrame(meta_rows, index=df.index)
    for col in meta_df.columns:
        df[col] = meta_df[col]

    missing_company = df["company"].isna().sum()
    missing_date = df["event_date"].isna().sum()
    if missing_company:
        logger.warning("Company still missing for %d transcripts", missing_company)
    if missing_date:
        logger.warning("Event date still missing for %d transcripts", missing_date)
    return df
