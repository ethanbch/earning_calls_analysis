"""Remove scraping artifacts from transcript text.

Operates on the ``raw_text`` column and produces ``clean_text``.
Pure function: DataFrame in → DataFrame out.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd


def _clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return "" if pd.isna(text) else str(text)

    t = text.replace("\x00", " ")
    # Strip control characters (keep \n and \t)
    t = re.sub(r"[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]", " ", t)
    # Strip residual HTML tags
    t = re.sub(r"<[^>]+>", " ", t)
    # Remove common Koyfin scraping artifacts
    t = re.sub(r"\bView Summary\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bClick here for webcast\b", " ", t, flags=re.IGNORECASE)
    # Normalise line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []
    prev = None
    for line in t.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            cleaned_lines.append("")
            prev = None
            continue
        # Drop accidental consecutive duplicate header lines
        if prev and line == prev and len(line) < 140:
            continue
        cleaned_lines.append(line)
        prev = line

    t = "\n".join(cleaned_lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def clean(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Add ``clean_text`` column with scraping artifacts removed."""
    df = df.copy()
    df["clean_text"] = df["raw_text"].map(_clean_text)
    n_empty = (df["clean_text"].str.strip() == "").sum()
    if n_empty:
        logger.warning("Cleaning produced %d empty transcripts", n_empty)
    return df
