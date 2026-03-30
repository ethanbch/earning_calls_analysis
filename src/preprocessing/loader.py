"""Streaming JSONL loader with schema inference.

Reads a large JSONL file in configurable batches and yields DataFrames.
Never loads the entire file into memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd

# Candidate column names used for automatic schema mapping.
# Parquet (R2K/SP) equivalents: body→text, transcript_subheader→company,
# subheader→date, speakers→participants.
SCHEMA_MAP = {
    "text": ["transcript_text", "raw_text", "transcript", "text", "content", "body"],
    "company": ["company", "company_name", "transcript_subheader"],
    "ticker": ["ticker", "symbol"],
    "date": ["derived_event_date", "datetime", "date", "event_date", "subheader"],
    "title": ["title"],
    "participants": ["participants", "speakers"],
}


def _normalize_col(col: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def _resolve_column(df_columns: List[str], candidates: List[str]) -> str | None:
    """Return the first matching column name from *candidates* that exists in *df_columns*."""
    for c in candidates:
        if c in df_columns:
            return c
    return None


def _infer_text_column(df: pd.DataFrame) -> str | None:
    """Heuristic: pick the object column with the longest average string length."""
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    best, best_score = None, -1.0
    for c in object_cols:
        non_null = df[c].dropna()
        if non_null.empty:
            continue
        score = float(non_null.astype(str).str.len().mean())
        if score > best_score:
            best_score, best = score, c
    return best if best_score >= 80 else None


def _generate_transcript_id(row: pd.Series, global_idx: int, source_file: str) -> str:
    basis = f"{source_file}|{str(row.get('raw_text', ''))}|{global_idx}"
    return hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()[:16]


def apply_schema(
    df: pd.DataFrame,
    source_file: str,
    logger: logging.Logger,
    line_offset: int = 0,
) -> pd.DataFrame:
    """Standardise column names and ensure all required columns exist.

    Returns a new DataFrame with canonical columns:
        raw_text, company, ticker, title, derived_event_date, datetime,
        participants, source_file, transcript_id

    Parameters
    ----------
    line_offset : int
        Global line number of the first record in this batch.  Used to
        generate deterministic, globally-unique transcript IDs.
    """
    df = df.copy()
    # Normalise raw column names
    rename = {c: _normalize_col(c) for c in df.columns}
    df = df.rename(columns=rename)

    # --- raw_text ---
    text_col = _resolve_column(list(df.columns), SCHEMA_MAP["text"])
    if text_col is None:
        text_col = _infer_text_column(df)
        if text_col:
            logger.info("Inferred text column from heuristic: %s", text_col)
    if text_col is None:
        logger.warning("No text column found – creating empty raw_text.")
        df["raw_text"] = ""
    elif text_col != "raw_text":
        df["raw_text"] = df[text_col].astype(str)

    # --- other canonical columns ---
    for canon, candidates in SCHEMA_MAP.items():
        if canon == "text":
            continue
        col = _resolve_column(list(df.columns), candidates)
        if col and col != canon:
            df[canon] = df[col]
        if canon not in df.columns:
            df[canon] = np.nan

    df["source_file"] = source_file

    # --- transcript_id ---
    ids = [
        _generate_transcript_id(row, line_offset + idx, source_file)
        for idx, (_, row) in enumerate(df.iterrows())
    ]
    df["transcript_id"] = ids

    return df


def iter_batches_parquet(
    path: Path,
    batch_size: int,
    logger: logging.Logger,
    start_row: int = 0,
) -> Generator[Tuple[pd.DataFrame, int], None, None]:
    """Yield (batch_df, rows_yielded_so_far) from a Parquet directory or single file.

    Parameters
    ----------
    path :
        Directory containing ``*.parquet`` files, or a single ``.parquet`` file.
    batch_size : int
        Records per batch.
    start_row : int
        Number of rows to skip at the start (for resume).
    """
    import pyarrow.parquet as pq

    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        logger.warning("No parquet files found in %s", path)
        return

    all_df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    if start_row > 0:
        all_df = all_df.iloc[start_row:].reset_index(drop=True)

    total = len(all_df)
    for start in range(0, total, batch_size):
        batch = all_df.iloc[start : start + batch_size].copy()
        rows_done = start_row + start + len(batch)
        yield apply_schema(
            batch, str(path), logger, line_offset=start_row + start
        ), rows_done


def iter_batches(
    path: Path,
    batch_size: int,
    logger: logging.Logger,
    start_line: int = 0,
) -> Generator[Tuple[pd.DataFrame, int], None, None]:
    """Yield (batch_df, last_line_read) tuples from a JSONL file.

    Parameters
    ----------
    path : Path
        JSONL file to read.
    batch_size : int
        Number of records per batch.
    logger : logging.Logger
    start_line : int
        1-based line to resume from (0 = start).
    """
    records: List[Dict[str, Any]] = []
    last_line = start_line
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            if i <= start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Skipping malformed JSONL line %d in %s", i, path)
                continue
            if isinstance(obj, dict):
                records.append(obj)
                last_line = i
            if len(records) >= batch_size:
                batch_df = pd.DataFrame(records)
                batch_start = last_line - len(records) + 1
                yield apply_schema(
                    batch_df, str(path), logger, line_offset=batch_start
                ), last_line
                records = []
    if records:
        batch_df = pd.DataFrame(records)
        batch_start = last_line - len(records) + 1
        yield apply_schema(
            batch_df, str(path), logger, line_offset=batch_start
        ), last_line
