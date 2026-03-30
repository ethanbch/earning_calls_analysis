"""Conservative deduplication with audit trail.

Three rules (applied in order):
1. Exact duplicate ``raw_text``.
2. Same company + event_date with near-identical title (SequenceMatcher ≥ 0.96).
3. Same company + year_quarter with ≤ 1 % length difference and identical 250-char prefix.

Returns (all_df_with_flags, duplicates_only_df).
DataFrame in → DataFrame out.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Tuple

import pandas as pd


def deduplicate(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flag duplicates and return (full_df, duplicates_audit_df)."""
    if df.empty:
        out = df.copy()
        out["is_duplicate"] = False
        out["duplicate_reason"] = ""
        return out, out.iloc[0:0].copy()

    out = df.copy()
    out["is_duplicate"] = False
    out["duplicate_reason"] = ""

    # Rule 1 – exact raw_text
    exact_dup = out.duplicated(subset=["raw_text"], keep="first")
    out.loc[exact_dup, "is_duplicate"] = True
    out.loc[exact_dup, "duplicate_reason"] = "exact_raw_text"

    # Rule 2 – same company + date + highly similar title
    out["_title_norm"] = (
        out["title"].fillna("").str.lower().str.replace(r"\W+", "", regex=True)
    )
    for _, idxs in out.groupby(["company", "event_date"], dropna=False).groups.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue
        for i in range(1, len(idxs)):
            if out.at[idxs[i], "is_duplicate"]:
                continue
            t_curr = out.at[idxs[i], "_title_norm"]
            for j in range(i):
                t_prev = out.at[idxs[j], "_title_norm"]
                if not t_curr or not t_prev:
                    continue
                if SequenceMatcher(None, t_curr, t_prev).ratio() >= 0.96:
                    out.at[idxs[i], "is_duplicate"] = True
                    _append_reason(out, idxs[i], "same_company_date_similar_title")
                    break

    # Rule 3 – same company + quarter + similar length + same prefix
    out["_clean_len"] = out["clean_text"].fillna("").str.len()
    out["_prefix"] = (
        out["clean_text"]
        .fillna("")
        .str[:250]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    for _, idxs in out.groupby(
        ["company", "year_quarter"], dropna=False
    ).groups.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue
        for i in range(1, len(idxs)):
            if out.at[idxs[i], "is_duplicate"]:
                continue
            for j in range(i):
                if out.at[idxs[j], "is_duplicate"]:
                    continue
                len_i = out.at[idxs[i], "_clean_len"]
                len_j = out.at[idxs[j], "_clean_len"]
                if min(len_i, len_j) == 0:
                    continue
                if (
                    abs(len_i - len_j) / max(len_i, len_j) <= 0.01
                    and out.at[idxs[i], "_prefix"] == out.at[idxs[j], "_prefix"]
                ):
                    out.at[idxs[i], "is_duplicate"] = True
                    _append_reason(
                        out, idxs[i], "same_company_quarter_similar_len_prefix"
                    )
                    break

    duplicates = out[out["is_duplicate"]].copy()
    logger.info("Duplicates flagged: %d / %d", len(duplicates), len(out))

    # Drop temporary columns
    out = out.drop(columns=["_title_norm", "_clean_len", "_prefix"], errors="ignore")
    duplicates = duplicates.drop(
        columns=["_title_norm", "_clean_len", "_prefix"], errors="ignore"
    )
    return out, duplicates


def _append_reason(df: pd.DataFrame, idx: int, reason: str) -> None:
    existing = df.at[idx, "duplicate_reason"]
    df.at[idx, "duplicate_reason"] = f"{existing};{reason}" if existing else reason
