#!/usr/bin/env python3
"""Post-preprocessing dataset checks.

Reads the 4 parquet outputs and prints a health report:
shape, nulls, token stats, section distribution, duplicates, referential integrity.

Usage:
    uv run python src/run_dataset_checks.py [--data-dir data/clean]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ANSI helpers
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> str:
    return f"  {GREEN}✓{RESET} {msg}"


def warn(msg: str) -> str:
    return f"  {YELLOW}⚠{RESET} {msg}"


def fail(msg: str) -> str:
    return f"  {RED}✗{RESET} {msg}"


def header(title: str) -> str:
    return f"\n{BOLD}{'─'*60}\n  {title}\n{'─'*60}{RESET}"


def run_checks(data_dir: Path) -> int:
    issues = 0

    # ── Load ──────────────────────────────────────────────────
    files = {
        "chunks": data_dir / "chunks.parquet",
        "segments": data_dir / "segments.parquet",
        "transcripts": data_dir / "transcripts_deduplicated.parquet",
        "duplicates": data_dir / "duplicates_audit.parquet",
    }

    dfs: dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        if not path.exists():
            print(fail(f"{path} NOT FOUND"))
            issues += 1
            continue
        dfs[name] = pq.read_table(path).to_pandas()

    if len(dfs) < 4:
        print(fail("Missing parquet files — aborting"))
        return issues

    chunks = dfs["chunks"]
    segments = dfs["segments"]
    transcripts = dfs["transcripts"]
    duplicates = dfs["duplicates"]

    # ── 1. Shape ──────────────────────────────────────────────
    print(header("1. Shape"))
    for name, df in dfs.items():
        print(ok(f"{name:25s}  {df.shape[0]:>9,} rows × {df.shape[1]:>2} cols"))

    # ── 2. Null check ─────────────────────────────────────────
    print(header("2. Null check (critical columns)"))
    critical = {
        "chunks": [
            "transcript_id",
            "segment_id",
            "chunk_id",
            "chunk_text",
            "n_tokens",
            "section",
            "speaker_type",
        ],
        "segments": [
            "transcript_id",
            "segment_id",
            "segment_text",
            "section",
            "speaker_type",
        ],
        "transcripts": ["transcript_id", "company", "clean_text"],
    }
    for name, cols in critical.items():
        df = dfs[name]
        for col in cols:
            if col not in df.columns:
                print(warn(f"{name}.{col} — column missing"))
                issues += 1
                continue
            n_null = df[col].isna().sum()
            if n_null > 0:
                print(
                    warn(f"{name}.{col} — {n_null:,} nulls ({100*n_null/len(df):.1f}%)")
                )
                issues += 1
            else:
                print(ok(f"{name}.{col} — 0 nulls"))

    # ── 3. Token stats ────────────────────────────────────────
    print(header("3. Token stats (chunks)"))
    max_tokens = 200
    desc = chunks["n_tokens"].describe()
    print(ok(f"count   = {int(desc['count']):>10,}"))
    print(ok(f"mean    = {desc['mean']:>10.1f}"))
    print(ok(f"median  = {desc['50%']:>10.0f}"))
    print(ok(f"p95     = {chunks['n_tokens'].quantile(0.95):>10.0f}"))
    print(ok(f"max     = {desc['max']:>10.0f}"))
    violations = (chunks["n_tokens"] > max_tokens).sum()
    if violations:
        print(fail(f"{violations:,} chunks exceed {max_tokens} tokens"))
        issues += 1
    else:
        print(ok(f"0 chunks exceed {max_tokens} tokens"))

    # ── 4. Section distribution (segments) ────────────────────
    print(header("4. Section distribution"))
    sec_counts = segments["section"].value_counts()
    total_seg = len(segments)
    for sec in ["Prepared", "Q", "A", "O"]:
        n = sec_counts.get(sec, 0)
        pct = 100 * n / total_seg
        print(ok(f"{sec:>8s}  {n:>9,} segments  ({pct:5.1f}%)"))

    # Text-volume based distribution
    segments_with_len = segments.copy()
    segments_with_len["_len"] = segments_with_len["segment_text"].str.len()
    vol = segments_with_len.groupby("section")["_len"].sum()
    total_vol = vol.sum()
    print()
    for sec in ["Prepared", "Q", "A", "O"]:
        v = vol.get(sec, 0)
        pct = 100 * v / total_vol
        print(ok(f"{sec:>8s}  {v:>12,} chars  ({pct:5.1f}% text volume)"))

    # ── 5. Speaker types ──────────────────────────────────────
    print(header("5. Speaker types"))
    st_counts = segments["speaker_type"].value_counts()
    for st, n in st_counts.items():
        print(ok(f"{st:>12s}  {n:>9,}"))

    # ── 6. Temporal coverage ──────────────────────────────────
    print(header("6. Temporal coverage"))
    if "year_quarter" in transcripts.columns:
        yq = transcripts["year_quarter"].dropna()
        if len(yq) > 0:
            print(ok(f"Range: {yq.min()} → {yq.max()}"))
            print(ok(f"Unique year_quarters: {yq.nunique()}"))
        else:
            print(warn("No year_quarter values"))
    if "year" in transcripts.columns:
        yr = transcripts["year"].dropna()
        if len(yr) > 0:
            counts_per_year = transcripts.groupby("year").size()
            print(
                ok(
                    f"Transcripts per year (min/max): {counts_per_year.min()} / {counts_per_year.max()}"
                )
            )

    # ── 7. Company coverage ───────────────────────────────────
    print(header("7. Company coverage"))
    n_companies = transcripts["company"].nunique()
    print(ok(f"Unique companies: {n_companies:,}"))
    ticker_fill = transcripts["ticker"].notna().sum()
    print(
        ok(
            f"Ticker fill rate: {ticker_fill:,}/{len(transcripts):,} ({100*ticker_fill/len(transcripts):.1f}%)"
        )
    )

    # ── 8. Duplicates audit ───────────────────────────────────
    print(header("8. Duplicates audit"))
    n_dup = (
        duplicates["is_duplicate"].sum() if "is_duplicate" in duplicates.columns else 0
    )
    print(ok(f"Flagged duplicates: {n_dup:,}"))
    if n_dup > 0 and "duplicate_reason" in duplicates.columns:
        reasons = duplicates[duplicates["is_duplicate"]][
            "duplicate_reason"
        ].value_counts()
        for reason, cnt in reasons.items():
            print(ok(f"  {reason}: {cnt:,}"))

    # ── 9. Referential integrity ──────────────────────────────
    print(header("9. Referential integrity"))

    # All chunk transcript_ids should exist in segments
    chunk_tids = set(chunks["transcript_id"].unique())
    seg_tids = set(segments["transcript_id"].unique())
    trans_tids = set(transcripts["transcript_id"].unique())

    orphan_chunks = chunk_tids - seg_tids
    if orphan_chunks:
        print(fail(f"{len(orphan_chunks)} chunk transcript_ids not in segments"))
        issues += 1
    else:
        print(ok("All chunk transcript_ids exist in segments"))

    orphan_segs = seg_tids - trans_tids
    if orphan_segs:
        print(fail(f"{len(orphan_segs)} segment transcript_ids not in transcripts"))
        issues += 1
    else:
        print(ok("All segment transcript_ids exist in transcripts"))

    # All segment_ids in chunks should exist in segments
    chunk_sids = set(chunks["segment_id"].unique())
    seg_sids = set(segments["segment_id"].unique())
    orphan_sids = chunk_sids - seg_sids
    if orphan_sids:
        print(fail(f"{len(orphan_sids)} chunk segment_ids not in segments"))
        issues += 1
    else:
        print(ok("All chunk segment_ids exist in segments"))

    # ── 10. Unique IDs ────────────────────────────────────────
    print(header("10. ID uniqueness"))
    for name, col in [
        ("transcripts", "transcript_id"),
        ("segments", "segment_id"),
        ("chunks", "chunk_id"),
    ]:
        df = dfs[name]
        n_total = len(df)
        n_unique = df[col].nunique()
        if n_total == n_unique:
            print(ok(f"{name}.{col} — all unique ({n_unique:,})"))
        else:
            print(fail(f"{name}.{col} — {n_total - n_unique:,} duplicates"))
            issues += 1

    # ── Summary ───────────────────────────────────────────────
    print(header("Summary"))
    if issues == 0:
        print(ok("All checks passed"))
    else:
        print(warn(f"{issues} issue(s) found — review above"))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset health checks")
    parser.add_argument(
        "--data-dir", default="data/clean", help="Path to parquet files"
    )
    args = parser.parse_args()
    issues = run_checks(Path(args.data_dir))
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
