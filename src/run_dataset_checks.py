#!/usr/bin/env python3
"""Post-preprocessing dataset checks.

Default mode is memory-safe (`quick`) for laptop-class machines.

Usage:
    uv run python src/run_dataset_checks.py --data-dir outputs/r1000/clean
    uv run python src/run_dataset_checks.py --data-dir outputs/r1000/clean --mode full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

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


def _read_sample(
    path: Path, columns: list[str], sample_rows: int, batch_size: int = 50_000
) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    parts: list[pd.DataFrame] = []
    seen = 0
    for rb in pf.iter_batches(batch_size=batch_size, columns=columns):
        chunk = rb.to_pandas()
        parts.append(chunk)
        seen += len(chunk)
        if seen >= sample_rows:
            break
    if not parts:
        return pd.DataFrame(columns=columns)
    df = pd.concat(parts, ignore_index=True)
    return df.head(sample_rows)


def _load_full(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "chunks": data_dir / "chunks.parquet",
        "segments": data_dir / "segments.parquet",
        "transcripts": data_dir / "transcripts_deduplicated.parquet",
        "duplicates": data_dir / "duplicates_audit.parquet",
    }
    dfs: dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        if not path.exists():
            continue
        dfs[name] = pq.read_table(path).to_pandas()
    return dfs


def _load_quick(
    data_dir: Path, sample_rows: int
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    files = {
        "chunks": data_dir / "chunks.parquet",
        "segments": data_dir / "segments.parquet",
        "transcripts": data_dir / "transcripts_deduplicated.parquet",
        "duplicates": data_dir / "duplicates_audit.parquet",
    }

    shapes: dict[str, tuple[int, int]] = {}
    dfs: dict[str, pd.DataFrame] = {}
    sample_cols = {
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
        "transcripts": [
            "transcript_id",
            "company",
            "ticker",
            "clean_text",
            "year",
            "year_quarter",
        ],
        "duplicates": ["is_duplicate", "duplicate_reason"],
    }

    for name, path in files.items():
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        shapes[name] = (pf.metadata.num_rows, pf.metadata.num_columns)
        cols = [c for c in sample_cols[name] if c in pf.schema.names]
        dfs[name] = _read_sample(path, cols, sample_rows=sample_rows)

    return dfs, shapes


def run_checks(data_dir: Path, mode: str, sample_rows: int) -> int:
    issues = 0
    files = {
        "chunks": data_dir / "chunks.parquet",
        "segments": data_dir / "segments.parquet",
        "transcripts": data_dir / "transcripts_deduplicated.parquet",
        "duplicates": data_dir / "duplicates_audit.parquet",
    }

    print(header(f"Dataset checks ({mode}) — {data_dir}"))
    for name, path in files.items():
        if not path.exists():
            print(fail(f"{path} NOT FOUND"))
            issues += 1

    if issues:
        return issues

    if mode == "full":
        dfs = _load_full(data_dir)
        shapes = {k: (len(v), v.shape[1]) for k, v in dfs.items()}
    else:
        dfs, shapes = _load_quick(data_dir, sample_rows=sample_rows)
        print(
            warn(f"Quick mode uses first {sample_rows:,} rows/file for content checks")
        )

    chunks = dfs["chunks"]
    segments = dfs["segments"]
    transcripts = dfs["transcripts"]
    duplicates = dfs["duplicates"]

    print(header("1. Shape"))
    for name, (n_rows, n_cols) in shapes.items():
        print(ok(f"{name:25s}  {n_rows:>9,} rows × {n_cols:>2} cols"))

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
                print(warn(f"{name}.{col} — column missing in sampled/full frame"))
                issues += 1
                continue
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                print(warn(f"{name}.{col} — {n_null:,} nulls in checked rows"))
                issues += 1
            else:
                print(ok(f"{name}.{col} — 0 nulls in checked rows"))

    print(header("3. Token stats (chunks, checked rows)"))
    if "n_tokens" in chunks.columns and len(chunks) > 0:
        desc = chunks["n_tokens"].describe()
        print(ok(f"count   = {int(desc['count']):>10,}"))
        print(ok(f"mean    = {desc['mean']:>10.1f}"))
        print(ok(f"median  = {desc['50%']:>10.0f}"))
        print(ok(f"p95     = {chunks['n_tokens'].quantile(0.95):>10.0f}"))
        print(ok(f"max     = {desc['max']:>10.0f}"))
        violations = int((chunks["n_tokens"] > 200).sum())
        if violations:
            print(warn(f"{violations:,} chunks exceed 200 tokens in checked rows"))
    else:
        print(warn("n_tokens unavailable"))

    print(header("4. Section distribution (checked rows)"))
    if "section" in segments.columns and len(segments) > 0:
        sec_counts = segments["section"].value_counts()
        total_seg = len(segments)
        for sec in ["Prepared", "Q", "A", "O"]:
            n = int(sec_counts.get(sec, 0))
            pct = 100 * n / total_seg
            print(ok(f"{sec:>8s}  {n:>9,} segments  ({pct:5.1f}%)"))

    print(header("5. Temporal/company coverage (checked rows)"))
    if "year_quarter" in transcripts.columns:
        yq = transcripts["year_quarter"].dropna()
        if len(yq) > 0:
            print(ok(f"Range: {yq.min()} → {yq.max()}"))
            print(ok(f"Unique year_quarters: {yq.nunique()}"))
    if "company" in transcripts.columns:
        print(
            ok(f"Unique companies (checked rows): {transcripts['company'].nunique():,}")
        )
    if "ticker" in transcripts.columns:
        fill = int(transcripts["ticker"].notna().sum())
        print(
            ok(
                f"Ticker fill (checked rows): {fill:,}/{len(transcripts):,} ({100*fill/max(1,len(transcripts)):.1f}%)"
            )
        )

    print(header("6. Duplicates audit (checked rows)"))
    if "is_duplicate" in duplicates.columns:
        n_dup = int(duplicates["is_duplicate"].sum())
        print(ok(f"Flagged duplicates: {n_dup:,}"))
    else:
        print(warn("duplicates schema missing is_duplicate"))

    print(header("7. Integrity / uniqueness"))
    if mode == "full":
        chunk_tids = set(chunks["transcript_id"].unique())
        seg_tids = set(segments["transcript_id"].unique())
        trans_tids = set(transcripts["transcript_id"].unique())
        orphan_chunks = chunk_tids - seg_tids
        orphan_segs = seg_tids - trans_tids
        if orphan_chunks:
            print(fail(f"{len(orphan_chunks)} chunk transcript_ids not in segments"))
            issues += 1
        else:
            print(ok("All chunk transcript_ids exist in segments"))
        if orphan_segs:
            print(fail(f"{len(orphan_segs)} segment transcript_ids not in transcripts"))
            issues += 1
        else:
            print(ok("All segment transcript_ids exist in transcripts"))
    else:
        print(warn("Skipped in quick mode (requires full-table scan)"))

    print(header("Summary"))
    if issues == 0:
        print(ok("All checks passed"))
    else:
        print(warn(f"{issues} issue(s) found — review above"))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset health checks")
    parser.add_argument(
        "--data-dir", default="outputs/r1000/clean", help="Path to parquet files"
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--sample-rows", type=int, default=100_000)
    args = parser.parse_args()
    issues = run_checks(
        Path(args.data_dir), mode=args.mode, sample_rows=args.sample_rows
    )
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
