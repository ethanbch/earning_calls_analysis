"""Extend company_ticker_map.json with R2K and SP companies.

Reads unique company names from parquet datasets, filters out entries
already in the existing map, queries yfinance, and merges results.

Run:
    uv run python _extend_ticker_map.py [--dry-run]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yfinance as yf
from tqdm import tqdm

US_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "OPR"}
MAP_PATH = Path("src/pipeline/company_ticker_map.json")

SOURCES = {
    "R2K": Path("raw/rawrussel2K"),
    "SP": Path("raw/sp"),
}


def search_ticker(company: str) -> str | None:
    try:
        s = yf.Search(company, max_results=5)
        for q in s.quotes:
            if q.get("quoteType") == "EQUITY" and q.get("exchange") in US_EXCHANGES:
                return q["symbol"]
        # Fallback: first equity result without foreign suffix
        for q in s.quotes:
            if q.get("quoteType") == "EQUITY":
                sym = q.get("symbol", "")
                if "." not in sym and ".BA" not in sym:
                    return sym
        if s.quotes:
            sym = s.quotes[0].get("symbol", "")
            return sym if sym else None
    except Exception as e:
        print(f"  ERROR {company}: {e}", file=sys.stderr)
    return None


def collect_companies() -> set[str]:
    companies: set[str] = set()
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        files = sorted(path.glob("*.parquet"))
        df = pd.concat(
            [
                pq.read_table(f, columns=["transcript_subheader"]).to_pandas()
                for f in files
            ],
            ignore_index=True,
        )
        unique = set(df["transcript_subheader"].dropna().unique())
        print(f"  {name}: {len(unique)} unique companies")
        companies |= unique
    return companies


def main():
    dry_run = "--dry-run" in sys.argv

    # Load existing map
    existing: dict[str, str] = {}
    if MAP_PATH.exists():
        existing = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    print(f"Existing map: {len(existing)} entries")

    # Collect all companies from parquet sources
    print("\nCollecting companies from parquet sources:")
    all_companies = collect_companies()
    missing = sorted(c for c in all_companies if c.strip() and c not in existing)
    print(f"\nTotal unique companies across sources: {len(all_companies)}")
    print(f"Already mapped: {len(all_companies) - len(missing)}")
    print(f"Need lookup:    {len(missing)}")

    if dry_run:
        print("\n[dry-run] First 20 missing companies:")
        for c in missing[:20]:
            print(f"  {c}")
        return

    if not missing:
        print("Nothing to do — all companies already mapped.")
        return

    print(f"\nLooking up {len(missing)} companies (~{len(missing)*0.35/60:.1f} min)...")

    new_map: dict[str, str] = {}
    failed: list[str] = []

    pbar = tqdm(
        missing,
        unit="co",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] found={postfix[0]} miss={postfix[1]}",
        postfix=[0, 0],
    )
    for company in pbar:
        ticker = search_ticker(company)
        if ticker:
            new_map[company] = ticker
            pbar.postfix[0] = len(new_map)
        else:
            failed.append(company)
            pbar.postfix[1] = len(failed)
        time.sleep(0.35)

    # Merge into existing map (existing entries take priority)
    merged = {**new_map, **existing}

    MAP_PATH.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\nDone: {len(new_map)}/{len(missing)} new tickers found ({100*len(new_map)/len(missing):.1f}%)"
    )
    print(f"Map now has {len(merged)} entries (was {len(existing)})")

    if failed:
        print(f"\nNot found ({len(failed)}):")
        for c in failed[:30]:
            print(f"  - {c}")
        if len(failed) > 30:
            print(f"  ... and {len(failed)-30} more")


if __name__ == "__main__":
    main()
