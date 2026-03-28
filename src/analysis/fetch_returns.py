"""Fetch stock returns around earnings-call dates via yfinance.

For each (ticker, event_date) pair found in the preprocessed transcripts,
downloads daily adjusted-close prices in a window [-2, +10] trading days
and computes cumulative returns at +1d, +3d, +5d, +10d relative to the
call date.

Usage
-----
    uv run python -m src.analysis.fetch_returns \
        --transcripts outputs/r1000/clean/transcripts_deduplicated.parquet \
                      outputs/r2k/clean/transcripts_deduplicated.parquet \
                      outputs/sp/clean/transcripts_deduplicated.parquet \
        --output outputs/returns.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Windows (in trading days) for cumulative returns
RETURN_WINDOWS = [1, 3, 5, 10]
# Price buffer around event date (calendar days) for yfinance download
CALENDAR_BUFFER_BEFORE = 10  # ~5 trading days before
CALENDAR_BUFFER_AFTER = 20  # ~10 trading days after


def _load_events(paths: list[Path]) -> pd.DataFrame:
    """Load and deduplicate (ticker, event_date) pairs from transcript parquets."""
    frames = []
    for p in paths:
        df = pq.read_table(
            p, columns=["transcript_id", "ticker", "event_date"]
        ).to_pandas()
        frames.append(df)
    events = pd.concat(frames, ignore_index=True)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["ticker", "event_date"])
    return events


def _fetch_ticker_prices(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download daily adjusted close for one ticker."""
    try:
        data = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
        )
        if data.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[["Close"]].rename(columns={"Close": "adj_close"})
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", ticker, e)
        return None


def _compute_returns(
    prices: pd.DataFrame, event_date: pd.Timestamp
) -> dict[str, float | None]:
    """Compute cumulative returns at each window relative to event_date."""
    # Find the trading day on or just after event_date (T=0)
    future = prices.loc[prices.index >= event_date]
    if future.empty:
        return {f"ret_{w}d": None for w in RETURN_WINDOWS}

    t0_idx = prices.index.get_loc(future.index[0])
    p0 = prices.iloc[t0_idx]["adj_close"]
    if pd.isna(p0) or p0 == 0:
        return {f"ret_{w}d": None for w in RETURN_WINDOWS}

    results: dict[str, float | None] = {}
    for w in RETURN_WINDOWS:
        target_idx = t0_idx + w
        if target_idx < len(prices):
            pw = prices.iloc[target_idx]["adj_close"]
            results[f"ret_{w}d"] = float((pw - p0) / p0) if pd.notna(pw) else None
        else:
            results[f"ret_{w}d"] = None
    return results


def fetch_all_returns(events: pd.DataFrame) -> pd.DataFrame:
    """Fetch returns for all unique (ticker, event_date) pairs.

    Groups events by ticker to batch downloads efficiently.
    """
    # Unique (ticker, event_date) to avoid redundant fetches
    unique_events = events[["ticker", "event_date"]].drop_duplicates()
    tickers = unique_events["ticker"].unique()

    all_rows: list[dict] = []

    for ticker in tqdm(tickers, desc="Fetching returns", unit="ticker"):
        ticker_events = unique_events[unique_events["ticker"] == ticker]
        min_date = ticker_events["event_date"].min() - pd.Timedelta(
            days=CALENDAR_BUFFER_BEFORE
        )
        max_date = ticker_events["event_date"].max() + pd.Timedelta(
            days=CALENDAR_BUFFER_AFTER
        )

        prices = _fetch_ticker_prices(
            ticker, min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")
        )

        for _, row in ticker_events.iterrows():
            rec = {"ticker": ticker, "event_date": row["event_date"]}
            if prices is not None and not prices.empty:
                rec.update(_compute_returns(prices, row["event_date"]))
            else:
                rec.update({f"ret_{w}d": None for w in RETURN_WINDOWS})
            all_rows.append(rec)

        # Rate limit: avoid yfinance throttling
        time.sleep(0.1)

    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch stock returns around earnings calls"
    )
    parser.add_argument(
        "--transcripts",
        nargs="+",
        required=True,
        help="Paths to transcripts_deduplicated.parquet files",
    )
    parser.add_argument("--output", default="outputs/returns.parquet")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    paths = [Path(p) for p in args.transcripts]
    for p in paths:
        if not p.exists():
            logger.warning("File not found, skipping: %s", p)
    paths = [p for p in paths if p.exists()]

    if not paths:
        logger.error("No transcript files found.")
        return

    events = _load_events(paths)
    logger.info(
        "Loaded %d events across %d tickers", len(events), events["ticker"].nunique()
    )

    returns_df = fetch_all_returns(events)

    # Merge returns back to transcript_ids
    merged = events.merge(returns_df, on=["ticker", "event_date"], how="left")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    logger.info(
        "Saved %d rows to %s | fill rates: %s",
        len(merged),
        out,
        {
            c: f"{merged[c].notna().mean():.1%}"
            for c in merged.columns
            if c.startswith("ret_")
        },
    )


if __name__ == "__main__":
    main()
