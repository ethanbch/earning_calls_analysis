"""Build the analysis panel: merge sentiment divergence with stock returns.

Reads:
    - sentiment_aggregated.parquet  (transcript-level exec/analyst scores)
    - returns.parquet               (ticker × event_date → ret_1d..ret_10d)
    - transcripts_deduplicated.parquet (metadata: company, year, quarter)

Outputs:
    - outputs/panel.parquet

Panel columns:
    transcript_id, ticker, company, event_date, year, quarter, year_quarter,
    score_exec, score_analyst, divergence,
    n_chunks_exec, n_chunks_analyst,
    ret_1d, ret_3d, ret_5d, ret_10d,
    is_smallcap (1 if source is R2K, 0 otherwise),
    source (r1000 / r2k / sp)

Usage
-----
    uv run python -m src.analysis.build_panel \
        --sentiment outputs/sentiment_aggregated.parquet \
        --returns outputs/returns.parquet \
        --transcripts outputs/r1000/clean/transcripts_deduplicated.parquet \
                      outputs/r2k/clean/transcripts_deduplicated.parquet \
                      outputs/sp/clean/transcripts_deduplicated.parquet \
        --output outputs/panel.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _infer_source(path: str) -> str:
    """Infer dataset source from file path."""
    p = path.lower()
    if "r2k" in p or "russel2k" in p or "russell2k" in p:
        return "r2k"
    if "sp" in p:
        return "sp"
    return "r1000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis panel")
    parser.add_argument(
        "--sentiment", required=True, help="Path to sentiment_aggregated.parquet"
    )
    parser.add_argument("--returns", required=True, help="Path to returns.parquet")
    parser.add_argument(
        "--transcripts",
        nargs="+",
        required=True,
        help="transcripts_deduplicated.parquet files",
    )
    parser.add_argument("--output", default="outputs/panel.parquet")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # ── Load sentiment aggregates ─────────────────────────────
    sent = pq.read_table(args.sentiment).to_pandas()
    logger.info("Sentiment: %d rows", len(sent))

    # Pivot: one row per (transcript_id, ticker, event_date) with exec + analyst columns
    exec_df = sent[sent["speaker_role"] == "exec"][
        ["transcript_id", "ticker", "event_date", "mean_score", "n_chunks"]
    ].rename(columns={"mean_score": "score_exec", "n_chunks": "n_chunks_exec"})

    analyst_df = sent[sent["speaker_role"] == "analyst"][
        ["transcript_id", "ticker", "event_date", "mean_score", "n_chunks"]
    ].rename(columns={"mean_score": "score_analyst", "n_chunks": "n_chunks_analyst"})

    panel = exec_df.merge(
        analyst_df, on=["transcript_id", "ticker", "event_date"], how="outer"
    )

    # Divergence = exec - analyst
    panel["divergence"] = panel["score_exec"] - panel["score_analyst"]

    logger.info(
        "Panel after pivot: %d rows | exec fill: %.1f%% | analyst fill: %.1f%%",
        len(panel),
        panel["score_exec"].notna().mean() * 100,
        panel["score_analyst"].notna().mean() * 100,
    )

    # ── Load returns ──────────────────────────────────────────
    returns = pq.read_table(args.returns).to_pandas()
    returns["event_date"] = pd.to_datetime(returns["event_date"])
    # Deduplicate returns per (ticker, event_date)
    returns = returns.drop_duplicates(subset=["ticker", "event_date"])
    ret_cols = [c for c in returns.columns if c.startswith("ret_")]
    returns = returns[["transcript_id", "ticker", "event_date"] + ret_cols]
    logger.info("Returns: %d rows", len(returns))

    # Merge
    panel["event_date"] = pd.to_datetime(panel["event_date"])
    panel = panel.merge(
        returns.drop(columns=["transcript_id"], errors="ignore"),
        on=["ticker", "event_date"],
        how="left",
    )

    # ── Load transcript metadata ──────────────────────────────
    meta_frames = []
    for p in args.transcripts:
        path = Path(p)
        if not path.exists():
            continue
        df = pq.read_table(
            path,
            columns=[
                "transcript_id",
                "company",
                "ticker",
                "year",
                "quarter",
                "year_quarter",
                "source_file",
            ],
        ).to_pandas()
        df["source"] = _infer_source(p)
        meta_frames.append(df)

    meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates(
        subset=["transcript_id"]
    )
    panel = panel.merge(
        meta[["transcript_id", "company", "year", "quarter", "year_quarter", "source"]],
        on="transcript_id",
        how="left",
    )

    # is_smallcap flag
    panel["is_smallcap"] = (panel["source"] == "r2k").astype(int)

    # ── Final shape ───────────────────────────────────────────
    col_order = [
        "transcript_id",
        "ticker",
        "company",
        "event_date",
        "year",
        "quarter",
        "year_quarter",
        "source",
        "is_smallcap",
        "score_exec",
        "score_analyst",
        "divergence",
        "n_chunks_exec",
        "n_chunks_analyst",
    ] + [f"ret_{w}d" for w in [1, 3, 5, 10]]

    panel = panel[[c for c in col_order if c in panel.columns]]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)

    logger.info(
        "Panel saved: %s (%d rows, %d cols)", out, len(panel), len(panel.columns)
    )
    logger.info(
        "Divergence stats: mean=%.3f, std=%.3f, non-null=%d",
        panel["divergence"].mean(),
        panel["divergence"].std(),
        panel["divergence"].notna().sum(),
    )


if __name__ == "__main__":
    main()
