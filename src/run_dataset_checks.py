"""Run descriptive sanity checks on cleaned earnings call datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dataset_checks")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def read_table(path: Path, logger: logging.Logger) -> pd.DataFrame:
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded CSV %s with shape=%s", csv_path, df.shape)
        return df
    if path.exists():
        try:
            df = pd.read_parquet(path)
            logger.info("Loaded Parquet %s with shape=%s", path, df.shape)
            return df
        except Exception as e:
            logger.warning("Could not read parquet %s (%s).", path, e)
    logger.warning("Missing table: %s", path)
    return pd.DataFrame()


def save_plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, top_n: int = 25) -> None:
    if df.empty:
        return
    plot_df = df.head(top_n)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df[x].astype(str), plot_df[y])
    plt.xticks(rotation=75, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def quarter_sort_key(yq: str) -> Tuple[int, int]:
    if not isinstance(yq, str) or "Q" not in yq:
        return (9999, 9)
    y, q = yq.split("Q")
    return (int(y), int(q))


def full_quarter_range(min_yq: str, max_yq: str) -> List[str]:
    y1, q1 = quarter_sort_key(min_yq)
    y2, q2 = quarter_sort_key(max_yq)
    if y1 == 9999 or y2 == 9999:
        return []
    out: List[str] = []
    y, q = y1, q1
    while (y < y2) or (y == y2 and q <= q2):
        out.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def run_checks(output_dir: Path) -> None:
    checks_dir = output_dir / "checks"
    clean_dir = output_dir / "clean"
    logger = setup_logger(Path("logs") / "checks.log")
    checks_dir.mkdir(parents=True, exist_ok=True)

    transcripts = read_table(clean_dir / "transcripts_deduplicated.parquet", logger)
    segments = read_table(clean_dir / "segments.parquet", logger)
    chunks = read_table(clean_dir / "chunks.parquet", logger)

    if transcripts.empty:
        logger.error("No transcripts dataset found. Exiting checks.")
        return

    # 1) Earnings calls per quarter
    ec_per_quarter = (
        transcripts.dropna(subset=["year_quarter"]).groupby("year_quarter")["transcript_id"].nunique().reset_index(name="n_earnings_calls")
    )
    ec_per_quarter = ec_per_quarter.sort_values("year_quarter", key=lambda s: s.map(quarter_sort_key))
    ec_per_quarter.to_csv(checks_dir / "ec_per_quarter.csv", index=False)
    save_plot_bar(ec_per_quarter, "year_quarter", "n_earnings_calls", "Earnings Calls per Quarter", checks_dir / "ec_per_quarter.png", top_n=len(ec_per_quarter))

    # 2) Average calls per company stats
    calls_per_company = transcripts.groupby("company")["transcript_id"].nunique().rename("n_calls").reset_index()
    calls_per_company = calls_per_company.sort_values("n_calls", ascending=False)
    calls_per_company.to_csv(checks_dir / "calls_per_company.csv", index=False)

    avg_stats = {
        "mean": float(calls_per_company["n_calls"].mean()) if not calls_per_company.empty else 0.0,
        "median": float(calls_per_company["n_calls"].median()) if not calls_per_company.empty else 0.0,
        "p25": float(calls_per_company["n_calls"].quantile(0.25)) if not calls_per_company.empty else 0.0,
        "p75": float(calls_per_company["n_calls"].quantile(0.75)) if not calls_per_company.empty else 0.0,
        "max": int(calls_per_company["n_calls"].max()) if not calls_per_company.empty else 0,
    }
    with (checks_dir / "avg_calls_per_company_summary.json").open("w", encoding="utf-8") as f:
        json.dump(avg_stats, f, indent=2)

    # 3) Companies present in every quarter
    yq_values = sorted(transcripts["year_quarter"].dropna().unique(), key=quarter_sort_key)
    if yq_values:
        full_range = full_quarter_range(yq_values[0], yq_values[-1])
    else:
        full_range = []

    presence = transcripts.dropna(subset=["company", "year_quarter"]).groupby("company")["year_quarter"].apply(lambda s: set(s)).reset_index(name="quarters")
    in_all = presence[presence["quarters"].apply(lambda qs: set(full_range).issubset(qs))].copy()
    in_all_companies = in_all[["company"]].sort_values("company")
    in_all_companies.to_csv(checks_dir / "companies_in_every_quarter.csv", index=False)
    with (checks_dir / "companies_in_every_quarter_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"count": int(len(in_all_companies)), "min_quarter": yq_values[0] if yq_values else None, "max_quarter": yq_values[-1] if yq_values else None}, f, indent=2)

    # 4) Most cited executives
    if not segments.empty:
        exec_df = segments[(segments["speaker_type"] == "Executive") & segments["speaker_name_normalized"].notna()].copy()
        top_exec_segments = exec_df.groupby("speaker_name_normalized").size().reset_index(name="n_segments").sort_values("n_segments", ascending=False).head(25)
        top_exec_calls = exec_df.groupby("speaker_name_normalized")["transcript_id"].nunique().reset_index(name="n_distinct_calls").sort_values("n_distinct_calls", ascending=False).head(25)
    else:
        top_exec_segments = pd.DataFrame(columns=["speaker_name_normalized", "n_segments"])
        top_exec_calls = pd.DataFrame(columns=["speaker_name_normalized", "n_distinct_calls"])

    top_exec_segments.to_csv(checks_dir / "top_executives_by_segments.csv", index=False)
    top_exec_calls.to_csv(checks_dir / "top_executives_by_distinct_calls.csv", index=False)
    save_plot_bar(top_exec_segments, "speaker_name_normalized", "n_segments", "Top Executives by Segments", checks_dir / "top_executives_by_segments.png")
    save_plot_bar(top_exec_calls, "speaker_name_normalized", "n_distinct_calls", "Top Executives by Distinct Calls", checks_dir / "top_executives_by_distinct_calls.png")

    # 5) Most cited analysts
    if not segments.empty:
        an_df = segments[(segments["speaker_type"] == "Analyst") & (segments["section"] == "Q") & segments["speaker_name_normalized"].notna()].copy()
        top_analyst_segments = an_df.groupby("speaker_name_normalized").size().reset_index(name="n_question_segments").sort_values("n_question_segments", ascending=False).head(25)
        top_analyst_calls = an_df.groupby("speaker_name_normalized")["transcript_id"].nunique().reset_index(name="n_distinct_calls").sort_values("n_distinct_calls", ascending=False).head(25)
    else:
        top_analyst_segments = pd.DataFrame(columns=["speaker_name_normalized", "n_question_segments"])
        top_analyst_calls = pd.DataFrame(columns=["speaker_name_normalized", "n_distinct_calls"])

    top_analyst_segments.to_csv(checks_dir / "top_analysts_by_question_segments.csv", index=False)
    top_analyst_calls.to_csv(checks_dir / "top_analysts_by_distinct_calls.csv", index=False)
    save_plot_bar(top_analyst_segments, "speaker_name_normalized", "n_question_segments", "Top Analysts by Q Segments", checks_dir / "top_analysts_by_question_segments.png")
    save_plot_bar(top_analyst_calls, "speaker_name_normalized", "n_distinct_calls", "Top Analysts by Distinct Calls", checks_dir / "top_analysts_by_distinct_calls.png")

    # 6) Top companies by number of calls
    top_companies = calls_per_company.head(25)
    top_companies.to_csv(checks_dir / "top_companies_by_calls.csv", index=False)
    save_plot_bar(top_companies, "company", "n_calls", "Top Companies by Number of Calls", checks_dir / "top_companies_by_calls.png")

    # 7) Distribution of calls per company
    if not calls_per_company.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(calls_per_company["n_calls"], bins=30)
        plt.title("Distribution of Calls per Company")
        plt.tight_layout()
        plt.savefig(checks_dir / "calls_per_company_distribution.png", dpi=150)
        plt.close()

    # 8) Share of transcripts with both Q and A detected
    if not segments.empty:
        qa_flags = segments.groupby("transcript_id")["section"].agg(lambda s: ("Q" in set(s)) and ("A" in set(s)))
        share_both_qa = float(qa_flags.mean())
    else:
        share_both_qa = 0.0

    # 9) Ratio of Q to A segments per transcript and overall
    if not segments.empty:
        q_counts = segments[segments["section"] == "Q"].groupby("transcript_id").size().rename("q_segments")
        a_counts = segments[segments["section"] == "A"].groupby("transcript_id").size().rename("a_segments")
        qa_ratio = pd.concat([q_counts, a_counts], axis=1).fillna(0).reset_index()
        qa_ratio["q_to_a_ratio"] = np.where(qa_ratio["a_segments"] > 0, qa_ratio["q_segments"] / qa_ratio["a_segments"], np.nan)
        qa_ratio.to_csv(checks_dir / "qa_ratio_per_transcript.csv", index=False)
        overall_ratio = float(q_counts.sum() / a_counts.sum()) if a_counts.sum() > 0 else None
    else:
        qa_ratio = pd.DataFrame(columns=["transcript_id", "q_segments", "a_segments", "q_to_a_ratio"])
        qa_ratio.to_csv(checks_dir / "qa_ratio_per_transcript.csv", index=False)
        overall_ratio = None

    # 10) Segment count per transcript
    if not segments.empty:
        seg_count = segments.groupby("transcript_id").size().reset_index(name="n_segments")
    else:
        seg_count = pd.DataFrame(columns=["transcript_id", "n_segments"])
    seg_count.to_csv(checks_dir / "segment_count_per_transcript.csv", index=False)

    # 11) Chunk count per transcript
    if not chunks.empty:
        chunk_count = chunks.groupby("transcript_id").size().reset_index(name="n_chunks")
    else:
        chunk_count = pd.DataFrame(columns=["transcript_id", "n_chunks"])
    chunk_count.to_csv(checks_dir / "chunk_count_per_transcript.csv", index=False)

    # 12) Most common missing metadata fields
    metadata_fields = ["company", "ticker", "event_date", "year", "quarter", "year_quarter", "title"]
    miss = pd.DataFrame({
        "field": metadata_fields,
        "missing_count": [int(transcripts[c].isna().sum()) if c in transcripts.columns else int(len(transcripts)) for c in metadata_fields],
    }).sort_values("missing_count", ascending=False)
    miss.to_csv(checks_dir / "missing_metadata_fields.csv", index=False)

    # 13) Coverage table + heatmap (top 50 companies)
    cov = transcripts.dropna(subset=["company", "year_quarter"]).groupby(["company", "year_quarter"]).size().unstack(fill_value=0)
    cov.to_csv(checks_dir / "company_quarter_coverage_table.csv")
    if not cov.empty:
        top50 = cov.sum(axis=1).sort_values(ascending=False).head(50).index
        cov_plot = cov.loc[top50]
        plt.figure(figsize=(12, 10))
        plt.imshow(cov_plot.values > 0, aspect="auto", interpolation="nearest")
        plt.yticks(range(len(cov_plot.index)), cov_plot.index)
        plt.xticks(range(len(cov_plot.columns)), cov_plot.columns, rotation=90)
        plt.title("Company x Quarter Coverage (Top 50 Companies)")
        plt.tight_layout()
        plt.savefig(checks_dir / "company_quarter_coverage_heatmap.png", dpi=150)
        plt.close()

    # 14) Top speaker names raw vs normalized
    if not segments.empty:
        top_raw = segments["speaker_name_raw"].fillna("").value_counts().head(50).rename_axis("speaker_name_raw").reset_index(name="count")
        top_norm = segments["speaker_name_normalized"].fillna("").value_counts().head(50).rename_axis("speaker_name_normalized").reset_index(name="count")
    else:
        top_raw = pd.DataFrame(columns=["speaker_name_raw", "count"])
        top_norm = pd.DataFrame(columns=["speaker_name_normalized", "count"])
    top_raw.to_csv(checks_dir / "top_speaker_names_raw.csv", index=False)
    top_norm.to_csv(checks_dir / "top_speaker_names_normalized.csv", index=False)

    # 15) Transcript length distribution
    t_len = transcripts[["transcript_id", "clean_text"]].copy()
    t_len["chars"] = t_len["clean_text"].fillna("").str.len()
    t_len["words"] = t_len["clean_text"].fillna("").str.split().map(len)

    chunk_tokens = chunks.groupby("transcript_id")["n_tokens"].sum().rename("tokens_from_chunks") if not chunks.empty else pd.Series(dtype=float)
    chunk_counts = chunks.groupby("transcript_id").size().rename("n_chunks") if not chunks.empty else pd.Series(dtype=float)

    t_len = t_len.merge(chunk_tokens, on="transcript_id", how="left")
    t_len = t_len.merge(chunk_counts, on="transcript_id", how="left")
    t_len["tokens_from_chunks"] = t_len["tokens_from_chunks"].fillna(0)
    t_len["n_chunks"] = t_len["n_chunks"].fillna(0)
    t_len.to_csv(checks_dir / "transcript_length_distribution.csv", index=False)

    for metric in ["chars", "words", "n_chunks", "tokens_from_chunks"]:
        plt.figure(figsize=(8, 4))
        plt.hist(t_len[metric], bins=40)
        plt.title(f"Transcript Length Distribution: {metric}")
        plt.tight_layout()
        plt.savefig(checks_dir / f"transcript_length_{metric}.png", dpi=150)
        plt.close()

    summary = {
        "n_transcripts": int(transcripts["transcript_id"].nunique()),
        "n_companies": int(transcripts["company"].nunique(dropna=True)),
        "n_segments": int(len(segments)),
        "n_chunks": int(len(chunks)),
        "share_transcripts_with_both_q_and_a": share_both_qa,
        "overall_q_to_a_ratio": overall_ratio,
        "avg_calls_per_company": avg_stats,
    }
    with (checks_dir / "dataset_checks_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Checks complete. Summary: %s", json.dumps(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run descriptive checks on cleaned earnings call outputs")
    parser.add_argument("--output-dir", default="outputs", help="Output base directory used in preprocessing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_checks(Path(args.output_dir))


if __name__ == "__main__":
    main()
