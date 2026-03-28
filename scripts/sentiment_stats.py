from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    scores_path = Path("outputs/sentiment_scores.parquet")
    agg_path = Path("outputs/sentiment_aggregated.parquet")

    if not scores_path.exists():
        print("MISSING: outputs/sentiment_scores.parquet")
        return

    scores = pd.read_parquet(scores_path)
    print("=== CHUNK-LEVEL ===")
    print(f"rows={len(scores):,}")

    label_counts = scores["sentiment_label"].value_counts(dropna=False)
    print("\nlabel_counts:")
    for label, count in label_counts.items():
        pct = (count / len(scores)) * 100 if len(scores) else 0.0
        print(f"  {label}: {count:,} ({pct:.2f}%)")

    s = scores["sentiment_score"]
    print("\nsentiment_score summary:")
    print(
        "  "
        f"mean={s.mean():.4f}  std={s.std():.4f}  "
        f"min={s.min():.4f}  p25={s.quantile(0.25):.4f}  "
        f"median={s.median():.4f}  p75={s.quantile(0.75):.4f}  max={s.max():.4f}"
    )

    max_prob = scores[["prob_positive", "prob_neutral", "prob_negative"]].max(axis=1)
    print("\nconfidence (max class probability):")
    print(
        "  "
        f"mean={max_prob.mean():.4f}  median={max_prob.median():.4f}  "
        f"p90={max_prob.quantile(0.90):.4f}"
    )

    if "speaker_type" in scores.columns:
        print("\nby speaker_type (mean sentiment_score, n):")
        by_type = (
            scores.groupby("speaker_type", dropna=False)["sentiment_score"]
            .agg(["mean", "median", "count"])
            .sort_values("count", ascending=False)
        )
        for speaker_type, row in by_type.iterrows():
            print(
                f"  {speaker_type}: mean={row['mean']:.4f}, "
                f"median={row['median']:.4f}, n={int(row['count']):,}"
            )

    if not agg_path.exists():
        print("\nMISSING: outputs/sentiment_aggregated.parquet")
        return

    agg = pd.read_parquet(agg_path)
    print("\n=== TRANSCRIPT-LEVEL AGG ===")
    print(f"rows={len(agg):,}")

    if "speaker_role" in agg.columns:
        by_role = (
            agg.groupby("speaker_role")["mean_score"]
            .agg(["mean", "median", "count"])
            .sort_values("count", ascending=False)
        )
        print("by speaker_role (mean_score):")
        for role, row in by_role.iterrows():
            print(
                f"  {role}: mean={row['mean']:.4f}, "
                f"median={row['median']:.4f}, n={int(row['count']):,}"
            )

    pivot = agg.pivot_table(
        index=["transcript_id", "ticker", "event_date"],
        columns="speaker_role",
        values="mean_score",
        aggfunc="first",
    )
    if {"exec", "analyst"}.issubset(pivot.columns):
        div = (pivot["exec"] - pivot["analyst"]).dropna()
        print("\nexec-analyst divergence (exec - analyst):")
        if len(div):
            print(
                "  "
                f"n={len(div):,}  mean={div.mean():.4f}  median={div.median():.4f}  "
                f"std={div.std():.4f}  p10={div.quantile(0.10):.4f}  "
                f"p90={div.quantile(0.90):.4f}"
            )
        else:
            print("  n=0")


if __name__ == "__main__":
    main()
