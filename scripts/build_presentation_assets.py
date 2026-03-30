from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _load_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_parquet(path)


def build_summary_stats(
    sentiment_scores: pd.DataFrame,
    sentiment_agg: pd.DataFrame,
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    regressions_csv: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # Global KPIs
    unique_transcripts = (
        sentiment_scores["transcript_id"].nunique()
        if "transcript_id" in sentiment_scores.columns
        else 0
    )
    unique_tickers = returns["ticker"].nunique() if "ticker" in returns.columns else 0

    kpis = [
        ("Chunk rows", int(len(sentiment_scores))),
        ("Transcript-level sentiment rows", int(len(sentiment_agg))),
        ("Panel rows", int(len(panel))),
        ("Unique transcripts scored", int(unique_transcripts)),
        ("Unique tickers in returns", int(unique_tickers)),
    ]

    for col in ["ret_1d", "ret_3d", "ret_5d", "ret_10d"]:
        if col in returns.columns:
            fill = float(returns[col].notna().mean() * 100)
            kpis.append((f"Returns fill rate {col}", round(fill, 2)))

    if "divergence" in panel.columns:
        div = panel["divergence"].dropna()
        if len(div):
            kpis.extend(
                [
                    ("Divergence mean", round(float(div.mean()), 4)),
                    ("Divergence median", round(float(div.median()), 4)),
                    ("Divergence std", round(float(div.std()), 4)),
                    ("Divergence non-null", int(len(div))),
                ]
            )

    kpi_df = pd.DataFrame(kpis, columns=["metric", "value"])

    # Speaker role stats
    role_stats = pd.DataFrame()
    if {"speaker_role", "mean_score"}.issubset(sentiment_agg.columns):
        role_stats = (
            sentiment_agg.groupby("speaker_role", dropna=False)["mean_score"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .sort_values("count", ascending=False)
        )

    # Slide-ready bullets
    bullets: list[str] = []
    bullets.append(
        f"Coverage: {unique_transcripts:,} transcripts scored, {unique_tickers:,} tickers with returns."
    )

    if "sentiment_label" in sentiment_scores.columns and len(sentiment_scores):
        labels = sentiment_scores["sentiment_label"].value_counts(normalize=True) * 100
        top = labels.head(3)
        parts = [f"{k}: {v:.1f}%" for k, v in top.items()]
        bullets.append("Sentiment mix (chunk-level): " + ", ".join(parts) + ".")

    if "divergence" in panel.columns:
        div = panel["divergence"].dropna()
        if len(div):
            bullets.append(
                f"Executive-analyst divergence: mean {div.mean():.3f}, std {div.std():.3f}, n={len(div):,}."
            )

    if regressions_csv is not None and not regressions_csv.empty:
        sel = regressions_csv[
            regressions_csv["variable"].isin(["divergence", "div_x_small"])
        ].copy()
        sel = sel.sort_values("p_value", ascending=True).head(2)
        for _, row in sel.iterrows():
            bullets.append(
                f"Regression signal: {row['model']} | {row['variable']} coef={row['coef']:.4f}, p={row['p_value']:.4g}."
            )

    return kpi_df, role_stats, bullets


def plot_sentiment_label_mix(scores: pd.DataFrame, out_path: Path) -> None:
    if "sentiment_label" not in scores.columns or scores.empty:
        return

    order = scores["sentiment_label"].value_counts(ascending=False).index.tolist()
    df = (
        scores["sentiment_label"]
        .value_counts(normalize=True)
        .mul(100)
        .rename_axis("sentiment_label")
        .reset_index(name="pct")
    )

    plt.figure(figsize=(7, 4.5))
    sns.barplot(
        data=df,
        x="sentiment_label",
        y="pct",
        hue="sentiment_label",
        order=order,
        palette="Set2",
        legend=False,
    )
    plt.title("Chunk-Level Sentiment Label Mix")
    plt.xlabel("Label")
    plt.ylabel("Share (%)")
    _save_fig(out_path)


def plot_sentiment_score_distribution(scores: pd.DataFrame, out_path: Path) -> None:
    if "sentiment_score" not in scores.columns or scores.empty:
        return

    plt.figure(figsize=(8, 4.8))
    sns.histplot(scores["sentiment_score"].dropna(), bins=40, kde=True, color="#2a9d8f")
    plt.title("Distribution of Chunk-Level Sentiment Score")
    plt.xlabel("Sentiment score")
    plt.ylabel("Count")
    _save_fig(out_path)


def plot_role_mean_scores(agg: pd.DataFrame, out_path: Path) -> None:
    if not {"speaker_role", "mean_score"}.issubset(agg.columns) or agg.empty:
        return

    role_df = (
        agg.groupby("speaker_role", dropna=False)["mean_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(7, 4.5))
    sns.barplot(
        data=role_df,
        x="speaker_role",
        y="mean_score",
        hue="speaker_role",
        palette="viridis",
        legend=False,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Average Sentiment by Speaker Role")
    plt.xlabel("Speaker role")
    plt.ylabel("Mean score")
    _save_fig(out_path)


def plot_divergence_over_time(panel: pd.DataFrame, out_path: Path) -> None:
    required = {"year_quarter", "divergence"}
    if not required.issubset(panel.columns) or panel.empty:
        return

    ts = panel.dropna(subset=["divergence", "year_quarter"]).copy()
    if ts.empty:
        return

    ts = ts.groupby("year_quarter", as_index=False)["divergence"].mean()

    plt.figure(figsize=(12, 4.8))
    sns.lineplot(data=ts, x="year_quarter", y="divergence", marker="o", linewidth=1.5)
    plt.xticks(rotation=70, fontsize=8)
    plt.title("Mean Executive-Analyst Divergence Over Time")
    plt.xlabel("Year-Quarter")
    plt.ylabel("Mean divergence")
    _save_fig(out_path)


def plot_divergence_vs_ret5(panel: pd.DataFrame, out_path: Path) -> None:
    required = {"divergence", "ret_5d"}
    if not required.issubset(panel.columns) or panel.empty:
        return

    df = panel.dropna(subset=["divergence", "ret_5d"]).copy()
    if df.empty:
        return

    if len(df) > 12000:
        df = df.sample(12000, random_state=42)

    plt.figure(figsize=(7.5, 5))
    sns.regplot(
        data=df,
        x="divergence",
        y="ret_5d",
        scatter_kws={"alpha": 0.25, "s": 15},
        line_kws={"color": "#e76f51", "linewidth": 2},
    )
    plt.title("Divergence vs 5-Day Return")
    plt.xlabel("Executive - analyst sentiment divergence")
    plt.ylabel("Return at +5d")
    _save_fig(out_path)


def plot_return_horizon_boxplot(panel: pd.DataFrame, out_path: Path) -> None:
    ret_cols = [
        c for c in ["ret_1d", "ret_3d", "ret_5d", "ret_10d"] if c in panel.columns
    ]
    if not ret_cols:
        return

    df = panel[ret_cols].melt(var_name="horizon", value_name="return").dropna()
    if df.empty:
        return

    plt.figure(figsize=(7.5, 5))
    sns.boxplot(
        data=df,
        x="horizon",
        y="return",
        hue="horizon",
        showfliers=False,
        palette="pastel",
        legend=False,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Distribution of Post-Earnings Returns by Horizon")
    plt.xlabel("Horizon")
    plt.ylabel("Return")
    _save_fig(out_path)


def plot_corr_heatmap(panel: pd.DataFrame, out_path: Path) -> None:
    candidates = [
        "score_exec",
        "score_analyst",
        "divergence",
        "n_chunks_exec",
        "n_chunks_analyst",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "is_smallcap",
    ]
    cols = [c for c in candidates if c in panel.columns]
    if len(cols) < 2:
        return

    corr = panel[cols].corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Correlation Heatmap (Panel Variables)")
    _save_fig(out_path)


def load_regression_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def write_presentation_notes(path: Path, bullets: list[str]) -> None:
    lines = ["# Presentation-ready key bullets", ""]
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stats + visual assets for presentation"
    )
    parser.add_argument("--scores", default="outputs/sentiment_scores.parquet")
    parser.add_argument("--agg", default="outputs/sentiment_aggregated.parquet")
    parser.add_argument("--panel", default="outputs/panel.parquet")
    parser.add_argument("--returns", default="outputs/returns.parquet")
    parser.add_argument("--reg-csv", default="outputs/regression_results.csv")
    parser.add_argument("--out-dir", default="outputs/presentation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    scores = _load_parquet(Path(args.scores), "sentiment_scores")
    agg = _load_parquet(Path(args.agg), "sentiment_aggregated")
    panel = _load_parquet(Path(args.panel), "panel")
    returns = _load_parquet(Path(args.returns), "returns")
    reg = load_regression_csv(Path(args.reg_csv))

    kpi_df, role_df, bullets = build_summary_stats(scores, agg, panel, returns, reg)

    kpi_df.to_csv(out_dir / "summary_kpis.csv", index=False)
    if not role_df.empty:
        role_df.to_csv(out_dir / "speaker_role_stats.csv", index=False)

    plot_sentiment_label_mix(scores, out_dir / "fig01_sentiment_label_mix.png")
    plot_sentiment_score_distribution(
        scores, out_dir / "fig02_sentiment_score_distribution.png"
    )
    plot_role_mean_scores(agg, out_dir / "fig03_role_mean_scores.png")
    plot_divergence_over_time(panel, out_dir / "fig04_divergence_over_time.png")
    plot_divergence_vs_ret5(panel, out_dir / "fig05_divergence_vs_ret5.png")
    plot_return_horizon_boxplot(panel, out_dir / "fig06_returns_boxplot_by_horizon.png")
    plot_corr_heatmap(panel, out_dir / "fig07_panel_corr_heatmap.png")

    write_presentation_notes(out_dir / "presentation_bullets.md", bullets)

    print(f"Wrote assets to: {out_dir}")
    print("- summary_kpis.csv")
    print("- speaker_role_stats.csv (if role columns present)")
    print("- presentation_bullets.md")
    print("- fig01..fig07 PNG charts")


if __name__ == "__main__":
    main()
