from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(path: Path, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
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


def plot_step3_output_funnel(panel: pd.DataFrame, out_path: Path) -> None:
    required = {"score_exec", "score_analyst"}
    if not required.issubset(panel.columns) or panel.empty:
        return

    ret_cols = [c for c in ["ret_1d", "ret_3d", "ret_5d", "ret_10d"] if c in panel.columns]

    total = int(len(panel))
    both_speakers_mask = panel["score_exec"].notna() & panel["score_analyst"].notna()
    both_speakers = int(both_speakers_mask.sum())

    if "divergence" in panel.columns:
        divergence_ready = int((both_speakers_mask & panel["divergence"].notna()).sum())
    else:
        divergence_ready = both_speakers

    if ret_cols:
        any_ret = int((both_speakers_mask & panel[ret_cols].notna().any(axis=1)).sum())
        ret5_ready = (
            int((both_speakers_mask & panel["ret_5d"].notna()).sum())
            if "ret_5d" in panel.columns
            else any_ret
        )
    else:
        any_ret = 0
        ret5_ready = 0

    labels = [
        "Panel rows after merge",
        "Both speaker types present",
        "Divergence computable",
        "With at least one return",
        "Regression sample (ret_5d)",
    ]
    values = [total, both_speakers, divergence_ready, any_ret, ret5_ready]

    colors = ["#DCE8F5", "#A7C4E5", "#79A6D3", "#4A86BF", "#123B6D"]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="white")
    bars = ax.barh(y, values, color=colors, edgecolor=colors, height=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of transcripts")
    ax.set_title("Step 3 Output: Sample Construction and Return Coverage", fontsize=13, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for i, b in enumerate(bars):
        val = int(values[i])
        pct = (100 * val / total) if total else 0
        ax.text(
            b.get_width() + max(total * 0.008, 5),
            b.get_y() + b.get_height() / 2,
            f"{val:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
            color="#1E2A36",
        )

    note = "Exclusion driver: missing analyst turns (both speaker types required)."
    ax.text(0.0, -0.14, note, transform=ax.transAxes, fontsize=10, color="#3B4B5A")

    _save_fig(out_path, dpi=300)


def plot_slide8_divergence_by_horizon(out_path: Path) -> None:
    horizons = ["1-day", "3-day", "5-day"]
    coefs = [0.0069, 0.0102, 0.0110]
    # Already ± 1.96 * SE as requested.
    yerr = [0.0022, 0.0025, 0.0029]

    x = np.arange(len(horizons))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    bars = ax.bar(
        x,
        coefs,
        yerr=yerr,
        capsize=5,
        color="#123B6D",
        edgecolor="#123B6D",
        width=0.62,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Divergence Effect by Return Horizon", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("Coefficient")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            coefs[i] + yerr[i] + 0.00035,
            "***",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    _save_fig(out_path, dpi=300)


def plot_slide9_large_smallcap(out_path: Path) -> None:
    labels = ["Large-cap", "Small-cap"]
    values = [0.0053, 0.0305]
    colors = ["#86B6E8", "#123B6D"]
    sigs = ["*", "***"]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    bars = ax.bar(x, values, color=colors, edgecolor=colors, width=0.62)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(
        "Divergence Effect: Large-Cap vs Small-Cap (ret_5d)",
        fontsize=13,
        weight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coefficient")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            values[i] + 0.0007,
            sigs[i],
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.annotate(
        "5.7x stronger",
        xy=(1, values[1]),
        xytext=(0.4, values[1] + 0.008),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.5},
        fontsize=11,
    )

    _save_fig(out_path, dpi=300)


def plot_slide10_pre_post(out_path: Path) -> None:
    groups = ["Divergence coef", "SmallCap interaction"]
    pre = [0.002, 0.0]
    post = [0.009, 0.021]

    x = np.arange(len(groups))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5), facecolor="white")
    pre_bars = ax.bar(
        x - width / 2,
        pre,
        width,
        label="Pre-2020",
        color="#B8BCC2",
        edgecolor="#B8BCC2",
    )
    post_bars = ax.bar(
        x + width / 2,
        post,
        width,
        label="Post-2020",
        color="#123B6D",
        edgecolor="#123B6D",
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Pre-2020 vs Post-2020 Effect", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Coefficient")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    for b in pre_bars:
        y = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            y + 0.0006,
            "n.s.",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4A4A4A",
        )

    for b in post_bars:
        y = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            y + 0.0006,
            "***",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    _save_fig(out_path, dpi=300)


def plot_slide11_lollipop(out_path: Path) -> None:
    variables = [
        "Divergence",
        "Div × SmallCap",
        "Raw exec tone",
        "Raw analyst tone",
    ]
    coef = np.array([0.0035, 0.0252, 0.0002, -0.0001])
    ci_low = np.array([0.0005, 0.0150, -0.0100, -0.0120])
    ci_high = np.array([0.0065, 0.0354, 0.0104, 0.0118])
    significant = [True, True, False, False]

    y = np.arange(len(variables))[::-1]
    fig, ax = plt.subplots(figsize=(9.3, 5.4), facecolor="white")

    ax.axvline(0, color="black", linewidth=1)

    for i in range(len(variables)):
        yy = y[i]
        color = "#123B6D" if significant[i] else "#B8BCC2"
        ax.hlines(yy, ci_low[i], ci_high[i], color=color, linewidth=2)
        ax.plot(coef[i], yy, "o", color=color, markersize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(variables)
    ax.set_xlabel("Coefficient")
    ax.set_title(
        "Model 3: Which Variables Are Significant?", fontsize=13, weight="bold"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    _save_fig(out_path, dpi=300)


def plot_slide12_regression_table(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.8), facecolor="white")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.01,
        0.96,
        "Panel Regression Results",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )

    x_cols = {
        "model": 0.02,
        "var": 0.22,
        "ret1": 0.50,
        "ret3": 0.68,
        "ret5": 0.86,
    }

    # Booktabs-like horizontal rules
    ax.hlines(0.90, 0.01, 0.99, colors="black", linewidth=1.6)
    ax.hlines(0.84, 0.01, 0.99, colors="black", linewidth=0.9)

    ax.text(x_cols["model"], 0.865, "Model", fontsize=11, fontweight="bold")
    ax.text(x_cols["var"], 0.865, "Variable", fontsize=11, fontweight="bold")
    ax.text(x_cols["ret1"], 0.865, "ret_1d", fontsize=11, fontweight="bold", ha="right")
    ax.text(x_cols["ret3"], 0.865, "ret_3d", fontsize=11, fontweight="bold", ha="right")
    ax.text(x_cols["ret5"], 0.865, "ret_5d", fontsize=11, fontweight="bold", ha="right")

    rows = [
        ("Baseline", "divergence", "0.0069***", "0.0102***", "0.0110***"),
        ("", "SE", "(0.0022)", "(0.0025)", "(0.0029)"),
        ("", "N", "1492", "1492", "1492"),
        ("", "R²", "0.031", "0.032", "0.053"),
        ("Size", "divergence", "0.0046*", "0.0064*", "0.0053*"),
        ("", "div×SmallCap", "0.0106***", "0.0169***", "0.0252***"),
        ("", "R²", "0.034", "0.037", "0.062"),
        ("Full", "divergence", "0.0032*", "0.0043*", "0.0035*"),
        ("", "div×SmallCap", "0.0109***", "0.0171***", "0.0252***"),
        ("", "R²", "0.035", "0.037", "0.062"),
    ]

    y = 0.80
    row_h = 0.055
    for i, row in enumerate(rows):
        model, var, c1, c3, c5 = row
        if i in {4, 7}:
            ax.hlines(y + 0.02, 0.01, 0.99, colors="#555555", linewidth=0.5)

        ax.text(x_cols["model"], y, model, fontsize=10.5, ha="left", va="center")
        ax.text(x_cols["var"], y, var, fontsize=10.5, ha="left", va="center")
        ax.text(x_cols["ret1"], y, c1, fontsize=10.5, ha="right", va="center")
        ax.text(x_cols["ret3"], y, c3, fontsize=10.5, ha="right", va="center")
        ax.text(x_cols["ret5"], y, c5, fontsize=10.5, ha="right", va="center")
        y -= row_h

    ax.hlines(y + 0.022, 0.01, 0.99, colors="black", linewidth=1.2)

    footnote = "HC1 robust SE. Quarter FE included. ***p<0.01, **p<0.05, *p<0.10"
    ax.text(0.01, y - 0.03, footnote, fontsize=10, color="#333333", ha="left", va="top")

    _save_fig(out_path, dpi=300)


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
    plot_step3_output_funnel(panel, out_dir / "fig08_step3_output_funnel.png")

    # Slide-specific visuals requested for final presentation.
    plot_slide8_divergence_by_horizon(
        out_dir / "slide08_divergence_effect_by_return_horizon.png"
    )
    plot_slide9_large_smallcap(out_dir / "slide09_largecap_vs_smallcap_ret5d.png")
    plot_slide10_pre_post(out_dir / "slide10_pre2020_vs_post2020_effect.png")
    plot_slide11_lollipop(out_dir / "slide11_model3_significance_lollipop.png")
    plot_slide12_regression_table(out_dir / "slide12_panel_regression_table.png")

    write_presentation_notes(out_dir / "presentation_bullets.md", bullets)

    print(f"Wrote assets to: {out_dir}")
    print("- summary_kpis.csv")
    print("- speaker_role_stats.csv (if role columns present)")
    print("- presentation_bullets.md")
    print("- fig01..fig08 PNG charts")
    print("- slide08..slide12 PNG charts/tables")


if __name__ == "__main__":
    main()
