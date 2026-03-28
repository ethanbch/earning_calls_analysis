"""Panel regressions: sentiment divergence → post-announcement returns.

Models estimated:
    1. Baseline:        ret_Xd = α + β₁·div + FE_quarter + ε
    2. Size interaction: ret_Xd = α + β₁·div + β₂·div×SmallCap + β₃·SmallCap + FE_quarter + ε
    3. Full:            ret_Xd = α + β₁·div + β₂·div×SmallCap + β₃·SmallCap
                                 + γ₁·score_exec + γ₂·score_analyst
                                 + γ₃·n_chunks_exec + FE_quarter + ε

For each model, we run OLS with heteroskedasticity-robust (HC1) standard errors
and optional company fixed effects (absorbed via dummies or within-transform).

Also runs the same regressions on pre-2020 vs post-2020 subsamples.

Usage
-----
    uv run python -m src.analysis.run_regressions \
        --panel outputs/panel.parquet \
        --output outputs/regression_results.txt
"""

from __future__ import annotations

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

logger = logging.getLogger(__name__)

RETURN_HORIZONS = ["ret_1d", "ret_3d", "ret_5d"]


def _prepare_panel(path: Path) -> pd.DataFrame:
    """Load panel and create regression-ready variables."""
    df = pq.read_table(path).to_pandas()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["year"] = df["year"].astype("Int64")
    df["quarter"] = df["quarter"].astype("Int64")

    # Interaction term
    df["div_x_small"] = df["divergence"] * df["is_smallcap"]

    # Quarter dummies for FE
    df["year_quarter"] = df["year_quarter"].astype(str)

    return df


def _run_ols(
    df: pd.DataFrame, y_col: str, x_cols: list[str], fe_col: str | None = None
) -> sm.OLS:
    """Estimate OLS with HC1 robust SE. Optionally absorbs fixed effects via dummies."""
    subset = df[x_cols + [y_col] + ([fe_col] if fe_col else [])].dropna()
    if len(subset) < 30:
        return None

    X = subset[x_cols].copy()

    if fe_col:
        dummies = pd.get_dummies(
            subset[fe_col], prefix="fe", drop_first=True, dtype=float
        )
        X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X, has_constant="add")
    y = subset[y_col]

    model = sm.OLS(y, X).fit(cov_type="HC1")
    return model


def _format_results(
    models: dict[str, sm.OLS], title: str, collector: list[dict[str, object]]
) -> str:
    """Format regression table."""
    buf = StringIO()
    buf.write(f"\n{'='*80}\n{title}\n{'='*80}\n\n")

    valid = {k: v for k, v in models.items() if v is not None}
    if not valid:
        buf.write("  No valid regressions (insufficient data)\n")
        return buf.getvalue()

    for name, model in valid.items():
        buf.write(f"\n--- {name} ---\n")
        # Show only key coefficients (skip FE dummies)
        params = model.params
        se = model.bse
        pvals = model.pvalues
        key_vars = [v for v in params.index if not v.startswith("fe_")]

        buf.write(
            f"  N={model.nobs:.0f}  R²={model.rsquared:.4f}  Adj-R²={model.rsquared_adj:.4f}\n\n"
        )
        buf.write(
            f"  {'Variable':<25} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8} {'Sig':>5}\n"
        )
        buf.write(f"  {'-'*71}\n")
        for v in key_vars:
            sig = ""
            if pvals[v] < 0.01:
                sig = "***"
            elif pvals[v] < 0.05:
                sig = "**"
            elif pvals[v] < 0.10:
                sig = "*"

            collector.append(
                {
                    "section": title,
                    "model": name,
                    "variable": v,
                    "coef": float(params[v]),
                    "std_err": float(se[v]),
                    "t_stat": float(model.tvalues[v]),
                    "p_value": float(pvals[v]),
                    "sig": sig,
                    "nobs": int(model.nobs),
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }
            )

            buf.write(
                f"  {v:<25} {params[v]:>10.5f} {se[v]:>10.5f} {model.tvalues[v]:>8.2f} {pvals[v]:>8.4f} {sig:>5}\n"
            )
        buf.write("\n")

    return buf.getvalue()


def run_all_regressions(panel: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Run all regression specifications and return formatted output + table."""
    output = StringIO()
    rows: list[dict[str, object]] = []
    output.write("PANEL REGRESSION RESULTS\n")
    output.write(f"Observations: {len(panel):,}\n")
    output.write(f"Tickers: {panel['ticker'].nunique():,}\n")
    output.write(
        f"Period: {panel['event_date'].min():%Y-%m-%d} to {panel['event_date'].max():%Y-%m-%d}\n"
    )
    output.write(f"Small caps: {panel['is_smallcap'].sum():,} / {len(panel):,}\n")

    # ── Model 1: Baseline ─────────────────────────────────────
    for ret_col in RETURN_HORIZONS:
        models = {}
        models[f"Baseline ({ret_col})"] = _run_ols(
            panel, ret_col, ["divergence"], fe_col="year_quarter"
        )
        output.write(
            _format_results(
                models,
                f"MODEL 1 — Baseline: {ret_col} ~ divergence + FE_quarter",
                rows,
            )
        )

    # ── Model 2: Size interaction ─────────────────────────────
    for ret_col in RETURN_HORIZONS:
        models = {}
        models[f"Size interaction ({ret_col})"] = _run_ols(
            panel,
            ret_col,
            ["divergence", "div_x_small", "is_smallcap"],
            fe_col="year_quarter",
        )
        output.write(
            _format_results(
                models,
                f"MODEL 2 — Size interaction: {ret_col} ~ div + div×SmallCap + FE",
                rows,
            )
        )

    # ── Model 3: Full ─────────────────────────────────────────
    for ret_col in RETURN_HORIZONS:
        models = {}
        full_x = [
            "divergence",
            "div_x_small",
            "is_smallcap",
            "score_exec",
            "score_analyst",
            "n_chunks_exec",
        ]
        models[f"Full ({ret_col})"] = _run_ols(
            panel, ret_col, full_x, fe_col="year_quarter"
        )
        output.write(
            _format_results(
                models,
                f"MODEL 3 — Full: {ret_col} ~ div + controls + FE",
                rows,
            )
        )

    # ── Subsample: pre-2020 vs post-2020 ──────────────────────
    for period_name, mask in [
        ("Pre-2020", panel["year"] < 2020),
        ("Post-2020", panel["year"] >= 2020),
    ]:
        sub = panel[mask]
        if len(sub) < 50:
            continue
        for ret_col in RETURN_HORIZONS:
            models = {}
            models[f"{period_name} ({ret_col})"] = _run_ols(
                sub,
                ret_col,
                ["divergence", "div_x_small", "is_smallcap"],
                fe_col="year_quarter",
            )
            output.write(
                _format_results(
                    models,
                    f"SUBSAMPLE — {period_name}: {ret_col} ~ div + div×SmallCap + FE",
                    rows,
                )
            )

    return output.getvalue(), pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run panel regressions")
    parser.add_argument("--panel", required=True, help="Path to panel.parquet")
    parser.add_argument("--output", default="outputs/regression_results.txt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    panel = _prepare_panel(Path(args.panel))
    valid = panel.dropna(subset=["divergence"])
    logger.info(
        "Panel: %d total, %d with divergence, %d with ret_5d",
        len(panel),
        len(valid),
        valid["ret_5d"].notna().sum(),
    )

    results, table_df = run_all_regressions(panel)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(results, encoding="utf-8")
    logger.info("Results saved to %s", out)

    csv_out = out.with_suffix(".csv")
    table_df.to_csv(csv_out, index=False)
    logger.info("Regression table saved to %s", csv_out)

    # Also print to stdout
    print(results)


if __name__ == "__main__":
    main()
