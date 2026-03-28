#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# run_analysis.sh — Full analysis pipeline orchestrator
# Usage: bash scripts/run_analysis.sh
# ──────────────────────────────────────────────────────────
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
RESET="\033[0m"

step() { echo -e "\n${BOLD}${CYAN}[$1/5]${RESET} ${BOLD}$2${RESET}\n"; }

OUTPUTS="outputs"
CHUNKS_DIR="${OUTPUTS}/chunks"
FINBERT_DIR="${OUTPUTS}/finbert_finetuned"

mkdir -p "$OUTPUTS"

# ── Step 1: Fine-tune FinBERT ──────────────────────────────
if [ -d "$FINBERT_DIR" ] && [ -f "$FINBERT_DIR/config.json" ]; then
    echo -e "${GREEN}✓ Fine-tuned FinBERT already exists at ${FINBERT_DIR}, skipping.${RESET}"
else
    step 1 "Fine-tuning FinBERT on FinancialPhraseBank"
    uv run python -m src.analysis.finetune_finbert \
        --output "$FINBERT_DIR" \
        --epochs 3 --batch-size 32 --lr 2e-5
fi

# ── Step 2: Score sentiment ────────────────────────────────
if [ -f "${OUTPUTS}/sentiment_aggregated.parquet" ]; then
    echo -e "${GREEN}✓ Sentiment scores already exist, skipping.${RESET}"
else
    step 2 "Scoring sentiment on all chunks"
    uv run python -m src.analysis.score_sentiment \
        --chunks "$CHUNKS_DIR" \
        --model "$FINBERT_DIR" \
        --output-scores "${OUTPUTS}/sentiment_scores.parquet" \
        --output-agg "${OUTPUTS}/sentiment_aggregated.parquet" \
        --batch-size 128
fi

# ── Step 3: Fetch returns ─────────────────────────────────
if [ -f "${OUTPUTS}/returns.parquet" ]; then
    echo -e "${GREEN}✓ Returns already fetched, skipping.${RESET}"
else
    step 3 "Fetching stock returns from yfinance"
    uv run python -m src.analysis.fetch_returns \
        --transcripts "${OUTPUTS}/transcripts" \
        --output "${OUTPUTS}/returns.parquet"
fi

# ── Step 4: Build panel ───────────────────────────────────
if [ -f "${OUTPUTS}/panel.parquet" ]; then
    echo -e "${GREEN}✓ Panel already built, skipping.${RESET}"
else
    step 4 "Building panel dataset"
    uv run python -m src.analysis.build_panel \
        --sentiment "${OUTPUTS}/sentiment_aggregated.parquet" \
        --returns "${OUTPUTS}/returns.parquet" \
        --transcripts "${OUTPUTS}/transcripts" \
        --output "${OUTPUTS}/panel.parquet"
fi

# ── Step 5: Run regressions ───────────────────────────────
step 5 "Running panel regressions"
uv run python -m src.analysis.run_regressions \
    --panel "${OUTPUTS}/panel.parquet" \
    --output "${OUTPUTS}/regression_results.txt"

echo -e "\n${BOLD}${GREEN}✓ Analysis pipeline complete.${RESET}"
echo -e "  Panel:       ${OUTPUTS}/panel.parquet"
echo -e "  Regressions: ${OUTPUTS}/regression_results.txt"
