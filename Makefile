# ──────────────────────────────────────────────────────────
# Makefile — Preprocessing & Analysis pipelines
# ──────────────────────────────────────────────────────────
#
# Usage:
#   make preprocess          # Preprocess all 3 datasets
#   make preprocess-r1000    # Russell 1000 only
#   make analysis            # Full analysis pipeline (5 steps)
#   make finetune            # Step 1: fine-tune FinBERT
#   make sentiment           # Step 2: score sentiment
#   make returns             # Step 3: fetch stock returns
#   make panel               # Step 4: build panel
#   make regressions         # Step 5: run regressions
#   make stats-sentiment     # Quick sentiment stats summary
#   make checks              # Run dataset validation
#   make clean-analysis      # Remove analysis outputs
#   make help                # Show all targets
#
# ──────────────────────────────────────────────────────────

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Paths ──────────────────────────────────────────────────
RAW_R1000   := raw/koyfin_transcripts_full_2006_2026.jsonl
RAW_R2K     := raw/rawrussel2K
RAW_SP      := raw/sp

OUT_R1000   := data
OUT_R2K     := data/r2k
OUT_SP      := outputs/sp
OUTPUTS     := outputs

FINBERT_DIR := $(OUTPUTS)/finbert_finetuned
CHUNKS_FILES := outputs/*/clean/chunks.parquet
TRANSCRIPTS_FILES := outputs/*/clean/transcripts_deduplicated.parquet

# ── Preprocessing hyperparams (8GB RAM defaults) ──────────
MAX_TOKENS  ?= 200
BATCH_SIZE  ?= 250
WORKERS     ?= 2

# ── Analysis hyperparams (8GB RAM defaults) ───────────────
EPOCHS      ?= 3
FT_BATCH    ?= 4
LR          ?= 2e-5
SCORE_BATCH ?= 8
SCORE_READ_BATCH ?= 2000

# ── Colors ─────────────────────────────────────────────────
CYAN  := \033[0;36m
GREEN := \033[0;32m
BOLD  := \033[1m
RESET := \033[0m

# ================================================================
#  PREPROCESSING
# ================================================================

.PHONY: preprocess preprocess-r1000 preprocess-r2k preprocess-sp checks

preprocess: preprocess-r1000 preprocess-r2k preprocess-sp ## Preprocess all 3 datasets
	@echo -e "$(GREEN)$(BOLD)✓ All preprocessing complete.$(RESET)"

preprocess-r1000: ## Preprocess Russell 1000 (JSONL)
	@echo -e "$(CYAN)$(BOLD)[Preprocessing] Russell 1000$(RESET)"
	uv run python -m src.run_preprocessing \
		--input $(RAW_R1000) --output-dir $(OUT_R1000) \
		--max-tokens $(MAX_TOKENS) --batch-size $(BATCH_SIZE) --workers $(WORKERS) \
		--force

preprocess-r2k: ## Preprocess Russell 2000 (Parquet)
	@echo -e "$(CYAN)$(BOLD)[Preprocessing] Russell 2000$(RESET)"
	uv run python -m src.run_preprocessing \
		--input $(RAW_R2K) --output-dir $(OUT_R2K) \
		--max-tokens $(MAX_TOKENS) --batch-size $(BATCH_SIZE) --workers $(WORKERS) \
		--force

preprocess-sp: ## Preprocess S&P 500 (Parquet)
	@echo -e "$(CYAN)$(BOLD)[Preprocessing] S&P 500$(RESET)"
	uv run python -m src.run_preprocessing \
		--input $(RAW_SP) --output-dir $(OUT_SP) \
		--max-tokens $(MAX_TOKENS) --batch-size $(BATCH_SIZE) --workers $(WORKERS) \
		--force

checks: ## Run dataset validation checks
	@echo -e "$(CYAN)$(BOLD)[Checks] Running dataset validation$(RESET)"
	uv run python src/run_dataset_checks.py --data-dir outputs/r1000/clean --mode quick --sample-rows 100000
	uv run python src/run_dataset_checks.py --data-dir outputs/r2k/clean --mode quick --sample-rows 100000
	uv run python src/run_dataset_checks.py --data-dir outputs/sp/clean --mode quick --sample-rows 100000

# ================================================================
#  ANALYSIS
# ================================================================

.PHONY: analysis finetune sentiment returns panel regressions stats-sentiment

analysis: finetune sentiment returns panel regressions ## Run full analysis pipeline
	@echo -e "$(GREEN)$(BOLD)✓ Analysis pipeline complete.$(RESET)"
	@echo "  Panel:       $(OUTPUTS)/panel.parquet"
	@echo "  Regressions: $(OUTPUTS)/regression_results.txt"

$(FINBERT_DIR)/config.json:
	@echo -e "$(CYAN)$(BOLD)[1/5] Fine-tuning FinBERT on FinancialPhraseBank$(RESET)"
	@mkdir -p $(OUTPUTS)
	uv run python -m src.analysis.finetune_finbert \
		--output $(FINBERT_DIR) --epochs $(EPOCHS) --batch-size $(FT_BATCH) --lr $(LR)

finetune: $(FINBERT_DIR)/config.json ## Step 1: Fine-tune FinBERT

$(OUTPUTS)/sentiment_aggregated.parquet: $(FINBERT_DIR)/config.json
	@echo -e "$(CYAN)$(BOLD)[2/5] Scoring sentiment on all chunks$(RESET)"
	uv run python -m src.analysis.score_sentiment \
		--chunks $(CHUNKS_FILES) --model $(FINBERT_DIR) \
		--output $(OUTPUTS)/sentiment_scores.parquet \
		--output-agg $(OUTPUTS)/sentiment_aggregated.parquet \
		--batch-size $(SCORE_BATCH) \
		--read-batch-size $(SCORE_READ_BATCH)

sentiment: $(OUTPUTS)/sentiment_aggregated.parquet ## Step 2: Score sentiment

$(OUTPUTS)/returns.parquet:
	@echo -e "$(CYAN)$(BOLD)[3/5] Fetching stock returns from yfinance$(RESET)"
	uv run python -m src.analysis.fetch_returns \
		--transcripts $(TRANSCRIPTS_FILES) --output $(OUTPUTS)/returns.parquet

returns: $(OUTPUTS)/returns.parquet ## Step 3: Fetch stock returns

$(OUTPUTS)/panel.parquet: $(OUTPUTS)/sentiment_aggregated.parquet $(OUTPUTS)/returns.parquet
	@echo -e "$(CYAN)$(BOLD)[4/5] Building panel dataset$(RESET)"
	uv run python -m src.analysis.build_panel \
		--sentiment $(OUTPUTS)/sentiment_aggregated.parquet \
		--returns $(OUTPUTS)/returns.parquet \
		--transcripts $(TRANSCRIPTS_FILES) \
		--output $(OUTPUTS)/panel.parquet

panel: $(OUTPUTS)/panel.parquet ## Step 4: Build panel

$(OUTPUTS)/regression_results.txt: $(OUTPUTS)/panel.parquet
	@echo -e "$(CYAN)$(BOLD)[5/5] Running panel regressions$(RESET)"
	uv run python -m src.analysis.run_regressions \
		--panel $(OUTPUTS)/panel.parquet --output $(OUTPUTS)/regression_results.txt

regressions: $(OUTPUTS)/regression_results.txt ## Step 5: Run regressions

stats-sentiment: ## Quick descriptive stats on sentiment outputs
	@echo -e "$(CYAN)$(BOLD)[Stats] Sentiment output summary$(RESET)"
	uv run python scripts/sentiment_stats.py

# ================================================================
#  UTILITIES
# ================================================================

.PHONY: install clean-analysis clean-preprocess clean help

install: ## Install all dependencies
	uv sync

clean-analysis: ## Remove analysis outputs (keeps preprocessing)
	rm -f $(OUTPUTS)/sentiment_scores.parquet \
	      $(OUTPUTS)/sentiment_aggregated.parquet \
	      $(OUTPUTS)/returns.parquet \
	      $(OUTPUTS)/panel.parquet \
	      $(OUTPUTS)/regression_results.txt \
	      $(OUTPUTS)/regression_results.csv
	rm -rf $(FINBERT_DIR)
	@echo -e "$(GREEN)✓ Analysis outputs cleaned.$(RESET)"

clean-preprocess: ## Remove preprocessing outputs
	rm -rf $(OUT_R1000)/clean $(OUT_R2K)/clean $(OUT_SP)/clean
	rm -f $(OUT_R1000)/.checkpoint $(OUT_R2K)/.checkpoint $(OUT_SP)/.checkpoint
	@echo -e "$(GREEN)✓ Preprocessing outputs cleaned.$(RESET)"

clean: clean-analysis clean-preprocess ## Remove all generated outputs

help: ## Show available targets
	@echo -e "$(BOLD)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
