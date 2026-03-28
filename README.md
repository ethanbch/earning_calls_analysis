# Earnings Call Sentiment Analysis

Sentiment analysis on ~115 000 earnings-call transcripts (2006–2026) covering
Russell 1000, Russell 2000, and the S&P 500.
The preprocessing pipeline produces clean, FinBERT-ready text chunks and structured
metadata for panel regressions studying executive-vs-analyst sentiment divergence
and post-announcement stock returns.

## Project structure

```
Makefile                          # Build orchestration (make preprocess, make analysis)
raw/
  koyfin_transcripts_full_2006_2026.jsonl  # R1000 JSONL corpus (~48 K transcripts)
  rawrussel2K/                             # R2K Parquet directory (~32 K transcripts)
  sp/                                      # S&P 500 Parquet directory (~34 K transcripts)
src/
  pipeline/                       # Modular preprocessing pipeline
    loader.py                     #   Streaming reader: JSONL + Parquet, schema inference
    cleaner.py                    #   Scraping-artifact removal
    metadata.py                   #   Ticker / date / company extraction
    company_ticker_map.json       #   2604-entry company → ticker mapping (R1000+R2K+SP)
    deduplicator.py               #   Conservative deduplication + audit trail
    segmenter.py                  #   Speaker-turn segmentation (multiprocessing)
    sectioner.py                  #   Prepared / Q / A / O labelling
    chunker.py                    #   Token-aware ≤200-token chunking (multiprocessing)
    writer.py                     #   Parquet output with fixed pa.Schema constants
  run_preprocessing.py            # CLI entrypoint for the pipeline
  run_dataset_checks.py           # Post-processing dataset validation
  preprocess_earnings_calls.py    # Legacy monolithic preprocessing script (kept for reference)
  analysis/                       # Sentiment & regression analysis
    finetune_finbert.py           #   Fine-tune FinBERT on FinancialPhraseBank
    score_sentiment.py            #   Batch inference → chunk + transcript-level scores
    fetch_returns.py              #   Download stock returns around earnings dates
    build_panel.py                #   Merge sentiment divergence + returns → panel
    run_regressions.py            #   OLS panel regressions with FE
  scraper/                        # Koyfin transcript scraper
    run_missing_r1000.py          #   Russell 1000 missing-only scraper
    koyfin_fast/                  #   Scraper support module
scripts/                          # Shell launcher scripts
data/                             # Generated pipeline outputs (gitignored)
  clean/
    segments.parquet              #   All speaker turns
    chunks.parquet                #   NLP-ready ≤200-token chunks
    transcripts_deduplicated.parquet
    duplicates_audit.parquet
  .checkpoint                     #   Resumability state
```

## Setup

Requires **Python ≥ 3.11**.

```bash
# With uv (recommended)
uv sync

# Or with pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For the scraper only
python -m playwright install chromium
```

---

## Preprocessing pipeline

The pipeline supports two raw data formats:

| Dataset      | Format      | Location                                      | Transcripts       |
| ------------ | ----------- | --------------------------------------------- | ----------------- |
| Russell 1000 | JSONL       | `raw/koyfin_transcripts_full_2006_2026.jsonl` | ~48 K             |
| Russell 2000 | Parquet dir | `raw/rawrussel2K/`                            | ~32 K (2019–2026) |
| S&P 500      | Parquet dir | `raw/sp/`                                     | ~34 K (2006–2026) |

The format is auto-detected: a directory path triggers the Parquet reader
(`iter_batches_parquet`); a `.jsonl` file triggers the streaming JSONL reader.

The pipeline applies cleaning → metadata extraction → deduplication →
speaker segmentation → section labelling → token-aware chunking, and writes
incremental Parquet output with fixed schemas.

```bash
# All 3 datasets at once
make preprocess

# Or individually
make preprocess-r1000    # Russell 1000 (JSONL)
make preprocess-r2k      # Russell 2000 (Parquet)
make preprocess-sp       # S&P 500 (Parquet)

# Resume an interrupted run (manual)
uv run python -m src.run_preprocessing \
  --input raw/rawrussel2K --output-dir data/r2k --resume
```

| Flag           | Description                                            |
| -------------- | ------------------------------------------------------ |
| `--input`      | Path to JSONL file **or** Parquet directory (required) |
| `--output-dir` | Output directory (default: `outputs`)                  |
| `--max-tokens` | Max BERT tokens per chunk (default: `200`)             |
| `--batch-size` | Transcripts per batch (default: `1000`)                |
| `--workers`    | Parallel workers for segmenter/chunker (default: `4`)  |
| `--resume`     | Resume from last checkpoint                            |
| `--force`      | Ignore checkpoint, restart from scratch                |

### Pipeline steps

1. **Loader** (`loader.py`): Auto-detects input format. For JSONL, streams line by
   line; for Parquet directories, reads all files into memory (manageable: ≤35 files
   × ~2 000 rows each). Column names differ between formats but are normalised via
   `SCHEMA_MAP` (e.g. Parquet `body` → `raw_text`, `transcript_subheader` → `company`,
   `subheader` → `date`). Assigns a deterministic `transcript_id` (SHA-1 hash of
   source file + full raw text + global line/row number) to each record.

2. **Cleaner** (`cleaner.py`): Removes HTML tags, control characters, Koyfin-specific
   scraping artifacts, and normalises whitespace. Produces a `clean_text` column while
   preserving `raw_text` for auditability.

3. **Metadata** (`metadata.py`): Extracts `company`, `ticker`, `event_date`, `year`,
   `quarter`, `year_quarter`, and `title`. See the [ticker resolution](#ticker-resolution)
   section below for details on how tickers are resolved.

4. **Deduplicator** (`deduplicator.py`): Flags duplicates using three conservative rules
   (exact raw text match, same company + date + title similarity, same company + quarter
   - length + prefix match). Flagged records are written to `duplicates_audit.parquet`
     for traceability; only non-duplicates continue downstream.

5. **Segmenter** (`segmenter.py`): Splits each transcript into individual speaker turns.
   Handles the Koyfin inline format (e.g. `"Jon RavivExecutive"`) where speaker name and
   role are concatenated without a separator. Uses multiprocessing for throughput.

6. **Sectioner** (`sectioner.py`): Labels each segment with a section tag:
   - **Prepared**: Executive speaking before the Q&A session
   - **Q**: Analyst asking a question during Q&A
   - **A**: Executive answering during Q&A
   - **O**: Operator / moderator

   The Q&A boundary is detected when the first Analyst turn appears, or when the Operator
   introduces a question-and-answer segment. To avoid false triggers,
   the Operator can only flip the Q&A flag after at least one Executive turn has been seen
   (prevents `[Operator Instructions]` at the top of the call from activating Q&A
   prematurely).

7. **Chunker** (`chunker.py`): Splits segments into chunks of ≤ 200 BERT tokens
   (configurable). Respects sentence boundaries; falls back to word-level splitting for
   sentences that exceed the limit. Uses `tokenizers` (Rust-based) with `encode_batch`
   for throughput — all sentences in a segment are tokenized in a single batch call.
   A persistent `multiprocessing.Pool` is shared across batches so the tokenizer is
   loaded once per worker (not once per batch).

8. **Writer** (`writer.py`): Writes to Parquet files with fixed `pyarrow.Schema`
   constants, ensuring column types are consistent across batches. Supports checkpointing
   so runs can be resumed with `--resume`.

### Output columns

| Column                            | Description                            |
| --------------------------------- | -------------------------------------- |
| `transcript_id`                   | Unique hash per transcript             |
| `ticker`                          | Stock ticker (e.g. AAPL)               |
| `company`                         | Company name                           |
| `event_date`                      | Earnings call date (YYYY-MM-DD)        |
| `year`, `quarter`, `year_quarter` | e.g. 2023, 3, 2023Q3                   |
| `speaker_type`                    | Executive / Analyst / Operator / Other |
| `section`                         | Prepared / Q / A / O                   |
| `segment_text`                    | One full speaker turn                  |
| `chunk_text`                      | Segment split into ≤200 token chunks   |
| `n_tokens`                        | BERT token count per chunk             |

---

## Ticker resolution

The raw Koyfin data **does not include a ticker field** — records only provide a
company name and a title. No `(TICKER)`, `NYSE:`, or `NASDAQ:` pattern appears in
the data.

### Why we need tickers

Tickers are essential for downstream analysis:

- **Panel regressions** join sentiment scores to CRSP/Compustat financial data via ticker
- **Event studies** map earnings call dates to stock return windows
- **Cross-sectional grouping** by firm requires a stable identifier (company names vary
  across filings: `"The Kroger Co."` vs `"Kroger"`)

### How we built the mapping

**R1000 (original):**

1. Extracted all 987 unique company names from the JSONL corpus
2. Queried the Yahoo Finance search API (`yfinance.Search`) for each name,
   filtering to US equity exchanges (NYSE, NASDAQ, AMEX)
3. **982 / 987 resolved automatically** (99.5% hit rate)
4. 5 remaining entities were mapped manually (subsidiaries/MLPs without own listings)

**R2K extension:**

1. Collected 1 620 unique company names from `raw/rawrussel2K/`
2. Filtered out the 7 already in the R1000 map → **1 617 new lookups**
3. Ran the same `yfinance.Search` logic via `_extend_ticker_map.py` → **1 613 / 1 617 resolved automatically** (99.8%)
4. 4 remaining entities mapped manually: `Collier Creek Holdings` → `CCH`, `Green Plains Partners LP` → `GPP`, `Rbb Asset Management Company` → `RBB`, `Royal Business Bank` → `RBB`
5. Map merged additively (existing entries take priority)

**SP — no extension needed:** all 506 unique S&P companies were already present
in the R1000 map (S&P 500 is a strict subset of R1000 by construction).

### Where the mapping lives

The mapping is stored at `src/pipeline/company_ticker_map.json`. It is loaded once
at import time by `metadata.py`. The lookup order is:

1. Parenthesised ticker in the transcript header, e.g. `(AAPL)`
2. Explicit `Ticker:` / `Symbol:` label in the header
3. **Company name → ticker JSON mapping** (primary source for this dataset)

To add or correct entries, edit `company_ticker_map.json` directly.
To rebuild the R1000 base map from scratch, use `_build_ticker_map.py`.
To extend the map with new Parquet sources, use `_extend_ticker_map.py`.

---

## Design decisions

### Why `tokenizers` instead of `transformers.AutoTokenizer`

We use the Rust-based `tokenizers` library directly (not via HuggingFace
`transformers`). This avoids pulling in PyTorch/TensorFlow as a dependency and
gives us `encode_batch` for vectorized tokenization. The `bert-base-uncased`
WordPiece vocabulary is context-free at the word level, so summing per-sentence
token counts equals the count for the concatenated text — no accuracy loss from
batching.

### Why a persistent multiprocessing pool

The chunker needs BERT tokenization, which requires loading a ~230 KB vocabulary
file. Without a persistent pool, each batch would spawn N workers × load the
tokenizer from disk (or network) — on 49 batches with 4 workers, that's 196
initializations. The persistent pool loads the tokenizer **4 times total** (once
per worker at startup) and reuses it across all batches.

### Parquet vs JSONL source format

The R1000 corpus was scraped to JSONL (one JSON object per line), while the R2K
and SP datasets come as directories of Parquet files (~2 000 rows each).
The pipeline auto-detects the format:

- **Directory path** → `loader.iter_batches_parquet()`: reads all `*.parquet` files
  at once (≤35 files × 2 000 rows = manageable, ~150 MB peak), normalises column
  names via `SCHEMA_MAP`, then yields batches.
- **`.jsonl` file** → `loader.iter_batches()`: true streaming, one-line-at-a-time.

**Parquet column mapping** (Koyfin parquet schema → internal canonical name):

| Parquet column         | Canonical name | Notes                                     |
| ---------------------- | -------------- | ----------------------------------------- |
| `body`                 | `raw_text`     | same inline speaker format                |
| `transcript_subheader` | `company`      | company name only                         |
| `subheader`            | `date`         | e.g. `"Friday, January 3, 2020 11:00 AM"` |
| `speakers`             | `participants` | comma-separated speaker list              |
| `title`                | `title`        | same                                      |
| `list_item`            | _(dropped)_    | Koyfin search snippet, unused             |

The `subheader` date string includes a weekday prefix that `pd.to_datetime()` cannot
parse directly; a regex fallback in `metadata.py` extracts `January 3, 2020` and
converts it correctly.

### Why segment counts ≠ text volume percentages

A naive "trigger Q&A when Operator mentions 'question'" fails because many calls
start with an Operator segment containing `[Operator Instructions]` — which
mentions questions before any Executive has spoken. The `has_seen_executive` guard
ensures the Q&A flag only activates after prepared remarks have actually begun,
giving a Prepared text volume of ~20–27% per transcript (validated on 10 transcripts).

### Why segment counts ≠ text volume percentages

Prepared remarks represent only ~3% of segments but ~20–25% of text volume. This is
expected: prepared remarks are long monologues (~5 700 chars/segment on average),
while Q&A responses are short exchanges (~460 chars for analysts, ~1 600 chars for
executives). The section distribution is validated by text volume, not segment count.

### Transcript ID generation

Each `transcript_id` is a 16-character hex string (truncated SHA-1) derived from:
`source_file + full_raw_text + global_JSONL_line_number`. Using the full text (not
a prefix) and the global line number (not the batch-local index) prevents hash
collisions that would cascade into duplicate `segment_id` and `chunk_id` values.

---

## Dataset checks

After preprocessing, run the validation suite:

```bash
make checks
```

Validated shape, critical nulls, token stats, section distribution,
temporal/company coverage, ticker fill rate, and duplicates audit

### Latest checks snapshot (`make checks`)

**All checks passed** on all three indexes.

| Index | Transcripts | Chunks | Period       |
| ----- | ----------: | -----: | ------------ |
| R1000 |      48,250 |   9.3M | 2006 -> 2026 |
| R2K   |      32,786 |   2.2M | 2020 -> 2026 |
| SP    |      34,171 |   3.6M | 2006 -> 2026 |

**Total: 115,207 transcripts**.

Duplicates flagged by audit are negligible and already handled in the cleaned output:

- R1000: 19
- R2K: 23
- SP: 109

All are <0.3% of transcript volume, and deduplication is tracked via
`transcripts_deduplicated.parquet` + `duplicates_audit.parquet`.

---

## Analysis pipeline

The analysis pipeline measures whether **executive-vs-analyst sentiment divergence**
during earnings calls predicts **post-announcement stock returns**, and whether the
effect differs between large-cap (R1000/SP) and small-cap (R2K) firms.

### Pipeline steps

| Step | Script                | Input                              | Output                                                             |
| ---- | --------------------- | ---------------------------------- | ------------------------------------------------------------------ |
| 1    | `finetune_finbert.py` | HuggingFace `financial_phrasebank` | `outputs/finbert_finetuned/`                                       |
| 2    | `score_sentiment.py`  | Chunks + fine-tuned model          | `outputs/sentiment_scores.parquet`, `sentiment_aggregated.parquet` |
| 3    | `fetch_returns.py`    | Transcripts metadata               | `outputs/returns.parquet`                                          |
| 4    | `build_panel.py`      | Sentiment + returns                | `outputs/panel.parquet`                                            |
| 5    | `run_regressions.py`  | Panel                              | `outputs/regression_results.txt`                                   |

### Quick start

```bash
make install      # Install dependencies (uv sync)
make analysis     # Run the full pipeline (skips already-completed steps)
```

Or run steps individually:

```bash
make finetune     # 1. Fine-tune FinBERT (3 epochs, MPS on Mac)
make sentiment    # 2. Score sentiment on all chunks
make returns      # 3. Fetch stock returns around earnings dates
make panel        # 4. Build the regression panel
make regressions  # 5. Run regressions
```

Override hyperparameters:

```bash
make finetune EPOCHS=5 FT_BATCH=16 LR=1e-5
make sentiment SCORE_BATCH=64
```

### Sentiment model

We fine-tune **ProsusAI/finbert** on Financial PhraseBank (all-agree subset)
using the parquet-native mirror:

- Dataset: `gtfintechlab/financial_phrasebank_sentences_allagree` (config `5768`)
- Split: 85% train / 15% validation (seed=42)
- Labels: 3 classes (`negative`, `neutral`, `positive`)
- Inference device is configurable in scoring (`--device cpu|mps|cuda`, default `cpu`)

What this step does in practice:

1. Loads phrase-level financial sentiment examples (`sentence`, `labels`)
2. Tokenizes with FinBERT tokenizer (`max_length=128`, truncation + fixed padding)
3. Fine-tunes `ProsusAI/finbert` for 3 epochs (cross-entropy objective)
4. Evaluates at the end of each epoch (accuracy + macro F1 + eval loss)
5. Saves the best checkpoint and final model artifacts to `outputs/finbert_finetuned/`

The fine-tuned model classifies each chunk as positive / neutral / negative and
assigns a continuous sentiment score in [-1, +1].

#### FinBERT training results (latest run)

Source: `outputs/finbert_finetuned/checkpoint-507/trainer_state.json`

| Epoch | Eval accuracy | Eval F1 macro | Eval loss |
| ----- | ------------- | ------------- | --------- |
| 1     | 0.97899       | 0.96653       | 0.09335   |
| 2     | 0.96639       | 0.96655       | 0.13215   |
| 3     | 0.97479       | **0.96778**   | 0.11982   |

Best checkpoint (selected by `f1_macro`):

- `checkpoint-507`
- `best_metric`: `0.9677767925`
- `train_batch_size`: `8`
- `num_train_epochs`: `3`

At the transcript level, chunks are grouped by `speaker_role` (exec / analyst / other)
and we compute mean, median, std, and count per group.

#### Low-RAM / fast scoring mode (8 GB machine)

Because the chunk corpora are very large (`r1000` alone has ~9.29M chunks), a full
pass can be extremely slow on a laptop-class machine (empirical estimate was ~44h
for R1000 in our setup). For iterative research/debug cycles, we use a bounded,
reproducible scoring mode:

1. Streaming parquet read with `ParquetFile.iter_batches(...)`
2. Per-batch random sampling with fixed seed (`SAMPLE_FRAC = 0.01`, `random_state=42`)
3. Early stop per file with `--max-batches N`

This gives predictable runtime and bounded memory while preserving representative
coverage for exploratory analysis.

Run used for the current artifacts:

```bash
uv run python -m src.analysis.score_sentiment \
  --chunks outputs/*/clean/chunks.parquet \
  --model outputs/finbert_finetuned \
  --output outputs/sentiment_scores.parquet \
  --output-agg outputs/sentiment_aggregated.parquet \
  --batch-size 256 --read-batch-size 50000 --max-batches 10 --device cpu
```

Observed logs/results for this run:

- Processed 3 files (`r1000`, `r2k`, `sp`)
- Read cap per file: `10 × 50,000 = 500,000` rows before stop
- Sampling: 1% per read batch (fixed seed)
- Output chunk-level rows: `15,000`
- Output transcript-level rows: `11,080`

Most informative score diagnostics (from `outputs/sentiment_scores.parquet` and
`outputs/sentiment_aggregated.parquet`):

- Chunk-level label mix: `neutral 52.83%`, `negative 36.95%`, `positive 10.21%`
- Chunk-level sentiment score: mean `-0.2674`, std `0.6326`, median `0.0000`
- Confidence (max class probability): mean `0.9500`, median `0.9953`, p90 `0.9985`
- By speaker type (chunk-level mean score):
  - Executive: `-0.3518` (n=`9,732`)
  - Analyst: `-0.1586` (n=`3,429`)
  - Operator: `-0.0018` (n=`1,683`)
- Transcript-level mean score by role:
  - exec: `-0.3456` (n=`6,566`)
  - analyst: `-0.1581` (n=`2,841`)
  - other: `-0.0198` (n=`1,673`)
- Exec-analyst divergence (`exec - analyst`): n=`1,505`, mean `-0.1759`, std `0.8028`

Note: this mode is designed for fast iteration. For final production estimates,
increase `--max-batches` (or remove it) and optionally raise sample fraction in code.

#### Returns fetching diagnostics (latest run)

`fetch_returns.py` was run on the full transcript universe and completed successfully.

- Tickers processed: `2,571`
- Rows written: `115,172` (`outputs/returns.parquet`)
- Fill rate: `98.9%` for `ret_1d`, `ret_3d`, `ret_5d`, and `ret_10d`

Yahoo Finance failures were limited and expected for a small subset of symbols,
typically due to delisted names, stale ticker mappings, or non-equity symbols
with no historical price series for the requested event window.

Examples seen in logs: `ARS`, `TAPE`, `UAHC`, `EMEA`, `CFO`, `ECRR`, `ACSI`.

Practical implication: with ~99% return coverage, the panel remains highly usable for
regression analysis without aggressive imputation.

### Panel construction

**Divergence** = `score_exec − score_analyst` per transcript.

Each observation is one earnings call, enriched with:

- Cumulative stock returns at +1d, +3d, +5d, +10d (from yfinance)
- `is_smallcap` flag (1 if source is R2K directory)
- Time identifiers (year, quarter, year_quarter)

Latest `make panel` run diagnostics:

- Sentiment input rows: `11,080`
- Rows after exec/analyst pivot: `7,902`
- Exec score fill rate: `83.1%`
- Analyst score fill rate: `36.0%`
- Returns rows loaded: `86,860`
- Final panel shape: `7,902 × 18` (`outputs/panel.parquet`)
- Divergence (`exec - analyst`): mean `-0.176`, std `0.803`, non-null `1,505`

### Regression specifications

Three models estimated with HC1 robust standard errors + quarter fixed effects:

1. **Baseline**: `ret_Xd = α + β₁·div + FE_quarter + ε`
2. **Size interaction**: `ret_Xd = α + β₁·div + β₂·div×SmallCap + β₃·SmallCap + FE`
3. **Full**: adds `score_exec`, `score_analyst`, `n_chunks_exec` as controls

Each model is run on `ret_1d`, `ret_3d`, `ret_5d`. Subsample analysis splits
pre-2020 vs post-2020.

### Key empirical results

Main finding:

- Executive-analyst sentiment divergence significantly predicts post-announcement
  returns at 1, 3, and 5 days.
- On `ret_5d`, the baseline divergence coefficient is `+0.011` (`p=0.0001`).
- Interpretation: when executives are less pessimistic (or more optimistic) than
  analysts, subsequent returns are higher.

Key coefficients:

| Model               | Variable                        | Coef   | p-value       |
| ------------------- | ------------------------------- | ------ | ------------- |
| Baseline            | divergence -> ret_5d            | +0.011 | 0.0001 \*\*\* |
| Size interaction    | divergence x SmallCap -> ret_5d | +0.025 | 0.0013 \*\*\* |
| Post-2020 subsample | divergence x SmallCap -> ret_5d | +0.021 | 0.014 \*\*    |

Three takeaways for the report:

1. The effect is stronger for small caps, consistent with lower analyst coverage
   and higher marginal value of call-level information.
2. The effect is weak pre-2020 but much stronger post-2020, potentially linked to
   the expansion of NLP-driven and sentiment-based institutional trading.
3. `R² ~ 5-6%` for `ret_5d` is economically meaningful in return prediction,
   where even low explanatory power is often material.

Suggested report phrasing:

> "A one-point increase in executive-analyst divergence is associated with an
> additional 1.1% return over 5 days. This effect is approximately doubled for
> small-cap firms (+2.5%), suggesting that information embedded in earnings calls
> is particularly valuable for less-covered stocks."

---

## Russell 1000 scraper

Requires an authenticated Koyfin Playwright storage state and the existing full
JSONL corpus to compare against.

```bash
# Foreground
python src/scraper/run_missing_r1000.py

# Background with keep-awake
./scripts/start_r1000_scraper.sh
```
