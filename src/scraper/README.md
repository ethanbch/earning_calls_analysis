# Earnings Call Project - Final Shareable Version

This folder is the clean handoff version of the project.
It contains the latest working scraper, the preprocessing pipeline, and the EDA script in a structure that is easy to share and run on another computer.

## What is included
- `src/scrape_r1000_missing.py`
  - latest scraper for missing Russell 1000 earnings call transcripts
  - resumes from checkpoints
  - skips transcripts already present in the existing raw corpus
- `src/preprocess_earnings_calls.py`
  - preprocessing pipeline from raw transcripts to cleaned transcript, segment, and chunk datasets
- `src/run_full_corpus_eda.py`
  - memory-safe EDA script for the full raw JSONL corpus
- `scripts/`
  - small shell helpers to run each step
- `checks/`
  - example EDA outputs generated from the current corpus

## What is not bundled
The raw JSONL corpus and cleaned CSV outputs are not copied into this folder because they are very large.
This keeps the handoff portable and safe to share.

Whoever receives this folder should place the large data files into the expected locations below.

## Expected structure
- `raw/`
  - place `koyfin_transcripts_full_2006_2026.jsonl` here
  - scraper writes additional transcript JSON files into `raw/r1000_missing_raw/`
- `cleaned/`
  - preprocessing outputs are written here
- `checks/`
  - EDA outputs are written here
- `logs/`
  - runtime logs
- `state/`
  - Playwright auth state and scraper checkpoints

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Input requirements
### For preprocessing and EDA
Place the raw corpus here:
- `raw/koyfin_transcripts_full_2006_2026.jsonl`

### For scraping missing transcripts
Place an authenticated Playwright storage state file here:
- `state/playwright_storage_state.json`

## Run commands
### 1. Preprocess the raw corpus
```bash
./scripts/run_preprocess.sh
```

### 2. Run the EDA
```bash
./scripts/run_eda.sh
```

### 3. Run the missing-only Russell 1000 scraper
Foreground:
```bash
PYTHON_BIN=.venv/bin/python ./scripts/start_scraper.sh --foreground
```

Background with keep-awake:
```bash
./scripts/start_scraper.sh
```

## What each step produces
### Preprocessing
Writes to `cleaned/`:
- `transcripts_clean.csv`
- `transcripts_deduplicated.csv`
- `duplicates_audit.csv`
- `segments.csv`
- `chunks.csv`

### EDA
Writes to `checks/`:
- `eda_redo_summary.json`
- `eda_redo_report.md`
- `eda_calls_per_year_full.png`
- `eda_calls_per_quarter_full.png`
- additional CSV summaries and plots

### Scraper
Writes to:
- `raw/r1000_missing_raw/` for newly scraped transcript JSON files
- `state/r1000_missing_resume.json` for resume progress
- `logs/r1000_missing.log` for scraper logs

## Notes for collaborators
- The EDA script is streaming and safe for very large JSONL files.
- The preprocessing script can create very large CSVs, so make sure there is enough disk space before running it.
- The scraper does not re-fetch transcripts already present in the raw JSONL corpus.
- The scraper is resume-friendly: if it stops, rerun the same command and it will continue from the checkpoint.
