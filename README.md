# Shareable Earnings Call Project

## What is included
- `raw/`: original raw corpus and the additional Russell 1000 missing transcripts that were scraped later.
- `cleaned/`: transcript-level, segment-level, chunk-level, and duplicate-audit cleaned outputs.
- `checks/`: validation summaries, EDA tables, and plots.
- `src/preprocess_earnings_calls.py`: preprocessing pipeline.
- `src/run_dataset_checks.py`: EDA / dataset-check pipeline.
- `src/scraper/run_missing_r1000.py`: current working Russell 1000 missing-only scraper.
- `src/scraper/koyfin_fast/`: minimal support module used by the scraper.
- `requirements.txt`: Python dependencies.
- `scripts/`: simple launcher scripts.

## Important note on disk usage
This handoff folder uses hardlinks for the very large data files so it does not duplicate the raw and cleaned datasets on this machine. That keeps the project shareable without consuming another ~50GB locally.

## Recommended setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Preprocessing / EDA
```bash
python src/preprocess_earnings_calls.py --input raw/koyfin_transcripts_full_2006_2026.jsonl --output-dir .
python src/run_dataset_checks.py --output-dir .
```

## Russell 1000 missing-only scraper
This scraper expects an authenticated Koyfin Playwright storage state and an existing full JSONL corpus to compare against.

Run in foreground:
```bash
python src/scraper/run_missing_r1000.py
```

Run in background with keep-awake:
```bash
./scripts/start_r1000_scraper.sh
```

## Current known limitations
- The Russell 1000 scraper is working but not complete; the last run added 56 new transcripts and also produced many row-opening failures.
- The scraper depends on the saved Koyfin search state having the correct `Russell 1000 (Large-Cap)` filter active.
- The cleaned CSV outputs are very large.
