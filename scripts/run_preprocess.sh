#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 src/preprocess_earnings_calls.py --input raw/koyfin_transcripts_full_2006_2026.jsonl --output-dir .
python3 src/run_dataset_checks.py --output-dir .
