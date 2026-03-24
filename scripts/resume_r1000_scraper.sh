#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
nohup /usr/bin/caffeinate -dimsu python3 src/scraper/run_missing_r1000.py > scrape_bg.out 2>&1 < /dev/null &
echo $!
