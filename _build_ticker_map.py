"""Build company_name → ticker mapping using yfinance Search API.

Reads unique company names from _companies.txt and writes a JSON mapping.
Filters to US exchanges only. Rate-limited to avoid throttling.
"""

import json
import sys
import time
from pathlib import Path

import yfinance as yf

US_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "OPR"}


def search_ticker(company: str) -> str | None:
    try:
        s = yf.Search(company, max_results=5)
        for q in s.quotes:
            if q.get("quoteType") == "EQUITY" and q.get("exchange") in US_EXCHANGES:
                return q["symbol"]
        # Fallback: first equity result
        for q in s.quotes:
            if q.get("quoteType") == "EQUITY" and ".BA" not in q.get("symbol", ""):
                sym = q["symbol"]
                if "." not in sym:  # Skip foreign listings like ABC.TO
                    return sym
        if s.quotes:
            return s.quotes[0].get("symbol")
    except Exception as e:
        print(f"  ERROR: {company}: {e}", file=sys.stderr)
    return None


def main():
    companies_file = Path("_companies.txt")
    output_file = Path("src/preprocessing/company_ticker_map.json")

    companies = []
    with open(companies_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                companies.append(line)

    print(f"Looking up {len(companies)} companies...")

    mapping = {}
    failed = []

    for i, company in enumerate(companies):
        ticker = search_ticker(company)
        if ticker:
            mapping[company] = ticker
            status = f"✓ {ticker}"
        else:
            failed.append(company)
            status = "✗ not found"

        if (i + 1) % 50 == 0 or i == len(companies) - 1:
            print(
                f"  [{i+1}/{len(companies)}] {len(mapping)} found, {len(failed)} missing"
            )

        # Rate limit: ~3 requests/sec
        time.sleep(0.35)

    # Save mapping
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(
        f"\nDone: {len(mapping)}/{len(companies)} mapped ({100*len(mapping)/len(companies):.1f}%)"
    )
    print(f"Saved to {output_file}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for c in failed:
            print(f"  - {c}")


if __name__ == "__main__":
    main()
