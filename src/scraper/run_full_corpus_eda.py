"""Memory-safe transcript-level EDA for the full earnings call corpus.

Reads a raw JSONL corpus in streaming mode and writes summary tables/plots to
`<output-dir>/checks`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def norm_company(value: str | None) -> str:
    text = (value or "").strip().lower().replace("&", "and")
    return " ".join(text.split())


def word_bin(n: int) -> str:
    lo = (n // 1000) * 1000
    return f"{lo}-{lo + 999}"


def char_bin(n: int) -> str:
    lo = (n // 10000) * 10000
    return f"{lo}-{lo + 9999}"


def parse_event_datetime(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10 and raw[4] == "-":
            return datetime.fromisoformat(raw)
        return datetime.strptime(raw, "%A, %B %d, %Y %I:%M %p")
    except Exception:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow([row.get(h, "") for h in header])
            else:
                writer.writerow(row)


def run_eda(input_path: Path, output_dir: Path, index_name: str) -> dict:
    checks_dir = output_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)

    company_counts = Counter()
    company_norm_counts = Counter()
    year_quarter_counts = Counter()
    year_counts = Counter()
    month_counts = Counter()
    weekday_counts = Counter()
    title_counts = Counter()
    text_hash_counts = Counter()
    company_quarters = defaultdict(set)
    word_bins = Counter()
    char_bins = Counter()
    raw_variants = defaultdict(set)
    missing_counts = Counter()

    records = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            records += 1
            company = (obj.get("company") or obj.get("company_name") or "").strip()
            title = (obj.get("title") or "").strip()
            text = obj.get("transcript_text") or obj.get("text") or ""
            dt_raw = (obj.get("derived_event_date") or obj.get("datetime") or "").strip()

            if not company:
                missing_counts["company"] += 1
            if not title:
                missing_counts["title"] += 1
            if not dt_raw:
                missing_counts["event_date"] += 1
            if not text:
                missing_counts["transcript_text"] += 1

            if company:
                company_counts[company] += 1
                company_norm = norm_company(company)
                company_norm_counts[company_norm] += 1
                raw_variants[company_norm].add(company)

            if title:
                title_counts[title] += 1

            if text:
                text_hash_counts[hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()] += 1
                word_bins[word_bin(len(text.split()))] += 1
                char_bins[char_bin(len(text))] += 1

            event_dt = parse_event_datetime(dt_raw)
            if event_dt is not None:
                year = event_dt.year
                quarter = ((event_dt.month - 1) // 3) + 1
                year_quarter = f"{year}Q{quarter}"
                year_quarter_counts[year_quarter] += 1
                year_counts[str(year)] += 1
                month_counts[event_dt.strftime("%Y-%m")] += 1
                weekday_counts[event_dt.strftime("%A")] += 1
                if company:
                    company_quarters[company].add(year_quarter)

    variant_rows = []
    for normalized, variants in raw_variants.items():
        if len(variants) > 1:
            variant_rows.append(
                {
                    "normalized_company": normalized,
                    "n_variants": len(variants),
                    "variants": " | ".join(sorted(variants)),
                }
            )
    variant_rows.sort(key=lambda x: (-x["n_variants"], x["normalized_company"]))

    n_quarters_total = len(year_quarter_counts)
    coverage_rows = []
    for company, quarters in company_quarters.items():
        coverage_rows.append(
            {
                "company": company,
                "n_quarters_present": len(quarters),
                "coverage_ratio": round(len(quarters) / n_quarters_total, 6) if n_quarters_total else 0,
            }
        )
    coverage_rows.sort(key=lambda x: (-x["n_quarters_present"], x["company"]))
    coverage_rows = coverage_rows[:100]

    freq_values = sorted(company_counts.values())
    summary = {
        "n_transcripts": records,
        "n_unique_companies_raw": len(company_counts),
        "n_unique_companies_normalized": len(company_norm_counts),
        "n_unique_titles": len(title_counts),
        "missing_counts": dict(missing_counts),
        "company_frequency": {
            "min": min(freq_values) if freq_values else 0,
            "median": freq_values[len(freq_values) // 2] if freq_values else 0,
            "max": max(freq_values) if freq_values else 0,
            "freq1_count": sum(1 for v in freq_values if v == 1),
            "freq1_share": round(sum(1 for v in freq_values if v == 1) / len(freq_values), 4) if freq_values else 0,
        },
        "exact_duplicate_transcripts": sum(v for v in text_hash_counts.values() if v > 1),
        "exact_duplicate_groups": sum(1 for v in text_hash_counts.values() if v > 1),
        "date_range": {
            "min": min(year_counts) if year_counts else None,
            "max": max(year_counts) if year_counts else None,
        },
        "n_year_quarters": len(year_quarter_counts),
        "n_company_variant_groups": len(variant_rows),
        "top_companies": company_counts.most_common(25),
        "top_year_quarters": sorted(year_quarter_counts.items(), key=lambda x: (-x[1], x[0]))[:20],
        "weekday_distribution": dict(weekday_counts),
        "index_name": index_name,
    }

    (checks_dir / "eda_redo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (checks_dir / "eda_extra_summary.json").write_text(
        json.dumps(
            {
                "duplicate_titles_count": sum(1 for v in title_counts.values() if v > 1),
                "top_12_months": month_counts.most_common(12),
                "top_10_years": sorted(year_counts.items(), key=lambda x: int(x[0]))[:10],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(checks_dir / "eda_top_companies_full.csv", ["company", "n_calls"], company_counts.most_common())
    write_csv(checks_dir / "eda_calls_per_quarter_full.csv", ["year_quarter", "n_calls"], sorted(year_quarter_counts.items(), key=lambda x: (-x[1], x[0])))
    write_csv(checks_dir / "eda_calls_per_year_full.csv", ["year", "n_calls"], sorted(year_counts.items(), key=lambda x: int(x[0])))
    write_csv(checks_dir / "eda_calls_per_month_full.csv", ["year_month", "n_calls"], sorted(month_counts.items()))
    write_csv(checks_dir / "eda_calls_by_weekday_full.csv", ["weekday", "n_calls"], weekday_counts.items())
    write_csv(checks_dir / "eda_company_variant_groups_full.csv", ["normalized_company", "n_variants", "variants"], variant_rows)
    write_csv(checks_dir / "eda_company_coverage_top100_full.csv", ["company", "n_quarters_present", "coverage_ratio"], coverage_rows)
    write_csv(checks_dir / "eda_transcript_word_bins_full.csv", ["word_bin", "n_calls"], sorted(word_bins.items(), key=lambda x: int(x[0].split("-")[0])))
    write_csv(checks_dir / "eda_transcript_char_bins_full.csv", ["char_bin", "n_calls"], sorted(char_bins.items(), key=lambda x: int(x[0].split("-")[0])))
    write_csv(checks_dir / "eda_top_duplicate_titles_full.csv", ["title", "n_rows"], [(k, v) for k, v in title_counts.items() if v > 1])

    subtitle = f"{index_name} | Date Range: {summary['date_range']['min']} to {summary['date_range']['max']}"

    quarter_items = sorted(year_quarter_counts.items(), key=lambda x: x[0])
    if quarter_items:
        plt.figure(figsize=(14, 6))
        plt.plot([k for k, _ in quarter_items], [v for _, v in quarter_items], linewidth=1.8)
        plt.suptitle("Earnings Calls per Quarter", fontsize=14, y=0.98)
        plt.title(subtitle, fontsize=10, pad=10)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(checks_dir / "eda_calls_per_quarter_full.png", dpi=150, bbox_inches="tight")
        plt.close()

    year_items = sorted(year_counts.items(), key=lambda x: int(x[0]))
    if year_items:
        plt.figure(figsize=(15, 5))
        plt.bar([k for k, _ in year_items], [v for _, v in year_items])
        plt.suptitle("Earnings Calls per Year", fontsize=14, y=0.98)
        plt.title(subtitle, fontsize=10, pad=10)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(checks_dir / "eda_calls_per_year_full.png", dpi=150, bbox_inches="tight")
        plt.close()

    report_lines = [
        "# Earnings Call Dataset EDA (Full Corpus)",
        "",
        f"- Number of transcripts: {summary['n_transcripts']:,}",
        f"- Unique companies: {summary['n_unique_companies_raw']:,}",
        f"- Unique titles: {summary['n_unique_titles']:,}",
        f"- Date range: {summary['date_range']['min']} to {summary['date_range']['max']}",
        f"- Distinct year-quarters: {summary['n_year_quarters']}",
        f"- Exact duplicate transcript groups: {summary['exact_duplicate_groups']}",
        "",
        "## Company Frequency",
        f"- Min: {summary['company_frequency']['min']}",
        f"- Median: {summary['company_frequency']['median']}",
        f"- Max: {summary['company_frequency']['max']}",
        f"- Companies with exactly one call: {summary['company_frequency']['freq1_count']}",
        f"- Share with exactly one call: {summary['company_frequency']['freq1_share']}",
        "",
        "## Top Companies",
    ]
    for company, n_calls in summary["top_companies"][:10]:
        report_lines.append(f"- {company}: {n_calls}")
    report_lines.extend(["", "## Top Quarters"])
    for year_quarter, n_calls in summary["top_year_quarters"][:10]:
        report_lines.append(f"- {year_quarter}: {n_calls}")
    report_lines.extend(["", "## Weekday Distribution"])
    for weekday, n_calls in summary["weekday_distribution"].items():
        report_lines.append(f"- {weekday}: {n_calls}")
    (checks_dir / "eda_redo_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory-safe transcript-level EDA on the raw JSONL corpus")
    parser.add_argument("--input", required=True, help="Path to raw JSONL transcript corpus")
    parser.add_argument("--output-dir", default=".", help="Base output directory; EDA files go to <output-dir>/checks")
    parser.add_argument("--index-name", default="Russell 1000 (Large-Cap)", help="Label to print in plot titles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eda(Path(args.input), Path(args.output_dir), args.index_name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
