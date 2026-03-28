"""Preprocess scraped earnings call transcripts.

Pipeline (simple, robust):
1) Load file/folder input and infer schema.
2) Clean scraping artifacts only (preserve language/content).
3) Extract/standardize metadata.
4) Conservative deduplication with audit trail.
5) Speaker segmentation.
6) Section labeling (Prepared/Q/A/O).
7) Chunking after segmentation.
8) Validation reports and plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


TEXT_CANDIDATES = ["transcript_text", "raw_text", "transcript", "text", "content", "body"]
META_CANDIDATES = ["company", "company_name", "ticker", "symbol", "date", "title", "source_url", "id"]
ROLE_WORDS = {
    "ceo",
    "cfo",
    "coo",
    "cto",
    "chief",
    "executive",
    "president",
    "chairman",
    "chair",
    "director",
    "vp",
    "svp",
    "evp",
    "analyst",
    "operator",
    "moderator",
    "investor",
    "relations",
    "officer",
    "founder",
    "partner",
    "research",
    "capital",
    "securities",
    "management",
}

SPEAKER_SUFFIXES = ["Executive", "Analyst", "Operator", "Speaker", "Moderator", "Unknown Analyst"]

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with", "we", "you", "your", "our", "from", "can", "could", "would",
}

SEGMENT_COLUMNS = [
    "transcript_id",
    "segment_id",
    "segment_order",
    "speaker_raw",
    "speaker_name",
    "speaker_name_raw",
    "speaker_name_normalized",
    "speaker_role_raw",
    "speaker_type",
    "segment_text",
    "segment_text_nlp",
    "qa_started",
    "section",
    "company",
    "ticker",
    "event_date",
    "year",
    "quarter",
    "year_quarter",
    "title",
    "source_file",
]

CHUNK_COLUMNS = [
    "transcript_id",
    "segment_id",
    "chunk_id",
    "company",
    "ticker",
    "event_date",
    "year",
    "quarter",
    "year_quarter",
    "speaker_name",
    "speaker_name_raw",
    "speaker_name_normalized",
    "speaker_type",
    "section",
    "chunk_text",
    "chunk_text_nlp",
    "n_tokens",
    "chunk_order_within_transcript",
]


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


@dataclass
class ParseStats:
    total_files_seen: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    records_loaded: int = 0


class TextNormalizer:
    """Optional NLP normalizer (stemming + lemmatization + stopword removal)."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.enabled = False
        self.stopwords: Set[str] = set(FALLBACK_STOPWORDS)
        self.stemmer = None
        self.lemmatizer = None

        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer, WordNetLemmatizer

            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stopwords = set(stopwords.words("english"))
            except LookupError:
                self.logger.warning("NLTK stopwords corpus not found; using fallback stopword list.")
            self.enabled = True
            self.logger.info("NLP normalization enabled (stemming + lemmatization + stopword removal).")
        except Exception as e:
            self.logger.warning("NLTK not available (%s). NLP normalization disabled.", e)

    def normalize(self, text: Any) -> str:
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        if not text.strip():
            return ""
        if not self.enabled:
            return text

        tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", text, flags=re.UNICODE)
        out_tokens: List[str] = []
        for tok in tokens:
            if re.match(r"[^\w\s]", tok):
                out_tokens.append(tok)
                continue

            low = tok.lower()
            if low in self.stopwords:
                continue

            lemma = low
            if self.lemmatizer is not None:
                try:
                    lemma = self.lemmatizer.lemmatize(low)
                except LookupError:
                    # WordNet not available locally; keep token.
                    lemma = low
            if self.stemmer is not None:
                lemma = self.stemmer.stem(lemma)

            out_tokens.append(lemma)

        text_out = " ".join(out_tokens)
        text_out = re.sub(r"\s+([.,!?;:])", r"\1", text_out)
        text_out = re.sub(r"\s+", " ", text_out).strip()
        return text_out


def normalize_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def safe_read_parquet(path: Path, logger: logging.Logger) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pragma: no cover - depends on runtime engine
        logger.error("Could not read parquet %s: %s", path, e)
        raise


def read_single_file(path: Path, logger: logging.Logger) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return pd.DataFrame([{"raw_text": text}])
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return safe_read_parquet(path, logger)
    if suffix == ".jsonl":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            # Fallback: parse valid lines and skip malformed rows.
            records: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except Exception:
                        logger.warning("Skipping malformed JSONL line %d in %s", i, path)
            return pd.DataFrame(records)
    if suffix == ".json":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                return pd.DataFrame(obj["data"])
            return pd.DataFrame([obj])
        return pd.DataFrame()
    raise ValueError(f"Unsupported input file: {path}")


def iter_jsonl_batches(path: Path, batch_size: int, logger: logging.Logger, start_line: int = 0):
    records: List[Dict[str, Any]] = []
    last_line = start_line
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            if i <= start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                logger.warning("Skipping malformed JSONL line %d in %s", i, path)
                continue
            if isinstance(obj, dict):
                records.append(obj)
                last_line = i
            if len(records) >= batch_size:
                yield pd.DataFrame(records), last_line
                records = []
    if records:
        yield pd.DataFrame(records), last_line


def parse_folder(input_dir: Path, logger: logging.Logger, stats: ParseStats) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    valid_ext = {".txt", ".json", ".jsonl", ".csv", ".parquet"}
    files = sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext])

    for file_path in tqdm(files, desc="Loading files"):
        stats.total_files_seen += 1
        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                rows.append({"raw_text": text, "source_file": str(file_path)})
            else:
                df = read_single_file(file_path, logger)
                if df.empty:
                    logger.warning("Parsed empty file: %s", file_path)
                else:
                    df = df.copy()
                    df["source_file"] = str(file_path)
                    rows.extend(df.to_dict("records"))
            stats.files_parsed += 1
        except Exception as e:
            stats.files_failed += 1
            logger.warning("Failed parsing file %s: %s", file_path, e)

    result = pd.DataFrame(rows)
    stats.records_loaded = len(result)
    return result


def load_input(input_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, ParseStats]:
    stats = ParseStats()
    if input_path.is_dir():
        return parse_folder(input_path, logger, stats), stats

    stats.total_files_seen = 1
    try:
        df = read_single_file(input_path, logger)
        df["source_file"] = str(input_path)
        stats.files_parsed = 1
        stats.records_loaded = len(df)
        return df, stats
    except Exception as e:
        stats.files_failed = 1
        logger.error("Cannot parse input %s: %s", input_path, e)
        return pd.DataFrame(), stats


def discover_input_source(explicit_input: Optional[str], logger: logging.Logger) -> Path:
    """Resolve input path from argument or common raw transcript locations."""
    if explicit_input:
        return Path(explicit_input)

    candidates = [
        Path("koyfin_transcripts_full_2006_2026.jsonl"),
        Path("koyfin_transcripts_full_2006_2026.csv"),
        Path("koyfin_transcripts_consolidated_2006_2026.jsonl"),
        Path("koyfin_transcripts_consolidated_2006_2026.csv"),
        Path("data/raw"),
    ]
    for p in candidates:
        if p.exists():
            logger.info("Auto-detected input source: %s", p)
            return p

    # Broader fallback scan for likely transcript files.
    patterns = ["*transcript*.jsonl", "*transcript*.csv", "*transcript*.parquet", "*.jsonl", "*.csv", "*.parquet"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path(".").glob(pat))
    files = [f for f in files if f.is_file()]
    if files:
        chosen = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.info("Auto-selected latest candidate input: %s", chosen)
        return chosen

    raise FileNotFoundError("No input source found. Provide --input path.")


def infer_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    rename_map = {c: normalize_col(c) for c in df.columns}
    df = df.rename(columns=rename_map)

    text_col = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
    if text_col is None:
        # Fallback: choose the column with the largest average string length.
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        best_col = None
        best_score = -1.0
        for c in object_cols:
            non_null = df[c].dropna()
            if non_null.empty:
                continue
            as_str = non_null.astype(str)
            score = float(as_str.str.len().mean())
            if score > best_score:
                best_score = score
                best_col = c
        if best_col is not None and best_score >= 80:
            text_col = best_col
            logger.info("Inferred text column from string length heuristic: %s", text_col)

    if text_col is None:
        logger.warning("No likely text column found. Creating empty raw_text.")
        df["raw_text"] = ""
    else:
        df["raw_text"] = df[text_col].astype(str)

    # Standardize metadata aliases if present.
    if "company_name" in df.columns and "company" not in df.columns:
        df["company"] = df["company_name"]
    if "symbol" in df.columns and "ticker" not in df.columns:
        df["ticker"] = df["symbol"]

    for c in META_CANDIDATES:
        if c not in df.columns:
            df[c] = np.nan

    if "source_file" not in df.columns:
        df["source_file"] = np.nan

    if "id" in df.columns and df["id"].notna().any():
        df["transcript_id"] = df["id"].astype(str)
    else:
        ids: List[str] = []
        for idx, row in df.iterrows():
            basis = f"{row.get('source_file', '')}|{str(row.get('raw_text', ''))[:500]}|{idx}"
            ids.append(hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()[:16])
        df["transcript_id"] = ids

    return df


def clean_scraping_artifacts(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    t = text.replace("\x00", " ")
    t = re.sub(r"[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\bView Summary\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bClick here for webcast\b", " ", t, flags=re.IGNORECASE)
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: List[str] = []
    prev = None
    for line in t.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            cleaned_lines.append("")
            prev = None
            continue
        # Drop only obvious accidental consecutive duplicate header lines.
        if prev and line == prev and len(line) < 140:
            continue
        cleaned_lines.append(line)
        prev = line

    t = "\n".join(cleaned_lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parse_date_from_text(text: str) -> Optional[pd.Timestamp]:
    patterns = [
        r"\b(20\d{2}-\d{1,2}-\d{1,2})\b",
        r"\b(\d{1,2}/\d{1,2}/20\d{2})\b",
        r"\b([A-Za-z]+\s+\d{1,2},\s+20\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            dt = pd.to_datetime(m.group(1), errors="coerce")
            if pd.notna(dt):
                return dt
    return None


def infer_quarter_from_date(dt: Optional[pd.Timestamp]) -> Optional[int]:
    if dt is None or pd.isna(dt):
        return None
    return int((int(dt.month) - 1) // 3 + 1)


def extract_metadata(row: pd.Series) -> Dict[str, Any]:
    raw_text = row.get("raw_text", "") or ""
    header = "\n".join(str(raw_text).splitlines()[:20])
    title = row.get("title")
    company = row.get("company")
    ticker = row.get("ticker")

    if pd.isna(title) or not str(title).strip():
        first_nonempty = next((ln.strip() for ln in str(raw_text).splitlines() if ln.strip()), "")
        title = first_nonempty[:200] if first_nonempty else None

    if pd.isna(company) or not str(company).strip():
        if isinstance(title, str) and "-" in title:
            company = title.split("-")[0].strip()
        else:
            m = re.search(r"\b([A-Z][A-Za-z0-9&.,'\- ]{2,60})\b(?:\s+Q[1-4]|\s+Earnings|\s+Conference)", header)
            company = m.group(1).strip() if m else None

    if pd.isna(ticker) or not str(ticker).strip():
        m = re.search(r"\(([A-Z]{1,6}(?:\.[A-Z])?)\)", header)
        if not m:
            m = re.search(r"\b(?:Ticker|Symbol)\s*[:\-]\s*([A-Z]{1,6}(?:\.[A-Z])?)\b", header, flags=re.IGNORECASE)
        ticker = m.group(1) if m else None

    event_date_imputed = False
    event_date = pd.to_datetime(row.get("derived_event_date"), errors="coerce")
    if pd.isna(event_date):
        event_date = pd.to_datetime(row.get("datetime"), errors="coerce")
    if pd.isna(event_date):
        event_date = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(event_date):
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
    if pd.isna(event_date):
        event_date = parse_date_from_text(header)
    if pd.isna(event_date):
        event_date = pd.to_datetime(row.get("scraped_at"), errors="coerce")

    quarter = None
    q_match = re.search(r"\bQ([1-4])\b", header, flags=re.IGNORECASE)
    if q_match:
        quarter = int(q_match.group(1))

    year = int(event_date.year) if event_date is not None and pd.notna(event_date) else None
    if year is None:
        y_match = re.search(r"\b(20\d{2})\b", header)
        year = int(y_match.group(1)) if y_match else None

    if quarter is None:
        quarter = infer_quarter_from_date(event_date)

    if (company is None or str(company).strip() == "") and pd.notna(row.get("source_file")):
        stem = Path(str(row.get("source_file"))).stem
        company = re.sub(r"[_-]+", " ", stem).strip()[:80]

    year_quarter = f"{year}Q{quarter}" if year is not None and quarter is not None else None

    return {
        "company": str(company).strip() if company is not None else None,
        "ticker": str(ticker).strip() if ticker is not None else None,
        "event_date": event_date.strftime("%Y-%m-%d") if event_date is not None and pd.notna(event_date) else None,
        "year": year,
        "quarter": quarter,
        "year_quarter": year_quarter,
        "title": str(title).strip() if title is not None else None,
        "event_date_imputed": event_date_imputed,
    }


def canonicalize(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["clean_text"] = df["raw_text"].map(clean_scraping_artifacts)

    meta_rows: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting metadata"):
        meta_rows.append(extract_metadata(row))
    meta_df = pd.DataFrame(meta_rows)
    for c in meta_df.columns:
        df[c] = meta_df[c]

    canonical_cols = [
        "transcript_id",
        "company",
        "ticker",
        "event_date",
        "year",
        "quarter",
        "year_quarter",
        "title",
        "source_file",
        "raw_text",
        "clean_text",
    ]
    for c in canonical_cols:
        if c not in df.columns:
            df[c] = np.nan

    missing_company = df["company"].isna().sum()
    if missing_company > 0:
        logger.warning("Company missing after extraction for %d rows", missing_company)
    return df[canonical_cols + [c for c in df.columns if c not in canonical_cols]]


def conservative_deduplicate(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        out = df.copy()
        out["is_duplicate"] = False
        out["duplicate_reason"] = ""
        return out, out.iloc[0:0].copy()

    out = df.copy()
    out["is_duplicate"] = False
    out["duplicate_reason"] = ""

    # Rule 1: exact duplicate raw text
    exact_dup = out.duplicated(subset=["raw_text"], keep="first")
    out.loc[exact_dup, "is_duplicate"] = True
    out.loc[exact_dup, "duplicate_reason"] = "exact_raw_text"

    # Rule 2: same company + date + highly similar title
    out["title_norm"] = out["title"].fillna("").str.lower().str.replace(r"\W+", "", regex=True)
    grouped = out.groupby(["company", "event_date"], dropna=False)
    for _, idxs in grouped.groups.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue
        for i in range(1, len(idxs)):
            if out.at[idxs[i], "is_duplicate"]:
                continue
            t_curr = out.at[idxs[i], "title_norm"]
            found_dup = False
            for j in range(0, i):
                t_prev = out.at[idxs[j], "title_norm"]
                if not t_curr or not t_prev:
                    continue
                ratio = SequenceMatcher(None, t_curr, t_prev).ratio()
                if ratio >= 0.96:
                    out.at[idxs[i], "is_duplicate"] = True
                    reason = out.at[idxs[i], "duplicate_reason"]
                    out.at[idxs[i], "duplicate_reason"] = (reason + ";" if reason else "") + "same_company_date_similar_title"
                    found_dup = True
                    break
            if found_dup:
                continue

    # Rule 3: same company + quarter + similar length + same prefix
    out["clean_len"] = out["clean_text"].fillna("").str.len()
    out["prefix"] = out["clean_text"].fillna("").str[:250].str.lower().str.replace(r"\s+", " ", regex=True)
    grouped2 = out.groupby(["company", "year_quarter"], dropna=False)
    for _, idxs in grouped2.groups.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue
        for i in range(1, len(idxs)):
            if out.at[idxs[i], "is_duplicate"]:
                continue
            for j in range(0, i):
                if out.at[idxs[j], "is_duplicate"]:
                    continue
                len_i, len_j = out.at[idxs[i], "clean_len"], out.at[idxs[j], "clean_len"]
                if min(len_i, len_j) == 0:
                    continue
                if abs(len_i - len_j) / max(len_i, len_j) <= 0.01 and out.at[idxs[i], "prefix"] == out.at[idxs[j], "prefix"]:
                    out.at[idxs[i], "is_duplicate"] = True
                    reason = out.at[idxs[i], "duplicate_reason"]
                    out.at[idxs[i], "duplicate_reason"] = (reason + ";" if reason else "") + "same_company_quarter_similar_len_prefix"
                    break

    duplicates = out[out["is_duplicate"]].copy()
    logger.info("Duplicates flagged: %d / %d", len(duplicates), len(out))
    out = out.drop(columns=["title_norm", "clean_len", "prefix"], errors="ignore")
    duplicates = duplicates.drop(columns=["title_norm", "clean_len", "prefix"], errors="ignore")
    return out, duplicates


def is_probable_speaker_line(line: str) -> bool:
    l = line.strip()
    if not l or len(l) > 120:
        return False
    if re.search(r"[.!?]$", l):
        return False
    if l.lower() in {"operator", "management", "analysts", "question-and-answer session", "q&a"}:
        return True
    if re.match(r"^[A-Z][A-Za-z .,'\-]{1,80}:$", l):
        return True
    if re.match(r"^[A-Z][A-Za-z .,'\-]{1,80}\s*[-|:]\s*[A-Za-z].*$", l):
        return True

    words = l.replace(":", " ").split()
    if 1 <= len(words) <= 10:
        titlecase_share = sum(1 for w in words if re.match(r"^[A-Z][A-Za-z'\-.]*$", w)) / max(1, len(words))
        has_role = any(w.lower().strip(".,") in ROLE_WORDS for w in words)
        if titlecase_share >= 0.6 and (has_role or len(words) <= 4):
            return True
    return False


def strip_participant_header(text: str) -> str:
    """Remove the top roster block before the actual transcript begins."""
    if not text:
        return text

    text = re.sub(r"^\s*View Summary\s*\n?", "", text, flags=re.IGNORECASE)
    lines = text.splitlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if re.search(r"(OperatorOperator|[A-Z][A-Za-z .,'\-]+(?:Executive|Analyst|Operator))", line):
            start_idx = i
            break

    trimmed = "\n".join(lines[start_idx:]).strip()
    return trimmed if trimmed else text


def split_inline_speaker_markers(text: str) -> str:
    """Turn concatenated markers like 'Jacquie RossExecutive' into standalone lines."""
    if not text:
        return text

    out = text
    out = out.replace("OperatorOperator", "\nOperator\n")
    out = re.sub(
        r"(?<!\n)([A-Z][A-Za-z0-9.'&,\- ]{1,80}?)(Executive|Analyst|Operator)(?=[A-Z])",
        r"\n\1\2\n",
        out,
    )
    out = re.sub(
        r"(?<!\n)([A-Z][A-Za-z0-9.'&,\- ]{1,80}?)(Executive|Analyst|Operator)\b",
        r"\n\1\2\n",
        out,
    )
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def normalize_speaker_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name or "").strip(" -:|")
    words = name.split()
    while words and words[-1].lower().strip(".,") in ROLE_WORDS:
        words.pop()
    norm = " ".join(words).strip()
    norm = " ".join(w.capitalize() if w.islower() or w.isupper() else w for w in norm.split())
    return norm


def parse_speaker_fields(speaker_raw: str) -> Tuple[str, str, str, str]:
    sr = re.sub(r"\s+", " ", (speaker_raw or "").strip())
    sr = sr.strip("-|:")
    for suffix in SPEAKER_SUFFIXES:
        if sr.endswith(suffix) and sr != suffix:
            sr = f"{sr[:-len(suffix)].strip()} {suffix}"
            break
    parts = sr.split()

    name_words = parts[:]
    role_words = []
    while name_words and name_words[-1].lower().strip(".,") in ROLE_WORDS:
        role_words.insert(0, name_words.pop())

    speaker_name_raw = " ".join(name_words).strip() if name_words else sr
    speaker_name_normalized = normalize_speaker_name(speaker_name_raw)
    speaker_role_raw = " ".join(role_words).strip()

    low = sr.lower()
    if "operator" in low or "moderator" in low:
        speaker_type = "Operator"
    elif any(k in low for k in ["analyst", "research", "jpmorgan", "goldman", "morgan", "barclays", "securities", "capital", "ubs", "bofa"]):
        speaker_type = "Analyst"
    elif any(k in low for k in ["ceo", "cfo", "coo", "chief", "president", "executive", "officer", "investor relations", "director", "chair", "founder"]):
        speaker_type = "Executive"
    else:
        speaker_type = "Other"

    return speaker_name_raw, speaker_name_normalized, speaker_role_raw, speaker_type


def segment_transcript(text: str) -> List[Tuple[str, str]]:
    if not isinstance(text, str) or not text.strip():
        return []

    prepared_text = split_inline_speaker_markers(strip_participant_header(text))
    lines = [ln.rstrip() for ln in prepared_text.splitlines()]
    segments: List[Tuple[str, str]] = []
    current_speaker = "Unknown"
    buff: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buff:
                buff.append("")
            continue

        if is_probable_speaker_line(stripped):
            if buff:
                seg_text = "\n".join(buff).strip()
                if seg_text:
                    segments.append((current_speaker, seg_text))
            current_speaker = stripped
            buff = []
        else:
            buff.append(stripped)

    if buff:
        seg_text = "\n".join(buff).strip()
        if seg_text:
            segments.append((current_speaker, seg_text))

    if not segments and text.strip():
        return [("Unknown", text.strip())]
    return segments


def build_segments(transcripts_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df), desc="Segmenting transcripts"):
        segments = segment_transcript(row.get("clean_text", ""))
        for i, (speaker_raw, seg_text) in enumerate(segments, start=1):
            speaker_name_raw, speaker_name_normalized, speaker_role_raw, speaker_type = parse_speaker_fields(speaker_raw)
            rows.append(
                {
                    "transcript_id": row["transcript_id"],
                    "segment_id": f"{row['transcript_id']}_s{i}",
                    "segment_order": i,
                    "speaker_raw": speaker_raw,
                    "speaker_name": speaker_name_normalized,
                    "speaker_name_raw": speaker_name_raw,
                    "speaker_name_normalized": speaker_name_normalized,
                    "speaker_role_raw": speaker_role_raw,
                    "speaker_type": speaker_type,
                    "segment_text": seg_text,
                    "company": row.get("company"),
                    "ticker": row.get("ticker"),
                    "event_date": row.get("event_date"),
                    "year": row.get("year"),
                    "quarter": row.get("quarter"),
                    "year_quarter": row.get("year_quarter"),
                    "title": row.get("title"),
                    "source_file": row.get("source_file"),
                }
            )

    seg_df = pd.DataFrame(rows)
    if seg_df.empty:
        seg_df = pd.DataFrame(columns=SEGMENT_COLUMNS)
    else:
        for c in SEGMENT_COLUMNS:
            if c not in seg_df.columns:
                seg_df[c] = np.nan
    if seg_df.empty:
        logger.warning("No segments extracted from deduplicated transcripts.")
    return seg_df


def label_sections(segments_df: pd.DataFrame) -> pd.DataFrame:
    for c in SEGMENT_COLUMNS:
        if c not in segments_df.columns:
            segments_df[c] = np.nan
    if segments_df.empty:
        segments_df["qa_started"] = pd.Series(dtype=bool)
        segments_df["section"] = pd.Series(dtype=str)
        return segments_df

    out = segments_df.sort_values(["transcript_id", "segment_order"]).copy()
    qa_flags: List[bool] = []
    sections: List[str] = []

    for _, grp in out.groupby("transcript_id", sort=False):
        qa_started = False
        for _, row in grp.iterrows():
            spk = row["speaker_type"]
            seg_text = str(row.get("segment_text", ""))

            if spk == "Analyst":
                qa_started = True
                sec = "Q"
            elif spk == "Executive":
                sec = "A" if qa_started else "Prepared"
            elif spk == "Operator":
                sec = "O"
                # If operator introduces Q&A, mark start for following turns.
                if re.search(r"question|q&a|questions", seg_text, flags=re.IGNORECASE):
                    qa_started = True
            else:
                sec = "O" if qa_started else "Prepared"

            qa_flags.append(qa_started)
            sections.append(sec)

    out["qa_started"] = qa_flags
    out["section"] = sections
    return out


class TokenCounter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.logger.info("Using transformers tokenizer for token counting.")
        except Exception:
            self.logger.info("transformers not available. Using regex token count fallback.")

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [s.strip() for s in sents if s.strip()]


def hard_split_by_words(text: str, token_counter: TokenCounter, max_tokens: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    buf: List[str] = []
    for w in words:
        candidate = " ".join(buf + [w])
        if buf and token_counter.count(candidate) > max_tokens:
            chunks.append(" ".join(buf))
            buf = [w]
        else:
            buf.append(w)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_segments(
    segments_df: pd.DataFrame,
    max_tokens: int,
    logger: logging.Logger,
    text_normalizer: Optional[TextNormalizer] = None,
) -> pd.DataFrame:
    if segments_df.empty:
        return pd.DataFrame(columns=CHUNK_COLUMNS)

    counter = TokenCounter(logger)
    rows: List[Dict[str, Any]] = []

    for transcript_id, grp in tqdm(segments_df.groupby("transcript_id", sort=False), desc="Chunking segments"):
        order = 0
        for _, row in grp.sort_values("segment_order").iterrows():
            segment_text = str(row.get("segment_text", "") or "").strip()
            if not segment_text:
                continue

            sentence_candidates = split_sentences(segment_text)
            if not sentence_candidates:
                sentence_candidates = [segment_text]

            chunk_texts: List[str] = []
            buf: List[str] = []

            for sent in sentence_candidates:
                if counter.count(sent) > max_tokens:
                    if buf:
                        chunk_texts.append(" ".join(buf).strip())
                        buf = []
                    chunk_texts.extend(hard_split_by_words(sent, counter, max_tokens))
                    continue

                candidate = " ".join(buf + [sent]).strip()
                if buf and counter.count(candidate) > max_tokens:
                    chunk_texts.append(" ".join(buf).strip())
                    buf = [sent]
                else:
                    buf.append(sent)

            if buf:
                chunk_texts.append(" ".join(buf).strip())

            for i, chunk_text in enumerate(chunk_texts, start=1):
                if not chunk_text:
                    continue
                order += 1
                rows.append(
                    {
                        "transcript_id": row["transcript_id"],
                        "segment_id": row["segment_id"],
                        "chunk_id": f"{row['segment_id']}_c{i}",
                        "company": row.get("company"),
                        "ticker": row.get("ticker"),
                        "event_date": row.get("event_date"),
                        "year": row.get("year"),
                        "quarter": row.get("quarter"),
                        "year_quarter": row.get("year_quarter"),
                        "speaker_name": row.get("speaker_name"),
                        "speaker_name_raw": row.get("speaker_name_raw"),
                        "speaker_name_normalized": row.get("speaker_name_normalized"),
                        "speaker_type": row.get("speaker_type"),
                        "section": row.get("section"),
                        "chunk_text": chunk_text,
                        "chunk_text_nlp": text_normalizer.normalize(chunk_text) if text_normalizer else chunk_text,
                        "n_tokens": counter.count(chunk_text),
                        "chunk_order_within_transcript": order,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=CHUNK_COLUMNS)
    else:
        for c in CHUNK_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan
    return out


def save_table(df: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:  # pragma: no cover
        logger.warning("Parquet write failed for %s (%s). Writing CSV fallback.", path, e)
        csv_fallback = path.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def validate_and_report(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    parse_stats: ParseStats,
    checks_dir: Path,
    max_tokens: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    checks_dir.mkdir(parents=True, exist_ok=True)
    if "transcript_id" not in seg_df.columns:
        seg_df = seg_df.copy()
        seg_df["transcript_id"] = pd.Series(dtype=str)
    if "transcript_id" not in chunk_df.columns:
        chunk_df = chunk_df.copy()
        chunk_df["transcript_id"] = pd.Series(dtype=str)
    if "section" not in seg_df.columns:
        seg_df = seg_df.copy()
        seg_df["section"] = pd.Series(dtype=str)
    if "segment_text" not in seg_df.columns:
        seg_df = seg_df.copy()
        seg_df["segment_text"] = pd.Series(dtype=str)
    if "n_tokens" not in chunk_df.columns:
        chunk_df = chunk_df.copy()
        chunk_df["n_tokens"] = pd.Series(dtype=float)

    parse_success_rate = (parse_stats.files_parsed / parse_stats.total_files_seen) if parse_stats.total_files_seen else 0.0
    metadata_success_rate = float(
        (
            dedup_df["company"].notna()
            & dedup_df["event_date"].notna()
            & dedup_df["year_quarter"].notna()
        ).mean()
    ) if not dedup_df.empty else 0.0

    raw_chars = raw_df["raw_text"].fillna("").str.len().sum() if not raw_df.empty else 0
    cleaned_chars = clean_df["clean_text"].fillna("").str.len().sum() if not clean_df.empty else 0
    segmented_chars = seg_df["segment_text"].fillna("").str.len().sum() if not seg_df.empty else 0

    drop_after_clean = (1 - cleaned_chars / raw_chars) if raw_chars else 0.0
    drop_after_seg = (1 - segmented_chars / cleaned_chars) if cleaned_chars else 0.0

    qa_by_transcript = (
        seg_df.groupby("transcript_id")["section"].agg(lambda s: ("Q" in set(s)) and ("A" in set(s))).mean()
        if not seg_df.empty
        else 0.0
    )

    section_dist = seg_df["section"].value_counts(dropna=False).rename_axis("section").reset_index(name="count") if not seg_df.empty else pd.DataFrame(columns=["section", "count"])
    if not section_dist.empty:
        section_dist.to_csv(checks_dir / "section_distribution.csv", index=False)
        plt.figure(figsize=(6, 4))
        plt.bar(section_dist["section"].astype(str), section_dist["count"])
        plt.title("Section Distribution")
        plt.tight_layout()
        plt.savefig(checks_dir / "section_distribution.png", dpi=150)
        plt.close()

    token_bins = [0, 25, 50, 100, 150, 200, 400, np.inf]
    chunk_token_dist = pd.DataFrame(columns=["token_bin", "count"])
    if not chunk_df.empty:
        chunk_df = chunk_df.copy()
        chunk_df["token_bin"] = pd.cut(chunk_df["n_tokens"], bins=token_bins, include_lowest=True)
        chunk_token_dist = chunk_df["token_bin"].value_counts().sort_index().rename_axis("token_bin").reset_index(name="count")
        chunk_token_dist["token_bin"] = chunk_token_dist["token_bin"].astype(str)
        chunk_token_dist.to_csv(checks_dir / "chunk_token_distribution.csv", index=False)
        plt.figure(figsize=(8, 4))
        plt.hist(chunk_df["n_tokens"], bins=30)
        plt.title("Chunk Token Length Distribution")
        plt.tight_layout()
        plt.savefig(checks_dir / "chunk_token_distribution.png", dpi=150)
        plt.close()

    zero_segments = int((~dedup_df["transcript_id"].isin(seg_df["transcript_id"]) if not dedup_df.empty else pd.Series(dtype=bool)).sum()) if not dedup_df.empty else 0
    zero_chunks = int((~dedup_df["transcript_id"].isin(chunk_df["transcript_id"]) if not dedup_df.empty else pd.Series(dtype=bool)).sum()) if not dedup_df.empty else 0

    suspicious_short = int((chunk_df["n_tokens"] < 5).sum()) if not chunk_df.empty else 0
    max_violations = int((chunk_df["n_tokens"] > max_tokens).sum()) if not chunk_df.empty else 0

    summary = {
        "transcript_parse_success_rate": round(parse_success_rate, 4),
        "metadata_extraction_success_rate": round(metadata_success_rate, 4),
        "duplicate_count": int(len(dup_df)),
        "cleaning_text_drop_rate": round(float(drop_after_clean), 4),
        "segmentation_text_drop_rate": round(float(drop_after_seg), 4),
        "share_transcripts_with_detected_qa": round(float(qa_by_transcript), 4),
        "section_distribution": section_dist.set_index("section")["count"].to_dict() if not section_dist.empty else {},
        "chunk_token_distribution": chunk_token_dist.set_index("token_bin")["count"].to_dict() if not chunk_token_dist.empty else {},
        "max_token_limit": int(max_tokens),
        "max_token_limit_violations": int(max_violations),
        "suspiciously_short_chunks_lt5_tokens": int(suspicious_short),
        "transcripts_with_zero_segments": int(zero_segments),
        "transcripts_with_zero_chunks": int(zero_chunks),
        "records_loaded": int(parse_stats.records_loaded),
        "files_seen": int(parse_stats.total_files_seen),
        "files_failed": int(parse_stats.files_failed),
    }

    with (checks_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if max_violations > 0:
        logger.warning("Found %d chunks above max token limit=%d", max_violations, max_tokens)
    if zero_segments > 0:
        logger.warning("Transcripts with zero extracted segments: %d", zero_segments)
    if zero_chunks > 0:
        logger.warning("Transcripts with zero extracted chunks: %d", zero_chunks)

    return summary


def run_pipeline(input_path: Path, output_dir: Path, max_tokens: int, resume: bool = False) -> None:
    clean_dir = output_dir / "clean"
    checks_dir = output_dir / "checks"
    log_dir = Path("logs")

    logger = setup_logger(log_dir / "preprocess.log")
    logger.info("Starting preprocessing pipeline")
    logger.info("Input: %s | Output: %s | max_tokens=%d | resume=%s", input_path, output_dir, max_tokens, resume)
    text_normalizer = TextNormalizer(logger)

    if input_path.suffix.lower() == ".jsonl" and input_path.stat().st_size > 500_000_000:
        run_large_jsonl_pipeline(input_path, output_dir, max_tokens, text_normalizer, logger, resume=resume)
        return

    raw_df, parse_stats = load_input(input_path, logger)
    if raw_df.empty:
        logger.error("No records loaded. Exiting.")
        return

    inferred_df = infer_columns(raw_df, logger)
    transcripts_clean = canonicalize(inferred_df, logger)
    transcripts_all, duplicates_audit = conservative_deduplicate(transcripts_clean, logger)
    transcripts_dedup = transcripts_all[~transcripts_all["is_duplicate"]].copy()

    segments = build_segments(transcripts_dedup, logger)
    segments = label_sections(segments)
    segments["segment_text_nlp"] = segments["segment_text"].map(text_normalizer.normalize) if not segments.empty else ""

    chunks = chunk_segments(segments, max_tokens=max_tokens, logger=logger, text_normalizer=text_normalizer)

    save_table(transcripts_all, clean_dir / "transcripts_clean.parquet", logger)
    save_table(transcripts_dedup, clean_dir / "transcripts_deduplicated.parquet", logger)
    save_table(segments, clean_dir / "segments.parquet", logger)
    save_table(chunks, clean_dir / "chunks.parquet", logger)
    save_table(duplicates_audit, clean_dir / "duplicates_audit.parquet", logger)

    summary = validate_and_report(
        raw_df=inferred_df,
        clean_df=transcripts_all,
        dedup_df=transcripts_dedup,
        seg_df=segments,
        chunk_df=chunks,
        dup_df=duplicates_audit,
        parse_stats=parse_stats,
        checks_dir=checks_dir,
        max_tokens=max_tokens,
        logger=logger,
    )

    logger.info("Preprocessing complete. Records=%d, dedup=%d, segments=%d, chunks=%d", len(transcripts_all), len(transcripts_dedup), len(segments), len(chunks))
    logger.info("Validation summary: %s", json.dumps(summary, ensure_ascii=False))


def run_large_jsonl_pipeline(
    input_path: Path,
    output_dir: Path,
    max_tokens: int,
    text_normalizer: TextNormalizer,
    logger: logging.Logger,
    batch_size: int = 250,
    resume: bool = False,
) -> None:
    clean_dir = output_dir / "clean"
    checks_dir = output_dir / "checks"

    transcript_clean_csv = clean_dir / "transcripts_clean.csv"
    transcript_dedup_csv = clean_dir / "transcripts_deduplicated.csv"
    segments_csv = clean_dir / "segments.csv"
    chunks_csv = clean_dir / "chunks.csv"
    dup_csv = clean_dir / "duplicates_audit.csv"
    stream_state = checks_dir / "stream_resume_state.json"

    start_line = 0
    if resume and stream_state.exists():
        try:
            state = json.loads(stream_state.read_text(encoding="utf-8"))
            start_line = int(state.get("last_line", 0))
            logger.info("Resuming from line %d based on %s", start_line, stream_state)
        except Exception as e:
            logger.warning("Could not read resume state %s (%s). Starting from scratch.", stream_state, e)

    if not resume:
        for path in [transcript_clean_csv, transcript_dedup_csv, segments_csv, chunks_csv, dup_csv]:
            if path.exists():
                path.unlink()
        if stream_state.exists():
            stream_state.unlink()

    stats = ParseStats(total_files_seen=1, files_parsed=1)
    total_records = 0
    total_dedup = 0
    total_segments = 0
    total_chunks = 0
    batch_no = 0

    logger.info("Large JSONL detected. Switching to streaming mode with batch_size=%d", batch_size)
    logger.info("For full-scale streaming, NLP-normalized segment/chunk columns are skipped to keep runtime manageable.")

    for raw_batch, last_line in iter_jsonl_batches(input_path, batch_size=batch_size, logger=logger, start_line=start_line):
        batch_no += 1
        raw_batch["source_file"] = str(input_path)
        stats.records_loaded += len(raw_batch)

        inferred_df = infer_columns(raw_batch, logger)
        transcripts_clean = canonicalize(inferred_df, logger)
        transcripts_all, duplicates_audit = conservative_deduplicate(transcripts_clean, logger)
        transcripts_dedup = transcripts_all[~transcripts_all["is_duplicate"]].copy()

        segments = build_segments(transcripts_dedup, logger)
        segments = label_sections(segments)
        if not segments.empty:
            segments["segment_text_nlp"] = ""

        chunks = chunk_segments(segments, max_tokens=max_tokens, logger=logger, text_normalizer=None)

        append_csv(transcripts_all, transcript_clean_csv)
        append_csv(transcripts_dedup, transcript_dedup_csv)
        append_csv(segments, segments_csv)
        append_csv(chunks, chunks_csv)
        if not duplicates_audit.empty:
            append_csv(duplicates_audit, dup_csv)

        total_records += len(transcripts_all)
        total_dedup += len(transcripts_dedup)
        total_segments += len(segments)
        total_chunks += len(chunks)
        logger.info(
            "Batch %d complete. records=%d dedup=%d segments=%d chunks=%d",
            batch_no,
            total_records,
            total_dedup,
            total_segments,
            total_chunks,
        )
        checks_dir.mkdir(parents=True, exist_ok=True)
        stream_state.write_text(
            json.dumps(
                {
                    "last_line": last_line,
                    "batch_no": batch_no,
                    "records_loaded": int(stats.records_loaded),
                    "total_records": int(total_records),
                    "total_dedup": int(total_dedup),
                    "total_segments": int(total_segments),
                    "total_chunks": int(total_chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = {
        "mode": "streaming_large_jsonl",
        "records_loaded": int(stats.records_loaded),
        "total_records_written": int(total_records),
        "total_deduplicated_written": int(total_dedup),
        "total_segments_written": int(total_segments),
        "total_chunks_written": int(total_chunks),
        "last_line_processed": int(last_line if stats.records_loaded > 0 else start_line),
    }
    checks_dir.mkdir(parents=True, exist_ok=True)
    with (checks_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Streaming preprocessing complete. Records=%d, dedup=%d, segments=%d, chunks=%d",
        total_records,
        total_dedup,
        total_segments,
        total_chunks,
    )
    logger.info("Validation summary: %s", json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess scraped earnings call transcripts")
    parser.add_argument("--input", required=False, default=None, help="Input file or folder path. If omitted, source is auto-detected.")
    parser.add_argument("--output-dir", default="outputs", help="Output base directory (default: outputs)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens per chunk")
    parser.add_argument("--resume", action="store_true", help="Resume large JSONL streaming from last saved state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_logger = setup_logger(Path("logs") / "preprocess.log")
    resolved_input = discover_input_source(args.input, bootstrap_logger)
    run_pipeline(input_path=resolved_input, output_dir=Path(args.output_dir), max_tokens=args.max_tokens, resume=args.resume)


if __name__ == "__main__":
    main()
