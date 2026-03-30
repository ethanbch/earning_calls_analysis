"""Speaker-turn segmentation of transcript text.

Splits ``clean_text`` into one row per speaker turn, detecting the Koyfin
inline format where speaker names are concatenated with role tags like
``Jon RavivExecutive`` or ``OperatorOperator``.

Designed for parallel execution via ``multiprocessing.Pool``.
DataFrame in → DataFrame out.
"""

from __future__ import annotations

import logging
import re
from multiprocessing import Pool
from typing import Dict, List, Tuple

import pandas as pd

# Known role suffixes that appear glued onto speaker names in Koyfin HTML.
_ROLE_SUFFIXES = [
    "Executive",
    "Analyst",
    "Operator",
    "Speaker",
    "Moderator",
    "Unknown Analyst",
]

_ROLE_WORDS = {
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


# ------------------------------------------------------------------
# Text preparation
# ------------------------------------------------------------------


def _strip_participant_header(text: str) -> str:
    """Remove the roster block before the actual transcript."""
    if not text:
        return text
    text = re.sub(r"^\s*View Summary\s*\n?", "", text, flags=re.IGNORECASE)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(
            r"(OperatorOperator|[A-Z][A-Za-z .,'\-]+(?:Executive|Analyst|Operator))",
            line,
        ):
            trimmed = "\n".join(lines[i:]).strip()
            return trimmed if trimmed else text
    return text


# Regex for a proper person name: 1-5 capitalised words (no commas/articles).
# Matches "Jon Raviv", "James Reagan", "Prabu Natarajan" but NOT
# "Jim Reagan, our Chief" which is mid-sentence text.
_NAME_RE = r"[A-Z][A-Za-z.'\-]+(?:\s[A-Z][A-Za-z.'\-]+){0,4}"


def _split_inline_markers(text: str) -> str:
    """Turn concatenated markers (e.g. ``Jon RavivExecutive``) into separate lines."""
    if not text:
        return text
    out = text.replace("OperatorOperator", "\nOperator\n")
    # Match "FirstName LastNameExecutive" glued together, followed by uppercase (next speaker)
    out = re.sub(
        rf"(?<!\n)({_NAME_RE})(Executive|Analyst|Operator)(?=[A-Z])",
        r"\n\1\2\n",
        out,
    )
    # Same but at word boundary (end of text or before space)
    out = re.sub(
        rf"(?<!\n)({_NAME_RE})(Executive|Analyst|Operator)\b",
        r"\n\1\2\n",
        out,
    )
    return re.sub(r"\n{3,}", "\n\n", out).strip()


# ------------------------------------------------------------------
# Speaker-line detection
# ------------------------------------------------------------------


def _is_speaker_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if re.search(r"[.!?]$", s):
        return False
    if s.lower() in {
        "operator",
        "management",
        "analysts",
        "question-and-answer session",
        "q&a",
    }:
        return True
    # "Name – Role:" pattern
    if re.match(r"^[A-Z][A-Za-z .,'\-]{1,80}:$", s):
        return True
    if re.match(r"^[A-Z][A-Za-z .,'\-]{1,80}\s*[-|:]\s*[A-Za-z].*$", s):
        return True
    words = s.replace(":", " ").split()
    if 1 <= len(words) <= 10:
        titlecase = sum(1 for w in words if re.match(r"^[A-Z][A-Za-z'\-.]*$", w)) / max(
            1, len(words)
        )
        has_role = any(w.lower().strip(".,") in _ROLE_WORDS for w in words)
        if titlecase >= 0.6 and (has_role or len(words) <= 4):
            return True
    return False


# ------------------------------------------------------------------
# Speaker-field parsing
# ------------------------------------------------------------------


def _normalize_speaker_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name or "").strip(" -:|")
    words = name.split()
    while words and words[-1].lower().strip(".,") in _ROLE_WORDS:
        words.pop()
    norm = " ".join(words).strip()
    return " ".join(
        w.capitalize() if w.islower() or w.isupper() else w for w in norm.split()
    )


def _parse_speaker(speaker_raw: str) -> Dict[str, str]:
    sr = re.sub(r"\s+", " ", (speaker_raw or "").strip()).strip("-|:")
    for suffix in _ROLE_SUFFIXES:
        if sr.endswith(suffix) and sr != suffix:
            sr = f"{sr[: -len(suffix)].strip()} {suffix}"
            break
    parts = sr.split()

    name_words = parts[:]
    role_words: list[str] = []
    while name_words and name_words[-1].lower().strip(".,") in _ROLE_WORDS:
        role_words.insert(0, name_words.pop())

    name_raw = " ".join(name_words).strip() if name_words else sr
    name_norm = _normalize_speaker_name(name_raw)
    role_raw = " ".join(role_words).strip()

    low = sr.lower()
    if "operator" in low or "moderator" in low:
        stype = "Operator"
    elif any(
        k in low
        for k in [
            "analyst",
            "research",
            "jpmorgan",
            "goldman",
            "morgan",
            "barclays",
            "securities",
            "capital",
            "ubs",
            "bofa",
        ]
    ):
        stype = "Analyst"
    elif any(
        k in low
        for k in [
            "ceo",
            "cfo",
            "coo",
            "chief",
            "president",
            "executive",
            "officer",
            "investor relations",
            "director",
            "chair",
            "founder",
        ]
    ):
        stype = "Executive"
    else:
        stype = "Other"

    return {
        "speaker_name_raw": name_raw,
        "speaker_name_normalized": name_norm,
        "speaker_role_raw": role_raw,
        "speaker_type": stype,
    }


# ------------------------------------------------------------------
# Core segmentation (single transcript)
# ------------------------------------------------------------------


def _segment_one(text: str) -> List[Tuple[str, str]]:
    """Return a list of (speaker_raw, segment_text) pairs."""
    if not isinstance(text, str) or not text.strip():
        return []

    prepared = _split_inline_markers(_strip_participant_header(text))
    lines = [ln.rstrip() for ln in prepared.splitlines()]
    segments: List[Tuple[str, str]] = []
    current_speaker = "Unknown"
    buf: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buf:
                buf.append("")
            continue
        if _is_speaker_line(stripped):
            if buf:
                seg = "\n".join(buf).strip()
                if seg:
                    segments.append((current_speaker, seg))
            current_speaker = stripped
            buf = []
        else:
            buf.append(stripped)

    if buf:
        seg = "\n".join(buf).strip()
        if seg:
            segments.append((current_speaker, seg))

    if not segments and text.strip():
        return [("Unknown", text.strip())]
    return segments


# ------------------------------------------------------------------
# Parallel worker
# ------------------------------------------------------------------


def _worker(args: Tuple[str, str, dict]) -> List[dict]:
    """Serialisable worker for multiprocessing."""
    transcript_id, clean_text, meta = args
    turns = _segment_one(clean_text)
    rows: List[dict] = []
    for i, (speaker_raw, seg_text) in enumerate(turns, start=1):
        fields = _parse_speaker(speaker_raw)
        rows.append(
            {
                "transcript_id": transcript_id,
                "segment_id": f"{transcript_id}_s{i}",
                "segment_order": i,
                "speaker_raw": speaker_raw,
                "speaker_name": fields["speaker_name_normalized"],
                "speaker_type": fields["speaker_type"],
                "segment_text": seg_text,
                **meta,
            }
        )
    return rows


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def segment(
    df: pd.DataFrame,
    logger: logging.Logger,
    n_workers: int = 4,
) -> pd.DataFrame:
    """Segment transcripts into speaker turns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``transcript_id``, ``clean_text``, and metadata columns.
    logger : logging.Logger
    n_workers : int
        Number of parallel workers.  Set to 1 for debugging.
    """
    if df.empty:
        return pd.DataFrame()

    meta_cols = [
        "company",
        "ticker",
        "event_date",
        "year",
        "quarter",
        "year_quarter",
    ]
    tasks = []
    for _, row in df.iterrows():
        meta = {c: row.get(c) for c in meta_cols}
        tasks.append((row["transcript_id"], row.get("clean_text", ""), meta))

    all_rows: List[dict] = []
    if n_workers > 1 and len(tasks) > 10:
        with Pool(processes=n_workers) as pool:
            for batch_rows in pool.imap_unordered(_worker, tasks, chunksize=64):
                all_rows.extend(batch_rows)
    else:
        for t in tasks:
            all_rows.extend(_worker(t))

    if not all_rows:
        logger.warning("No segments extracted")
        return pd.DataFrame()

    seg_df = pd.DataFrame(all_rows)
    logger.info("Segmented %d transcripts → %d speaker turns", len(df), len(seg_df))
    return seg_df
