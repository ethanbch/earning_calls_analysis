"""Token-aware chunking of speaker segments.

Each segment is split into chunks of at most ``max_tokens`` BERT tokens using
the fast Rust-based ``tokenizers`` library (``encode_batch`` for throughput).
Sentence boundaries are respected where possible.

Designed for parallel execution via ``multiprocessing.Pool``.
DataFrame in → DataFrame out.
"""

from __future__ import annotations

import logging
import re
from multiprocessing import Pool
from typing import List, Tuple

import pandas as pd
from tokenizers import Tokenizer

# Module-level tokenizer (loaded once per process via _init_worker).
_tokenizer: Tokenizer | None = None


def _init_worker() -> None:
    global _tokenizer
    _tokenizer = Tokenizer.from_pretrained("bert-base-uncased")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    assert _tokenizer is not None
    return len(_tokenizer.encode(text, add_special_tokens=False).ids)


def _count_tokens_batch(texts: List[str]) -> List[int]:
    if not texts:
        return []
    assert _tokenizer is not None
    results = _tokenizer.encode_batch(texts, add_special_tokens=False)
    return [len(r.ids) for r in results]


# ------------------------------------------------------------------
# Sentence splitting
# ------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [s.strip() for s in sents if s.strip()]


def _hard_split(text: str, max_tokens: int) -> List[Tuple[str, int]]:
    """Brute-force word-level splitting when a single sentence exceeds *max_tokens*."""
    words = text.split()
    if not words:
        return []
    assert _tokenizer is not None
    word_counts = [
        len(e.ids) for e in _tokenizer.encode_batch(words, add_special_tokens=False)
    ]
    chunks: List[Tuple[str, int]] = []
    buf: List[str] = []
    buf_tokens = 0
    for w, c in zip(words, word_counts):
        if buf and buf_tokens + c > max_tokens:
            chunks.append((" ".join(buf), buf_tokens))
            buf, buf_tokens = [], 0
        buf.append(w)
        buf_tokens += c
    if buf:
        chunks.append((" ".join(buf), buf_tokens))
    return chunks


# ------------------------------------------------------------------
# Single-segment chunking
# ------------------------------------------------------------------


def _chunk_segment(segment_text: str, max_tokens: int) -> List[Tuple[str, int]]:
    """Return list of (chunk_text, n_tokens) for one segment."""
    sentences = _split_sentences(segment_text)
    if not sentences:
        sentences = [segment_text]

    # Single encode_batch for all sentences instead of per-sentence encode()
    assert _tokenizer is not None
    encodings = _tokenizer.encode_batch(sentences, add_special_tokens=False)
    sent_counts = [len(e.ids) for e in encodings]

    chunks: List[Tuple[str, int]] = []
    buf_sents: List[str] = []
    buf_tokens = 0

    for sent, count in zip(sentences, sent_counts):
        if count > max_tokens:
            if buf_sents:
                chunks.append((" ".join(buf_sents), buf_tokens))
                buf_sents, buf_tokens = [], 0
            chunks.extend(_hard_split(sent, max_tokens))
            continue

        if buf_sents and buf_tokens + count > max_tokens:
            chunks.append((" ".join(buf_sents), buf_tokens))
            buf_sents, buf_tokens = [], 0

        buf_sents.append(sent)
        buf_tokens += count

    if buf_sents:
        chunks.append((" ".join(buf_sents), buf_tokens))

    return chunks


# ------------------------------------------------------------------
# Parallel worker
# ------------------------------------------------------------------


def _worker(args: Tuple[dict, int]) -> List[dict]:
    """Process a single segment row → multiple chunk rows."""
    row, max_tokens = args
    seg_text = str(row.get("segment_text", "") or "").strip()
    if not seg_text:
        return []

    pairs = _chunk_segment(seg_text, max_tokens)
    out: List[dict] = []
    for i, (chunk_text, n_tokens) in enumerate(pairs, start=1):
        if not chunk_text:
            continue
        out.append(
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
                "speaker_type": row.get("speaker_type"),
                "section": row.get("section"),
                "chunk_text": chunk_text,
                "n_tokens": n_tokens,
            }
        )
    return out


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def chunk(
    df: pd.DataFrame,
    max_tokens: int,
    logger: logging.Logger,
    n_workers: int = 4,
    pool: Pool | None = None,
) -> pd.DataFrame:
    """Chunk segments into ≤ *max_tokens* pieces.

    Parameters
    ----------
    df : pd.DataFrame
        Segments DataFrame (output of ``segmenter`` + ``sectioner``).
    max_tokens : int
    logger : logging.Logger
    n_workers : int
        Parallel workers.  Set to 1 for debugging.
    pool : Pool | None
        Optional pre-initialised worker pool.  When supplied the pool is
        **not** closed by this function — the caller owns its lifecycle.
    """
    if df.empty:
        return pd.DataFrame()

    tasks = [(row.to_dict(), max_tokens) for _, row in df.iterrows()]

    all_rows: List[dict] = []
    _own_pool = pool is None and n_workers > 1 and len(tasks) > 50
    if _own_pool:
        pool = Pool(processes=n_workers, initializer=_init_worker)
    try:
        if pool is not None:
            for batch_rows in pool.imap_unordered(_worker, tasks, chunksize=256):
                all_rows.extend(batch_rows)
        else:
            _init_worker()
            for t in tasks:
                all_rows.extend(_worker(t))
    finally:
        if _own_pool and pool is not None:
            pool.close()
            pool.join()

    if not all_rows:
        logger.warning("No chunks produced")
        return pd.DataFrame()

    chunk_df = pd.DataFrame(all_rows)
    over = (chunk_df["n_tokens"] > max_tokens).sum()
    if over:
        logger.warning("%d chunks exceed max_tokens=%d", over, max_tokens)
    logger.info(
        "Chunked %d segments → %d chunks (median %d tokens)",
        len(df),
        len(chunk_df),
        int(chunk_df["n_tokens"].median()),
    )
    return chunk_df
