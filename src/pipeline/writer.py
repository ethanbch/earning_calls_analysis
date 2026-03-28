"""Parquet output management with explicit schemas.

Uses ``pyarrow.parquet.ParquetWriter`` with fixed ``pa.Schema`` constants
so every batch appended is type-consistent.  Outputs:

  outputs/clean/segments.parquet
  outputs/clean/chunks.parquet
  outputs/clean/transcripts_deduplicated.parquet
  outputs/clean/duplicates_audit.parquet
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------------------------------------------------
# Schema definitions
# ------------------------------------------------------------------

SEGMENTS_SCHEMA = pa.schema(
    [
        pa.field("transcript_id", pa.string()),
        pa.field("segment_id", pa.string()),
        pa.field("segment_order", pa.int32()),
        pa.field("speaker_raw", pa.string()),
        pa.field("speaker_name", pa.string()),
        pa.field("speaker_type", pa.string()),
        pa.field("segment_text", pa.string()),
        pa.field("section", pa.string()),
        pa.field("company", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("event_date", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("quarter", pa.int8()),
        pa.field("year_quarter", pa.string()),
    ]
)

CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("transcript_id", pa.string()),
        pa.field("segment_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("company", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("event_date", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("quarter", pa.int8()),
        pa.field("year_quarter", pa.string()),
        pa.field("speaker_name", pa.string()),
        pa.field("speaker_type", pa.string()),
        pa.field("section", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("n_tokens", pa.int32()),
    ]
)

TRANSCRIPTS_SCHEMA = pa.schema(
    [
        pa.field("transcript_id", pa.string()),
        pa.field("company", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("event_date", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("quarter", pa.int8()),
        pa.field("year_quarter", pa.string()),
        pa.field("title", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("clean_text", pa.string()),
    ]
)

DUPLICATES_SCHEMA = pa.schema(
    [
        pa.field("transcript_id", pa.string()),
        pa.field("company", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("event_date", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("quarter", pa.int8()),
        pa.field("year_quarter", pa.string()),
        pa.field("title", pa.string()),
        pa.field("is_duplicate", pa.bool_()),
        pa.field("duplicate_reason", pa.string()),
    ]
)

# ------------------------------------------------------------------
# Coerce helper
# ------------------------------------------------------------------


def _coerce_to_schema(df: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    """Select & order columns to match *schema*, adding missing cols as None."""
    out = df.copy()
    for field in schema:
        if field.name not in out.columns:
            out[field.name] = None
    return out[[f.name for f in schema]]


# ------------------------------------------------------------------
# Incremental Parquet writers
# ------------------------------------------------------------------


class IncrementalWriter:
    """Wraps ``pq.ParquetWriter`` for batch-by-batch appending."""

    def __init__(self, path: Path, schema: pa.Schema, logger: logging.Logger):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.logger = logger
        self._writer = pq.ParquetWriter(str(path), schema=schema)
        self._rows_written = 0

    def write_batch(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        coerced = _coerce_to_schema(df, self.schema)
        table = pa.Table.from_pandas(coerced, schema=self.schema, preserve_index=False)
        self._writer.write_table(table)
        self._rows_written += len(df)

    def close(self) -> None:
        self._writer.close()
        self.logger.info("Wrote %d rows → %s", self._rows_written, self.path)

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------
# Checkpoint management
# ------------------------------------------------------------------

CHECKPOINT_FILE = ".checkpoint"


def load_checkpoint(output_dir: Path) -> int:
    """Return last completed batch index (0 if none)."""
    cp = output_dir / CHECKPOINT_FILE
    if cp.exists():
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            return int(data.get("last_batch", 0))
        except Exception:
            return 0
    return 0


def save_checkpoint(
    output_dir: Path, batch_index: int, extra: dict | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"last_batch": batch_index}
    if extra:
        payload.update(extra)
    (output_dir / CHECKPOINT_FILE).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def clear_checkpoint(output_dir: Path) -> None:
    cp = output_dir / CHECKPOINT_FILE
    if cp.exists():
        cp.unlink()
