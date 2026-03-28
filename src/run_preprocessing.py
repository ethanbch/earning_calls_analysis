#!/usr/bin/env python3
"""CLI entrypoint – orchestrates the full preprocessing pipeline.

Usage
-----
    # Russell 1000 JSONL
    python -m src.run_preprocessing --input raw/koyfin_transcripts_full_2006_2026.jsonl

    # Russell 2000 or S&P parquet directory
    python -m src.run_preprocessing --input raw/rawrussel2K --output-dir outputs/r2k
    python -m src.run_preprocessing --input raw/sp --output-dir outputs/sp

    # Resume an interrupted run
    python -m src.run_preprocessing --input raw/rawrussel2K --resume

    # Restart from scratch, ignoring checkpoints
    python -m src.run_preprocessing --input raw/rawrussel2K --force
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from src.pipeline import (  # noqa: F401 – used below
    chunker,
    cleaner,
    deduplicator,
    loader,
    metadata,
    sectioner,
    segmenter,
    writer,
)
from src.pipeline.chunker import _init_worker
from src.pipeline.writer import (
    CHUNKS_SCHEMA,
    DUPLICATES_SCHEMA,
    SEGMENTS_SCHEMA,
    TRANSCRIPTS_SCHEMA,
    IncrementalWriter,
    clear_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------


def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_dir / "preprocess.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------


def _count_records(input_path: Path) -> int:
    """Count total records: lines for JSONL, rows for Parquet directory/file."""
    if input_path.is_dir() or input_path.suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        files = (
            sorted(input_path.glob("*.parquet"))
            if input_path.is_dir()
            else [input_path]
        )
        return sum(pq.read_metadata(f).num_rows for f in files)
    with open(input_path, "rb") as f:
        return sum(1 for _ in f)


def run(
    input_path: Path,
    output_dir: Path,
    max_tokens: int = 200,
    batch_size: int = 1000,
    n_workers: int = 4,
    resume: bool = False,
    force: bool = False,
) -> None:
    is_parquet = input_path.is_dir() or input_path.suffix in (".parquet", ".pq")

    clean_dir = output_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(Path("logs"))

    # Count total records for progress bar
    total_lines = _count_records(input_path)
    total_batches = math.ceil(total_lines / batch_size)

    logger.info(
        "Pipeline start | input=%s output=%s batch=%d max_tokens=%d workers=%d resume=%s force=%s total_lines=%d",
        input_path,
        output_dir,
        batch_size,
        max_tokens,
        n_workers,
        resume,
        force,
        total_lines,
    )

    # --- Checkpoint logic ---
    start_batch = 0
    if force:
        clear_checkpoint(output_dir)
        # Remove old output files so writers create fresh ones
        for f in clean_dir.glob("*.parquet"):
            f.unlink()
        logger.info("Force mode: cleared checkpoint and old outputs")
    elif resume:
        start_batch = load_checkpoint(output_dir)
        if start_batch > 0:
            logger.info("Resuming from batch %d", start_batch)

    # When resuming we must open writers in append mode.
    # pyarrow.ParquetWriter doesn't natively append, so for resumed runs
    # we write to a temp file per batch and merge at the end.  For fresh
    # runs we stream directly with IncrementalWriter.
    fresh_run = start_batch == 0

    seg_path = clean_dir / "segments.parquet"
    chunk_path = clean_dir / "chunks.parquet"
    trans_path = clean_dir / "transcripts_deduplicated.parquet"
    dup_path = clean_dir / "duplicates_audit.parquet"

    if fresh_run:
        seg_writer = IncrementalWriter(seg_path, SEGMENTS_SCHEMA, logger)
        chunk_writer = IncrementalWriter(chunk_path, CHUNKS_SCHEMA, logger)
        trans_writer = IncrementalWriter(trans_path, TRANSCRIPTS_SCHEMA, logger)
        dup_writer = IncrementalWriter(dup_path, DUPLICATES_SCHEMA, logger)
    else:
        # For resumed runs, collect tables and merge at end
        _seg_tables, _chunk_tables, _trans_tables, _dup_tables = [], [], [], []

    t0 = time.perf_counter()
    total_transcripts = 0
    total_segments = 0
    total_chunks = 0
    total_dups = 0

    # Compute start_line for loader (skip already-processed batches)
    start_line = 0
    if start_batch > 0:
        start_line = start_batch * batch_size  # approximate

    remaining_batches = total_batches - start_batch
    pbar = tqdm(
        total=remaining_batches,
        desc="Preprocessing",
        unit="batch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]",
    )

    # Persistent worker pool — tokenizer loaded once per worker, reused
    # across all batches instead of being re-spawned each time.
    chunk_pool = (
        Pool(processes=n_workers, initializer=_init_worker) if n_workers > 1 else None
    )

    batch_idx = 0
    _batch_iter = (
        loader.iter_batches_parquet(
            input_path, batch_size=batch_size, logger=logger, start_row=start_line
        )
        if is_parquet
        else loader.iter_batches(
            input_path, batch_size=batch_size, logger=logger, start_line=start_line
        )
    )
    for batch_df, last_line in _batch_iter:
        batch_idx += 1
        global_batch = start_batch + batch_idx
        bt = time.perf_counter()

        # 1. Clean
        batch_df = cleaner.clean(batch_df, logger)

        # 2. Metadata
        batch_df = metadata.extract_metadata(batch_df, logger)

        # 3. Deduplication
        all_df, dup_df = deduplicator.deduplicate(batch_df, logger)
        dedup_df = all_df[~all_df["is_duplicate"]].copy()

        # 4. Segment
        seg_df = segmenter.segment(dedup_df, logger, n_workers=n_workers)

        # 5. Section labelling
        if not seg_df.empty:
            seg_df = sectioner.label_sections(seg_df, logger)

        # 6. Chunk (uses persistent pool)
        chunk_df = chunker.chunk(
            seg_df,
            max_tokens=max_tokens,
            logger=logger,
            n_workers=n_workers,
            pool=chunk_pool,
        )

        # 7. Write
        if fresh_run:
            seg_writer.write_batch(seg_df)
            chunk_writer.write_batch(chunk_df)
            trans_writer.write_batch(dedup_df)
            dup_writer.write_batch(dup_df)
        else:
            import pyarrow as pa

            if not seg_df.empty:
                _seg_tables.append(seg_df)
            if not chunk_df.empty:
                _chunk_tables.append(chunk_df)
            if not dedup_df.empty:
                _trans_tables.append(dedup_df)
            if not dup_df.empty:
                _dup_tables.append(dup_df)

        total_transcripts += len(dedup_df)
        total_segments += len(seg_df)
        total_chunks += len(chunk_df)
        total_dups += len(dup_df)

        save_checkpoint(
            output_dir,
            global_batch,
            {
                "last_line": last_line,
                "total_transcripts": total_transcripts,
                "total_segments": total_segments,
                "total_chunks": total_chunks,
            },
        )

        elapsed = time.perf_counter() - bt
        pbar.set_postfix(
            T=total_transcripts, S=total_segments, C=total_chunks, ordered=True
        )
        pbar.update(1)
        logger.info(
            "Batch %d done (%.1fs) | transcripts=%d segments=%d chunks=%d | cumul: T=%d S=%d C=%d",
            global_batch,
            elapsed,
            len(dedup_df),
            len(seg_df),
            len(chunk_df),
            total_transcripts,
            total_segments,
            total_chunks,
        )

    pbar.close()

    # Shut down persistent worker pool
    if chunk_pool is not None:
        chunk_pool.close()
        chunk_pool.join()

    # --- Close / merge ---
    if fresh_run:
        seg_writer.close()
        chunk_writer.close()
        trans_writer.close()
        dup_writer.close()
    else:
        import pyarrow as pa

        def _append_to_parquet(
            existing: Path, new_dfs: list, schema: pa.Schema
        ) -> None:
            import pyarrow.parquet as pq

            tables = []
            if existing.exists():
                tables.append(pq.read_table(existing))
            for d in new_dfs:
                coerced = writer._coerce_to_schema(d, schema)
                tables.append(
                    pa.Table.from_pandas(coerced, schema=schema, preserve_index=False)
                )
            if tables:
                merged = pa.concat_tables(tables)
                pq.write_table(merged, existing)

        _append_to_parquet(seg_path, _seg_tables, SEGMENTS_SCHEMA)
        _append_to_parquet(chunk_path, _chunk_tables, CHUNKS_SCHEMA)
        _append_to_parquet(trans_path, _trans_tables, TRANSCRIPTS_SCHEMA)
        _append_to_parquet(dup_path, _dup_tables, DUPLICATES_SCHEMA)
        logger.info("Appended resumed batches to existing parquet files")

    total_time = time.perf_counter() - t0
    logger.info(
        "Pipeline complete in %.1fs | transcripts=%d segments=%d chunks=%d duplicates=%d",
        total_time,
        total_transcripts,
        total_segments,
        total_chunks,
        total_dups,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess earnings call transcripts")
    p.add_argument(
        "--input",
        required=True,
        help="Path to JSONL file or directory of Parquet files",
    )
    p.add_argument(
        "--output-dir", default="outputs", help="Output directory (default: outputs)"
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens per chunk (default: 200)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Transcripts per batch (default: 1000)",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="Parallel workers (default: 4)"
    )
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument(
        "--force",
        action="store_true",
        help="Restart from scratch, ignoring checkpoints",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        n_workers=args.workers,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
