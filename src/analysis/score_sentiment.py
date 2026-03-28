"""Score preprocessed chunks with fine-tuned FinBERT.

Reads chunks.parquet from each dataset, runs inference with the fine-tuned
model, and outputs a scored parquet with sentiment columns.

Outputs per chunk:
    sentiment_label  : positive / neutral / negative
    sentiment_score  : float in [-1, 1]  (positive=+1, negative=-1)
    prob_positive, prob_neutral, prob_negative

Also aggregates per-transcript scores by speaker_type (exec vs analyst).

Usage
-----
    uv run python -m src.analysis.score_sentiment \
        --chunks outputs/r1000/clean/chunks.parquet \
                 outputs/r2k/clean/chunks.parquet \
                 outputs/sp/clean/chunks.parquet \
        --model outputs/finbert_finetuned \
        --output outputs/sentiment_scores.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

LABEL_NAMES = ["positive", "neutral", "negative"]
# Mapping to [-1, 1] score: positive=+1, neutral=0, negative=-1
LABEL_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
BATCH_SIZE = 128
MAX_LENGTH = 128
READ_BATCH_SIZE = 5000
SAMPLE_FRAC = 0.01
SAMPLE_RANDOM_STATE = 42


def score_chunks(
    texts: list[str], model, tokenizer, device: str, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Run inference on a list of texts, return sentiment columns."""
    if not texts:
        return pd.DataFrame(
            {
                "sentiment_label": [],
                "sentiment_score": [],
                "prob_positive": [],
                "prob_neutral": [],
                "prob_negative": [],
            }
        )

    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            text_batch = texts[i : i + batch_size]
            encodings = tokenizer(
                text_batch,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in encodings.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

            if device == "mps":
                torch.mps.empty_cache()

    probs = np.concatenate(all_probs, axis=0)  # (N, 3)
    labels = np.array(LABEL_NAMES)[probs.argmax(axis=1)]
    scores = np.array([LABEL_SCORE[l] for l in labels])

    return pd.DataFrame(
        {
            "sentiment_label": labels,
            "sentiment_score": scores,
            "prob_positive": probs[:, 0],
            "prob_neutral": probs[:, 1],
            "prob_negative": probs[:, 2],
        }
    )


def aggregate_transcript_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chunk-level scores to transcript-level by speaker_type.

    Returns one row per (transcript_id, speaker_role) with columns:
        speaker_role:  exec | analyst | other
        mean_score, median_score, n_chunks
    """
    # Map speaker_type → role
    role_map = {"Executive": "exec", "Analyst": "analyst"}
    scored = scored.copy()
    scored["speaker_role"] = scored["speaker_type"].map(role_map).fillna("other")

    agg = (
        scored.groupby(["transcript_id", "ticker", "event_date", "speaker_role"])
        .agg(
            mean_score=("sentiment_score", "mean"),
            median_score=("sentiment_score", "median"),
            std_score=("sentiment_score", "std"),
            n_chunks=("sentiment_score", "count"),
        )
        .reset_index()
    )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Score chunks with fine-tuned FinBERT")
    parser.add_argument(
        "--chunks", nargs="+", required=True, help="Paths to chunks.parquet"
    )
    parser.add_argument(
        "--model", default="outputs/finbert_finetuned", help="Fine-tuned model dir"
    )
    parser.add_argument("--output", default="outputs/sentiment_scores.parquet")
    parser.add_argument("--output-agg", default="outputs/sentiment_aggregated.parquet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--read-batch-size", type=int, default=READ_BATCH_SIZE)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of read batches per file (default: no limit)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Inference device (default: cpu)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but unavailable, falling back to cpu")
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, falling back to cpu")
        device = "cpu"
    logger.info("Device: %s", device)

    # Load model
    model_path = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    logger.info("Loaded model from %s", model_path)

    # Stream chunks file-by-file and batch-by-batch to avoid RAM spikes
    valid_paths = [Path(p) for p in args.chunks if Path(p).exists()]
    for p in args.chunks:
        if not Path(p).exists():
            logger.warning("Not found, skipping: %s", p)
    if not valid_paths:
        logger.error("No chunk files found.")
        return

    logger.info(
        "Scoring %d chunk files (read_batch_size=%d)",
        len(valid_paths),
        args.read_batch_size,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0
    agg_running = None

    try:
        for path in valid_paths:
            pf = pq.ParquetFile(path)
            n_batches = max(
                1,
                (pf.metadata.num_rows + args.read_batch_size - 1)
                // args.read_batch_size,
            )
            logger.info("Processing %s (%d rows)", path, pf.metadata.num_rows)

            for batch_idx, rb in enumerate(
                tqdm(
                    pf.iter_batches(batch_size=args.read_batch_size),
                    total=n_batches,
                    desc=f"File {path.parent.parent.name}",
                    unit="rbatch",
                )
            ):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    logger.info(
                        "Reached --max-batches=%d for %s, stopping early",
                        args.max_batches,
                        path,
                    )
                    break

                chunks = rb.to_pandas()
                # Reproducible 10% sampling per read batch to cap RAM/compute.
                chunks = chunks.sample(
                    frac=SAMPLE_FRAC, random_state=SAMPLE_RANDOM_STATE
                )
                if chunks.empty:
                    continue

                texts = chunks["chunk_text"].fillna("").astype(str).tolist()
                scores_df = score_chunks(
                    texts, model, tokenizer, device, args.batch_size
                )
                scored_batch = pd.concat(
                    [chunks.reset_index(drop=True), scores_df], axis=1
                )

                table = pa.Table.from_pandas(scored_batch, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema)
                writer.write_table(table)
                total_rows += len(scored_batch)

                # Batch-level aggregation, then fold incrementally (memory-safe)
                agg_batch = aggregate_transcript_sentiment(scored_batch)
                agg_fold = agg_batch[
                    [
                        "transcript_id",
                        "ticker",
                        "event_date",
                        "speaker_role",
                        "mean_score",
                        "n_chunks",
                    ]
                ].copy()
                agg_fold["weighted_mean_sum"] = (
                    agg_fold["mean_score"] * agg_fold["n_chunks"]
                )
                agg_fold = agg_fold.drop(columns=["mean_score"])

                if agg_running is None:
                    agg_running = agg_fold
                else:
                    agg_running = pd.concat([agg_running, agg_fold], ignore_index=True)
                    agg_running = agg_running.groupby(
                        ["transcript_id", "ticker", "event_date", "speaker_role"],
                        as_index=False,
                    ).agg(
                        weighted_mean_sum=("weighted_mean_sum", "sum"),
                        n_chunks=("n_chunks", "sum"),
                    )
    finally:
        if writer is not None:
            writer.close()

    if writer is None or total_rows == 0:
        logger.error(
            "No rows were scored after sampling. Increase --read-batch-size or sampling fraction in code."
        )
        return

    logger.info("Chunk-level scores saved: %s (%d rows)", out, total_rows)

    # Aggregate transcript-level from folded batch aggregates
    agg = agg_running
    agg["mean_score"] = agg["weighted_mean_sum"] / agg["n_chunks"].replace(0, np.nan)
    agg["median_score"] = np.nan
    agg["std_score"] = np.nan
    agg = agg[
        [
            "transcript_id",
            "ticker",
            "event_date",
            "speaker_role",
            "mean_score",
            "median_score",
            "std_score",
            "n_chunks",
        ]
    ]

    agg_out = Path(args.output_agg)
    agg.to_parquet(agg_out, index=False)
    logger.info(
        "Transcript-level aggregated scores saved: %s (%d rows)", agg_out, len(agg)
    )


if __name__ == "__main__":
    main()
