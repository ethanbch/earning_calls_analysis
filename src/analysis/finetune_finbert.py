"""Fine-tune FinBERT on FinancialPhraseBank for 3-class sentiment.

Downloads the dataset from HuggingFace, fine-tunes ProsusAI/finbert for
3 epochs, and saves the model + tokenizer to outputs/finbert_finetuned/.

Uses MPS (Apple Silicon M3) when available, else CPU.

Usage
-----
    uv run python -m src.analysis.finetune_finbert [--epochs 3] [--batch-size 32] [--lr 2e-5]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP = {"positive": 0, "neutral": 1, "negative": 2}
OUTPUT_DIR = Path("outputs/finbert_finetuned")


def _get_device() -> str:
    """Return best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


class TQDMProgressCallback(TrainerCallback):
    """tqdm progress bar at batch level + epoch-end eval summary."""

    def __init__(self):
        self._pbar: tqdm | None = None
        self._current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._current_epoch = int(state.epoch or 0) + 1
        steps_per_epoch = state.max_steps // args.num_train_epochs
        self._pbar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {self._current_epoch}/{args.num_train_epochs}",
            unit="batch",
            leave=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._pbar is not None and logs and "loss" in logs:
            self._pbar.set_postfix(loss=f"{logs['loss']:.4f}", refresh=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self._pbar is not None:
            self._pbar.update(1)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        if metrics:
            loss = metrics.get("eval_loss", float("nan"))
            acc = metrics.get("eval_accuracy", float("nan"))
            f1 = metrics.get("eval_f1_macro", float("nan"))
            print(
                f"  → Epoch {self._current_epoch} eval: "
                f"loss={loss:.4f}  accuracy={acc:.4f}  f1_macro={f1:.4f}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self._pbar is not None:
            self._pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune FinBERT on FinancialPhraseBank"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    device = _get_device()
    device_label = {"mps": "MPS (Apple Silicon)", "cuda": "CUDA (GPU)", "cpu": "CPU"}
    print(f"\n🔧 Using device: {device_label[device]}\n")
    logger.info("Device: %s", device)

    # ── Load dataset ──────────────────────────────────────────
    logger.info("Loading FinancialPhraseBank (sentences_allagree)...")
    raw = load_dataset(
        "gtfintechlab/financial_phrasebank_sentences_allagree",
        "5768",
        split="train",
    )
    # Columns: "sentence" + "labels" (0=negative, 1=neutral, 2=positive)
    # Same data as takala/financial_phrasebank, native Parquet format
    ds = raw.train_test_split(test_size=0.15, seed=42)
    logger.info("Train: %d  Val: %d", len(ds["train"]), len(ds["test"]))

    # ── Tokenize ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"], truncation=True, padding="max_length", max_length=128
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    # ── Model ─────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
    )

    # ── Training ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=False,  # MPS doesn't support fp16 training
        report_to="none",
        disable_tqdm=True,  # we use our own TQDMProgressCallback
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=_compute_metrics,
        callbacks=[TQDMProgressCallback()],
    )

    logger.info(
        "Starting fine-tuning (%d epochs, batch=%d, lr=%s)...",
        args.epochs,
        args.batch_size,
        args.lr,
    )
    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────
    metrics = trainer.evaluate()
    logger.info("Final eval: %s", metrics)

    # ── Save ──────────────────────────────────────────────────
    out = Path(args.output_dir)
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    logger.info("Model + tokenizer saved to %s", out)


if __name__ == "__main__":
    main()
