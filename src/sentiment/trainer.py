"""
Fine-tune ProsusAI/finbert on Financial PhraseBank.

Dataset: takala/financial_phrasebank (HuggingFace Hub)
  - sentences_allagree: only sentences where all annotators agreed (highest quality)
  - 3 classes: positive (2), negative (0), neutral (1)

Usage:
    python -m src.sentiment.trainer --output_dir outputs/finbert-finetuned
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "ProsusAI/finbert"
DATASET_NAME = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_allagree"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class FinBERTTrainerConfig:
    model_name: str = BASE_MODEL
    dataset_split: str = DATASET_CONFIG
    output_dir: str = "outputs/finbert-finetuned"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    test_size: float = 0.15
    seed: int = 42


def load_and_split(tokenizer, test_size: float = 0.15, seed: int = 42):
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)

    # Financial PhraseBank only has a train split — create train/val/test
    splits = raw["train"].train_test_split(test_size=test_size * 2, seed=seed)
    val_test = splits["test"].train_test_split(test_size=0.5, seed=seed)

    dataset = {
        "train": splits["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=512)

    def rename_label(batch):
        batch["label"] = batch["label"]
        return batch

    tokenized = {}
    for split, ds in dataset.items():
        tokenized[split] = ds.map(tokenize, batched=True).remove_columns(["sentence"])

    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "accuracy": float((preds == labels).mean()),
    }


def train(output_dir: str = "outputs/finbert-finetuned", epochs: int = 3):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    dataset = load_and_split(tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on held-out test set
    test_results = trainer.evaluate(dataset["test"])
    print("\n=== Test Set Results ===")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\nModel saved to {output_path}")
    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/finbert-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args.output_dir, args.epochs)
