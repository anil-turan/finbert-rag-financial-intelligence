"""
Evaluation utilities for the fine-tuned FinBERT model.

Reports: accuracy, macro-F1, per-class F1, confusion matrix, calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABELS = ["negative", "neutral", "positive"]


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> dict:
    """Return a structured dict of classification metrics."""
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in LABELS
        },
    }


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("FinBERT Confusion Matrix — Financial PhraseBank Test Set", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_confidence_distribution(results, save_path: str | None = None):
    """Histogram of max-probability scores, split by correct/incorrect predictions."""
    confidences = [r.score for r in results]
    correct = [r.label == getattr(r, "true_label", r.label) for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        [c for c, ok in zip(confidences, correct) if ok],
        bins=20, alpha=0.7, label="Correct", color="#0f766e",
    )
    ax.hist(
        [c for c, ok in zip(confidences, correct) if not ok],
        bins=20, alpha=0.7, label="Incorrect", color="#dc2626",
    )
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("FinBERT Prediction Confidence Distribution")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
