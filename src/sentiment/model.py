"""
FinBERT sentiment wrapper.

Base model: ProsusAI/finbert (pre-trained on financial text).
Fine-tuned on Financial PhraseBank (positive / negative / neutral).

Labels follow Financial PhraseBank convention:
  0 → negative, 1 → neutral, 2 → positive
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "ProsusAI/finbert"
LABELS = ["negative", "neutral", "positive"]
MAX_LENGTH = 512


@dataclass
class SentimentResult:
    text: str
    label: str
    score: float
    scores: dict[str, float]


class FinBERTSentiment:
    """Thin inference wrapper around a fine-tuned FinBERT checkpoint."""

    def __init__(self, model_path: str = BASE_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: str | List[str]) -> List[SentimentResult]:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for text, prob_row in zip(texts, probs):
            idx = int(prob_row.argmax())
            results.append(
                SentimentResult(
                    text=text,
                    label=LABELS[idx],
                    score=float(prob_row[idx]),
                    scores={label: float(p) for label, p in zip(LABELS, prob_row)},
                )
            )
        return results

    def predict_one(self, text: str) -> SentimentResult:
        return self.predict([text])[0]
