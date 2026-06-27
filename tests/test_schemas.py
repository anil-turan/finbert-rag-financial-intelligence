"""Schema validation tests — run without GPU or model files."""
import pytest
from pydantic import ValidationError

from src.api.schemas import SentimentRequest, RegQueryRequest


def test_sentiment_request_valid():
    req = SentimentRequest(texts=["Profits rose sharply.", "Revenue fell 10%."])
    assert len(req.texts) == 2


def test_sentiment_request_empty_list_rejected():
    with pytest.raises(ValidationError):
        SentimentRequest(texts=[])


def test_reg_query_too_short_rejected():
    with pytest.raises(ValidationError):
        RegQueryRequest(question="ok")


def test_reg_query_valid():
    req = RegQueryRequest(question="What is the minimum CET1 ratio under Basel III?")
    assert "Basel" in req.question
