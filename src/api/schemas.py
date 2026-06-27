from pydantic import BaseModel, Field
from typing import List, Optional


class SentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of financial sentences to analyse")

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "The company reported record profits, beating analyst expectations.",
                    "Revenues declined 12% year-on-year amid weakening demand.",
                    "The board approved a share buyback programme of £500 million.",
                ]
            }
        }
    }


class SentimentItem(BaseModel):
    text: str
    label: str
    score: float
    scores: dict[str, float]


class SentimentResponse(BaseModel):
    results: List[SentimentItem]
    model_version: str


class RegQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Regulatory compliance question")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the minimum capital requirements under Basel III?"
            }
        }
    }


class RegQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    sentiment_model: str
    rag_documents: Optional[int]
    rag_mode: str
