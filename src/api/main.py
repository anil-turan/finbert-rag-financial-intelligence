"""
Financial Intelligence API

Endpoints:
  GET  /health              — service status
  POST /sentiment           — FinBERT sentiment for a list of sentences
  POST /ask-regulation      — RAG query against regulatory documents
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    RegQueryRequest,
    RegQueryResponse,
    SentimentRequest,
    SentimentResponse,
    SentimentItem,
)

_sentiment_model = None
_rag = None
_rag_doc_count: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sentiment_model, _rag, _rag_doc_count

    model_path = os.getenv("FINBERT_MODEL_PATH", "ProsusAI/finbert")
    try:
        from src.sentiment.model import FinBERTSentiment
        _sentiment_model = FinBERTSentiment(model_path=model_path)
        print(f"FinBERT loaded from: {model_path}")
    except Exception as e:
        print(f"WARNING: Could not load FinBERT — {e}")

    try:
        from src.rag.retriever import RegulatoryRAG
        _rag = RegulatoryRAG()
        _rag_doc_count = _rag.store._collection.count()
        print(f"RAG loaded: {_rag_doc_count} vectors")
    except Exception as e:
        print(f"WARNING: Could not load RAG — {e}")

    yield


app = FastAPI(
    title="Financial Intelligence API",
    description=(
        "FinBERT sentiment analysis + regulatory RAG pipeline. "
        "Fine-tuned on Financial PhraseBank · RAG over FCA/Basel III documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    rag_mode = "retrieval+llm" if (_rag and _rag.llm) else "retrieval-only"
    return HealthResponse(
        status="ok" if _sentiment_model else "degraded",
        sentiment_model=os.getenv("FINBERT_MODEL_PATH", "ProsusAI/finbert"),
        rag_documents=_rag_doc_count,
        rag_mode=rag_mode,
    )


@app.post("/sentiment", response_model=SentimentResponse, tags=["Sentiment"])
def sentiment(request: SentimentRequest):
    if _sentiment_model is None:
        raise HTTPException(status_code=503, detail="FinBERT model not loaded.")
    if len(request.texts) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 texts per request.")

    results = _sentiment_model.predict(request.texts)
    return SentimentResponse(
        results=[
            SentimentItem(
                text=r.text,
                label=r.label,
                score=round(r.score, 4),
                scores={k: round(v, 4) for k, v in r.scores.items()},
            )
            for r in results
        ],
        model_version=os.getenv("FINBERT_MODEL_PATH", "ProsusAI/finbert"),
    )


@app.post("/ask-regulation", response_model=RegQueryResponse, tags=["RAG"])
def ask_regulation(request: RegQueryRequest):
    if _rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not loaded. Run scripts/build_vectorstore.py first.",
        )

    response = _rag.ask(request.question)
    return RegQueryResponse(
        question=response.question,
        answer=response.answer,
        sources=response.sources,
        chunks_retrieved=len(response.chunks_used),
    )
