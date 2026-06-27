# FinBERT RAG Financial Intelligence

> Domain-adapted NLP pipeline for financial sentiment analysis and regulatory document Q&A.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FinBERT](https://img.shields.io/badge/model-ProsusAI%2Ffinbert-orange)](https://huggingface.co/ProsusAI/finbert)
[![LangChain](https://img.shields.io/badge/RAG-LangChain%20%2B%20ChromaDB-purple)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com/)

---

## Overview

This project demonstrates two complementary NLP capabilities relevant to financial services:

| Capability | Technology | Use Case |
|---|---|---|
| **Sentiment Analysis** | FinBERT (ProsusAI) | Classify financial text as positive / neutral / negative |
| **Regulatory Q&A** | RAG (LangChain + ChromaDB) | Answer compliance questions grounded in regulatory documents |

Both capabilities are served via a **FastAPI REST API** and a **Streamlit demo app**.

---

## Results

### Sentiment Analysis (Financial PhraseBank)

| Metric | Baseline (pre-trained) | Fine-tuned |
|---|---|---|
| Accuracy | ~0.72 | ~0.89 |
| Macro-F1 | ~0.68 | ~0.87 |
| Weighted-F1 | ~0.74 | ~0.89 |

Fine-tuned on `sentences_allagree` split (100% annotator agreement) — highest-quality subset.

### RAG Pipeline

| Metric | Value |
|---|---|
| Documents ingested | 3 regulatory docs (Basel III, FCA Conduct Rules, AML/KYC) |
| Vector chunks | 28 @ 384 dimensions |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Retrieval Precision@1 | 100% (6/6 annotated queries) |
| Single-text latency | ~180 ms (CPU) |

---

## Project Structure

```
finbert-rag-financial-intelligence/
├── src/
│   ├── sentiment/
│   │   ├── model.py          # FinBERTSentiment inference class
│   │   ├── trainer.py        # HuggingFace Trainer fine-tuning pipeline
│   │   └── evaluator.py      # Metrics, confusion matrix, calibration plots
│   ├── rag/
│   │   ├── ingestion.py      # Document loading + recursive chunking
│   │   ├── vectorstore.py    # ChromaDB build + load
│   │   └── retriever.py      # RegulatoryRAG class (retrieval-only or + LLM)
│   └── api/
│       └── main.py           # FastAPI: /sentiment, /ask-regulation, /health
├── app/
│   └── streamlit_app.py      # Interactive demo UI
├── notebooks/
│   ├── 01_sentiment_finbert.ipynb   # EDA + baseline evaluation
│   ├── 02_rag_pipeline.ipynb        # RAG demo + similarity analysis
│   ├── 03_finbert_finetuning.ipynb  # Fine-tuning walkthrough + curves
│   └── 04_evaluation.ipynb          # End-to-end evaluation + benchmarks
├── data/
│   └── raw/regulatory_docs/         # Basel III, FCA, AML/KYC source texts
├── scripts/
│   └── build_vectorstore.py         # One-time vector store ingestion
└── tests/
    ├── test_sentiment.py
    ├── test_ingestion.py
    └── test_retriever.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Build the vector store (one-time)

```bash
python scripts/build_vectorstore.py
```

This loads the 3 regulatory documents, chunks them into 500-char overlapping segments, embeds with `all-MiniLM-L6-v2`, and persists to `data/processed/chroma_db/`.

### 3. Run the API

```bash
uvicorn src.api.main:app --reload
```

Endpoints:
- `POST /sentiment` — classify up to 50 texts
- `POST /ask-regulation` — RAG Q&A against regulatory docs
- `GET /health` — health check

### 4. Run the Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

### 5. Fine-tune FinBERT (optional)

```bash
python -m src.sentiment.trainer --output_dir outputs/finbert-finetuned --epochs 3
```

---

## Architecture

### Sentiment Pipeline

```
Input text(s)
    │
    ▼
AutoTokenizer (BERT WordPiece, max 512 tokens)
    │
    ▼
FinBERT (ProsusAI/finbert — fine-tuned on Financial PhraseBank)
    │
    ▼
Softmax → {positive, neutral, negative} + confidence score
```

### RAG Pipeline

```
User question
    │
    ▼
SentenceTransformer embedding (all-MiniLM-L6-v2, 384 dims)
    │
    ▼
ChromaDB similarity search → top-k chunks
    │
    ▼
[Retrieval-only mode]            [With LLM]
Return top chunk + source  →OR→  Format prompt → LLM → Synthesised answer
```

The retrieval-only mode runs **entirely offline** — no API key required. Connect any LangChain-compatible LLM for full generation.

---

## Regulatory Documents

| Document | Coverage |
|---|---|
| `basel_iii_capital_requirements.txt` | CET1/Tier1/Total capital ratios, LCR, NSFR, leverage ratio |
| `fca_conduct_rules.txt` | Principles for Business, Consumer Duty, SMCR individual conduct rules |
| `aml_kyc_requirements.txt` | CDD/EDD/SDD, PEP screening, SAR reporting, record retention |

---

## Tests

```bash
pytest tests/ -v --cov=src
```

7 tests covering ingestion, chunking, metadata preservation, sentiment prediction, batch handling, API endpoints, and RAG retrieval.

---

## Technical Highlights

- **Domain-adapted NLP**: FinBERT is pre-trained on financial corpora (Reuters, Bloomberg, FiQA), then fine-tuned on Financial PhraseBank for 3-class sentiment
- **RAG without API keys**: Retrieval-only mode surfaces the most relevant regulatory chunk with source attribution — extensible to GPT-4/Claude/Llama with one line
- **Calibration analysis**: Notebook 04 includes an Expected Calibration Error (ECE) plot — important for production risk systems where confidence scores drive decisions
- **Batch efficiency**: Per-item latency drops 3× from batch=1 to batch=50 due to GPU/CPU parallelism in the attention layers

---

## Portfolio Context

This is **Project 3** of 9 in a UK job market Data Science portfolio. It demonstrates:

- Transformer fine-tuning with HuggingFace Trainer API
- Production RAG architecture (LangChain + ChromaDB)
- Domain-adapted NLP for financial services
- FastAPI serving + Streamlit UI
- Evaluation beyond accuracy: calibration, latency, retrieval precision
