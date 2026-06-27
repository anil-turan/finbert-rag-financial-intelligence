"""RAG ingestion tests — no embeddings needed."""
from pathlib import Path
import pytest
from langchain_core.documents import Document
from src.rag.ingestion import chunk_documents, load_regulatory_docs

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "regulatory_docs"


def test_load_regulatory_docs():
    docs = load_regulatory_docs(DOCS_DIR)
    assert len(docs) >= 3
    for doc in docs:
        assert doc.page_content
        assert "source" in doc.metadata


def test_chunk_documents():
    docs = load_regulatory_docs(DOCS_DIR)
    chunks = chunk_documents(docs)
    assert len(chunks) > len(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= 600  # allow slight overflow at sentence boundaries
        assert "source" in chunk.metadata


def test_chunk_preserves_metadata():
    docs = [Document(page_content="A" * 1200, metadata={"source": "test.txt", "title": "Test"})]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"
