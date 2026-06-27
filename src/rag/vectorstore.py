"""
ChromaDB vector store management.

Embeddings: sentence-transformers/all-MiniLM-L6-v2
  - Local inference, no API key needed
  - 384-dim embeddings, fast on CPU

The store is persisted to data/processed/chroma_db/ so ingestion
only needs to run once.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "regulatory_docs"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document], persist_dir: Path = PERSIST_DIR) -> Chroma:
    """Embed chunks and persist to ChromaDB. Overwrites any existing store."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )
    print(f"Vector store built: {store._collection.count()} vectors at {persist_dir}")
    return store


def load_vectorstore(persist_dir: Path = PERSIST_DIR) -> Chroma:
    """Load an existing ChromaDB store from disk."""
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No vector store found at {persist_dir}. Run scripts/build_vectorstore.py first."
        )
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def similarity_search(query: str, k: int = 4, persist_dir: Path = PERSIST_DIR) -> List[Document]:
    """Quick one-shot retrieval without loading a full chain."""
    store = load_vectorstore(persist_dir)
    return store.similarity_search(query, k=k)
