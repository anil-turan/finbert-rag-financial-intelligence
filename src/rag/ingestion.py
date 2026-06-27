"""
Document ingestion for the regulatory RAG pipeline.

Loads text files from data/raw/regulatory_docs/, splits them into
overlapping chunks, and returns LangChain Document objects ready
for embedding and vector storage.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "regulatory_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_regulatory_docs(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """Load all .txt files from the regulatory docs directory."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Regulatory docs directory not found: {docs_dir}")

    documents = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "title": path.stem.replace("_", " ").title()},
            )
        )

    if not documents:
        raise ValueError(f"No .txt files found in {docs_dir}")

    print(f"Loaded {len(documents)} regulatory documents.")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks


def ingest(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """Full ingestion pipeline: load → chunk."""
    docs = load_regulatory_docs(docs_dir)
    return chunk_documents(docs)
