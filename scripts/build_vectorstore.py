"""
One-time script to ingest regulatory documents and build the ChromaDB vector store.

Run once before starting the API or Streamlit app:
    python scripts/build_vectorstore.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.ingestion import ingest
from src.rag.vectorstore import build_vectorstore

if __name__ == "__main__":
    print("Ingesting regulatory documents...")
    chunks = ingest()
    print(f"\nBuilding ChromaDB vector store...")
    store = build_vectorstore(chunks)
    print(f"\nDone. {store._collection.count()} vectors stored.")
    print("You can now start the API or Streamlit app.")
