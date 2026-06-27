"""
RAG retrieval chain.

Architecture:
  Query → ChromaDB similarity search (top-k chunks)
         → Prompt template (context + question)
         → LLM answer with source citations

LLM backend is configurable:
  - Default: template-based retrieval-only mode (no API key needed)
  - Optional: any LangChain-compatible LLM (OpenAI, Anthropic, local HuggingFace)

This lets the project run offline for demos while being production-extensible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from src.rag.vectorstore import load_vectorstore, PERSIST_DIR

PROMPT_TEMPLATE = """You are a financial regulatory compliance expert.
Answer the question below using ONLY the provided regulatory context.
If the context does not contain enough information, say "I cannot find this in the provided regulatory documents."
Always cite the source document for each key statement.

Context:
{context}

Question: {question}

Answer (cite sources):"""


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[str]
    chunks_used: List[Document] = field(default_factory=list)


def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] ({source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[str]:
    seen = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("title", doc.metadata.get("source", "unknown"))
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


class RegulatoryRAG:
    """
    Retrieval-Augmented Generation for financial regulatory documents.

    When no llm is provided, returns the retrieved context directly
    (retrieval-only mode). Pass any LangChain LLM to enable generation.
    """

    def __init__(
        self,
        persist_dir: Path = PERSIST_DIR,
        k: int = 4,
        llm=None,
    ):
        self.store = load_vectorstore(persist_dir)
        self.k = k
        self.llm = llm

    def ask(self, question: str) -> RAGResponse:
        docs = self.store.similarity_search(question, k=self.k)
        context = _format_context(docs)
        sources = _extract_sources(docs)

        if self.llm is not None:
            prompt = PROMPT_TEMPLATE.format(context=context, question=question)
            answer = self.llm.invoke(prompt)
            if hasattr(answer, "content"):
                answer = answer.content
        else:
            # Retrieval-only: surface the most relevant chunk with attribution
            top = docs[0] if docs else None
            if top:
                src = top.metadata.get("title", "regulatory document")
                answer = (
                    f"Based on {src}:\n\n{top.page_content}\n\n"
                    f"(Retrieval-only mode — connect an LLM for a synthesised answer)"
                )
            else:
                answer = "No relevant regulatory content found for this query."

        return RAGResponse(question=question, answer=answer, sources=sources, chunks_used=docs)
