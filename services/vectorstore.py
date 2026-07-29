import os
import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embeddings import get_embeddings

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION = "docuchat"

_store: Chroma | None = None
logger = logging.getLogger(__name__)


def _get_store() -> Chroma:
    global _store
    if _store is None:
        logger.info("Initializing Chroma vector store '%s' at %s", COLLECTION, PERSIST_DIR)
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIR,
        )
    return _store


def add_documents(docs: List[Document]) -> None:
    logger.info("Adding %d documents to vector store", len(docs))
    _get_store().add_documents(docs)


def similarity_search(query: str, k: int = 4, threshold: float = 0.25) -> list[dict]:
    logger.info("Running similarity search (query_chars=%d, k=%d, threshold=%.2f)", len(query), k, threshold)
    results = _get_store().similarity_search_with_relevance_scores(query, k=k)
    filtered = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk": doc.metadata.get("chunk", 0),
            "score": round(score, 3),
            "snippet": doc.page_content[:200].replace("\n", " ") + "...",
        }
        for doc, score in results
        if score > threshold
    ]
    logger.info("Similarity search completed with %d matches", len(filtered))
    return filtered


def reset_vectorstore() -> None:
    global _store
    logger.info("Deleting Chroma collection '%s'", COLLECTION)
    store = _get_store()
    store.delete_collection()
    _store = None
