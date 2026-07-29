import os
import tempfile
import logging
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .vectorstore import add_documents, reset_vectorstore

logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".csv":  CSVLoader,
}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
)


def ingest_file(uploaded_file: BinaryIO) -> int:
    """Load, chunk, and embed a Streamlit UploadedFile. Returns chunk count."""
    suffix = Path(uploaded_file.name).suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)
    if loader_cls is None:
        logger.warning("Unsupported upload type for %s: %s", uploaded_file.name, suffix)
        raise ValueError(f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        logger.info("Loading uploaded file %s (%s)", uploaded_file.name, suffix)
        loader = loader_cls(tmp_path)
        raw_docs: list[Document] = loader.load()

        for doc in raw_docs:
            doc.metadata["source"] = uploaded_file.name

        chunks = _splitter.split_documents(raw_docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = i

        add_documents(chunks)
        logger.info("Indexed uploaded file %s into %d chunks", uploaded_file.name, len(chunks))
        return len(chunks)
    except Exception:
        logger.exception("Failed to ingest uploaded file %s", uploaded_file.name)
        raise
    finally:
        os.unlink(tmp_path)


def clear_all_documents() -> None:
    logger.info("Resetting vector store for all uploaded documents")
    reset_vectorstore()
