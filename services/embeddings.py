import logging
import os

from huggingface_hub import close_session
from langchain_huggingface import HuggingFaceEmbeddings

from .config import configure_runtime, get_secret

_model: HuggingFaceEmbeddings | None = None
logger = logging.getLogger(__name__)


def _build_embeddings() -> HuggingFaceEmbeddings:
    configure_runtime()

    hf_token = get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached BGE-small embedding model (runs locally on CPU, free)."""
    global _model
    if _model is None:
        try:
            _model = _build_embeddings()
        except RuntimeError as exc:
            if "client has been closed" not in str(exc).lower():
                raise

            logger.warning("Resetting closed Hugging Face HTTP session and retrying embeddings init")
            close_session()
            try:
                _model = _build_embeddings()
            except RuntimeError as retry_exc:
                if "client has been closed" not in str(retry_exc).lower():
                    raise

                raise RuntimeError(
                    "Failed to download the embedding model from Hugging Face. "
                    "This usually means Python could not verify the SSL certificate chain for huggingface.co. "
                    "For local development, run `uv sync` after installing the updated dependencies and try again. "
                    "For Streamlit Community Cloud, add your secrets in Advanced settings and redeploy."
                ) from retry_exc
    return _model
