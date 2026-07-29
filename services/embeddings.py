from langchain_huggingface import HuggingFaceEmbeddings

_model: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached BGE-small embedding model (runs locally on CPU, free)."""
    global _model
    if _model is None:
        _model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _model
