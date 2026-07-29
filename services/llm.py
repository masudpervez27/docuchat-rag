import os
import logging
from typing import Generator

from groq import APIConnectionError, Groq

from .config import create_httpx_client, get_secret

_client: Groq | None = None
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions ONLY using the provided context.

Rules:
- Base every answer strictly on the context below.
- Be concise and specific. Prefer bullet points for lists.
- If the context does not contain the answer, say: "The documents don't contain information about this."
- Never fabricate information or use outside knowledge.
- Always be factual and cite which document the information comes from when possible."""


def _get_client() -> Groq:
    global _client
    if _client is None or _client.is_closed():
        logger.info("Initializing Groq client for DocuChat")
        _client = Groq(
            api_key=get_secret("GROQ_API_KEY", required=True),
            http_client=create_httpx_client(follow_redirects=True, timeout=None),
        )
    return _client


def _reset_client() -> Groq:
    global _client
    if _client is not None and not _client.is_closed():
        _client.close()
    _client = None
    return _get_client()


def stream_answer(question: str, context: str) -> Generator[str, None, None]:
    """Yield response tokens from Groq Llama-3.3-70b."""
    logger.info("Streaming DocuChat answer (question_chars=%d, context_chars=%d)", len(question), len(context))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context from uploaded documents:\n\n{context}\n\n"
                f"---\n\nQuestion: {question}"
            ),
        },
    ]

    try:
        stream = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            stream=True,
        )
    except APIConnectionError:
        logger.warning("Groq connection failed, recreating client and retrying once")
        stream = _reset_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            stream=True,
        )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
