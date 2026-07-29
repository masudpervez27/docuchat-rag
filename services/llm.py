import os
import logging
import time
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

    last_error: APIConnectionError | None = None

    # Retry streaming requests because transient network/proxy/TLS issues are common.
    for attempt in range(1, 4):
        try:
            stream = _get_client().chat.completions.create(
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
            return
        except APIConnectionError as exc:
            last_error = exc
            logger.warning(
                "Groq streaming connection failed (attempt %d/3). cause=%r",
                attempt,
                exc.__cause__,
            )
            _reset_client()
            if attempt < 3:
                time.sleep(attempt)

    # Fall back to non-streaming completion when streaming repeatedly fails.
    try:
        logger.warning("Falling back to non-streaming Groq completion")
        completion = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            stream=False,
        )
        text = completion.choices[0].message.content
        if text:
            yield text
            return
    except APIConnectionError as exc:
        logger.error("Groq non-streaming fallback failed. cause=%r", exc.__cause__)

    if last_error is not None:
        raise last_error
