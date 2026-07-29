import os
import ssl

from dotenv import load_dotenv

_runtime_configured = False
_ssl_context: ssl.SSLContext | None = None


def configure_runtime() -> None:
    global _runtime_configured
    if _runtime_configured:
        return

    load_dotenv()
    _configure_huggingface_http()
    _hydrate_env_from_streamlit_secrets(
        "GROQ_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    )
    _runtime_configured = True


def get_secret(name: str, *, required: bool = False) -> str | None:
    configure_runtime()

    value = os.getenv(name)
    if value:
        return value

    value = _get_streamlit_secret(name)
    if value:
        os.environ[name] = value
        return value

    if required:
        raise RuntimeError(
            f"Missing required secret '{name}'. Set it in .env for local development "
            f"or in .streamlit/secrets.toml / Streamlit Community Cloud app settings."
        )

    return None


def create_httpx_client(**kwargs):
    import httpx

    return httpx.Client(verify=get_ssl_context(), **kwargs)


def _hydrate_env_from_streamlit_secrets(*names: str) -> None:
    for name in names:
        if os.getenv(name):
            continue

        value = _get_streamlit_secret(name)
        if value:
            os.environ[name] = value


def _get_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st
    except Exception:
        return None

    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    if value is None:
        return None
    return str(value)


def _configure_huggingface_http() -> None:
    try:
        from huggingface_hub import set_client_factory
        from huggingface_hub.utils._http import hf_request_event_hook
    except Exception:
        return

    def client_factory():
        return create_httpx_client(
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )

    set_client_factory(client_factory)


def get_ssl_context() -> ssl.SSLContext:
    global _ssl_context
    if _ssl_context is None:
        _ssl_context = ssl.create_default_context()
        _ssl_context.load_default_certs()
    return _ssl_context