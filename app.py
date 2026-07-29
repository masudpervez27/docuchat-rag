import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.ingestion import clear_all_documents, ingest_file
from services.llm import stream_answer
from services.vectorstore import similarity_search

load_dotenv()


def _configure_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root_logger.setLevel(level)

    return logging.getLogger("docuchat.app")


logger = _configure_logging()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuChat — Chat with Your Documents",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .source-pill {
        background: #1e2130; border: 1px solid #6C63FF;
        border-radius: 12px; padding: 3px 10px;
        font-size: 12px; color: #a0aec0; margin: 2px;
        display: inline-block;
    }
    .score-badge { color: #68d391; font-size: 11px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_ready" not in st.session_state:
    st.session_state.docs_ready = False
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 DocuChat")
    st.caption("Chat with any document — PDFs, Word, CSV, and more.")
    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded = st.file_uploader(
        "Supported: PDF, DOCX, TXT, CSV, MD",
        type=["pdf", "docx", "txt", "csv", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚡ Process", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑 Clear", use_container_width=True)

    if process_btn:
        if uploaded:
            logger.info("Processing %d uploaded documents", len(uploaded))
            progress = st.progress(0, text="Processing…")
            new_names = []
            for i, f in enumerate(uploaded):
                if f.name not in st.session_state.ingested_files:
                    with st.spinner(f"Chunking {f.name}…"):
                        n = ingest_file(f)
                    logger.info("Ingested document %s into %d chunks", f.name, n)
                    new_names.append(f.name)
                    st.session_state.ingested_files.append(f.name)
                progress.progress((i + 1) / len(uploaded))
            progress.empty()
            if new_names:
                st.success(f"✅ Ingested: {', '.join(new_names)}")
            else:
                st.info("All files already processed.")
            st.session_state.docs_ready = True
        else:
            logger.warning("Process clicked without uploaded files")
            st.warning("Please upload at least one file.")

    if clear_btn:
        clear_all_documents()
        st.session_state.messages = []
        st.session_state.docs_ready = False
        st.session_state.ingested_files = []
        logger.info("Cleared uploaded documents and chat history")
        st.rerun()

    if st.session_state.ingested_files:
        st.divider()
        st.caption("**Loaded files:**")
        for name in st.session_state.ingested_files:
            st.caption(f"• {name}")

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks retrieved (k)", min_value=2, max_value=8, value=4)
    show_sources = st.toggle("Show source citations", value=True)
    score_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.25, 0.05)

    st.divider()
    st.caption("🔧 **Stack:** LangChain · ChromaDB · HF BGE · Groq Llama 3")
    st.caption("🚀 **Deploy:** Streamlit Community Cloud")

# ── Main area ──────────────────────────────────────────────────
st.title("💬 Ask your documents anything")
st.caption(
    "Upload documents in the sidebar → click **Process** → start chatting. "
    "Every answer includes highlighted source snippets."
)

if not st.session_state.docs_ready:
    st.info(
        "👈 Upload one or more documents in the sidebar and click **⚡ Process** to begin.",
        icon="📌",
    )
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source chunk(s)"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<span class="source-pill">📄 {s["source"]} — chunk {s["chunk"]}'
                        f'<span class="score-badge"> ({s["score"]})</span></span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"> {s['snippet']}")
                    st.divider()

# New user input
if prompt := st.chat_input("Ask a question about your documents…"):
    logger.info("Received chat question (chars=%d) with %d loaded files", len(prompt), len(st.session_state.ingested_files))
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            results = similarity_search(prompt, k=top_k, threshold=score_threshold)

        if not results:
            logger.warning("No retrieval results for chat question (chars=%d)", len(prompt))
            response = (
                "⚠️ No relevant content found in the uploaded documents for this question. "
                "Try rephrasing, or upload additional documents."
            )
            st.markdown(response)
            sources: list[dict] = []
        else:
            context = "\n\n---\n\n".join(r["content"] for r in results)
            sources = results
            logger.info("Streaming answer from %d retrieved sources", len(sources))
            response = st.write_stream(stream_answer(prompt, context))

        if show_sources and sources:
            with st.expander(f"📎 {len(sources)} source chunk(s)"):
                for s in sources:
                    st.markdown(
                        f'<span class="source-pill">📄 {s["source"]} — chunk {s["chunk"]}'
                        f'<span class="score-badge"> ({s["score"]})</span></span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"> {s['snippet']}")
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
