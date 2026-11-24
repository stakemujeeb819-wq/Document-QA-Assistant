from pathlib import Path

import streamlit as st

from src.document_store import DocumentStore
from src.rag import RAGPipeline

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄")
st.title("📄 Document Q&A Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None


def show_sources(sources):
    """Display the retrieved chunks with their similarity scores."""
    with st.expander("Sources"):
        for s in sources:
            st.caption(f"Similarity score: {s['score']}")
            st.text(s["text"][:300])


# Sidebar: upload files and build the index
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Build index", disabled=not files):
        for f in files:
            (UPLOAD_DIR / f.name).write_bytes(f.getbuffer())
        try:
            docs = DocumentStore(str(UPLOAD_DIR)).load()
            st.session_state.rag = RAGPipeline(docs, prompt_version="v3_refuse")
            st.session_state.messages = []
            st.success(f"Indexed {len(docs)} document(s)")
        except Exception as error:
            st.error(f"Could not build index: {error}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Main area: chat
if st.session_state.rag is None:
    st.info("Upload files in the sidebar and click Build index to start.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    question = st.chat_input("Ask a question about your documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.rag.ask(question)
                st.write(result["answer"])
                show_sources(result["sources"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as error:
                st.error(f"Question failed: {error}")