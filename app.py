"""Streamlit web UI for the RAG chatbot.

Run locally:
    streamlit run app.py

Deployment: works on Streamlit Community Cloud - the API key is read from
Streamlit secrets (Gemini_API_Key) and bridged into the environment before
rag.py is imported, so rag.py works identically locally (via .env) and in
the cloud (via st.secrets).
"""

import os

import streamlit as st

# --- Bridge Streamlit secrets -> environment variable (needed on cloud) -----
# Locally, rag.py reads the key from .env. On Streamlit Cloud there is no
# .env file, so we copy the secret into os.environ BEFORE importing rag.
if not os.getenv("Gemini_API_Key"):
    try:
        os.environ["Gemini_API_Key"] = st.secrets["Gemini_API_Key"]
    except Exception:
        pass  # handled below

if not os.getenv("Gemini_API_Key"):
    st.error(
        "Gemini_API_Key is missing. Locally: add it to the .env file. "
        "On Streamlit Cloud: Manage app -> Settings -> Secrets, then add "
        'Gemini_API_Key = "your-key"'
    )
    st.stop()

import rag  # noqa: E402  (import AFTER the env bridge above)

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chat - RSNA RAG", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Ask questions about the RSNA Knee Abnormality technical document.")

# --------------------------------------------------------------------------
# Vector store: cached so it is loaded/built only once per server process
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return rag.build_vectorstore()


# --------------------------------------------------------------------------
# Sidebar: rebuild button
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("🔁 Rebuild Vector Database"):
        st.cache_resource.clear()  # drop the cached vectorstore
        with st.spinner("Re-ingesting PDF and rebuilding vectors... this can take a few minutes."):
            rag.build_vectorstore(force=True)
        st.success("Vector database rebuilt!")
        st.rerun()

# --------------------------------------------------------------------------
# Chat history (persists across reruns within the session)
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous messages on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {msg['sources']}")

# --------------------------------------------------------------------------
# Chat input -> answer -> citations
# --------------------------------------------------------------------------
question = st.chat_input("Ask a question about the document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.ask(question, get_vectorstore())
                answer, sources = result["answer"], result["sources"]
            except Exception as e:
                answer, sources = f"Error: {e}", ""
        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {sources}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
