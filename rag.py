"""RAG pipeline for chatting with the RSNA Knee Abnormality PDF.

This file can be used in two ways:
  1. Import it as a module (used by app.py - the Streamlit web UI):
         import rag
         result = rag.ask("What is this document about?")
         rag.build_vectorstore(force=True)   # re-ingest the PDF

  2. Run it from the command line as a simple terminal chatbot:
         python rag.py             # start chat (uses existing chroma_db)
         python rag.py --rebuild   # delete + rebuild chroma_db, then chat

rag.ask(question) returns a dict: {"answer": str, "sources": "Page 3, Page 12"}
"""

import os
import shutil
import sys
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("Gemini_API_Key")
PDF_PATH = "RSNA_Knee_Abnormality_Winning_Blueprint_TechnicalDoc_v2.0_2026-08-09.pdf"
CHROMA_DIR = "chroma_db"
COLLECTION = "rsna_doc"

CHUNK_SIZE = 1000      # max characters per text chunk
CHUNK_OVERLAP = 200    # characters shared between consecutive chunks
TOP_K = 5              # how many chunks are retrieved per question

EMBEDDING_MODEL = "models/gemini-embedding-001"   # outputs 3072-dim vectors
LLM_MODEL = "gemini-3.6-flash"

# Free tier allows ~100 embedding requests/minute. We batch + pause + retry
# so bulk ingestion never hits a 429 RESOURCE_EXHAUSTED error.
EMBED_BATCH_SIZE = 50
EMBED_PAUSE_SECONDS = 3.0


# --------------------------------------------------------------------------
# Rate-limit-safe embedding wrapper
# --------------------------------------------------------------------------
class ThrottledEmbeddings:
    """Wraps GoogleGenerativeAIEmbeddings to respect free-tier rate limits.

    - embed_documents: embeds in batches of EMBED_BATCH_SIZE, sleeps between
      batches, and retries with exponential backoff if a 429 happens anyway.
    - embed_query: single request, passed straight through.
    """

    def __init__(self, base, batch_size=EMBED_BATCH_SIZE, pause=EMBED_PAUSE_SECONDS):
        self.base = base
        self.batch_size = batch_size
        self.pause = pause

    def embed_documents(self, texts):
        all_vectors = []
        for i in range(0, len(texts), self.batch_size):
            if i > 0:
                time.sleep(self.pause)
            batch = texts[i : i + self.batch_size]
            for attempt in range(5):
                try:
                    all_vectors.extend(self.base.embed_documents(batch))
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    wait = 2 ** attempt * 5
                    print(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
        return all_vectors

    def embed_query(self, text):
        return self.base.embed_query(text)


def get_embeddings():
    """Plain (non-throttled) embeddings - fine for single query vectors."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


# --------------------------------------------------------------------------
# Ingestion: PDF -> chunks -> vectors -> ChromaDB
# --------------------------------------------------------------------------
def build_vectorstore(force=False):
    """Load the PDF, chunk it, embed it, and persist everything in ChromaDB.

    force=True deletes any existing chroma_db folder first (full rebuild).
    Returns the Chroma vectorstore object.
    """
    if force and os.path.exists(CHROMA_DIR):
        print("Deleting existing vector database...")
        shutil.rmtree(CHROMA_DIR)

    if not os.path.exists(CHROMA_DIR):
        print("Loading PDF...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        print(f"Splitting {len(docs)} pages...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(docs)
        print(f"Total chunks: {len(chunks)}")

        print("Embedding and storing in Chroma (this can take a few minutes)...")
        Chroma.from_documents(
            documents=chunks,
            embedding=ThrottledEmbeddings(get_embeddings()),
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION,
        )
        print("Vector store created.")

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION,
    )


# --------------------------------------------------------------------------
# Query: retrieve relevant chunks + generate grounded answer with citations
# --------------------------------------------------------------------------
def format_docs(docs):
    """Turn retrieved chunks into one context string, tagged with page numbers.

    PyPDFLoader stores the page number (0-based) in doc.metadata["page"], so we
    add 1 to show the human-readable page number to the user.
    """
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_sources(docs):
    """Build a sorted, de-duplicated source list like 'Page 3, Page 12'."""
    pages = sorted({doc.metadata.get("page", 0) + 1 for doc in docs})
    return ", ".join(f"Page {p}" for p in pages)


SYSTEM_PROMPT = (
    "You are an expert assistant. Answer the question based only on the "
    "provided context. If the answer is not in the context, say so. "
    "The context sections are tagged with [Page N]; mention the page "
    "numbers you used at the end of your answer."
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

_generation_chain = (
    _prompt
    | ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)
    | StrOutputParser()
)


def ask(question, vectorstore=None):
    """Answer one question. Returns {"answer": str, "sources": str}."""
    if vectorstore is None:
        vectorstore = build_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)                 # retrieve top-k chunks
    answer = _generation_chain.invoke(                 # LLM answers from them
        {"context": format_docs(docs), "question": question}
    )
    return {"answer": answer, "sources": format_sources(docs)}


# --------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------
def main():
    force = "--rebuild" in sys.argv
    vectorstore = build_vectorstore(force=force)

    print("Ready! Type your question (or 'exit').")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        result = ask(q, vectorstore)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
