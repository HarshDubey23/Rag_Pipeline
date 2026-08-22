# Rag_Pipeline

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about a PDF document with page-level citations.

Built with: Python, LangChain, ChromaDB, Google Gemini API (gemini-embedding-001 + gemini-3.6-flash), Streamlit.

## Features
- Ingests a PDF -> chunks -> embeddings -> persistent ChromaDB vector store
- Rate-limit-safe embedding (batching + exponential backoff for free-tier 429s)
- Answers grounded ONLY in the document, with "Sources: Page X" citations
- Streamlit chat UI with session history and a one-click vector DB rebuild button

## Run locally
```bash
pip install -r requirements.txt
# put your key in .env:  Gemini_API_Key="your-key"
streamlit run app.py        # web app at http://localhost:8501
python rag.py               # terminal-only chatbot
python rag.py --rebuild     # re-ingest the PDF from scratch
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. share.streamlit.io -> Create app -> select repo, main file `app.py`.
3. App Settings -> Secrets -> add: `Gemini_API_Key = "your-key"`

See `about.txt` for full architecture documentation and how every part works.
