<div align="center">

# 🩺 RSNA Medical Document RAG Pipeline
### Domain-Specific Technical Blueprint Retrieval with Page-Level Clinical Citations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-1C3C3C?logo=langchain)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FF6F00)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

*An end-to-end Retrieval-Augmented Generation (RAG) system engineered to analyze and answer technical queries against dense clinical documentation (RSNA Knee Abnormality Winning Blueprint) with strict source attribution.*

---

</div>

## 📌 Overview

Clinical documents, competition blueprints, and medical research specifications contain high-density diagnostic details that standard language models struggle to parse reliably without hallucinating.

This repository implements a **Domain-Specific RAG Pipeline** designed for:
1. **Zero-Hallucination Retrieval**: Strict prompt grounding restricting answers exclusively to context retrieved from the technical document.
2. **Page-Level Grounding**: Every answer returns exact source attribution (e.g., `Sources: Page 3, Page 8`) allowing human clinicians to audit conclusions instantly.
3. **Dual Execution Modes**: Interactive Streamlit web interface for visual exploration and a lightweight CLI terminal chatbot for automated workflows.

---

## 🏗️ Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │   RSNA Technical Blueprint (PDF Document)                │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼ [ PyMuPDF / Document Loader ]
  ┌─────────────────────────────────────────────────────────┐
  │   Text Extraction & Recursive Character Chunking         │
  │   (Optimized chunk size & overlap for medical passages) │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼ [ Embedding Model ]
  ┌─────────────────────────────────────────────────────────┐
  │   ChromaDB Local Vector Database (Persistent Index)     │
  └────────────────────────────┬────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       [ 🌐 Streamlit Web UI ]       [ 💻 Terminal CLI ]
          (`python app.py`)             (`python rag.py`)
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼ [ Cosine Similarity Top-K ]
  ┌─────────────────────────────────────────────────────────┐
  │   Augmented Context + Strict Grounding Prompt           │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼ [ LLM Response Generator ]
  ┌─────────────────────────────────────────────────────────┐
  │   Final Clinical Answer + Page Citations (`Page X, Y`)  │
  └─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Document Ingestion Engine:** Automatic document chunking with metadata preservation (page numbers, section titles).
- **Persistent Vector Store:** ChromaDB indexing on disk so embeddings are generated once and queried in sub-second latency.
- **Dynamic Re-Indexing:** Simple CLI flag (`python rag.py --rebuild`) or UI button to re-embed updated versions of the PDF without code modifications.
- **Clinical Attribution:** Directly outputs the exact pages where evidence was discovered to ensure full auditability.
- **Web UI & CLI:** Interactive chat UI with conversation history powered by Streamlit, plus a standalone terminal loop.

---

## 🚀 Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/HarshDubey23/Rag_Pipeline.git
cd Rag_Pipeline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Pipeline

#### Option A: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Open http://localhost:8501 in your browser.*

#### Option B: Run in Terminal (CLI Mode)
```bash
# Chat with the existing vector store
python rag.py

# Or rebuild the vector store and then chat
python rag.py --rebuild
```

---

## 📂 Repository Structure

```
├── RSNA_Knee_Abnormality_Winning_Blueprint_TechnicalDoc_v2.0_2026-08-09.pdf  # Target Clinical PDF
├── app.py              # Streamlit Web Application & Chat Interface
├── rag.py              # Core RAG logic: ingestion, embedding, retrieval & CLI
├── requirements.txt    # Production dependencies
└── README.md           # Documentation
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
