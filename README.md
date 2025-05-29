# 🛰️ Domain-Specific Q&A Assistant

This project implements a **retrieval-augmented generation (RAG)** chatbot designed to answer domain-specific technical questions based on internal documents.  
It leverages **LangChain**, **FAISS**, and **Hugging Face Transformers** to deliver fast, accurate, and context-aware responses.

---

## ✨ Features

✅ End-to-end RAG pipeline using LangChain  
✅ Local GPU-accelerated LLM inference (Flan-T5)  
✅ Semantic search with SentenceTransformers + FAISS vector store  
✅ Streamlit web app for interactive Q&A  
✅ CLI interface for terminal-based interaction  
✅ Modular design supporting local and cloud backends

---

## 🏗️ Architecture Overview

- **Data ingestion** → Structured and unstructured `.txt` files are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector storage** → FAISS provides a high-speed local vector store for semantic retrieval.
- **LLM backend** → Local Hugging Face model (`google/flan-t5-base`) runs on GPU for fast, cost-free inference.
- **Orchestration** → LangChain connects the retriever and generator into a unified RAG pipeline.
- **Frontend** → Streamlit provides an interactive web UI; CLI mode is also supported.

---

## 🛠️ Setup Instructions

1️⃣ **Clone the repo**
```bash
git clone https://github.com/elaminemouhamadali/domain-qa-assistant.git
cd domain-qa-assistant
