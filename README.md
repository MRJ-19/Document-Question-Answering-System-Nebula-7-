# 🚀 Project Nebula-7: Local RAG Knowledge Base


[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/LLM-Gemma--3-blue?style=for-the-badge)](https://ollama.ai)

An advanced **Retrieval-Augmented Generation (RAG)** system that enables private, local conversations with your documents. Nebula-7 transforms raw text into a searchable vector brain using **ChromaDB** and **Gemma 3**.

---
The project supports:
- Document ingestion and vector indexing
- Semantic search using embeddings
- LLM-based answer generation
- REST API access via FastAPI
- A simple web-based chat interface

---

## Tech Stack
- Python
- LangChain
- LangGraph (conceptual state separation via chains)
- HuggingFace Embeddings
- Chroma Vector Database
- Ollama (Gemma 3 4B – local LLM)
- FastAPI
- HTML + Tailwind CSS (frontend)

---

## Project Files and Their Purpose

### main.py
FastAPI backend entry point.

What it does:
- Initializes the RAG pipeline (retriever + prompt + LLM)
- Exposes a `/ask` API endpoint
- Accepts user questions as JSON
- Returns answers generated using retrieved document context

This is the **main backend server**.

---

### ingest.py
Document ingestion and indexing script.

What it does:
- Loads text data from `dummy_data.txt`
- Splits and embeds the document content
- Stores embeddings in a persistent Chroma vector database (`vector_db`)

Run this file **before querying**, whenever documents change.

---

### chat.py
Command-line interactive chatbot.

What it does:
- Loads the same RAG pipeline
- Allows users to ask questions directly from the terminal
- Retrieves relevant context and prints LLM-generated answers

Useful for **local testing without API or UI**.

---

### query.py
Debug and validation script.

What it does:
- Tests whether document retrieval is working correctly
- Prints retrieved context from the vector database
- Helps debug ingestion or retrieval issues

---

### index.html
Simple frontend chat interface.

What it does:
- Provides a web-based UI for asking questions
- Sends user queries to the FastAPI backend
- Displays AI-generated answers in real time

This file communicates with the `/ask` API endpoint.

---

### dummy_data.txt
Sample document data.

What it contains:
- Text used for ingestion and indexing
- Acts as the knowledge base for question answering

You can replace or expand this file with your own documents.

---

### Test.py
Basic test file.

What it does:
- Simple placeholder script
- Not part of the core pipeline

---

## How the System Works

1. Documents are loaded and embedded using a sentence-transformer model.
2. Embeddings are stored in a persistent Chroma vector database.
3. When a user asks a question:
   - Relevant document chunks are retrieved using similarity search.
   - Retrieved context is injected into a prompt.
   - The LLM generates an answer strictly based on that context.
4. The answer is returned via API, CLI, or web UI.

---

## System Architecture

Document Text  
→ Embedding Generation  
→ Vector Database (Chroma)  
→ Semantic Retrieval  
→ Prompt Construction  
→ LLM Answer Generation  
→ API / CLI / Web UI  

---

## Setup Instructions

### 1. Install Dependencies

## 📂 System Components

| File | Role | Tech Stack |
| :--- | :--- | :--- |
| **`ingest.py`** | Document Vectorization | HuggingFace, ChromaDB |
| **`main.py`** | Production API | FastAPI, Pydantic |
| **`query.py`** | RAG Orchestration | LCEL (LangChain Expression Language) |
| **`index.html`** | User Interface | Tailwind CSS, JavaScript |
| **`chat.py`** | Terminal Interface | Interactive CLI |

---

## 🛠️ Installation & Quickstart

### 1. Clone & Install Dependencies
```bash
pip install -r requirements.txt
