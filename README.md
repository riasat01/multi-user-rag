# 📚 Multi-User RAG API with FastAPI & Google Gemini

A lightweight, multi-user Retrieval-Augmented Generation (RAG) backend built from scratch. It allows multiple users to upload documents (`.pdf`, `.txt`, `.docx`), index them with vector embeddings, and query their private context using Google Gemini with strict data isolation guardrails.

---

## 🌟 Key Features

* 📁 **Multi-Format Document Parsing:** Ingests `.pdf`, `.txt`, and `.docx` files seamlessly.
* 🔒 **User Data Isolation:** Uses ChromaDB metadata filters (`user_id` AND `doc_name`) so users can never access each other's uploaded data.
* 🧠 **Hallucination Guardrails:** Low-temperature Gemini prompts strictly bound answers to retrieved context.
* 🚀 **FastAPI Backend:** Fully asynchronous file uploads, document management, and chat endpoints with auto-generated OpenAPI documentation.

---

## 🛠️ Tech Stack

* **Framework:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (`langchain-chroma`)
* **LLM & Embeddings:** Google Gemini (`gemini-3.5-flash` & `gemini-embedding-001`) via `langchain-google-genai`
* **Orchestration:** [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-text-splitters`)

---

## 📁 Project Structure

```text
.
├── config.py         # Centralized configuration (API keys, models, chunk sizes)
├── loaders.py        # File loader & RecursiveCharacterTextSplitter logic
├── vectorstore.py    # ChromaDB indexing, similarity search, & delete filters
├── rag_chain.py      # Gemini LLM initialization & context prompt builder
├── main.py           # FastAPI HTTP endpoints
└── requirements.txt  # Dependencies list