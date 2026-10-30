# Agentic Chatbot using LangGraph & PostgreSQL RAG

A production-ready, modular, Human-in-the-Loop (HITL) agentic chatbot architecture built with **LangGraph**, **PostgreSQL** checkpoint persistence, **FAISS** vector retrieval, **Tesseract OCR** fallback with Unicode sanitization, **Ollama** embeddings (`nomic-embed-text`), and **vLLM / OpenAI-compatible** LLM inference.

---

## Architecture Overview

```text
Agentic-Chatbot-using-LangGraph/
│
├── app_hitl_postgres.py                  # Production Streamlit UI with streaming, multi-PDF uploads & HITL approvals
├── agentic_chatbot_hitl_backend_postgres.py  # LangGraph StateGraph facade with ConnectionPool & tools
├── requirements.txt                      # Pinned Python dependencies
├── .env.example                          # Safe environment configuration template
├── .gitignore                            # Comprehensive gitignore rules (secrets, venv, databases, indexes)
├── README.md                             # Project documentation and deployment guide
│
├── agent/                                # LangGraph workflow assembly
│   ├── __init__.py
│   ├── state.py                          # ChatState definition with add_messages reducer
│   ├── prompts.py                        # System instructions and prompt templates
│   ├── tool_registry.py                  # Central tool registry mapping
│   ├── routing.py                        # Deterministic tool routing and conditional edges
│   ├── nodes.py                          # chat_node and tool_node definitions
│   └── graph.py                          # StateGraph assembly compiled with PostgresSaver checkpointer
│
├── app/                                  # Application configuration & singletons
│   ├── __init__.py
│   ├── config.py                         # Settings dataclass & URL normalization
│   ├── dependencies.py                   # Service singletons (LLM, embeddings, DB connection)
│   └── logging_config.py                 # Structured logging configuration
│
├── database/                             # Database persistence layer
│   ├── __init__.py
│   ├── postgres.py                       # Connection pooling (psycopg_pool.ConnectionPool)
│   ├── checkpointing.py                  # LangGraph PostgresSaver checkpointer & thread manager
│   └── document_metadata_db.py           # PostgreSQL document_metadata table persistence
│
├── services/                             # Business logic and domain services
│   ├── __init__.py
│   ├── llm_service.py                    # Model initialization and tool binding
│   ├── embedding_service.py              # Ollama reachability and health checks
│   ├── pdf_ingestion_service.py          # PDF extraction, surrogate healing, chunking, and FAISS indexing
│   ├── document_service.py               # Thread-scoped multi-document discovery & retriever assembly
│   └── conversation_service.py           # Thread listing and checkpoint state inspection
│
├── storage/                              # Storage layout and file persistence
│   ├── __init__.py
│   ├── paths.py                          # Path resolution and thread sanitization
│   ├── document_metadata.py              # Dual persistence: PostgreSQL + local documents.json
│   └── faiss_store.py                    # FAISS vector store loading and serialization
│
├── tools/                                # Dedicated, modular tool packages
│   ├── __init__.py                       # Global tool registry
│   ├── calculator/                       # Safe mathematical evaluation
│   │   ├── __init__.py, tool.py, schemas.py, tests/test_calculator.py
│   ├── stock_price/                      # Alpha Vantage stock quotes
│   │   ├── __init__.py, tool.py, schemas.py, tests/test_stock_price.py
│   ├── purchase_stock/                   # HITL stock purchase simulation with interrupt()
│   │   ├── __init__.py, tool.py, schemas.py, tests/test_purchase_stock.py
│   ├── weather/                          # Real-time weather via Open-Meteo
│   │   ├── __init__.py, tool.py, schemas.py, tests/test_weather.py
│   ├── search/                           # Web search via Tavily
│   │   ├── __init__.py, tool.py, schemas.py, tests/test_search.py
│   ├── rag/                              # Thread-scoped vector search and smart retrieval
│   │   ├── __init__.py, tool.py, schemas.py, retrieval.py, tests/test_rag_tool.py
│   └── pdf/                              # PDF text extraction, OCR engine, and Unicode sanitizer
│       ├── __init__.py, extractor.py, ocr.py, schemas.py, tests/test_pdf_extraction.py
│
├── scripts/                              # Operations, diagnostics, and security audit CLI utilities
│   ├── check_ollama.py                   # Ollama connection and embedding model probe
│   ├── check_postgres.py                 # PostgreSQL connection and checkpoint inspector
│   ├── inspect_conversation.py           # Conversation history inspector for given thread ID
│   └── security_audit.py                 # Repository secret, token, and private IP scanner
│
└── tests/                                # Test suites
    ├── test_rag_postgres.py              # Integration tests for RAG indexing & persistence
    ├── test_multi_pdf_upload.py          # Multi-PDF sequential upload & isolation test
    ├── test_ollama_health.py             # Diagnostic tests for Ollama health check
    ├── test_csm_and_state_diagnostics.py # Scanned/vector-drawn PDF diagnostics with OCR
    └── test_full_workflow_live.py        # End-to-end live LLM & RAG workflow verification
```

---

## Key Features

1. **Modular Tool Architecture**:
   - Every tool lives in its own dedicated package with `tool.py`, Pydantic `schemas.py`, and isolated unit tests.
   - `tools/__init__.py` and `agent/tool_registry.py` provide a unified global registry.

2. **Thread-Isolated Multi-PDF RAG Pipeline**:
   - Each conversation thread maintains its own isolated FAISS indexes under `faiss_db/<thread_id>/documents/<doc_id>/`.
   - Documents are tracked with dual persistence: PostgreSQL `document_metadata` table + thread `documents.json` manifest.
   - Multiple PDFs can be uploaded to the same conversation thread without collisions or cross-talk.
   - Smart agentic retrieval targets specific documents when requested (e.g. *"explain this new pdf"*, *"explain OpenClaw"*), or merges all thread documents for cross-document synthesis.

3. **Fallback Extraction & Unicode Sanitizer**:
   - Three-tier extraction fallback:
     1. Native text extraction via `pypdf`.
     2. Vector-drawn / outlined text detection.
     3. High-resolution OCR fallback via Tesseract (`pypdfium2` / `pdf2image`).
   - Integrated `sanitize_unicode` engine heals paired UTF-16 surrogates (e.g., emojis `💡`) and strips lone/unpaired surrogates (`\ud800`–`\udfff`) and null bytes (`\x00`), preventing `UnicodeEncodeError` in downstream JSON serializers.

4. **Thread-Safe PostgreSQL Checkpointing**:
   - LangGraph `PostgresSaver` uses `psycopg_pool.ConnectionPool` (2–20 connections) to prevent pipeline mode collisions between background checkpointing tasks and Streamlit sidebar queries.
   - Preserves complete conversation history, thread activity, and interrupted execution states.

5. **Human-in-the-Loop (HITL)**:
   - Sensitive financial operations (`purchase_stock`) trigger LangGraph's `interrupt()`.
   - The Streamlit interface displays approval and rejection buttons (`Approve Purchase` / `Reject Purchase`), resuming graph execution via `Command(resume=...)`.

---

## Prerequisites

- **Python**: 3.10 to 3.12 (Python 3.12 recommended)
- **PostgreSQL**: 14+ running locally or remotely
- **Ollama**: Running locally or on the network with `nomic-embed-text`
- **LLM Server**: OpenAI-compatible server (vLLM, Ollama, LM Studio, or OpenAI)
- **Tesseract OCR**: (Optional but recommended for scanned or vector-outlined PDFs)

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/your-username/Agentic-Chatbot-using-LangGraph.git
cd Agentic-Chatbot-using-LangGraph

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the provided template:

```bash
# Windows PowerShell
cp .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and fill in your actual credentials and endpoints:

```ini
# OpenAI-compatible LLM Inference Endpoint (vLLM, Ollama, etc.)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=your_vllm_api_key_or_EMPTY
VLLM_MODEL=your_model_name

# Ollama Embeddings Service (nomic-embed-text)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# External API Keys (Optional based on tool usage)
TAVILY_API_KEY=your_tavily_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agentic_chatbot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password
```

> **Security Note:** The `.env` file is excluded in `.gitignore`. Never commit `.env` or any file containing private credentials to Git.

### 4. Setup Tesseract OCR (Optional for OCR Support)

- **Windows**: Install Tesseract from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) or place in `resources/ocr/tesseract/` or add to system `PATH`.
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
  ```
- **macOS**:
  ```bash
  brew install tesseract poppler
  ```

---

## Verifying the Setup

Run the diagnostic scripts to verify database and embedding connectivity:

```bash
# 1. Check Ollama reachability and embedding generation
python scripts/check_ollama.py

# 2. Check PostgreSQL connection and checkpoint table status
python scripts/check_postgres.py

# 3. Run security audit to verify no secrets are present in tracked files
python scripts/security_audit.py
```

---

## Running the Automated Test Suites

The project includes unit and integration tests covering tools, RAG, and live streaming:

```bash
# 1. Run all modular tool unit tests (calculator, search, stock, weather, RAG, PDF)
python -m unittest discover -s tools -v

# 2. Run multi-PDF sequential upload and thread isolation tests
python -m unittest tests/test_multi_pdf_upload.py -v

# 3. Run PostgreSQL RAG retrieval tests
python -m unittest tests/test_rag_postgres.py -v

# 4. Run Ollama health diagnostics
python -m unittest tests/test_ollama_health.py -v

# 5. Run vector-drawn & OCR fallback diagnostics
python tests/test_csm_and_state_diagnostics.py

# 6. Run full live end-to-end integration test (LLM + Embeddings + PostgreSQL + HITL)
python tests/test_full_workflow_live.py
```

---

## Launching the Application

Start the Streamlit user interface:

```bash
streamlit run app_hitl_postgres.py
```

Once running, navigate to `http://localhost:8501` in your browser.

---

## Troubleshooting

- **`psycopg.OperationalError: cannot enter pipeline mode`**:
  Ensure you are using `get_postgres_pool()` rather than sharing a single raw connection between LangGraph background tasks and Streamlit.
- **`UnicodeEncodeError: 'utf-8' codec can't encode character '\ud83d'`**:
  The built-in `sanitize_unicode` utility automatically heals surrogate pairs and strips lone surrogates extracted from malformed PDF fonts. Ensure your uploads pass through `services/pdf_ingestion_service.py`.
- **`Ollama connection timeout`**:
  On Windows, ensure `OLLAMA_BASE_URL` is set to `http://127.0.0.1:11434` instead of `localhost` to avoid IPv6 address resolution delays.
- **`OCR unavailable` warning**:
  If a PDF contains vector-drawn / outlined text without selectable characters, install Tesseract OCR and ensure `tesseract.exe` is in `PATH` or located under `resources/ocr/tesseract/`.

---

## License

This project is licensed under the Apache 2.0 License.
