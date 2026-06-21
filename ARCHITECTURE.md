# Hybrid RAG System - Architecture

## Overview

A production-grade Retrieval-Augmented Generation system with Docling document extraction, hybrid search (BM25 + vector), cross-encoder re-ranking, guardrails, and RAGAS evaluation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (CLI / FastAPI Web UI)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT GUARDRAILS                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Prompt     │  │   Harmful    │  │   Length               │ │
│  │  Injection   │  │   Content    │  │   Validation           │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  QUERY TRANSFORMATION                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Strategy   │  │   Multi-     │  │   Step-Back            │ │
│  │   Router     │  │   Query      │  │   Prompting            │ │
│  └──────┬──────┘  └──────────────┘  └────────────────────────┘ │
│         │                                                       │
│    ┌────┴────┐                                                  │
│    │ direct  │  → Skip LLM call, use original query             │
│    │rewrite  │  → Clarify vague questions                       │
│    │multi_q  │  → Generate alternative phrasings                │
│    │step_back│  → Broaden specific queries                      │
│    └─────────┘                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HYBRID RETRIEVAL                              │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │     BM25 Search     │      │   Vector Search     │          │
│  │   (Keyword-based)   │      │  (Semantic-based)   │          │
│  └──────────┬──────────┘      └──────────┬──────────┘          │
│             │                            │                      │
│             └──────────┬─────────────────┘                      │
│                        │                                        │
│                        ▼                                        │
│            ┌───────────────────────┐                            │
│            │   Reciprocal Rank    │                            │
│            │      Fusion (RRF)    │                            │
│            │  BM25: 0.4 weight    │                            │
│            │  Vector: 0.6 weight  │                            │
│            └───────────┬──────────┘                            │
│                        │                                        │
│                        ▼                                        │
│            ┌───────────────────────┐                            │
│            │  Cross-Encoder        │                            │
│            │  Re-ranking           │                            │
│            │  ms-marco-MiniLM      │                            │
│            │  Top 10 → Top 3       │                            │
│            └───────────┬──────────┘                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GENERATE                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ┌──────────────┐    ┌──────────────┐                      ││
│  │  │   LLM Call   │───▶│   Cache      │                      ││
│  │  │  (Groq)      │    │  (hit/miss)  │                      ││
│  │  └──────────────┘    └──────────────┘                      ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT GUARDRAILS                          │
│  ┌─────────────┐                                               │
│  │  Toxicity   │                                               │
│  │   Filter    │                                               │
│  └─────────────┘                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RESPONSE + METADATA                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Answer     │  │  Confidence  │  │   Trace                │ │
│  │              │  │    Score     │  │   (Langfuse)           │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Document Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     DOCUMENT UPLOAD                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  User uploads PDF/DOCX to MinIO                             ││
│  │  Folder: resume/ | in_progress_projects/ | completed_projects/ | uni_projects/ ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │ Webhook
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     WORKER SERVICE                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  1. Receive webhook event from MinIO                        ││
│  │  2. Download file to temp storage                           ││
│  │  3. Send to Docling for text extraction                     ││
│  │  4. Chunk extracted text (500 chars, 100 overlap)           ││
│  │  5. Build ChromaDB vectors + BM25 index                     ││
│  │  6. Cleanup temp files                                      ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DOCLING SERVICE                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  - PDF/DOCX/PPTX/HTML extraction                            ││
│  │  - Layout analysis + table structure                        ││
│  │  - OCR for scanned documents                                ││
│  │  - Returns structured Markdown                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag/
├── main.py                     # Entry point (CLI / API mode)
├── api.py                      # FastAPI web interface + REST API
├── worker.py                   # Document processing worker
├── Dockerfile                  # RAG app container
├── Dockerfile.worker           # Worker container
├── docker-compose.yml          # Multi-container deployment
├── requirements.txt            # RAG app dependencies
├── requirements-worker.txt     # Worker dependencies
├── .env                        # API keys (GROQ, Langfuse, MinIO)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py               # All settings & environment vars
│   ├── ingest.py               # Document loading from MinIO
│   ├── retrieval.py            # Hybrid search + cross-encoder re-ranking
│   ├── pipeline.py             # RAG chain + conversation memory
│   ├── graph.py                # Main RAG pipeline (linear flow)
│   ├── query_transform.py      # Query enhancement (rewrite, multi-query, step-back)
│   ├── guardrails.py           # Input/output safety guards
│   ├── cache.py                # LLM response caching
│   ├── storage.py              # MinIO object storage
│   ├── utils.py                # Shared utilities
│   └── langfuse_integration.py # Observability/tracing
│
├── evals/
│   ├── golden_dataset.json     # 10 golden Q&A pairs for evaluation
│   └── run_evals.py            # RAGAS evaluation script
│
└── chroma_db/                  # Auto-generated vector store (shared volume)
```

## Component Details

### 1. Document Extraction (Docling)

| Component | Technology |
|-----------|------------|
| Service | Docling Serve (Docker) |
| PDF Parsing | Layout analysis + table structure |
| OCR | For scanned documents |
| Output | Structured Markdown |
| API | REST endpoint at `http://docling:5001` |

### 2. Document Ingestion (Worker)

| Component | Technology |
|-----------|------------|
| Trigger | MinIO webhook on file upload/delete |
| PDF Loader | Docling (via API) |
| Chunking | RecursiveCharacterTextSplitter (500 chars, 100 overlap) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Vector Store | ChromaDB (persistent) |
| Keyword Index | rank-bm25 (in-memory) |

### 3. Hybrid Retrieval (`retrieval.py`)

| Component | Technology |
|-----------|------------|
| Keyword Search | BM25Retriever |
| Semantic Search | ChromaDB Vector Store |
| Fusion | EnsembleRetriever (RRF) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Weights | BM25: 0.4, Vector: 0.6 |

### 4. Query Transformation (`query_transform.py`)

| Strategy | Trigger | Action | LLM Call |
|----------|---------|--------|----------|
| `direct` | Clear, specific query | No transformation | No |
| `rewrite` | Ambiguous questions | LLM clarifies the question | Yes |
| `multi_query` | Vague/short queries | Generate 2-3 alternative phrasings | Yes |
| `step_back` | Specific how-to questions | Broaden for foundational context | Yes |

**Optimization:** Clear queries skip the LLM transform call entirely.

### 5. Guardrails (`guardrails.py`)

| Guard | Type | Action |
|-------|------|--------|
| Prompt Injection | Input | Block malicious instructions |
| Harmful Content | Input | Block dangerous queries |
| Length Validation | Input | Reject too short/long queries |
| Toxicity | Output | Filter offensive language |

### 6. Caching (`cache.py`)

| Feature | Description |
|---------|-------------|
| In-Memory Cache | LRU cache with max 1000 entries |
| Cache Keys | MD5 hash of query + context |
| Cache Hit | Skip LLM call, return cached response |

### 7. Object Storage (`storage.py`)

| Feature | Description |
|---------|-------------|
| Provider | MinIO (S3-compatible) |
| Bucket | `rag-documents` |
| Folders | `resume/`, `in_progress_projects/`, `completed_projects/`, `uni_projects/` |
| Webhook | Auto-index on upload/delete |

### 8. Observability (`langfuse_integration.py`)

| Feature | Description |
|---------|-------------|
| Tracing | Every query logged with full execution trace |
| Grading Scores | Answer quality scores logged |
| Guard Results | Input/output guard decisions logged |
| Cache Hits | Cached responses flagged in trace |
| Dashboard | https://us.cloud.langfuse.com |

### 9. Evaluation (`evals/run_evals.py`)

| Metric | What It Measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in context (no hallucination)? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are retrieved docs focused and relevant? |
| Context Recall | Did we retrieve all needed information? |
| Factual Correctness | Is answer factually accurate vs ground truth? |

## API Endpoints

### RAG App (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Ask a question |
| POST | `/upload?folder=` | Upload file to MinIO (folder: resume/in_progress_projects/completed_projects/uni_projects) |
| POST | `/upload-url?folder=` | Download from URL, store in MinIO |
| GET | `/files?folder=` | List files (optional folder filter) |
| DELETE | `/files/{folder}/{filename}` | Delete file from MinIO |
| POST | `/reindex` | Trigger full reindex on worker |
| POST | `/clear-memory` | Clear conversation history |
| GET | `/` | Web UI |

### Worker (port 9000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/webhook/minio` | MinIO event notification |
| POST | `/reindex` | Full reindex all documents |
| GET | `/health` | Health check |

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Object Storage | MinIO (S3-compatible) |
| Document Extraction | Docling Serve |
| Re-ranker | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| Framework | LangChain |
| API | FastAPI |
| Observability | Langfuse |
| Evaluation | RAGAS |
| Container | Docker + Docker Compose |

## Deployment

### Docker Compose

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Services:**

| Service | Port | Description |
|---------|------|-------------|
| rag | 8000 | RAG API + Web UI |
| worker | 9000 | Document processing worker |
| docling | 5001 | Document extraction API |
| minio | 9002 | S3 API |
| minio | 9003 | MinIO Console |

## Secrets Management

**`.env` file:**

```bash
GROQ_API_KEY=gsk_xxxxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_HOST=https://us.cloud.langfuse.com
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag-documents
MINIO_SECURE=false
RATE_LIMIT=5
RATE_WINDOW=60
```
