# Hybrid RAG System - Architecture

## Overview

A production-grade Agentic RAG system with LangGraph agent, Docling document extraction, OpenSearch hybrid search (BM25 + k-NN), Cohere reranking, guardrails, Upstash Redis caching, and RAGAS evaluation. Deployed on Oracle Cloud via Coolify.

## System Architecture

```
                            Recruiter Question
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │       INPUT GUARDRAILS        │
                    │  prompt injection detection   │
                    │  harmful content filter       │
                    │  length validation            │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │    PORTFOLIO CLASSIFIER       │
                    │  keyword check + LLM fallback │
                    │  (blocks general knowledge)   │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │      AGENT (LangGraph)        │
                    │  LLM decides which tools to   │
                    │  call based on query intent    │
                    │  max 2 retrieval rounds       │
                    └──────────────┬───────────────┘
                                   |
                    ┌──────────────┼───────────────┐
                    v              v               v
            ┌──────────┐  ┌──────────────┐  ┌──────────┐
            │search_all │  │search_projects│  │search_   │
            │ (general) │  │ (project PDFs)│  │skills    │
            └─────┬────┘  └──────┬───────┘  │(skill    │
                  │              │           │ PDFs)    │
                  v              v           └────┬─────┘
            ┌─────────────────────────────────────┐
            │        HYBRID RETRIEVAL              │
            │  ┌──────────┐  ┌──────────┐         │
            │  │  BM25    │  │  k-NN    │         │
            │  │ (keyword)│  │(semantic)│         │
            │  └────┬─────┘  └────┬─────┘         │
            │       └──────┬──────┘                │
            │              v                       │
            │     OpenSearch RRF Fusion            │
            │              |                       │
            │              v                       │
            │     Cohere rerank-v3.5               │
            │       Top 10 -> Top 3                │
            └──────────────┬──────────────────────┘
                           |
                           v
                    ┌──────────────────┐
                    │  AGENT ANALYZE   │
                    │  Enough info?    │
                    │  No -> retrieve  │
                    │  again (max 2)   │
                    │  Yes -> generate │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────────────────┐
                    │        GENERATION             │
                    │  Groq openai/gpt-oss-120b     │
                    │  + Redis conversation memory  │
                    │  + Redis LLM cache (1hr TTL)  │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │     OUTPUT GUARDRAILS         │
                    │     toxicity filter           │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │     ANSWER + SOURCES          │
                    │  + Langfuse trace             │
                    └──────────────────────────────┘
```

## Document Processing Pipeline

```
                    ┌──────────────────────────────┐
                    │      DOCUMENT UPLOAD          │
                    │  PDF uploaded to MinIO        │
                    └──────────────┬───────────────┘
                                   | webhook
                                   v
                    ┌──────────────────────────────┐
                    │      WORKER SERVICE           │
                    │  1. Receive MinIO webhook     │
                    │  2. Download file to /tmp     │
                    │  3. Docling text extraction   │
                    │  4. Parent-child chunking     │
                    │     (2000 char parents +      │
                    │      500 char children)       │
                    │  5. Gemini embed (768-dim)    │
                    │  6. Index to OpenSearch        │
                    │  7. Cleanup temp files        │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │      DOCLING SERVICE          │
                    │  PDF/DOCX/PPTX extraction     │
                    │  Layout analysis + tables     │
                    │  OCR for scanned docs         │
                    │  Returns structured Markdown  │
                    └──────────────────────────────┘
```

## Component Details

### 1. Document Extraction (Docling)

| Component | Technology |
|-----------|------------|
| Service | Docling Serve (Docker) |
| PDF Parsing | Layout analysis + table structure |
| OCR | For scanned documents |
| Output | Structured Markdown |
| API | REST at `http://docling:5001/v1/convert/file` |

### 2. Document Ingestion (Worker)

| Component | Technology |
|-----------|------------|
| Trigger | MinIO webhook on file upload/delete |
| PDF Loader | Docling (via API) |
| Chunking | Parent-child: 2000 char parents (LLM context) + 500 char children (search precision) |
| Embeddings | Gemini gemini-embedding-2 (768-dim) |
| Vector Store | OpenSearch k-NN index |
| Keyword Index | OpenSearch BM25 (built-in) |

### 3. Agentic RAG Agent (`graph.py`)

| Component | Technology |
|-----------|------------|
| Framework | LangGraph StateGraph |
| Agent LLM | Groq openai/gpt-oss-120b |
| Tools | 5 tools: search_all, search_projects, search_skills, search_source, list_documents |
| Multi-step | Max 2 retrieval rounds (retrieve -> analyze -> retrieve again if needed) |
| Tool Executor | Custom tool_executor with print logging |

**Agent Flow:**
1. LLM receives system prompt with available tools and document list
2. LLM decides which tools to call based on query intent
3. Tools execute OpenSearch hybrid search with Cohere reranking
4. Agent analyzes results and decides if more retrieval is needed
5. Max 2 rounds to prevent infinite loops
6. Final answer generated from all accumulated context

### 4. Agent Tools (`tools.py`)

| Tool | What it searches | When to use |
|------|-----------------|-------------|
| `search_all` | All documents (no source filter) | Default for most queries |
| `search_projects` | Project PDFs (fraud, RAG, all_projects) | Project-related queries |
| `search_skills` | Skill PDFs (technical, soft) | Skill-related queries |
| `search_source` | Specific document file | When agent needs details from one doc |
| `list_documents` | Index metadata | When agent needs to know what's available |

**Source names in OpenSearch** (include folder prefix for resume/ files):
- `resume/technical_skills.pdf` (14 chunks)
- `resume/all_projects.pdf` (12 chunks)
- `soft_skills.pdf` (42 chunks)
- `realtime_fraud_detection.pdf` (40 chunks)
- `hybrid_rag_project.pdf` (35 chunks)

### 5. Hybrid Retrieval (`retrieval.py`)

| Component | Technology |
|-----------|------------|
| Keyword Search | OpenSearch BM25 (built-in) |
| Semantic Search | OpenSearch k-NN (HNSW, cosine similarity) |
| Fusion | OpenSearch native hybrid query (bool filter) |
| Re-ranking | Cohere rerank-v3.5 (Top 10 -> Top 3) |
| Embeddings | Gemini gemini-embedding-2 (768-dim) |
| Source Filter | Filter by specific document filename |

### 6. Guardrails (`guardrails.py`)

| Guard | Type | Action |
|-------|------|--------|
| Prompt Injection | Input | Block malicious instructions |
| Harmful Content | Input | Block dangerous queries |
| Length Validation | Input | Reject too short/long queries |
| Portfolio Classifier | Input | Block general knowledge (keyword + LLM) |
| Toxicity | Output | Filter offensive language |

### 7. Caching (`cache.py`)

| Feature | Description |
|---------|-------------|
| Provider | Upstash Redis (REST API) |
| TTL | 1 hour (auto-expire stale cache) |
| Cache Keys | MD5 hash of query + context |
| Cache Hit | Skip LLM call, return cached response |
| Fallback | In-memory dict if Redis unavailable |

### 8. Conversation Memory (`pipeline.py`)

| Feature | Description |
|---------|-------------|
| Provider | Upstash Redis (REST API) |
| TTL | 24 hours |
| Session | Per-session storage (keyed by session_id) |
| Window | Last 5 message pairs (configurable) |
| Fallback | In-memory list if Redis unavailable |

### 9. Object Storage (`storage.py`)

| Feature | Description |
|---------|-------------|
| Provider | MinIO (S3-compatible) |
| Bucket | `rag-document` |
| Structure | Flat (root-level files, resume/ files have prefix in source name) |
| Webhook | Auto-index on upload/delete |

### 10. Observability (`langfuse_integration.py`)

| Feature | Description |
|---------|-------------|
| Tracing | Every query logged with full execution trace |
| Grading Scores | Answer quality scores logged |
| Guard Results | Input/output guard decisions logged |
| Cache Hits | Cached responses flagged in trace |
| Dashboard | https://us.cloud.langfuse.com |

### 11. Evaluation (`evals/run_evals.py`)

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
| POST | `/upload?folder=` | Upload file to MinIO |
| POST | `/upload-url?folder=` | Download from URL, store in MinIO |
| GET | `/files?folder=` | List files (optional folder filter) |
| DELETE | `/files/{folder}/{filename}` | Delete file from MinIO |
| POST | `/reindex` | Trigger full reindex on worker |
| POST | `/clear-memory` | Clear conversation history |
| GET | `/` | Health check |

### Worker (port 9000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/webhook/minio` | MinIO event notification |
| POST | `/reindex` | Full reindex all documents |
| GET | `/health` | Health check |

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Groq (openai/gpt-oss-120b) |
| Agent Framework | LangGraph (StateGraph) |
| Embeddings | Google Gemini (gemini-embedding-2, 768-dim) |
| Reranking | Cohere (rerank-v3.5) |
| Vector + Keyword | OpenSearch 2.19.0 (k-NN + BM25) |
| Object Storage | MinIO (S3-compatible) |
| Document Extraction | Docling Serve |
| Cache | Upstash Redis (REST API) |
| Memory | Upstash Redis (REST API) |
| Framework | LangChain + LangGraph |
| API | FastAPI |
| Frontend | Next.js (portfolio) + floating AI assistant |
| Observability | Langfuse |
| Evaluation | RAGAS |
| Infrastructure | Docker Compose, Oracle Cloud, Coolify |

## Deployment

### Docker Compose (6 services)

| Service | Description |
|---------|-------------|
| rag | FastAPI API + LangGraph agent |
| worker | Document processing (webhook-triggered) |
| opensearch | Vector + keyword search engine |
| opensearch-dashboards | Search analytics UI |
| minio | S3-compatible object storage |
| docling | Document extraction API |

### Production (Coolify on Oracle Cloud)

```bash
# Push to GitHub -> Coolify auto-builds and deploys
git push origin main

# Manual reindex after upload
curl -X POST http://<rag-url>/reindex
```

### Local Development

```bash
# Start infrastructure
docker run -d --name opensearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.19.0

docker run -d --name minio -p 9002:9000 -p 9003:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"

docker run -d --name docling -p 5001:5001 \
  ghcr.io/docling-project/docling-serve:latest

# Install and run
pip install -r requirements.txt
python main.py --api
```

## Secrets

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq LLM API key |
| `GEMINI_API_KEY` | Google Gemini embedding API key |
| `COHERE_API_KEY` | Cohere reranking API key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse observability |
| `LANGFUSE_SECRET_KEY` | Langfuse observability |
| `OPENSEARCH_HOST` | OpenSearch endpoint |
| `MINIO_ENDPOINT` | MinIO S3 endpoint |
| `MINIO_BUCKET` | Storage bucket name |
| `DOCLING_URL` | Docling extraction service URL |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis REST endpoint |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis auth token |
