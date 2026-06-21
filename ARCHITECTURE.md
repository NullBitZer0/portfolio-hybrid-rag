# Hybrid RAG System - Architecture

## Overview

A production-grade Retrieval-Augmented Generation system with Docling document extraction, OpenSearch hybrid search (BM25 + k-NN), Cohere reranking, guardrails, and RAGAS evaluation. Deployed on Oracle Cloud via Coolify.

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
                    │    QUERY TRANSFORMATION       │
                    │  ┌─────────┐ ┌────────────┐  │
                    │  │ direct  │ │  rewrite   │  │
                    │  │ (skip)  │ │ (clarify)  │  │
                    │  ├─────────┤ ├────────────┤  │
                    │  │multi_q  │ │ step_back  │  │
                    │  │(altern.)│ │ (broaden)  │  │
                    │  └─────────┘ └────────────┘  │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │     HYBRID RETRIEVAL          │
                    │  ┌──────────┐  ┌──────────┐  │
                    │  │  BM25    │  │  k-NN    │  │
                    │  │ (keyword)│  │(semantic)│  │
                    │  └────┬─────┘  └────┬─────┘  │
                    │       └──────┬──────┘         │
                    │              v                │
                    │     OpenSearch RRF Fusion      │
                    │              |                │
                    │              v                │
                    │     Cohere rerank-v3.5        │
                    │       Top 10 -> Top 3         │
                    └──────────────┬───────────────┘
                                   |
                                   v
                    ┌──────────────────────────────┐
                    │        GENERATION             │
                    │  Groq llama-3.3-70b           │
                    │  + conversation memory (5)    │
                    │  + in-memory LRU cache        │
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
                    │  4. Chunk (500 chars, 100)    │
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
| Chunking | RecursiveCharacterTextSplitter (500 chars, 100 overlap) |
| Embeddings | Gemini gemini-embedding-2 (768-dim) |
| Vector Store | OpenSearch k-NN index |
| Keyword Index | OpenSearch BM25 (built-in) |

### 3. Hybrid Retrieval (`retrieval.py`)

| Component | Technology |
|-----------|------------|
| Keyword Search | OpenSearch BM25 (built-in) |
| Semantic Search | OpenSearch k-NN (HNSW, cosine similarity) |
| Fusion | OpenSearch native hybrid query (bool filter) |
| Re-ranking | Cohere rerank-v3.5 (Top 10 -> Top 3) |
| Embeddings | Gemini gemini-embedding-2 (768-dim) |

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
| Portfolio Classifier | Input | Block general knowledge (keyword + LLM) |
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
| Bucket | `rag-document` |
| Structure | Flat (root-level files) |
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
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | Google Gemini (gemini-embedding-2, 768-dim) |
| Reranking | Cohere (rerank-v3.5) |
| Vector + Keyword | OpenSearch 2.19.0 (k-NN + BM25) |
| Object Storage | MinIO (S3-compatible) |
| Document Extraction | Docling Serve |
| Framework | LangChain (function chain) |
| API | FastAPI |
| Frontend | Next.js (portfolio) + floating AI assistant |
| Observability | Langfuse |
| Evaluation | RAGAS |
| Infrastructure | Docker Compose, Oracle Cloud, Coolify |

## Deployment

### Docker Compose (6 services)

| Service | Description |
|---------|-------------|
| rag | FastAPI API + web UI |
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
