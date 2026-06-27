# Agentic RAG Portfolio Assistant

A production-grade Retrieval-Augmented Generation system that answers questions about my projects, skills, resume, and experience. Deployed and live at [adeeshaperera.me](https://adeeshaperera.me).

**[Live Demo](https://uu046uuagdooowl3dmx12t97.140.245.59.209.sslip.io)** | **[Portfolio](https://adeeshaperera.me)** | **[GitHub](https://github.com/NullBitZer0/portfolio-hybrid-rag)**

---

## What It Does

A recruiter visits my portfolio, clicks the AI assistant button, and asks:

- *"What ML projects has Adeesha worked on?"*
- *"Tell me about his experience with NLP"*
- *"What's his background in cloud infrastructure?"*

The system retrieves relevant information from my documents and generates accurate, grounded answers in seconds.

---

## Architecture

```
Recruiter Question
       |
       v
[Input Guardrails] -- prompt injection, harmful content, length check
       |
       v
[Portfolio Classifier] -- keyword check + LLM fallback (blocks general knowledge)
       |
       v
[LangGraph Agent] -- LLM decides which tools to call based on query intent
       |
  ┌────┼────┬──────────┐
  v    v    v          v
search search search  list
_all  _projs _skills  docs
  |    |    |          |
  v    v    v          v
[Hybrid Retrieval] -- OpenSearch BM25 + k-NN dense vector search
       |
       v
[Cohere Rerank] -- top 10 results -> top 3 most relevant
       |
       v
[Agent Analyze] -- enough info? no -> retrieve again (max 2 rounds)
       |
       v
[LLM Generation] -- Groq gpt-oss-120b + Gemini fallback + Redis conversation memory
       |
       v
[Output Guardrails] -- toxicity filter
       |
       v
Answer + Sources
```

---

## Key Features

| Feature | Implementation |
|---|---|
| **Agentic RAG** | LangGraph StateGraph with 5 tools, multi-step retrieval |
| **Hybrid Search** | OpenSearch BM25 (keyword) + k-NN (semantic) with RRF fusion |
| **Reranking** | Cohere rerank-v3.5 (top 10 -> top 3) |
| **Embeddings** | Google Gemini gemini-embedding-2 (768-dim) |
| **Document Extraction** | Docling Serve (layout-aware PDF parsing) |
| **Parent-Child Chunking** | 2000 char parents (LLM context) + 500 char children (search precision) |
| **Query Expansion** | Synonym-based expansion for better BM25 recall |
| **Guardrails** | Input: injection, toxicity, length, portfolio classifier. Output: toxicity |
| **LLM Caching** | Upstash Redis with 1hr TTL (in-memory fallback) |
| **Conversation Memory** | Upstash Redis with 24hr TTL, sliding window of last 5 messages |
| **API Key Auth** | X-API-Key header for /query and admin endpoints |
| **Rate Limiting** | 5 requests/min per IP |
| **Gemini Fallback** | Groq primary, Gemini 2.5 Flash on rate limit/connection errors |
| **Observability** | Full pipeline traced with Langfuse |
| **Structured Logging** | logging module with per-module loggers |
| **Evaluation** | RAGAS metrics (faithfulness, relevancy, precision, recall) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq openai/gpt-oss-120b (primary), Gemini 2.5 Flash (fallback) |
| **Agent Framework** | LangGraph (StateGraph) |
| **Embeddings** | Google Gemini gemini-embedding-2 (768-dim) |
| **Reranking** | Cohere rerank-v3.5 |
| **Vector + Search** | OpenSearch 2.19.0 (k-NN + BM25) |
| **Object Storage** | MinIO (S3-compatible) |
| **Document Extraction** | Docling Serve |
| **Cache + Memory** | Upstash Redis (REST API) |
| **Backend** | Python 3.13, FastAPI |
| **Observability** | Langfuse |
| **Evaluation** | RAGAS |
| **Frontend** | Next.js (portfolio) with floating AI assistant |
| **Infrastructure** | Docker Compose, Oracle Cloud (23GB RAM), Coolify |

---

## How It Works

### Document Ingestion
1. PDFs uploaded to MinIO (resume, projects, certifications)
2. Worker service triggered via webhook
3. Worker extracts text with Docling (layout-aware, table structure, OCR)
4. Parent-child chunking: 2000 char parents (for LLM context) + 500 char children (for search precision)
5. Chunks embedded with Gemini gemini-embedding-2 (768-dim)
6. Indexed into OpenSearch with parent_content field for citation
7. Duplicate chunks deleted before re-indexing (deduplication)

### Query Processing
1. **Input guardrails** block prompt injection, harmful content, oversized inputs
2. **Portfolio classifier** ensures the question is about my work (keyword + LLM check)
3. **LangGraph agent** analyzes the query and decides which tools to call
4. **Agent tools** (search_all, search_projects, search_skills, search_source, list_documents) execute OpenSearch hybrid search with Cohere reranking
5. **Query expansion** improves BM25 recall with synonym dictionary
6. **Agent analyzes results** — if more info needed, retrieves again (max 2 rounds)
7. **LLM generation** produces a grounded answer with conversation context
8. **Output guardrails** filter toxicity before returning the response
9. **LLM cache** stores response for 1hr (same question returns instantly)

### Deployment
- **Infrastructure**: Oracle Cloud VM (23GB RAM, 4GB swap)
- **Orchestration**: Coolify with Docker Compose
- **Services**: rag API, worker, OpenSearch, MinIO, Docling, OpenSearch Dashboards
- **Auto-deploy**: Push to GitHub -> Coolify auto-builds and deploys

---

## API Endpoints

All endpoints except `/` require `X-API-Key` header when `API_KEY` env var is set.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check (OpenSearch, MinIO, Redis status) |
| `POST` | `/query` | Ask a question (body: `{question, source_filter?}`) |
| `POST` | `/upload?folder=` | Upload a document (max 20MB default) |
| `POST` | `/upload-url?folder=` | Download from URL, store in MinIO |
| `GET` | `/files?folder=` | List stored documents |
| `DELETE` | `/files/{folder}/{filename}` | Delete document + reindex |
| `POST` | `/reindex` | Full reindex from MinIO |
| `POST` | `/clear-memory` | Clear conversation history |
| `POST` | `/clear-cache` | Clear LLM response cache |

---

## Performance

| Metric | Value |
|---|---|
| Embedding Dimension | 768 |
| Chunk Sizes | Parent: 2000 chars, Child: 500 chars |
| Retrieval | Top 10 -> Top 3 (hybrid + rerank) |
| Agent Rounds | Max 2 retrieval rounds |
| LLM | Groq gpt-oss-120b (primary), Gemini 2.5 Flash (fallback) |
| Rate Limit | 5 requests/min per IP |
| LLM Cache | 1hr TTL (Upstash Redis) |
| Response Time | ~3-5 seconds end-to-end |

---

## Evaluation (RAGAS)

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are retrieved docs focused and relevant? |
| Context Recall | Did we retrieve all needed information? |
| Factual Correctness | Is the answer factually accurate? |

---

## Getting Started

### Local Development

```bash
# Start infrastructure (OpenSearch, MinIO, Docling)
docker run -d --name opensearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
  opensearchproject/opensearch:2.19.0

docker run -d --name minio -p 9002:9000 -p 9003:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"

docker run -d --name docling -p 5001:5001 \
  -e "DOCLING_SERVE_ENABLE_UI=true" \
  ghcr.io/docling-project/docling-serve:latest

# Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set environment variables (or use .env)
export OPENSEARCH_HOST=localhost:9200
export MINIO_ENDPOINT=localhost:9002
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
export COHERE_API_KEY=your_key
export UPSTASH_REDIS_REST_URL=your_url
export UPSTASH_REDIS_REST_TOKEN=your_token

# Index documents
python -c "from worker import reindex_all; reindex_all()"

# Start API server
python main.py --api
```

### Deploy (Coolify)

Push to GitHub. Coolify auto-builds and deploys all 6 services.

---

## Project Structure

```
rag/
  api.py                 # FastAPI endpoints + health check
  main.py                # Entry point (CLI / API mode)
  worker.py              # Document processing worker
  docker-compose.yml     # Local development
  docker-compose.coolify.yml  # Production (Coolify)
  src/
    config.py            # Settings, LLM setup, fallback logic
    graph.py             # LangGraph StateGraph + agent tools
    tools.py             # 5 agent tools with caching + query expansion
    retrieval.py         # OpenSearch hybrid search + Cohere rerank
    embeddings.py        # Gemini embedding wrapper
    opensearch_client.py # OpenSearch CRUD + hybrid search
    query_expansion.py   # Synonym-based query expansion
    pipeline.py          # Conversation memory (Redis-backed)
    guardrails.py        # Input/output safety + portfolio classifier
    cache.py             # LLM response cache (Redis-backed)
    storage.py           # MinIO client wrapper
    langfuse_integration.py  # Langfuse tracing
  evals/
    golden_dataset.json  # 10 golden Q&A pairs
    run_evals.py         # RAGAS evaluation script
  pdf/
    generate_hybrid_rag_pdf.py   # Project PDF generator
    generate_techskills_pdf.py   # Technical skills PDF
    generate_projects_pdf.py     # All projects PDF
  requirements.txt       # Python dependencies
  requirements-eval.txt  # Eval-only deps (adds ragas)
```

---

## Why This Project

This system demonstrates end-to-end ML engineering skills:

- **System Design**: Multi-service architecture with proper separation of concerns
- **Agentic RAG**: LangGraph agent with tool routing and multi-step retrieval
- **Search Engineering**: Hybrid retrieval combining keyword and semantic search
- **MLOps**: Automated ingestion, evaluation, and observability
- **Production Mindset**: Guardrails, rate limiting, caching, retry logic, fallbacks
- **Cloud Deployment**: Container orchestration on real infrastructure

---

## Author

**Adeesha Perera** - ML & AI Engineer
- [adeeshaperera.me](https://adeeshaperera.me)
- [GitHub](https://github.com/NullBitZer0)
- [LinkedIn](https://linkedin.com/in/adeesha-perera-b9a614290)
- [X/Twitter](https://x.com/MLwithAdeesha)
