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
[Query Transformation] -- direct / rewrite / multi_query / step_back
       |
       v
[Hybrid Retrieval] -- OpenSearch BM25 + k-NN dense vector search
       |
       v
[Cohere Rerank] -- top 10 results -> top 3 most relevant
       |
       v
[LLM Generation] -- Groq llama-3.3-70b with conversation memory
       |
       v
[Output Guardrails] -- toxicity filter
       |
       v
Answer to Recruiter
```

---

## Key Features

| Feature | Implementation |
|---|---|
| **Hybrid Search** | OpenSearch BM25 (keyword) + k-NN (semantic) with RRF fusion |
| **Reranking** | Cohere rerank-v3.5 (top 10 -> top 3) |
| **Embeddings** | Google Gemini gemini-embedding-2 (768-dim) |
| **Document Extraction** | Docling Serve (layout-aware PDF parsing) |
| **Query Strategies** | 4 modes: direct, rewrite, multi_query, step_back |
| **Guardrails** | Input: injection detection, toxicity, length. Output: toxicity filter |
| **Portfolio Classifier** | Blocks general knowledge, only answers about my work |
| **Conversation Memory** | Sliding window of last 5 messages |
| **Caching** | In-memory LRU cache (MD5 keys, 1000 entries) |
| **Observability** | Full pipeline traced with Langfuse |
| **Evaluation** | RAGAS metrics (faithfulness, relevancy, precision, recall) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq llama-3.3-70b-versatile |
| **Embeddings** | Google Gemini gemini-embedding-2 |
| **Reranking** | Cohere rerank-v3.5 |
| **Vector + Search** | OpenSearch 2.19.0 (k-NN + BM25) |
| **Object Storage** | MinIO (S3-compatible) |
| **Document Extraction** | Docling Serve |
| **Backend** | Python 3.13, FastAPI |
| **Orchestration** | LangChain (function chain) |
| **Observability** | Langfuse |
| **Evaluation** | RAGAS |
| **Frontend** | Next.js (portfolio) with floating AI assistant |
| **Infrastructure** | Docker Compose, Oracle Cloud (23GB RAM), Coolify |

---

## How It Works

### Document Ingestion
1. PDFs uploaded to MinIO (resume, projects, certifications)
2. MinIO webhook triggers the worker service
3. Worker extracts text with Docling (layout-aware, table structure, OCR)
4. Text chunked (500 chars, 100 overlap) with LangChain
5. Chunks embedded with Gemini gemini-embedding-2 (768-dim)
6. Indexed into OpenSearch (dense vectors + full-text)

### Query Processing
1. **Input guardrails** block prompt injection, harmful content, oversized inputs
2. **Portfolio classifier** ensures the question is about my work (keyword + LLM check)
3. **Query transformation** selects the best strategy (direct/rewrite/multi_query/step_back)
4. **Hybrid retrieval** combines BM25 keyword search + k-NN semantic search
5. **Cohere reranking** narrows top 10 results to top 3 most relevant
6. **LLM generation** produces a grounded answer with conversation context
7. **Output guardrails** filter toxicity before returning the response

### Deployment
- **Infrastructure**: Oracle Cloud VM (23GB RAM, 4GB swap)
- **Orchestration**: Coolify with Docker Compose
- **Services**: rag API, worker, OpenSearch, MinIO, Docling, OpenSearch Dashboards
- **Auto-deploy**: Push to GitHub -> Coolify auto-builds and deploys

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Ask a question |
| `POST` | `/upload?folder=` | Upload a document |
| `GET` | `/files?folder=` | List stored documents |
| `DELETE` | `/files/{folder}/{filename}` | Delete a document |
| `POST` | `/reindex` | Full reindex from MinIO |
| `POST` | `/clear-memory` | Clear conversation history |

---

## Performance

| Metric | Value |
|---|---|
| Embedding Dimension | 768 |
| Retrieval | Top 10 -> Top 3 (hybrid + rerank) |
| LLM Calls per Query | 2-3 |
| Rate Limit | 5 requests/min per IP |
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
  api.py              # FastAPI endpoints + web UI
  main.py             # Entry point (CLI / API mode)
  worker.py           # Document processing worker
  docker-compose.yml  # Local development
  docker-compose.coolify.yml  # Production (Coolify)
  src/
    config.py         # Settings and environment variables
    embeddings.py     # Gemini embedding wrapper
    opensearch_client.py  # OpenSearch CRUD + hybrid search
    retrieval.py      # HybridRetriever + Cohere rerank
    ingest.py         # Document ingestion pipeline
    graph.py          # Main RAG pipeline function
    pipeline.py       # Conversation memory + chain wrapper
    guardrails.py     # Input/output safety + portfolio classifier
    query_transform.py    # Strategy router (direct/rewrite/multi/step_back)
    cache.py          # In-memory LLM response cache
    storage.py        # MinIO client wrapper
    langfuse_integration.py  # Langfuse tracing
    utils.py          # Shared utilities
  evals/
    golden_dataset.json    # 10 golden Q&A pairs
    run_evals.py          # RAGAS evaluation script
  requirements.txt    # Python dependencies
  requirements-worker.txt  # Worker dependencies (lighter)
  .env.example        # Environment variable template
```

---

## Why This Project

This system demonstrates end-to-end ML engineering skills:

- **System Design**: Multi-service architecture with proper separation of concerns
- **Search Engineering**: Hybrid retrieval combining keyword and semantic search
- **MLOps**: Automated ingestion, evaluation, and observability
- **Production Mindset**: Guardrails, rate limiting, caching, error handling
- **Cloud Deployment**: Container orchestration on real infrastructure

---

## Author

**Adeesha Perera** - ML & AI Engineer
- [adeeshaperera.me](https://adeeshaperera.me)
- [GitHub](https://github.com/NullBitZer0)
- [LinkedIn](https://linkedin.com/in/adeesha-perera-b9a614290)
- [X/Twitter](https://x.com/MLwithAdeesha)
