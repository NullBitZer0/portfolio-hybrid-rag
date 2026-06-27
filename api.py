import os
import time
import logging
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import FOLDERS, ALLOWED_EXTENSIONS, API_KEY, MAX_UPLOAD_SIZE_MB
from src.pipeline import build_conversational_rag_chain, ConversationMemory
from src.storage import storage

# ── Logging ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag-api")

# ── App ────────────────────────────────────────────────────

app = FastAPI(title="Agentic RAG API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
rag_app = {"chain": None, "memory": None, "retriever": None, "cache": None}

# Rate limiting
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "5"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))
rate_limit_store = defaultdict(list)

# ── Auth ───────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    """Verify API key if configured. Skip if no API_KEY set."""
    if not API_KEY:
        return
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < RATE_WINDOW]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        return False
    rate_limit_store[client_ip].append(now)
    return True


# ── Models ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    source_filter: str = None


class QueryResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    mode: str = "unknown"
    sources: list[str] = []
    pages: list[str] = []
    flags: list[str] = []


# ── Startup ────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    """Initialize RAG system on server start."""
    logger.info("Initializing Agentic RAG system...")
    try:
        from src.opensearch_client import get_opensearch_client, ensure_index, get_doc_count

        client = get_opensearch_client()
        ensure_index(client)
        doc_count = get_doc_count(client)
        from src.graph import rag_graph
        from src.cache import llm_cache

        memory = ConversationMemory()
        chain = build_conversational_rag_chain(rag_graph, memory)
        rag_app["chain"] = chain
        rag_app["memory"] = memory
        rag_app["retriever"] = rag_graph
        rag_app["cache"] = llm_cache
        logger.info(f"Ready! {doc_count} chunks in OpenSearch. Graph compiled.")
    except Exception as e:
        logger.warning(f"Startup warning: {e}. Waiting for worker to index documents.")


@app.on_event("shutdown")
async def shutdown():
    """Flush Langfuse traces on shutdown."""
    from src.config import LANGFUSE_ENABLED

    if LANGFUSE_ENABLED:
        from langfuse import get_client

        get_client().flush()
        logger.info("Langfuse traces flushed.")


# ── Endpoints ──────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest, request: Request, _key: str = Depends(verify_api_key)):
    """Ask a question to the RAG system."""
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(429, f"Rate limit exceeded. Max {RATE_LIMIT} requests per {RATE_WINDOW}s.")

    if not rag_app["chain"]:
        raise HTTPException(503, "RAG system not initialized. Wait for worker to index documents.")

    # Check LLM cache
    cache = rag_app.get("cache")
    if cache:
        cached = cache.get(req.question)
        if cached:
            logger.info(f"Cache hit: {req.question[:50]}...")
            return QueryResponse(**cached)

    result = rag_app["chain"](req.question, source_filter=req.source_filter)

    response_data = {
        "answer": result["answer"],
        "confidence": result.get("confidence", 0.0),
        "mode": result.get("mode", "unknown"),
        "sources": result.get("sources", []),
        "pages": result.get("pages", []),
        "flags": result.get("flags", []),
    }

    # Store in cache
    if cache:
        cache.set(req.question, response_data)

    return QueryResponse(**response_data)


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    folder: str = "resume",
    _key: str = Depends(verify_api_key),
):
    """Upload a file to MinIO. Worker will auto-index."""
    allowed_folders = set(FOLDERS)
    if folder not in allowed_folders:
        raise HTTPException(400, f"Folder '{folder}' not allowed. Use: {sorted(allowed_folders - {''})}")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed. Use: {ALLOWED_EXTENSIONS}")

    # Check file size
    content = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(400, f"File too large. Max {MAX_UPLOAD_SIZE_MB}MB.")

    try:
        object_name = f"{folder}/{file.filename}" if folder else file.filename
        storage.upload_bytes(content, object_name, content_type=file.content_type or "application/octet-stream")
        logger.info(f"Uploaded {object_name} ({len(content)} bytes)")
        return {
            "filename": file.filename,
            "folder": folder,
            "status": "uploaded",
            "object_name": object_name,
            "message": "Worker will auto-index this file",
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.post("/upload-url")
async def upload_from_url(url: str, filename: str, _key: str = Depends(verify_api_key)):
    """Download from URL, store in MinIO. Worker will auto-index."""
    import httpx

    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed. Use: {ALLOWED_EXTENSIONS}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30)
            response.raise_for_status()

        if len(response.content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, f"File too large. Max {MAX_UPLOAD_SIZE_MB}MB.")

        object_name = f"uploads/{filename}"
        storage.upload_bytes(response.content, object_name, content_type="application/octet-stream")
        logger.info(f"Uploaded from URL: {object_name}")
        return {
            "filename": filename,
            "status": "uploaded",
            "object_name": object_name,
            "message": "Worker will auto-index this file",
        }
    except httpx.HTTPError as e:
        raise HTTPException(500, f"Download failed: {str(e)}")


@app.get("/files")
async def list_files(folder: str = None, _key: str = Depends(verify_api_key)):
    """List all files in MinIO storage."""
    try:
        if folder:
            files = storage.list_files(prefix=f"{folder}/")
        else:
            files = storage.list_files()
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/files/{folder}/{filename}")
async def delete_file(folder: str, filename: str, _key: str = Depends(verify_api_key)):
    """Delete a file from MinIO and remove its OpenSearch index."""
    allowed_folders = set(FOLDERS) - {""}
    if folder not in allowed_folders:
        raise HTTPException(400, f"Folder {folder} not allowed")

    object_name = f"{folder}/{filename}"
    if not storage.file_exists(object_name):
        raise HTTPException(404, f"File {filename} not found in {folder}")

    try:
        storage.delete_file(object_name)
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                await client.post("http://worker:9000/delete", json={"filename": object_name}, timeout=10)
        except Exception as e:
            logger.warning(f"Could not delete index via worker: {e}")

        logger.info(f"Deleted {object_name}")
        return {"filename": filename, "folder": folder, "status": "deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reindex")
async def reindex(_key: str = Depends(verify_api_key)):
    """Trigger full reindex on worker."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://worker:9000/reindex", timeout=60)
            return response.json()
    except Exception as e:
        raise HTTPException(500, f"Reindex failed: {str(e)}")


@app.post("/clear-memory")
async def clear_memory(_key: str = Depends(verify_api_key)):
    """Clear conversation history."""
    if rag_app["memory"]:
        rag_app["memory"].clear()
    return {"status": "memory cleared"}


@app.post("/clear-cache")
async def clear_cache(_key: str = Depends(verify_api_key)):
    """Clear LLM response cache."""
    if rag_app["cache"]:
        rag_app["cache"].clear()
    return {"status": "cache cleared"}


@app.get("/")
async def health_check():
    """Health check with dependency status (no auth required)."""
    status = {"status": "ok", "service": "Agentic RAG API", "version": "3.0"}
    deps = {}

    try:
        from src.opensearch_client import get_opensearch_client, get_doc_count

        client = get_opensearch_client()
        deps["opensearch"] = {"status": "ok", "chunks": get_doc_count(client)}
    except Exception as e:
        deps["opensearch"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    try:
        files = storage.list_files()
        deps["minio"] = {"status": "ok", "files": len(files)}
    except Exception as e:
        deps["minio"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    try:
        from src.config import UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN

        if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
            from upstash_redis import Redis

            r = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
            r.ping()
            deps["redis"] = {"status": "ok"}
        else:
            deps["redis"] = {"status": "not_configured"}
    except Exception as e:
        deps["redis"] = {"status": "error", "error": str(e)}

    status["dependencies"] = deps
    return status
