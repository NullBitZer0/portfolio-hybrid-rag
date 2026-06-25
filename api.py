import os
import time
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import FOLDERS, ALLOWED_EXTENSIONS
from src.retrieval import get_reranked_retriever
from src.pipeline import build_conversational_rag_chain, ConversationMemory
from src.storage import storage

app = FastAPI(title="Hybrid RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
rag_app = {"chain": None, "memory": None, "retriever": None}

# Rate limiting
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "5"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))
rate_limit_store = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if now - t < RATE_WINDOW
    ]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        return False
    rate_limit_store[client_ip].append(now)
    return True


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    mode: str = "unknown"
    sources: list[str] = []
    pages: list[str] = []
    flags: list[str] = []


@app.on_event("startup")
async def startup():
    """Initialize RAG system on server start."""
    print("Initializing Hybrid RAG system...")
    try:
        from src.ingest import ingest
        total_chunks = ingest()
        retriever = get_reranked_retriever()
        memory = ConversationMemory()
        chain = build_conversational_rag_chain(retriever, memory)
        rag_app["chain"] = chain
        rag_app["memory"] = memory
        rag_app["retriever"] = retriever
        print(f"Ready! {total_chunks} chunks loaded from OpenSearch.")
    except Exception as e:
        print(f"Warning: {e}. Waiting for worker to index documents.")


@app.post("/query")
async def query_rag(req: QueryRequest, request: Request):
    """Ask a question to the RAG system."""
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(429, f"Rate limit exceeded. Max {RATE_LIMIT} requests per {RATE_WINDOW}s.")

    if not rag_app["chain"]:
        raise HTTPException(503, "RAG system not initialized. Wait for worker to index documents.")
    result = rag_app["chain"](req.question)
    return QueryResponse(
        answer=result["answer"],
        confidence=result.get("confidence", 0.0),
        mode=result.get("mode", "unknown"),
        sources=result.get("sources", []),
        pages=result.get("pages", []),
        flags=result.get("flags", []),
    )


@app.on_event("shutdown")
async def shutdown():
    """Flush Langfuse traces on shutdown."""
    from src.config import LANGFUSE_ENABLED
    if LANGFUSE_ENABLED:
        from langfuse import get_client
        get_client().flush()
        print("Langfuse traces flushed.")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), folder: str = "resume"):
    """Upload a file to MinIO. Worker will auto-index."""
    allowed_folders = set(FOLDERS)
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed. Use: {ALLOWED_EXTENSIONS}")

    try:
        content = await file.read()
        object_name = f"{folder}/{file.filename}" if folder else file.filename
        storage.upload_bytes(content, object_name, content_type=file.content_type or "application/octet-stream")
        return {
            "filename": file.filename,
            "folder": folder,
            "status": "uploaded",
            "object_name": object_name,
            "message": "Worker will auto-index this file",
        }
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.post("/upload-url")
async def upload_from_url(url: str, filename: str):
    """Download from URL, store in MinIO. Worker will auto-index."""
    import httpx

    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed. Use: {ALLOWED_EXTENSIONS}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

        object_name = f"uploads/{filename}"
        storage.upload_bytes(response.content, object_name, content_type="application/octet-stream")
        return {
            "filename": filename,
            "status": "uploaded",
            "object_name": object_name,
            "message": "Worker will auto-index this file",
        }
    except httpx.HTTPError as e:
        raise HTTPException(500, f"Download failed: {str(e)}")


@app.get("/files")
async def list_files(folder: str = None):
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
async def delete_file(folder: str, filename: str):
    """Delete a file from MinIO. Worker will auto-remove index."""
    allowed_folders = set(FOLDERS) - {""}
    if folder not in allowed_folders:
        raise HTTPException(400, f"Folder {folder} not allowed")

    object_name = f"{folder}/{filename}"

    if not storage.file_exists(object_name):
        raise HTTPException(404, f"File {filename} not found in {folder}")

    try:
        storage.delete_file(object_name)
        return {"filename": filename, "folder": folder, "status": "deleted", "message": "Worker will auto-remove index"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reindex")
async def reindex():
    """Trigger full reindex on worker."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://worker:9000/reindex", timeout=60)
            return response.json()
    except Exception as e:
        raise HTTPException(500, f"Reindex failed: {str(e)}")


@app.post("/clear-memory")
async def clear_memory():
    """Clear conversation history."""
    if rag_app["memory"]:
        rag_app["memory"].clear()
    return {"status": "memory cleared"}


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Hybrid RAG API"}
