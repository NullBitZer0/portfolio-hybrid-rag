import os
import time
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.retrieval import get_reranked_retriever
from src.pipeline import build_conversational_rag_chain, ConversationMemory
from src.storage import storage

app = FastAPI(title="Hybrid RAG API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    flags: list[str] = []


@app.on_event("startup")
async def startup():
    """Initialize RAG system on server start."""
    print("Initializing Hybrid RAG system...")
    try:
        from src.ingest import ingest
        vectorstore, bm25_retriever, chunks = ingest()
        retriever = get_reranked_retriever(vectorstore, bm25_retriever)
        memory = ConversationMemory()
        chain = build_conversational_rag_chain(retriever, memory)
        rag_app["chain"] = chain
        rag_app["memory"] = memory
        rag_app["retriever"] = retriever
        print(f"Ready! {len(chunks)} chunks loaded.")
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
    sources = [doc.metadata.get("source", "") for doc in rag_app["retriever"].invoke(req.question)]
    return QueryResponse(
        answer=result["answer"],
        confidence=result.get("confidence", 0.0),
        mode=result.get("mode", "unknown"),
        sources=list(set(sources)),
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
    allowed_extensions = {".pdf", ".txt", ".docx", ".md"}
    allowed_folders = {"resume", "in_progress_projects", "completed_projects", "uni_projects"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"File type {file_ext} not allowed. Use: {allowed_extensions}")
    if folder not in allowed_folders:
        raise HTTPException(400, f"Folder {folder} not allowed. Use: {allowed_folders}")

    try:
        content = await file.read()
        object_name = f"{folder}/{file.filename}"
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

    allowed_extensions = {".pdf", ".txt", ".docx", ".md"}
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"File type {file_ext} not allowed. Use: {allowed_extensions}")

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
    allowed_folders = {"resume", "in_progress_projects", "completed_projects", "uni_projects"}
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


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the chat UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid RAG Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
.header{padding:16px 24px;background:#1e293b;border-bottom:1px solid #334155;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:18px;font-weight:600}
.header span{font-size:12px;color:#94a3b8}
.chat-container{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.message{max-width:70%;padding:12px 16px;border-radius:12px;line-height:1.5;font-size:14px}
.user{align-self:flex-end;background:#3b82f6;color:#fff;border-bottom-right-radius:4px}
.assistant{align-self:flex-start;background:#1e293b;border:1px solid #334155;border-bottom-left-radius:4px}
.input-area{padding:16px 24px;background:#1e293b;border-top:1px solid #334155;display:flex;gap:12px}
input[type=text]{flex:1;padding:12px 16px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;font-size:14px;outline:none}
input[type=text]:focus{border-color:#3b82f6}
input[type=file]{display:none}
button{padding:12px 20px;border-radius:8px;border:none;background:#3b82f6;color:#fff;font-weight:600;cursor:pointer;font-size:14px}
button:hover{background:#2563eb}
button:disabled{background:#475569;cursor:not-allowed}
.toolbar{padding:8px 24px;background:#0f172a;display:flex;gap:8px;border-bottom:1px solid #1e293b}
.toolbar button{font-size:12px;padding:6px 12px;background:#1e293b;border:1px solid #334155}
.status{font-size:12px;color:#64748b;padding:4px 8px}
.upload-btn{position:relative}
.upload-btn input[type=file]{position:absolute;opacity:0;width:100%;height:100%;cursor:pointer}
</style>
</head>
<body>
<div class="header">
<h1>Hybrid RAG</h1><span>BM25 + Vector + Cross-Encoder Re-ranking</span>
</div>
<div class="toolbar">
<button onclick="clearMemory()">Clear Memory</button>
<button onclick="reindex()">Reindex</button>
<select id="folderSelect" style="padding:6px 12px;border-radius:8px;background:#1e293b;border:1px solid #334155;color:#e2e8f0;font-size:12px">
<option value="resume">Resume</option>
<option value="in_progress_projects">In Progress Projects</option>
<option value="completed_projects">Completed Projects</option>
<option value="uni_projects">Uni Projects</option>
</select>
<div class="upload-btn"><button>Upload File</button><input type="file" id="fileInput" accept=".pdf,.txt,.docx,.md" onchange="uploadFile(this)"></div>
<span class="status" id="status"></span>
</div>
<div class="chat-container" id="chat"></div>
<div class="input-area">
<input type="text" id="input" placeholder="Ask a question..." onkeydown="if(event.key==='Enter')send()">
<button id="sendBtn" onclick="send()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),input=document.getElementById('input'),sendBtn=document.getElementById('sendBtn'),status=document.getElementById('status');
function addMsg(text,role){const d=document.createElement('div');d.className='message '+role;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
async function send(){const q=input.value.trim();if(!q)return;input.value='';addMsg(q,'user');sendBtn.disabled=true;status.textContent='Thinking...';try{const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const d=await r.json();addMsg(d.answer,'assistant');}catch(e){addMsg('Error: '+e.message,'assistant');}sendBtn.disabled=false;status.textContent='';}
async function clearMemory(){await fetch('/clear-memory',{method:'POST'});status.textContent='Memory cleared';}
async function reindex(){status.textContent='Reindexing...';const r=await fetch('/reindex',{method:'POST'});const d=await r.json();status.textContent='Done: '+d.chunks+' chunks';}
async function uploadFile(input){const file=input.files[0];if(!file)return;const folder=document.getElementById('folderSelect').value;status.textContent='Uploading '+file.name+' to '+folder+'...';const fd=new FormData();fd.append('file',file);try{const r=await fetch('/upload?folder='+folder,{method:'POST',body:fd});const d=await r.json();status.textContent=d.filename+' uploaded to '+folder+'. Indexing by worker...';setTimeout(()=>status.textContent='',3000);}catch(e){status.textContent='Upload failed: '+e.message;}input.value='';}
addMsg('Hello! Ask me anything about your documents.','assistant');
</script>
</body></html>"""
