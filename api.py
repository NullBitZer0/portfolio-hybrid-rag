from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.ingest import ingest, load_documents, split_documents, build_vectorstore, build_bm25_retriever
from src.retrieval import get_reranked_retriever
from src.pipeline import build_conversational_rag_chain, ConversationMemory

app = FastAPI(title="Hybrid RAG API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
rag_app = {"chain": None, "memory": None, "retriever": None}


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


@app.on_event("startup")
async def startup():
    """Initialize RAG system on server start."""
    print("Initializing Hybrid RAG system...")
    try:
        vectorstore, bm25_retriever, chunks = ingest("./data")
        retriever = get_reranked_retriever(vectorstore, bm25_retriever)
        memory = ConversationMemory()
        chain = build_conversational_rag_chain(retriever, memory)

        rag_app["chain"] = chain
        rag_app["memory"] = memory
        rag_app["retriever"] = retriever
        print(f"Ready! {len(chunks)} chunks loaded.")
    except ValueError as e:
        print(f"Warning: {e}. Add documents to ./data to start querying.")


@app.post("/query")
async def query_rag(req: QueryRequest):
    """Ask a question to the RAG system."""
    if not rag_app["chain"]:
        raise HTTPException(503, "RAG system not initialized. Add documents first.")
    answer = rag_app["chain"](req.question)
    sources = [doc.metadata.get("source", "") for doc in rag_app["retriever"].invoke(req.question)]
    return QueryResponse(answer=answer, sources=list(set(sources)))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF or TXT file to the knowledge base."""
    import os
    os.makedirs("./data/uploads", exist_ok=True)
    path = f"./data/uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "status": "uploaded"}


@app.post("/reindex")
async def reindex():
    """Rebuild the vector store after adding new documents."""
    try:
        vectorstore, bm25_retriever, chunks = ingest("./data")
        retriever = get_reranked_retriever(vectorstore, bm25_retriever)
        memory = ConversationMemory()
        chain = build_conversational_rag_chain(retriever, memory)

        rag_app["chain"] = chain
        rag_app["memory"] = memory
        rag_app["retriever"] = retriever
        return {"status": "reindexed", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))


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
input{flex:1;padding:12px 16px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;font-size:14px;outline:none}
input:focus{border-color:#3b82f6}
button{padding:12px 20px;border-radius:8px;border:none;background:#3b82f6;color:#fff;font-weight:600;cursor:pointer;font-size:14px}
button:hover{background:#2563eb}
button:disabled{background:#475569;cursor:not-allowed}
.toolbar{padding:8px 24px;background:#0f172a;display:flex;gap:8px;border-bottom:1px solid #1e293b}
.toolbar button{font-size:12px;padding:6px 12px;background:#1e293b;border:1px solid #334155}
.status{font-size:12px;color:#64748b;padding:4px 8px}
</style>
</head>
<body>
<div class="header">
<h1>Hybrid RAG</h1><span>BM25 + Vector + Cross-Encoder Re-ranking</span>
</div>
<div class="toolbar">
<button onclick="clearMemory()">Clear Memory</button>
<button onclick="reindex()">Reindex</button>
<span class="status" id="status"></span>
</div>
<div class="chat-container" id="chat"></div>
<div class="input-area">
<input id="input" placeholder="Ask a question..." onkeydown="if(event.key==='Enter')send()">
<button id="sendBtn" onclick="send()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),input=document.getElementById('input'),sendBtn=document.getElementById('sendBtn'),status=document.getElementById('status');
function addMsg(text,role){const d=document.createElement('div');d.className='message '+role;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
async function send(){const q=input.value.trim();if(!q)return;input.value='';addMsg(q,'user');sendBtn.disabled=true;status.textContent='Thinking...';try{const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const d=await r.json();addMsg(d.answer,'assistant');}catch(e){addMsg('Error: '+e.message,'assistant');}sendBtn.disabled=false;status.textContent='';}
async function clearMemory(){await fetch('/clear-memory',{method:'POST'});status.textContent='Memory cleared';}
async function reindex(){status.textContent='Reindexing...';const r=await fetch('/reindex',{method:'POST'});const d=await r.json();status.textContent='Done: '+d.chunks+' chunks';}
addMsg('Hello! Ask me anything about your documents.','assistant');
</script>
</body></html>"""
