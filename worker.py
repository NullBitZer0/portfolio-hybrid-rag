import os
import tempfile
import json
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minio import Minio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")
DOCLING_URL = os.getenv("DOCLING_URL", "http://docling:5001")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Folder categories
FOLDERS = ["resume", "in_progress_projects", "completed_projects", "uni_projects"]

# Globals
minio_client = None
embeddings = None


def get_minio_client():
    global minio_client
    if minio_client is None:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
    return minio_client


def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


def download_from_minio(filename: str, local_path: str):
    client = get_minio_client()
    client.fget_object(MINIO_BUCKET, filename, local_path)


def extract_with_docling(file_path: str) -> str:
    url = f"{DOCLING_URL}/v1/convert/file"
    with open(file_path, "rb") as f:
        files = {"files": (os.path.basename(file_path), f, "application/octet-stream")}
        data = {"output_formats": ["markdown"]}
        response = httpx.post(url, files=files, data=data, timeout=120)
        response.raise_for_status()
    result = response.json()
    document = result.get("document", {})
    return document.get("md_content", "")


def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents([text])


def build_vectorstore(chunks: list) -> Chroma:
    embeddings = get_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hybrid_rag",
        persist_directory=CHROMA_PERSIST_DIR,
    )


def build_bm25(chunks: list) -> BM25Retriever:
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    return bm25


def reindex_all():
    client = get_minio_client()
    objects = client.list_objects(MINIO_BUCKET)

    all_chunks = []
    for obj in objects:
        if not obj.object_name.endswith((".pdf", ".txt", ".docx", ".pptx")):
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(obj.object_name).suffix) as tmp:
            try:
                download_from_minio(obj.object_name, tmp.name)
                text = extract_with_docling(tmp.name)
                if text:
                    chunks = chunk_text(text)
                    for chunk in chunks:
                        chunk.metadata["source"] = obj.object_name
                    all_chunks.extend(chunks)
            finally:
                os.unlink(tmp.name)

    if all_chunks:
        build_vectorstore(all_chunks)
        build_bm25(all_chunks)
        print(f"Reindexed {len(all_chunks)} chunks from {len(objects)} files")
    else:
        print("No documents found to index")


def process_file(filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        try:
            download_from_minio(filename, tmp.name)
            text = extract_with_docling(tmp.name)
            if text:
                chunks = chunk_text(text)
                for chunk in chunks:
                    chunk.metadata["source"] = filename
                build_vectorstore(chunks)
                build_bm25(chunks)
                print(f"Processed {filename}: {len(chunks)} chunks")
        finally:
            os.unlink(tmp.name)


def delete_file_index(filename: str):
    vectorstore = Chroma(
        collection_name="hybrid_rag",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )
    vectorstore.delete(where={"source": filename})
    print(f"Deleted index for {filename}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Worker starting...")
    yield
    print("Worker shutting down...")


app = FastAPI(title="RAG Worker", lifespan=lifespan)


class WebhookEvent(BaseModel):
    EventName: str
    Key: str
    Records: list = []


@app.post("/webhook/minio")
async def minio_webhook(event: WebhookEvent):
    try:
        filename = event.Key
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check if file is in a valid folder
        folder = filename.split("/")[0] if "/" in filename else ""
        if folder not in FOLDERS:
            print(f"Skipping file outside valid folders: {filename}")
            return {"status": "skipped", "reason": "invalid folder"}

        if event.EventName == "s3:ObjectCreated:*":
            process_file(filename)
            return {"status": "processed", "filename": filename}
        elif event.EventName == "s3:ObjectRemoved:*":
            delete_file_index(filename)
            return {"status": "deleted", "filename": filename}
        else:
            return {"status": "ignored", "event": event.EventName}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex():
    try:
        reindex_all()
        return {"status": "reindexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
