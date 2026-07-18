import os
import tempfile
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minio import Minio
from src.config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET,
    DOCLING_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    FOLDERS,
    ALLOWED_EXTENSIONS,
)
from src.embeddings import embed_texts
from src.semantic_chunker import semantic_chunk_with_parents
from src.opensearch_client import (
    get_opensearch_client,
    ensure_index,
    bulk_index,
    delete_by_source,
    get_doc_count,
)

# Globals
minio_client = None


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


def download_from_minio(filename: str, local_path: str):
    client = get_minio_client()
    client.fget_object(MINIO_BUCKET, filename, local_path)


def extract_with_docling(file_path: str) -> str:
    url = f"{DOCLING_URL}/v1/convert/file"
    with open(file_path, "rb") as f:
        files = {"files": (os.path.basename(file_path), f, "application/pdf")}
        data = {"output_formats": '["markdown"]'}
        response = httpx.post(url, files=files, data=data, timeout=120)
        response.raise_for_status()
    result = response.json()
    document = result.get("document", {})
    return document.get("md_content", "")


def chunk_text(text: str) -> list:
    """Split text into child chunks with parent content using semantic chunking."""
    child_parent_pairs = semantic_chunk_with_parents(
        text,
        embed_fn=embed_texts,
        threshold=0.45,
        parent_max_chars=PARENT_CHUNK_SIZE,
        child_max_chars=CHUNK_SIZE,
    )

    from langchain_core.documents import Document

    child_chunks = []
    for child_text, parent_text in child_parent_pairs:
        doc = Document(page_content=child_text)
        doc.metadata["parent_content"] = parent_text
        child_chunks.append(doc)

    return child_chunks


def index_chunks_to_opensearch(chunks: list, source: str):
    """Embed child chunks with Gemini and index into OpenSearch with parent content."""
    client = get_opensearch_client()
    ensure_index(client)

    texts = [c.page_content for c in chunks]

    all_vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)

    documents = []
    for i, chunk in enumerate(chunks):
        doc_id = hashlib.md5(f"{source}_{i}_{chunk.page_content[:50]}".encode()).hexdigest()
        documents.append(
            {
                "id": doc_id,
                "content": chunk.page_content,
                "vector": all_vectors[i],
                "source": source,
                "page": chunk.metadata.get("page", 0),
                "chunk_id": i,
                "parent_content": chunk.metadata.get("parent_content", chunk.page_content),
            }
        )

    bulk_index(client, documents)


def reindex_all():
    client = get_minio_client()
    objects = client.list_objects(MINIO_BUCKET, recursive=True)

    from src.opensearch_client import delete_index

    os_client = get_opensearch_client()
    delete_index(os_client)
    ensure_index(os_client)

    all_count = 0
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
                    index_chunks_to_opensearch(chunks, obj.object_name)
                    all_count += len(chunks)
            finally:
                os.unlink(tmp.name)

    print(f"Reindexed {all_count} chunks into OpenSearch")


def process_file(filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        try:
            download_from_minio(filename, tmp.name)
            text = extract_with_docling(tmp.name)
            if text:
                chunks = chunk_text(text)
                for chunk in chunks:
                    chunk.metadata["source"] = filename

                # Deduplicate: delete existing chunks for this file before re-indexing
                client = get_opensearch_client()
                delete_by_source(client, filename)

                index_chunks_to_opensearch(chunks, filename)
                print(f"Processed {filename}: {len(chunks)} chunks")
        finally:
            os.unlink(tmp.name)


def delete_file_index(filename: str):
    client = get_opensearch_client()
    delete_by_source(client, filename)
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


class DeleteRequest(BaseModel):
    filename: str


@app.post("/webhook/minio")
async def minio_webhook(event: WebhookEvent):
    try:
        filename = event.Key
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

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


@app.post("/delete")
async def delete_index(req: DeleteRequest):
    """Delete OpenSearch chunks for a specific file."""
    try:
        delete_file_index(req.filename)
        return {"status": "deleted", "filename": req.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex():
    try:
        reindex_all()
        count = get_doc_count()
        return {"status": "reindexed", "chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
