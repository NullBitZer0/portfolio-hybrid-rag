import os
import hashlib
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.embeddings import embed_texts
from src.opensearch_client import get_opensearch_client, ensure_index, bulk_index, get_doc_count


def load_documents() -> list:
    """Load all documents from MinIO."""
    from src.storage import storage

    documents = []
    files = storage.list_files()

    for obj in files:
        name = obj['name']
        if not name.endswith(('.pdf', '.txt', '.docx')):
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
            try:
                storage.download_file(name, tmp.name)
                if name.endswith('.pdf'):
                    documents.extend(PyPDFLoader(tmp.name).load())
                elif name.endswith('.txt'):
                    documents.extend(TextLoader(tmp.name).load())
            finally:
                os.unlink(tmp.name)

    print(f"Loaded {len(documents)} documents from MinIO")
    return documents


def split_documents(documents: list) -> list:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def index_to_opensearch(chunks: list) -> int:
    """Index chunks into OpenSearch with Gemini embeddings."""
    client = get_opensearch_client()
    ensure_index(client)

    texts = [c.page_content for c in chunks]

    # Batch embed (Gemini supports batch)
    all_vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    documents = []
    for i, chunk in enumerate(chunks):
        doc_id = hashlib.md5(f"{chunk.metadata.get('source', '')}_{i}_{chunk.page_content[:50]}".encode()).hexdigest()
        documents.append({
            "id": doc_id,
            "content": chunk.page_content,
            "vector": all_vectors[i],
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", 0),
            "chunk_id": i,
        })

    bulk_index(client, documents)
    return len(documents)


def ingest():
    """Full ingestion pipeline: load from MinIO -> chunk -> index to OpenSearch."""
    client = get_opensearch_client()
    ensure_index(client)

    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in MinIO")

    chunks = split_documents(docs)
    count = index_to_opensearch(chunks)

    return count
