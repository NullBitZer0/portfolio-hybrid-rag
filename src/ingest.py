import os
import hashlib
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
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


def get_embeddings_model():
    """Get the HuggingFace embeddings model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def index_to_opensearch(chunks: list) -> int:
    """Index chunks into OpenSearch with dense vectors."""
    client = get_opensearch_client()
    ensure_index(client)

    embeddings = get_embeddings_model()
    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    documents = []
    for i, chunk in enumerate(chunks):
        doc_id = hashlib.md5(f"{chunk.metadata.get('source', '')}_{i}_{chunk.page_content[:50]}".encode()).hexdigest()
        documents.append({
            "id": doc_id,
            "content": chunk.page_content,
            "vector": vectors[i],
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

    return get_doc_count(client)
