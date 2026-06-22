import os
import hashlib
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE
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
                    docs = PyPDFLoader(tmp.name).load()
                    for doc in docs:
                        doc.metadata["source"] = name
                    documents.extend(docs)
                elif name.endswith('.txt'):
                    docs = TextLoader(tmp.name).load()
                    for doc in docs:
                        doc.metadata["source"] = name
                    documents.extend(docs)
            finally:
                os.unlink(tmp.name)

    print(f"Loaded {len(documents)} documents from MinIO")
    return documents


def split_documents(documents: list) -> list:
    """Split documents into parent chunks, then child chunks within each parent."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks = parent_splitter.split_documents(documents)
    print(f"Split into {len(parent_chunks)} parent chunks")

    child_chunks = []
    for parent in parent_chunks:
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_content"] = parent.page_content
        child_chunks.extend(children)

    print(f"Split into {len(child_chunks)} child chunks")
    return child_chunks


def index_to_opensearch(chunks: list) -> int:
    """Index child chunks into OpenSearch with Gemini embeddings and parent content."""
    client = get_opensearch_client()
    ensure_index(client)

    texts = [c.page_content for c in chunks]

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
            "parent_content": chunk.metadata.get("parent_content", chunk.page_content),
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
