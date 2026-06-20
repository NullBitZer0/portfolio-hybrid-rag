import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


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


def build_vectorstore(chunks: list) -> Chroma:
    """Create ChromaDB vector store from chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hybrid_rag",
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Vector store created with {len(chunks)} chunks")
    return vectorstore


def build_bm25_retriever(chunks: list, k: int = 5) -> BM25Retriever:
    """Create BM25 keyword retriever from chunks."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    return bm25


def ingest():
    """Full ingestion pipeline."""
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in MinIO")
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    bm25_retriever = build_bm25_retriever(chunks)
    return vectorstore, bm25_retriever, chunks
