from opensearchpy import OpenSearch, RequestsHttpConnection
from src.config import OPENSEARCH_HOST, OPENSEARCH_INDEX, EMBEDDING_DIM


def get_opensearch_client() -> OpenSearch:
    host, port = OPENSEARCH_HOST.split(":")
    return OpenSearch(
        hosts=[{"host": host, "port": int(port)}],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )


INDEX_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True,
            "knn.algo_param.ef_search": 128,
        },
    },
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "vector": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16,
                    },
                },
            },
            "source": {"type": "keyword"},
            "page": {"type": "integer"},
            "chunk_id": {"type": "integer"},
            "parent_content": {"type": "text", "analyzer": "standard"},
        }
    },
}


def ensure_index(client: OpenSearch = None):
    """Create the OpenSearch index if it doesn't exist."""
    if client is None:
        client = get_opensearch_client()
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        client.indices.create(index=OPENSEARCH_INDEX, body=INDEX_MAPPING)
        print(f"Created OpenSearch index: {OPENSEARCH_INDEX}")
    else:
        print(f"OpenSearch index exists: {OPENSEARCH_INDEX}")


def delete_index(client: OpenSearch = None):
    """Delete the OpenSearch index."""
    if client is None:
        client = get_opensearch_client()
    if client.indices.exists(index=OPENSEARCH_INDEX):
        client.indices.delete(index=OPENSEARCH_INDEX)
        print(f"Deleted OpenSearch index: {OPENSEARCH_INDEX}")


def bulk_index(client: OpenSearch, documents: list[dict]):
    """Bulk index documents into OpenSearch."""
    if not documents:
        return
    actions = []
    for doc in documents:
        actions.append({"index": {"_index": OPENSEARCH_INDEX, "_id": doc["id"]}})
        actions.append({
            "content": doc["content"],
            "vector": doc["vector"],
            "source": doc["source"],
            "page": doc["page"],
            "chunk_id": doc["chunk_id"],
            "parent_content": doc.get("parent_content", doc["content"]),
        })
    response = client.bulk(body=actions, refresh="wait_for")
    errors = response.get("errors", False)
    if errors:
        print(f"Bulk index errors: {response['items'][:3]}")
    else:
        print(f"Indexed {len(documents)} documents into OpenSearch")


def hybrid_search(client: OpenSearch, query_text: str, query_vector: list, k: int = 10) -> list[dict]:
    """Hybrid search: BM25 + k-NN with Reciprocal Rank Fusion."""
    from src.config import BM25_WEIGHT, VECTOR_WEIGHT

    search_body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                                "boost": BM25_WEIGHT,
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "knn": {
                            "vector": {
                                "vector": query_vector,
                                "k": k,
                            }
                        }
                    }
                ],
            }
        },
    }

    response = client.search(index=OPENSEARCH_INDEX, body=search_body)
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "content": hit["_source"].get("parent_content", hit["_source"]["content"]),
            "source": hit["_source"]["source"],
            "page": hit["_source"]["page"],
            "score": hit["_score"],
        })
    return results


def delete_by_source(client: OpenSearch, source: str):
    """Delete all documents with a given source filename."""
    client.delete_by_query(
        index=OPENSEARCH_INDEX,
        body={"query": {"term": {"source": source}}},
        refresh="wait_for",
    )
    print(f"Deleted documents with source: {source}")


def get_doc_count(client: OpenSearch = None) -> int:
    """Get total document count in the index."""
    if client is None:
        client = get_opensearch_client()
    try:
        response = client.count(index=OPENSEARCH_INDEX)
        return response["count"]
    except Exception:
        return 0
