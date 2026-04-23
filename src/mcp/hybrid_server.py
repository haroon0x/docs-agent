"""
MCP server with hybrid search: dense vector + BM25 sparse vector + RRF fusion.
Searches the unified docs_rag collection covering docs, issues, and code.
"""
import os
import sys
from pathlib import Path

from fastmcp import FastMCP
from pymilvus import AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.milvus.collection import get_client, COLLECTION_NAME, EMBEDDING_MODEL

mcp = FastMCP("Kubeflow Docs Hybrid Search")

_model: SentenceTransformer | None = None
_milvus_client = None


def _init():
    global _model, _milvus_client
    if _model is None:
        print(f"[MCP] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[MCP] Embedding model loaded")
    if _milvus_client is None:
        _milvus_client = get_client()
        print(f"[MCP] Milvus client connected to {COLLECTION_NAME}")


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5, source_filter: str = "all") -> str:
    """
    Search Kubeflow docs, issues, and code using hybrid search.

    Hybrid search combines:
      - Dense vector search (all-mpnet-base-v2): semantic similarity
      - Sparse / BM25 search: exact keyword matching
      - RRF (Reciprocal Rank Fusion): merges rankings from both searches

    Args:
        query: Search query — can be natural language or a K8s identifier.
        top_k: Number of results to return (default 5, max 20).
        source_filter: Filter by source type: "all" | "docs" | "issues" | "code".

    Returns:
        Formatted search results with source type, citation URL, and content.
    """
    _init()

    dense_vec = _model.encode(query).tolist()

    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=top_k * 2,
    )

    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=top_k * 2,
    )

    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "reqs": [dense_req, sparse_req],
        "ranker": RRFRanker(k=60),
        "limit": top_k,
        "output_fields": [
            "content_text",
            "citation_url",
            "file_path",
            "source_type",
            "file_name",
        ],
    }

    if source_filter != "all":
        filter_expr = f'source_type == "{source_filter}"'
        search_kwargs["filter"] = filter_expr

    hits = _milvus_client.hybrid_search(**search_kwargs)

    if not hits or not hits[0]:
        return "No results found for your query."

    results = []
    for i, hit in enumerate(hits[0], 1):
        entity = hit.get("entity", {})
        content = entity.get("content_text", "")
        if len(content) > 400:
            content = content[:400] + "..."

        entry = f"### Result {i} (distance: {hit.get('distance', 0):.4f})"
        entry += f"\n**Source:** {entity.get('source_type', 'unknown').upper()}"
        entry += f"\n**File:** {entity.get('file_path', '')}"
        entry += f"\n**URL:** {entity.get('citation_url', '')}"
        entry += f"\n\n{content}\n"
        results.append(entry)

    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
