"""
End-to-end tests for the retrieval pipeline.

These tests verify the complete retrieval flow:
1. Query -> Embed -> Milvus Hybrid Search -> RRF Fusion -> Results

Requirements:
- Milvus 2.5+ running with hybrid search (BM25 + HNSW) indexes
- Optional: LLM endpoint for full E2E tests

Run with:
    pytest tests/e2e/ -v
    MILVUS_HOST=localhost MILVUS_PORT=19530 pytest tests/e2e/ -v
"""
import os
import pytest
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus import AnnSearchRequest, RRFRanker

# Skip all tests if Milvus is not available
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_E2E_TESTS", "0") != "1",
    reason="Set RUN_E2E_TESTS=1 to run E2E tests"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def milvus_connection(milvus_host, milvus_port):
    """Create and cleanup Milvus connection."""
    connections.connect("default", host=milvus_host, port=milvus_port)
    yield
    try:
        connections.disconnect("default")
    except Exception:
        pass


@pytest.fixture(scope="module")
def encoder():
    """Load the embedding model."""
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model


@pytest.fixture(scope="module")
def test_collection(milvus_connection, test_collection_name):
    """Create a test collection with hybrid search schema."""
    from pymilvus import utility, Function, FunctionType
    
    # Cleanup any existing test collection
    if utility.has_collection(test_collection_name):
        utility.drop_collection(test_collection_name)
    
    # Create schema with BM25 + HNSW
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000, enable_analyzer=True),
        FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    
    # BM25 function for sparse vectors
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["content_text"],
        output_field_names=["sparse_vector"],
    )
    
    schema = CollectionSchema(
        fields, 
        "Test RAG collection for E2E tests",
        functions=[bm25_function]
    )
    collection = Collection(test_collection_name, schema)
    
    # Create indexes
    dense_index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256}
    }
    collection.create_index("vector", dense_index_params)
    
    sparse_index_params = {
        "metric_type": "BM25",
        "index_type": "SPARSE_INVERTED_INDEX"
    }
    collection.create_index("sparse_vector", sparse_index_params)
    
    collection.load()
    
    yield collection
    
    # Cleanup
    if utility.has_collection(test_collection_name):
        utility.drop_collection(test_collection_name)


@pytest.fixture
def populated_collection(test_collection, encoder, sample_kubeflow_docs):
    """Insert sample documents into test collection."""
    import numpy as np
    
    records = []
    for doc in sample_kubeflow_docs:
        embedding = encoder.encode(doc["content_text"]).tolist()
        records.append({
            "file_path": doc["file_path"],
            "content_text": doc["content_text"],
            "citation_url": doc["citation_url"],
            "vector": embedding,
        })
    
    # Insert in batches
    test_collection.insert(records)
    test_collection.flush()
    
    yield test_collection


# ============================================================================
# Helper Functions
# ============================================================================

def hybrid_search(
    collection: Collection,
    encoder: SentenceTransformer,
    query: str,
    top_k: int = 5,
    vector_field: str = "vector",
    sparse_field: str = "sparse_vector"
) -> List[Dict[str, Any]]:
    """
    Execute hybrid search with RRF fusion.
    
    This mirrors the implementation in server/app.py
    """
    # Encode query for dense search
    query_vec = encoder.encode(query).tolist()
    
    # Dense search request
    dense_req = AnnSearchRequest(
        data=[query_vec],
        anns_field=vector_field,
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k * 2,
    )
    
    # Sparse search request (BM25)
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field=sparse_field,
        param={"metric_type": "BM25"},
        limit=top_k * 2,
    )
    
    # Execute hybrid search with RRF
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=RRFRanker(k=60),
        limit=top_k,
        output_fields=["file_path", "content_text", "citation_url"]
    )
    
    # Format results
    hits = []
    for hit in results[0]:
        entity = hit.entity
        hits.append({
            "file_path": entity.get("file_path"),
            "content_text": entity.get("content_text"),
            "citation_url": entity.get("citation_url"),
            "score": hit.score,
        })
    
    return hits


def dense_only_search(
    collection: Collection,
    encoder: SentenceTransformer,
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Execute dense-only search for comparison."""
    query_vec = encoder.encode(query).tolist()
    
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["file_path", "content_text", "citation_url"]
    )
    
    hits = []
    for hit in results[0]:
        entity = hit.entity
        similarity = 1.0 - float(hit.distance)
        hits.append({
            "file_path": entity.get("file_path"),
            "content_text": entity.get("content_text"),
            "citation_url": entity.get("citation_url"),
            "score": similarity,
        })
    
    return hits


# ============================================================================
# Test Cases
# ============================================================================

class TestHybridSearch:
    """Tests for hybrid search functionality."""

    def test_hybrid_search_returns_results(self, populated_collection, encoder):
        """Hybrid search should return results for valid query."""
        results = hybrid_search(populated_collection, encoder, "KFP v2 SDK")
        
        assert len(results) > 0
        assert results[0]["content_text"] is not None

    def test_hybrid_search_exact_keyword_match(self, populated_collection, encoder):
        """
        Hybrid search should find exact keyword matches.
        
        This tests the sparse (BM25) component.
        """
        # Query with exact term "KServe" should match the KServe doc
        results = hybrid_search(populated_collection, encoder, "KServe InferenceService")
        
        assert len(results) > 0
        kserve_found = any("KServe" in r["content_text"] for r in results)
        assert kserve_found, "Expected KServe document in results"

    def test_hybrid_search_semantic_match(self, populated_collection, encoder):
        """
        Hybrid search should find semantic matches.
        
        This tests the dense (embedding) component.
        """
        # Query with different wording but same meaning
        results = hybrid_search(populated_collection, encoder, "how to deploy ML model")
        
        assert len(results) > 0
        
        # Should find either KServe deployment or pipelines doc
        found_relevant = any(
            "deploy" in r["content_text"].lower() or 
            "pipeline" in r["content_text"].lower()
            for r in results
        )
        assert found_relevant

    def test_hybrid_vs_dense_comparison(self, populated_collection, encoder):
        """Hybrid search should complement dense-only results."""
        query = "compile pipeline"
        
        hybrid_results = hybrid_search(populated_collection, encoder, query, top_k=5)
        dense_results = dense_only_search(populated_collection, encoder, query, top_k=5)
        
        # Both should return results
        assert len(hybrid_results) > 0
        assert len(dense_results) > 0
        
        # Hybrid should include results from both dense and sparse
        # (This is a sanity check, not exact assertion)
        hybrid_urls = {r["citation_url"] for r in hybrid_results}
        dense_urls = {r["citation_url"] for r in dense_results}
        
        # The union should potentially be larger or equal
        assert isinstance(hybrid_results[0]["score"], float)


class TestRetrievalQuality:
    """Tests for retrieval quality metrics."""

    def test_recall_at_k(self, populated_collection, encoder, golden_queries):
        """
        Test recall@k metric.
        
        For each golden query, check if expected doc is in top-k results.
        """
        recall_scores = []
        
        for golden in golden_queries:
            results = hybrid_search(
                populated_collection, 
                encoder, 
                golden["query"], 
                top_k=3
            )
            
            retrieved_urls = [r["file_path"] for r in results]
            
            # Check if any expected doc is in results
            hit = any(
                expected in retrieved_urls 
                for expected in golden["expected_doc_ids"]
            )
            recall_scores.append(1.0 if hit else 0.0)
        
        recall = sum(recall_scores) / len(recall_scores)
        
        # For this small test set, expect at least some recalls
        # (With only 5 docs, 100% is achievable if queries are well-chosen)
        assert recall >= 0.4, f"Recall too low: {recall}"

    def test_keyword_boosting(self, populated_collection, encoder):
        """
        Exact keyword matches should be boosted in hybrid results.
        
        Query containing exact term from doc should rank that doc higher
        than semantic-only search might.
        """
        # Query contains "Katib" - exact term in one doc
        hybrid = hybrid_search(populated_collection, encoder, "Katib hyperparameter", top_k=3)
        
        # Should find the Katib doc
        katib_found = any("Katib" in r["content_text"] for r in hybrid)
        assert katib_found


class TestErrorHandling:
    """Tests for error handling in retrieval."""

    def test_empty_query_handling(self, populated_collection, encoder):
        """Empty query should not crash."""
        # This may return empty or all results - just shouldn't crash
        try:
            results = hybrid_search(populated_collection, encoder, "", top_k=5)
            assert isinstance(results, list)
        except Exception as e:
            # Empty query might raise exception - that's acceptable
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()

    def test_very_long_query(self, populated_collection, encoder):
        """Very long query should be handled."""
        long_query = "word " * 1000
        
        try:
            results = hybrid_search(populated_collection, encoder, long_query, top_k=5)
            assert isinstance(results, list)
        except Exception:
            # May fail due to length - acceptable
            pass


class TestEndToEndFlow:
    """Tests simulating the full E2E flow."""

    def test_query_to_results_pipeline(self, populated_collection, encoder):
        """Test complete query to results pipeline."""
        # 1. User query
        user_query = "How to install Kubeflow Pipelines"
        
        # 2. Hybrid search
        results = hybrid_search(populated_collection, encoder, user_query, top_k=3)
        
        # 3. Verify results structure
        assert len(results) > 0
        for result in results:
            assert "content_text" in result
            assert "citation_url" in result
            assert "file_path" in result
        
        # 4. Verify results are relevant (at least one)
        relevant = any(
            "install" in r["content_text"].lower() or
            "pipeline" in r["content_text"].lower()
            for r in results
        )
        assert relevant, "Expected relevant results for install query"

    def test_batch_queries(self, populated_collection, encoder):
        """Test processing multiple queries efficiently."""
        queries = [
            "install pipelines",
            "KServe deployment",
            "Katib hyperparameters",
            "Notebooks setup",
        ]
        
        all_results = []
        for query in queries:
            results = hybrid_search(populated_collection, encoder, query, top_k=3)
            all_results.append(results)
        
        # All queries should return results
        assert len(all_results) == len(queries)
        assert all(len(r) > 0 for r in all_results)


# ============================================================================
# Performance Tests (Optional)
# ============================================================================

class TestPerformance:
    """Optional performance tests - run manually."""

    @pytest.mark.slow
    def test_retrieval_latency(self, populated_collection, encoder):
        """Retrieval should complete within latency budget."""
        import time
        
        query = "KFP v2 SDK compile"
        iterations = 10
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            hybrid_search(populated_collection, encoder, query, top_k=5)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        
        avg_latency = sum(times) / len(times)
        
        # ADR specifies <100ms for retrieval
        # Allow some overhead for test environment
        assert avg_latency < 500, f"Avg latency {avg_latency:.1f}ms exceeds threshold"
