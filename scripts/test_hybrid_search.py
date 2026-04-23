"""
End-to-end hybrid search benchmark with eval results saved to eval/ directory.

Usage:
    MILVUS_HOST=127.0.0.1 PYTHONPATH=. uv run python scripts/test_hybrid_search.py

Three phases:
    Eval 1: code-only baseline (this file, run first)
    Eval 2: after docs ingestion (same script after ingesting docs)
    Eval 3: LLM-augmented RAG (after adding Ollama config)
"""
import json
import time
import sys
from pathlib import Path

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.milvus.collection import COLLECTION_NAME, EMBEDDING_MODEL

COLLECTION = COLLECTION_NAME
MODEL = EMBEDDING_MODEL
EVAL_DIR = Path(__file__).parent.parent / "eval"
OUTPUT_FIELDS = ["content_text", "file_path", "source_type", "file_name"]

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[BENCH] Loading model: {MODEL}")
        _model = SentenceTransformer(MODEL, device="cpu")
        print("[BENCH] Model loaded")
    return _model


def search_dense(client, query_vec, top_k=5, partition_filter=None):
    req = AnnSearchRequest(
        data=[query_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=top_k,
    )
    kwargs = dict(
        collection_name=COLLECTION,
        reqs=[req],
        ranker=RRFRanker(k=1),
        limit=top_k,
        output_fields=OUTPUT_FIELDS,
    )
    if partition_filter:
        kwargs["filter"] = partition_filter
    return client.hybrid_search(**kwargs)[0]


def search_sparse(client, query_text, top_k=5, partition_filter=None):
    req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=top_k,
    )
    kwargs = dict(
        collection_name=COLLECTION,
        reqs=[req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=OUTPUT_FIELDS,
    )
    if partition_filter:
        kwargs["filter"] = partition_filter
    return client.hybrid_search(**kwargs)[0]


def search_hybrid(client, query_vec, query_text, top_k=5, partition_filter=None):
    d_req = AnnSearchRequest(
        data=[query_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=top_k * 2,
    )
    s_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=top_k * 2,
    )
    kwargs = dict(
        collection_name=COLLECTION,
        reqs=[d_req, s_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=OUTPUT_FIELDS,
    )
    if partition_filter:
        kwargs["filter"] = partition_filter
    return client.hybrid_search(**kwargs)[0]


def run_query(client, model, query_text, partition_filter=None):
    vec = model.encode(query_text).tolist()
    results = {}

    for req_type, search_fn, args in [
        ("dense", search_dense, (vec,)),
        ("sparse", search_sparse, (query_text,)),
        ("hybrid", search_hybrid, (vec, query_text)),
    ]:
        start = time.perf_counter()
        hits = search_fn(client, *args, partition_filter=partition_filter)
        latency_ms = (time.perf_counter() - start) * 1000
        results[req_type] = {
            "hits": [
                {"file": h["entity"]["file_name"], "source": h["entity"]["source_type"], "distance": h["distance"]}
                for h in hits
            ],
            "latency_ms": round(latency_ms, 2),
        }
    return results


def _fmt_hit(h):
    e = h.get("entity", {})
    return f"  [{h['distance']:.4f}] {e.get('file_name', '?')} | {e.get('content_text', '')[:80].replace(chr(10), ' ')}"


QUERIES = [
    ("katib controller deployment", "semantic/dense test — should find katib controller configs"),
    ("InferenceService scaleToZeroGracePeriod", "keyword/BM25 test — KServe CRD identifiers"),
    ("KFP pipeline component definition", "hybrid RRF test — Kubeflow Pipelines DSL"),
    ("PVC volume mount storage config", "partition filter test — code only, storage configs"),
    ("kubeflow pipeline DSL syntax", "cross-type retrieval test — both YAML and Python"),
]


def main():
    eval_id = sys.argv[1] if len(sys.argv) > 1 else "eval1_code_only"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_file = EVAL_DIR / f"{eval_id}_{timestamp}.json"
    EVAL_DIR.mkdir(exist_ok=True)

    client = MilvusClient(uri="http://127.0.0.1:19530")
    model = get_model()

    info = client.describe_collection(COLLECTION)
    entity_count = info.get("num_entities", "unknown")

    print(f"\n[BENCH] {'='*60}")
    print(f"[BENCH] Eval: {eval_id}")
    print(f"[BENCH] Collection: {COLLECTION} | Entities: {entity_count}")
    print(f"[BENCH] Output: {eval_file}")
    print(f"[BENCH] {'='*60}\n")

    all_results = {
        "eval_id": eval_id,
        "timestamp": timestamp,
        "collection": COLLECTION,
        "model": MODEL,
        "entity_count": entity_count,
        "queries": [],
    }

    partition_filter = None if "with_docs" in eval_id or "llm" in eval_id else 'source_type == "code"'
    for query_text, description in QUERIES:
        print(f"[BENCH] Q: {query_text}")
        print(f"       {description}")
        results = run_query(client, model, query_text, partition_filter=partition_filter)
        all_results["queries"].append({
            "query": query_text,
            "description": description,
            "partition_filter": partition_filter,
            "results": results,
        })
        for req_type, data in results.items():
            print(f"  {req_type:6s}: {len(data['hits'])} hits, {data['latency_ms']:.1f}ms")
            for h in data["hits"][:2]:
                print(f"         {h['file']} dist={h['distance']:.4f}")
        print()

    total_latency = sum(
        q["results"][rt]["latency_ms"]
        for q in all_results["queries"]
        for rt in ["dense", "sparse", "hybrid"]
    )
    avg_latency = total_latency / (len(all_results["queries"]) * 3)
    all_results["summary"] = {
        "total_queries": len(QUERIES),
        "avg_latency_ms": round(avg_latency, 2),
    }

    with open(eval_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[BENCH] Saved: {eval_file}")
    print(f"[BENCH] Avg latency: {avg_latency:.1f}ms across {len(QUERIES)*3} searches")
    print(f"[BENCH] Done!")


if __name__ == "__main__":
    main()