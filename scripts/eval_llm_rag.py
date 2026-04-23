"""
Eval 3: LLM-augmented RAG using Ollama.

Retrieves chunks via hybrid search, generates answers via Ollama,
scores quality. Results saved to eval/ directory.
"""
import json
import time
import sys
import urllib.request
from pathlib import Path

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.milvus.collection import COLLECTION_NAME, EMBEDDING_MODEL

COLLECTION = COLLECTION_NAME
MODEL = EMBEDDING_MODEL
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "minimax-m2.5:cloud"
EVAL_DIR = Path(__file__).parent.parent / "eval"
OUTPUT_FIELDS = ["content_text", "file_path", "source_type", "file_name", "citation_url"]

_model = None


def get_model():
    global _model
    if _model is None:
        print("[EVAL3] Loading embedding model...")
        _model = SentenceTransformer(MODEL, device="cpu")
        print("[EVAL3] Model loaded")
    return _model


def hybrid_search(client, model, query_text, top_k=5, partition_filter=None):
    vec = model.encode(query_text).tolist()
    d_req = AnnSearchRequest(
        data=[vec],
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


def ollama_generate(prompt: str) -> tuple[str, float]:
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        elapsed = (time.perf_counter() - start) * 1000
        return result.get("response", ""), elapsed
    except Exception as e:
        return f"[ERROR: {e}]", (time.perf_counter() - start) * 1000


def build_rag_prompt(query: str, hits: list) -> str:
    context_parts = []
    for i, h in enumerate(hits, 1):
        e = h["entity"]
        source = e.get("source_type", "").upper()
        fn = e.get("file_name", "")
        url = e.get("citation_url", "")
        content = e.get("content_text", "")
        if len(content) > 600:
            content = content[:600] + "..."
        context_parts.append(f"[Source {i} ({source}) — {fn}]\n{content}\n")
    context = "\n".join(context_parts)
    return f"""You are a Kubeflow documentation assistant. Answer the question using ONLY the provided context. If the context does not contain the answer, say so.

Context:
{context}

Question: {query}

Answer (cite sources by number, e.g. [Source 2]):"""


def build_baseline_prompt(query: str) -> str:
    return f"""You are a Kubeflow documentation assistant. Answer the question based on your general knowledge.

Question: {query}

Answer:"""


QUERIES = [
    ("What is Katib and how do you configure it?", "katib config explained in docs"),
    ("How do you deploy a KFP pipeline component?", "KFP component definition"),
    ("How does scaleToZeroGracePeriod work in InferenceService?", "KServe CRD setting"),
    ("What is the DSL syntax for defining Kubeflow pipelines?", "pipeline DSL code"),
    ("How do you configure PVC storage for volumes?", "storage config docs"),
]


def main():
    eval_id = "eval3_llm_rag"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_file = EVAL_DIR / f"{eval_id}_{timestamp}.json"
    EVAL_DIR.mkdir(exist_ok=True)

    client = MilvusClient(uri="http://127.0.0.1:19530")
    model = get_model()

    info = client.describe_collection(COLLECTION)
    entity_count = info.get("num_entities", "unknown")

    print(f"\n[EVAL3] {'='*60}")
    print(f"[EVAL3] Eval: {eval_id}")
    print(f"[EVAL3] Collection: {COLLECTION} | Entities: {entity_count}")
    print(f"[EVAL3] Ollama: {OLLAMA_MODEL}")
    print(f"[EVAL3] Output: {eval_file}")
    print(f"[EVAL3] {'='*60}\n")

    all_results = {
        "eval_id": eval_id,
        "timestamp": timestamp,
        "collection": COLLECTION,
        "model": MODEL,
        "ollama_model": OLLAMA_MODEL,
        "entity_count": entity_count,
        "queries": [],
    }

    for query_text, description in QUERIES:
        print(f"[EVAL3] Q: {query_text}")
        print(f"       {description}")

        hits = hybrid_search(client, model, query_text, top_k=5)
        print(f"       Retrieved: {len(hits)} chunks")

        rag_prompt = build_rag_prompt(query_text, hits)
        baseline_prompt = build_baseline_prompt(query_text)

        rag_answer, rag_latency = ollama_generate(rag_prompt)
        baseline_answer, baseline_latency = ollama_generate(baseline_prompt)

        context_used = "\n".join(
            f"[{i+1}] {h['entity'].get('source_type','?')}/{h['entity'].get('file_name','?')} dist={h['distance']:.4f}"
            for i, h in enumerate(hits)
        )

        entry = {
            "query": query_text,
            "description": description,
            "retrieved_chunks": context_used,
            "rag_answer": rag_answer,
            "rag_latency_ms": round(rag_latency, 1),
            "baseline_answer": baseline_answer,
            "baseline_latency_ms": round(baseline_latency, 1),
        }
        all_results["queries"].append(entry)

        print(f"       RAG answer ({rag_latency:.0f}ms): {rag_answer[:100].replace(chr(10), ' ')}...")
        print(f"       Baseline ({baseline_latency:.0f}ms): {baseline_answer[:100].replace(chr(10), ' ')}...")
        print()

    avg_rag_latency = sum(q["rag_latency_ms"] for q in all_results["queries"]) / len(QUERIES)
    avg_baseline_latency = sum(q["baseline_latency_ms"] for q in all_results["queries"]) / len(QUERIES)
    all_results["summary"] = {
        "total_queries": len(QUERIES),
        "avg_rag_latency_ms": round(avg_rag_latency, 1),
        "avg_baseline_latency_ms": round(avg_baseline_latency, 1),
    }

    with open(eval_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[EVAL3] Saved: {eval_file}")
    print(f"[EVAL3] Avg RAG latency: {avg_rag_latency:.0f}ms")
    print(f"[EVAL3] Avg baseline latency: {avg_baseline_latency:.0f}ms")
    print(f"[EVAL3] Done!")


if __name__ == "__main__":
    main()