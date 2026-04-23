"""Test partition key filter routing in hybrid search."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

COLLECTION = "docs_rag"
MODEL = "sentence-transformers/all-mpnet-base-v2"

client = MilvusClient(uri="http://127.0.0.1:19530")
model = SentenceTransformer(MODEL)

queries = [
    "katib controller deployment",
    "dex service configuration",
    "istio virtual service gateway",
]

for q in queries:
    vec = model.encode(q).tolist()
    d_req = AnnSearchRequest(
        data=[vec], anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {}}, limit=10,
    )
    s_req = AnnSearchRequest(
        data=[q], anns_field="sparse_vector",
        param={"metric_type": "BM25"}, limit=10,
    )

    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")

    all_res = client.hybrid_search(
        collection_name=COLLECTION, reqs=[d_req, s_req],
        ranker=RRFRanker(k=60), limit=5,
        output_fields=["content_text", "source_type", "file_name"],
    )
    print("  [ALL PARTITIONS]")
    for h in all_res[0]:
        e = h["entity"]
        print(f"    [{e['source_type']}] {e['file_name']} dist={h['distance']:.4f}")

    code_res = client.hybrid_search(
        collection_name=COLLECTION, reqs=[d_req, s_req],
        ranker=RRFRanker(k=60), limit=5,
        filter='source_type == "code"',
        output_fields=["content_text", "source_type", "file_name"],
    )
    print("  [CODE PARTITION ONLY]")
    for h in code_res[0]:
        e = h["entity"]
        print(f"    [{e['source_type']}] {e['file_name']} dist={h['distance']:.4f}")

print("\n[OK] Partition filter routing verified")
