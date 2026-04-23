"""Quick hybrid search test."""
import sys
sys.path.insert(0, "/home/g/Documents/contrib/docs-agent")

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION = "docs_rag"

client = MilvusClient(uri="http://127.0.0.1:19530")
model = SentenceTransformer(MODEL, device="cpu")

print("Connected. Testing hybrid search...")

q = "katib controller deployment"
vec = model.encode(q).tolist()

d_req = AnnSearchRequest(
    data=[vec],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {}},
    limit=5,
)
s_req = AnnSearchRequest(
    data=[q],
    anns_field="sparse_vector",
    param={"metric_type": "BM25"},
    limit=5,
)

all_res = client.hybrid_search(
    collection_name=COLLECTION, reqs=[d_req, s_req],
    ranker=RRFRanker(k=60), limit=5,
    output_fields=["source_type", "file_name"],
)
print(f"\nQ: {q}")
print(f"Results: {len(all_res[0])}")
for h in all_res[0]:
    e = h["entity"]
    print(f"  [{e['source_type']}] {e['file_name']} dist={h['distance']:.4f}")

code_res = client.hybrid_search(
    collection_name=COLLECTION, reqs=[d_req, s_req],
    ranker=RRFRanker(k=60), limit=5,
    filter='source_type == "code"',
    output_fields=["source_type", "file_name"],
)
print(f"\nCode-only results: {len(code_res[0])}")
for h in code_res[0]:
    e = h["entity"]
    print(f"  [{e['source_type']}] {e['file_name']} dist={h['distance']:.4f}")

print("\nAll tests passed.")
