"""
Legacy comparison script.

This script does NOT compare separate physical collections. It only compares
three query modes on the SAME `docs_rag` collection:
  Approach A: Single collection, unfiltered hybrid search
  Approach B: Single collection, code-only hybrid search
  Approach C: Single collection, docs-only hybrid search

For each query, compares:
  - Dense-only vs Sparse-only vs Hybrid rankings
  - MRR, Recall@K against ground truth
  - Latency per approach
  - RRF rank shift analysis (how docs/code-only rankings differ from mixed)

Ground truth: provided as {query: [relevant_file_names]} dict below.
If no ground truth provided, shows Hit@K and rank position without scoring.

Usage:
    MILVUS_HOST=127.0.0.1 PYTHONPATH=. uv run python scripts/eval_comparison.py
"""
import json
import time
import sys
from pathlib import Path
from collections import defaultdict

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.milvus.collection import COLLECTION_NAME, EMBEDDING_MODEL

COLLECTION = COLLECTION_NAME
MODEL = EMBEDDING_MODEL
EVAL_DIR = Path(__file__).parent.parent / "eval"
OUTPUT_FIELDS = ["content_text", "file_path", "source_type", "file_name", "citation_url"]

_model = None


def get_model():
    global _model
    if _model is None:
        print("[EVAL] Loading model: sentence-transformers/all-mpnet-base-v2 ...")
        _model = SentenceTransformer(MODEL, device="cpu")
        print("[EVAL] Model loaded")
    return _model


QUERIES = [
    {
        "id": 1,
        "query": "katib controller deployment",
        "description": "semantic/dense — find katib controller YAML configs",
        "ground_truth_code": ["controller.yaml", "kustomization.yaml", "rbac.yaml"],
        "ground_truth_docs": ["katib-config.md", "architecture.md"],
    },
    {
        "id": 2,
        "query": "InferenceService scaleToZeroGracePeriod",
        "description": "keyword/BM25 — KServe CRD identifiers",
        "ground_truth_code": ["kserve.yaml", "kserve_kubeflow.yaml"],
        "ground_truth_docs": ["webapp.md"],
    },
    {
        "id": 3,
        "query": "KFP pipeline component definition",
        "description": "hybrid RRF — Kubeflow Pipelines DSL",
        "ground_truth_code": ["kustomization.yaml", "_index.md"],
        "ground_truth_docs": ["component-development.md", "pipeline.md", "_index.md"],
    },
    {
        "id": 4,
        "query": "PVC volume mount storage config",
        "description": "partition filter — storage configs",
        "ground_truth_code": ["seaweedfs-pvc.yaml", "metrics-server_resource_table.py"],
        "ground_truth_docs": ["manipulate-resources.md", "platform-specific-features.md"],
    },
    {
        "id": 5,
        "query": "kubeflow pipeline DSL syntax",
        "description": "cross-type — YAML manifests + Python code",
        "ground_truth_code": ["pipeline_run_and_wait_kubeflow.py", "kustomization.yaml"],
        "ground_truth_docs": ["getting-started.md", "pipeline.md", "artifacts.md"],
    },
]


def hybrid_search(client, model, query_text, top_k=10, partition_filter=None):
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


def dense_only(client, model, query_text, top_k=10, partition_filter=None):
    vec = model.encode(query_text).tolist()
    req = AnnSearchRequest(
        data=[vec],
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


def sparse_only(client, query_text, top_k=10, partition_filter=None):
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


def mrr_at_k(hits, relevant_files, k=5):
    score = 0.0
    for i, h in enumerate(hits[:k]):
        fn = h["entity"].get("file_name", "")
        if fn in relevant_files:
            score = 1.0 / (i + 1)
            break
    return score


def recall_at_k(hits, relevant_files, k=5):
    retrieved = set(h["entity"].get("file_name", "") for i, h in enumerate(hits[:k]))
    relevant = set(relevant_files)
    if not relevant:
        return None
    return len(retrieved & relevant) / len(relevant)


def dcg_at_k(hits, relevant_files, k=5):
    score = 0.0
    for i, h in enumerate(hits[:k]):
        fn = h["entity"].get("file_name", "")
        if fn in relevant_files:
            score += 1.0 / (1 + i)
    return score


def ndcg_at_k(hits, relevant_files, k=5):
    dcg = dcg_at_k(hits, relevant_files, k)
    ideal = dcg_at_k([{"entity": {"file_name": f}} for f in relevant_files], relevant_files, k)
    if not ideal or ideal == 0:
        return None
    return dcg / ideal


def rank_shift(hits_a, hits_b, k=5):
    files_a = [h["entity"].get("file_name", "") for h in hits_a[:k]]
    files_b = [h["entity"].get("file_name", "") for h in hits_b[:k]]
    set_a, set_b = set(files_a), set(files_b)
    overlap = len(set_a & set_b)
    return {
        "top5_overlap": overlap,
        "top5_a_only": len(set_a - set_b),
        "top5_b_only": len(set_b - set_a),
        "a_files": files_a,
        "b_files": files_b,
    }


def format_hits(hits, label=""):
    lines = []
    if label:
        lines.append(f"  [{label}]")
    for i, h in enumerate(hits[:10]):
        e = h["entity"]
        src = e.get("source_type", "?").upper()
        fn = e.get("file_name", "?")
        dist = h.get("distance", 0)
        lines.append(f"    {i+1:2d}. [{src}] {fn:<45s} dist={dist:.4f}")
    return "\n".join(lines)


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_file = EVAL_DIR / f"eval_comparison_{timestamp}.json"
    EVAL_DIR.mkdir(exist_ok=True)

    client = MilvusClient(uri="http://127.0.0.1:19530")
    model = get_model()

    info = client.describe_collection(COLLECTION)
    entity_count = info.get("num_entities", "unknown")

    print(f"\n{'='*80}")
    print(f"[EVAL] Comparative: Single Collection Partition vs Separate Collections")
    print(f"[EVAL] Collection: {COLLECTION} | Entities: {entity_count}")
    print(f"[EVAL] Approaches: A=unfiltered | B=code partition | C=docs partition")
    print(f"[EVAL] Output: {eval_file}")
    print(f"{'='*80}\n")

    all_results = {
        "timestamp": timestamp,
        "collection": COLLECTION,
        "entity_count": entity_count,
        "model": MODEL,
        "queries": [],
        "summary": {},
    }

    approach_labels = {"A": "unfiltered", "B": "code_partition", "C": "docs_partition"}
    filters = {
        "A": None,
        "B": 'source_type == "code"',
        "C": 'source_type == "docs"',
    }

    for qdata in QUERIES:
        qid = qdata["id"]
        qtext = qdata["query"]
        desc = qdata["description"]
        gt_code = qdata.get("ground_truth_code", [])
        gt_docs = qdata.get("ground_truth_docs", [])
        gt_all = gt_code + gt_docs

        print(f"{'─'*80}")
        print(f"[Q{qid}] {qtext}")
        print(f"       {desc}")
        print(f"       Ground truth CODE  ({len(gt_code)}): {gt_code}")
        print(f"       Ground truth DOCS  ({len(gt_docs)}): {gt_docs}")
        print(f"{'─'*80}")

        query_result = {
            "query_id": qid,
            "query": qtext,
            "description": desc,
            "ground_truth_code": gt_code,
            "ground_truth_docs": gt_docs,
            "approaches": {},
            "rank_shifts": {},
        }

        hits_by_approach = {}
        latencies = {}

        for approach, label in approach_labels.items():
            filter_expr = filters[approach]
            t0 = time.perf_counter()
            hybrid_hits = hybrid_search(client, model, qtext, top_k=10, partition_filter=filter_expr)
            t1 = time.perf_counter()
            dense_hits = dense_only(client, model, qtext, top_k=10, partition_filter=filter_expr)
            t2 = time.perf_counter()
            sparse_hits = sparse_only(client, qtext, top_k=10, partition_filter=filter_expr)
            t3 = time.perf_counter()

            hybrid_ms = (t1 - t0) * 1000
            dense_ms = (t2 - t1) * 1000
            sparse_ms = (t3 - t2) * 1000
            latencies[approach] = {"hybrid": hybrid_ms, "dense": dense_ms, "sparse": sparse_ms}
            hits_by_approach[approach] = {
                "hybrid": hybrid_hits,
                "dense": dense_hits,
                "sparse": sparse_hits,
            }

            gt = gt_all if approach == "A" else (gt_code if approach == "B" else gt_docs)

            metrics = {}
            for search_type, hits in [("hybrid", hybrid_hits), ("dense", dense_hits), ("sparse", sparse_hits)]:
                mrr_5 = mrr_at_k(hits, gt, k=5)
                mrr_10 = mrr_at_k(hits, gt, k=10)
                rec_5 = recall_at_k(hits, gt, k=5)
                rec_10 = recall_at_k(hits, gt, k=10)
                ndcg_5 = ndcg_at_k(hits, gt, k=5)
                ndcg_10 = ndcg_at_k(hits, gt, k=10)
                top5_files = [h["entity"].get("file_name", "") for h in hits[:5]]
                top10_files = [h["entity"].get("file_name", "") for h in hits[:10]]
                metrics[search_type] = {
                    "mrr@5": round(mrr_5, 4) if mrr_5 is not None else None,
                    "mrr@10": round(mrr_10, 4) if mrr_10 is not None else None,
                    "recall@5": round(rec_5, 4) if rec_5 is not None else None,
                    "recall@10": round(rec_10, 4) if rec_10 is not None else None,
                    "ndcg@5": round(ndcg_5, 4) if ndcg_5 is not None else None,
                    "ndcg@10": round(ndcg_10, 4) if ndcg_10 is not None else None,
                    "top5_files": top5_files,
                    "top10_files": top10_files,
                }

            query_result["approaches"][approach] = {
                "label": label,
                "filter": filter_expr,
                "latencies_ms": latencies[approach],
                "metrics": metrics,
            }

        shifts = {}
        for (a1, a2) in [("A", "B"), ("A", "C"), ("B", "C")]:
            shift_key = f"{a1}_vs_{a2}"
            shifts[shift_key] = rank_shift(
                hits_by_approach[a1]["hybrid"],
                hits_by_approach[a2]["hybrid"],
                k=10
            )

        query_result["rank_shifts"] = shifts
        all_results["queries"].append(query_result)

        print(f"\n  Approach A — Unfiltered (mixed: code + docs):")
        print(f"    Latency: hybrid={latencies['A']['hybrid']:.1f}ms dense={latencies['A']['dense']:.1f}ms sparse={latencies['A']['sparse']:.1f}ms")
        for st in ["hybrid", "dense", "sparse"]:
            m = query_result["approaches"]["A"]["metrics"][st]
            print(f"    {st.upper():6s} | MRR@5={m['mrr@5']} MRR@10={m['mrr@10']} R@5={m['recall@5']} R@10={m['recall@10']} nDCG@5={m['ndcg@5']}")
        print(format_hits(hits_by_approach["A"]["hybrid"], "HYBRID-A"))

        print(f"\n  Approach B — Code partition only:")
        print(f"    Latency: hybrid={latencies['B']['hybrid']:.1f}ms dense={latencies['B']['dense']:.1f}ms sparse={latencies['B']['sparse']:.1f}ms")
        for st in ["hybrid", "dense", "sparse"]:
            m = query_result["approaches"]["B"]["metrics"][st]
            print(f"    {st.upper():6s} | MRR@5={m['mrr@5']} MRR@10={m['mrr@10']} R@5={m['recall@5']} R@10={m['recall@10']} nDCG@5={m['ndcg@5']}")
        print(format_hits(hits_by_approach["B"]["hybrid"], "HYBRID-B"))

        print(f"\n  Approach C — Docs partition only:")
        print(f"    Latency: hybrid={latencies['C']['hybrid']:.1f}ms dense={latencies['C']['dense']:.1f}ms sparse={latencies['C']['sparse']:.1f}ms")
        for st in ["hybrid", "dense", "sparse"]:
            m = query_result["approaches"]["C"]["metrics"][st]
            print(f"    {st.upper():6s} | MRR@5={m['mrr@5']} MRR@10={m['mrr@10']} R@5={m['recall@5']} R@10={m['recall@10']} nDCG@5={m['ndcg@5']}")
        print(format_hits(hits_by_approach["C"]["hybrid"], "HYBRID-C"))

        print(f"\n  Rank Shift A vs B: top10 overlap={shifts['A_vs_B']['top5_overlap']}/5 + extras B-only={shifts['A_vs_B']['top5_b_only']}")
        print(f"  Rank Shift A vs C: top10 overlap={shifts['A_vs_C']['top5_overlap']}/5 + extras C-only={shifts['A_vs_C']['top5_b_only']}")
        print(f"  Rank Shift B vs C: top10 overlap={shifts['B_vs_C']['top5_overlap']}/5")
        print()

    avg_metrics = defaultdict(lambda: defaultdict(list))
    for q in all_results["queries"]:
        for approach in ["A", "B", "C"]:
            for st in ["hybrid", "dense", "sparse"]:
                m = q["approaches"][approach]["metrics"][st]
                for metric in ["mrr@5", "mrr@10", "recall@5", "recall@10", "ndcg@5", "ndcg@10"]:
                    val = m[metric]
                    if val is not None:
                        avg_metrics[approach][st].append((metric, val))

    summary_lines = ["\n[SUMMARY] Per-approach average metrics:\n"]
    summary_data = {}
    for approach in ["A", "B", "C"]:
        label = approach_labels[approach]
        summary_data[approach] = {"label": label, "search_types": {}}
        summary_lines.append(f"  Approach {approach} ({label}):")
        for st in ["hybrid", "dense", "sparse"]:
            vals = avg_metrics[approach][st]
            if not vals:
                continue
            metric_avgs = {}
            for metric, val in vals:
                metric_avgs.setdefault(metric, []).append(val)
            st_summary = {}
            for metric, vals_list in metric_avgs.items():
                avg = sum(vals_list) / len(vals_list)
                st_summary[metric] = round(avg, 4)
            summary_data[approach]["search_types"][st] = st_summary
            summary_lines.append(
                f"    {st.upper():6s} | MRR@5={st_summary.get('mrr@5','N/A'):<8} MRR@10={st_summary.get('mrr@10','N/A'):<8} "
                f"R@5={st_summary.get('recall@5','N/A'):<8} R@10={st_summary.get('recall@10','N/A'):<8} "
                f"nDCG@5={st_summary.get('ndcg@5','N/A'):<8}"
            )

    all_results["summary"] = summary_data
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(eval_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[EVAL] Saved: {eval_file}")


if __name__ == "__main__":
    main()
