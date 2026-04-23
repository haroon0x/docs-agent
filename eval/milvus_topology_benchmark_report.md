# Milvus Topology Benchmark Report

Date: 2026-04-23
Corrected result file: `eval/eval_topology_models_20260423_175954.json`
Benchmark script: `scripts/eval_topology_models.py`

## TL;DR

The best retrieval design is **B hybrid**:

- separate `rag_code` and `rag_docs` collections
- MPNet dense embeddings for both
- Milvus BM25 sparse search
- RRF fusion across source collections

Why it wins:

- better mixed code+docs coverage
- highest overall retrieval quality
- cleaner source routing for RAG

Single collection with partition key (**A hybrid**) is close, and simpler, but it loses mixed-query coverage.

CodeBERT split (**C hybrid**) is not competitive. It hurts dense retrieval too much.

## What Was Tested

Corpus:

| Source | Count |
|---|---:|
| Code | 1,380 |
| Docs | 1,771 |
| Total | 3,151 |

Query set:

- 4 code-only cases
- 4 docs-only cases
- 6 mixed code+docs cases

Search modes:

- `dense`: embedding search only
- `sparse`: Milvus BM25 only
- `hybrid`: dense + BM25 fused with RRF

Topologies:

| Setup | Topology | Dense model |
|---|---|---|
| `A` | One collection: `rag_all` with `source_type` partition key | MPNet |
| `A_plus` | One collection: `rag_all_bge` with `source_type` partition key | BGE-base |
| `B` | Split collections: `rag_code` + `rag_docs` | MPNet |
| `B_plus` | Split collections: `rag_code_bge` + `rag_docs_bge` | BGE-base |
| `C` | Split collections, CodeBERT for code | CodeBERT + BGE-base |

## Metric Guide

| Metric | Meaning | Why it matters |
|---|---|---|
| MRR | Rank of first relevant hit | Good for "did we find something useful immediately?" |
| Recall@10 | Relevant docs found in top 10 | Important for RAG context breadth |
| nDCG@10 | Ranking quality with top-rank reward | Better than recall alone |
| Source purity | Share of top 10 from allowed source types | Detects wrong-source leakage |
| Source coverage | Allowed source types present in top 10 | Critical for mixed queries |
| Embed latency | Time spent building query vector | Major part of dense/hybrid cost |
| DB latency | Milvus search time | Database-side retrieval cost |
| Total latency | End-to-end retrieval time | User-visible latency |

Important:

- For mixed queries, source coverage matters more than purity.
- For code-only and docs-only queries, both purity and coverage matter.

## Results

Corrected aggregate summary:

| Setup | Search | MRR | Recall@10 | nDCG@10 | Coverage | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| `A` | dense | 0.9286 | 0.7095 | 0.7335 | 0.8929 | 135.05 ms |
| `A` | sparse | 0.7936 | 0.6679 | 0.6536 | 0.8929 | 7.27 ms |
| `A` | hybrid | 0.8810 | 0.7310 | 0.7301 | 0.8929 | 130.07 ms |
| `B` | dense | 0.8571 | 0.7691 | 0.7386 | 1.0000 | 184.56 ms |
| `B` | sparse | 0.7936 | 0.7274 | 0.6885 | 1.0000 | 10.46 ms |
| `B` | hybrid | 0.8988 | 0.7560 | 0.7612 | 1.0000 | 189.97 ms |
| `C` | dense | 0.3912 | 0.4155 | 0.3588 | 1.0000 | 184.30 ms |
| `C` | sparse | 0.7936 | 0.7274 | 0.6885 | 1.0000 | 12.20 ms |
| `C` | hybrid | 0.6155 | 0.6095 | 0.5571 | 1.0000 | 180.15 ms |

Bottom line:

- `B hybrid` is best on quality.
- `A hybrid` is best on speed among strong options.
- `C hybrid` is clearly worst of the three topologies.

## Why B Wins

### Mixed queries

This benchmark is about code + docs RAG. That is where `B` separates itself.

In `B`:

- code queries search `rag_code`
- docs queries search `rag_docs`
- mixed queries search both and fuse results

That gives both source types a fair chance before top-10 selection.

In `A`:

- code and docs share one ranking pool
- mixed queries can over-select one source type
- the top-10 can miss the other source entirely

That is why `A hybrid` only reaches `0.8929` source coverage overall, while `B hybrid` reaches `1.0000`.

### BM25 behavior

BM25 works better when code and docs are separated.

Why:

- code has many YAML keys and repeated config tokens
- docs have natural language and headings
- mixing them changes term statistics

That shows up in sparse search:

- `A sparse` recall: `0.6679`
- `B sparse` recall: `0.7274`

### Dense retrieval

`B dense` beats `A dense` on recall and nDCG:

- `A dense` recall: `0.7095`
- `B dense` recall: `0.7691`
- `A dense` nDCG: `0.7335`
- `B dense` nDCG: `0.7386`

The main reason is not one magical query result. It is that source-specific search is less noisy.

## Why A Still Matters

`A hybrid` is not bad. It is simpler and faster.

What it does well:

- code-only queries
- docs-only queries
- lower operational overhead

Where it loses:

- mixed queries need balanced source coverage
- some mixed answers need both implementation and explanation

Representative mixed cases:

| Case | `A hybrid` | `B hybrid` | Note |
|---|---|---|---|
| `mixed_storage` | 0.75 recall, 0.7877 nDCG | 1.0 recall, 1.0 nDCG | `B` finds both storage code and docs |
| `mixed_spark_application` | 0.40 recall, 0.4600 nDCG | 0.60 recall, 0.6740 nDCG | `B` better balanced |
| `mixed_component_definition` | 0.75 recall, 0.4939 nDCG | 0.25 recall, 0.1681 nDCG | `A` can win a case, but `B` still has better overall mixed coverage |

The key point:

- `A` can win a specific case
- `B` wins the system-level objective

## Why C Loses

`C` replaces MPNet on code with base CodeBERT.

That hurts a lot:

- `C hybrid` MRR: `0.6155`
- `C hybrid` recall: `0.6095`
- `C hybrid` nDCG: `0.5571`

Dense retrieval is the problem, not BM25.

Observed failure mode:

- CodeBERT often ranks generic or structurally similar YAML above the actual target file.
- Hybrid fusion cannot fully recover because dense still influences the final ranking.

Examples:

| Case | `B hybrid` top hit | `C hybrid` top hit |
|---|---|---|
| `code_katib_controller` | Katib controller manifest | Istio sidecar patch |
| `code_pipeline_runner_rbac` | pipeline runner RBAC file | unrelated `kustomization.yaml` |
| `mixed_pipeline_dsl` | actual pipeline DSL example | unrelated sidecar YAML |

Conclusion:

- Do not switch to CodeBERT for this corpus.
- If a code-specific model is tested later, use a retrieval-tuned code embedding model, not base CodeBERT.

## Why BGE-base Did Not Win

`A_plus` and `B_plus` tested BGE-base.

It did not beat MPNet here:

| Setup | Hybrid MRR | Hybrid Recall@10 | Hybrid nDCG@10 |
|---|---:|---:|---:|
| `A` | 0.8810 | 0.7310 | 0.7301 |
| `A_plus` | 0.7881 | 0.6512 | 0.6668 |
| `B` | 0.8988 | 0.7560 | 0.7612 |
| `B_plus` | 0.8571 | 0.6893 | 0.7008 |

Inference:

- this corpus likes MPNet more than BGE-base
- BGE-base may need different chunking or prompting to shine
- no evidence here justifies replacing MPNet

## Latency

Latency tradeoff is real:

| Setup | Hybrid mean latency |
|---|---:|
| `A` | 130.07 ms |
| `B` | 189.97 ms |
| `C` | 180.15 ms |

Sparse is much faster:

| Setup | Sparse mean latency |
|---|---:|
| `A` | 7.27 ms |
| `B` | 10.46 ms |
| `C` | 12.20 ms |

Why `B` is slower:

- mixed queries search two collections
- the benchmark currently embeds the query once per searched collection

What to optimize next:

- reuse one MPNet query embedding for both code and docs in `B`
- keep RRF fusion
- accept a small latency cost for much better mixed retrieval

## Important Correction

The older result file, `eval/eval_topology_models_20260423_172644.json`, should not be used for the final decision.

Reason:

- the partition filter was passed to `hybrid_search` at the wrong level
- PyMilvus expects the scalar filter on each `AnnSearchRequest(expr=...)`

The script is fixed now, and the corrected file is:

```text
eval/eval_topology_models_20260423_175954.json
```

## Final Recommendation

Use **B hybrid** in production.

Use **A hybrid** only if you want the simplest deployment and can accept weaker mixed-query coverage.

Do not use **C**.

Next improvement, if needed:

- optimize `B` query embedding reuse
- then test answer quality with an LLM on top of the best retrieval setup
