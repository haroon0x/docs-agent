# Milvus Topology Benchmark Report

Date: 2026-04-23  
Corrected result file: `eval/eval_topology_models_20260423_175954.json`  
Benchmark script: `scripts/eval_topology_models.py`

## Executive Summary

The benchmark answers one question:

What is better for docs-agent retrieval?

- one Milvus collection with `source_type` partitioning
- or separate physical code/docs collections

The answer from the corrected run is:

- **Best overall**: `B hybrid`
- **Best simple fallback**: `A hybrid`
- **Not recommended**: `C hybrid` with CodeBERT

Why `B hybrid` wins:

- mixed code+docs questions need both source types in the result set
- split collections keep code and docs from competing in one pool
- RRF fusion combines both sources after each has had a fair search pass
- source coverage is perfect in `B` and lower in `A`

If you only need one sentence:

> Separate code/docs collections with MPNet + BM25 are the strongest setup for this corpus.

## Tested Cases

These are the benchmark cases in human-readable form.

### Code Retrieval Cases

- **Katib controller manifests**
  Query: `katib controller deployment`
- **Scale-to-zero configuration**
  Query: `Knative serving scale to zero configuration`
- **KServe ServingRuntime resources**
  Query: `KServe ServingRuntime cluster resources`
- **Pipeline runner RBAC**
  Query: `Kubeflow Pipelines pipeline runner RBAC service account`

### Documentation Retrieval Cases

- **Katib configuration docs**
  Query: `What is Katib and how do you configure it?`
- **KServe scale-to-zero docs**
  Query: `How does scale to zero work in KServe?`
- **Katib early stopping docs**
  Query: `How does Katib early stopping work for experiments?`
- **Pipelines multi-user docs**
  Query: `How does Kubeflow Pipelines multi-user isolation work?`

### Mixed Retrieval Cases

- **Storage configuration across code and docs**
  Query: `PVC volume mount storage config`
- **InferenceService setup across code and docs**
  Query: `KServe InferenceService predictor configuration`
- **SparkApplication code plus user guidance**
  Query: `SparkApplication resources and user guide`
- **Model Registry implementation plus docs**
  Query: `Kubeflow Model Registry service and docs`
- **Pipeline DSL code plus docs**
  Query: `kubeflow pipeline DSL syntax`
- **Pipeline component definition across code and docs**
  Query: `KFP pipeline component definition`

## Setup Names

Plain English:

- `A` = one collection for everything, with `source_type` partitioning
- `A_plus` = same as `A`, but with BGE-base instead of MPNet
- `B` = separate code collection and docs collection, both MPNet
- `B_plus` = separate code collection and docs collection, both BGE-base
- `C` = separate code collection with CodeBERT, docs collection with BGE-base

## Corpus

| Source | Count |
|---|---:|
| Code | 1,380 |
| Docs | 1,771 |
| Total | 3,151 |

## Metrics

| Metric | Meaning | Why it matters |
|---|---|---|
| MRR | Rank of first relevant hit | Good for finding the first useful chunk quickly |
| Recall@10 | Relevant docs found in top 10 | Important for RAG context breadth |
| nDCG@10 | Ranking quality with higher-rank reward | Better than recall alone |
| Source purity | Share of top 10 from expected source types | Detects source leakage |
| Source coverage | Expected source types present in top 10 | Critical for mixed queries |
| Embed latency | Time spent building query vector | Major part of dense/hybrid cost |
| DB latency | Milvus search time | Database-side retrieval cost |
| Total latency | End-to-end retrieval time | User-visible latency |

For mixed questions, source coverage matters more than purity.  
For code-only and docs-only questions, both matter.

## Main Results

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

Short read:

- `B hybrid` has the best retrieval quality.
- `A hybrid` is faster, but weaker on mixed-source reliability.
- `C hybrid` is far behind and should not be used.

## Best Setup By Objective

Not every objective points to the same winner. This is the practical recommendation matrix.

| Objective | Best choice | Why |
|---|---|---|
| Best overall retrieval quality | `B hybrid` | Highest hybrid MRR, recall, and nDCG |
| Best mixed-source reliability | `B hybrid` | Perfect source coverage (`1.0000`) |
| Best code-only retrieval | `A hybrid` and `B hybrid` (tie) | Same average code-only hybrid recall (`0.8167`); `A` is simpler, `B` fits split production design |
| Best docs-only retrieval | `B hybrid` | Best docs-only hybrid recall (`0.8750`) and nDCG (`0.8578`) |
| Best simple production setup | `A hybrid` | One collection, good quality, lower operational overhead |
| Best lowest-latency option | `A sparse` | Fastest overall at `7.27 ms` mean latency |
| Best low-latency split setup | `B sparse` | Fast, and keeps perfect source coverage |
| Best dense-only recall | `B dense` | Better dense recall and nDCG than `A dense` |
| Best dense-only first-hit rank | `A dense` | Highest dense MRR with lower latency than `B dense` |
| Best docs+code answer grounding | `B hybrid` | Most dependable for returning both implementation and explanation |
| Setup to avoid | `C hybrid` | CodeBERT hurts retrieval quality too much |

Read this table as follows:

- If you care most about mixed-source RAG quality, choose `B hybrid`.
- If you only search code, `A hybrid` and `B hybrid` are effectively tied on quality.
- If you only search docs, `B hybrid` is the best option.
- If you care most about simplicity, choose `A hybrid`.
- If you need a very fast retrieval mode, choose sparse, not hybrid.

## If Mixed Search Is Not Required

This is an important special case.

If your real product behavior is:

- sometimes search code
- sometimes search docs
- but never combine both in one answer

then the benchmark should be interpreted differently.

In that scenario, the main weakness of `A hybrid` matters much less.

Why:

- `A hybrid` mostly lost because mixed queries can let one source type crowd out the other
- if mixed queries do not exist, that failure mode disappears
- code-only and docs-only queries can use a direct `source_type` filter
- partition-key routing then works exactly in the mode Milvus is good at

### What This Means In Practice

If you do **code-only** and **docs-only** search as two separate product paths:

- `A hybrid` becomes a very strong option
- `B hybrid` is no longer the obvious winner
- the decision becomes quality tradeoff vs operational simplicity

### Code-Only Search

For code-only hybrid retrieval, `A hybrid` and `B hybrid` were effectively tied in this benchmark.

Why the tie makes sense:

- both setups are searching only code
- `A` uses a `source_type == "code"` filter, so docs do not interfere
- `B` searches the dedicated `rag_code` collection
- once the source is isolated, the main structural difference between one collection and two collections matters much less

Practical conclusion for code-only search:

- if code retrieval is your main need and you want the simpler system, `A hybrid` is enough
- if you already prefer split ownership and split ops, `B hybrid` is also fine

### Docs-Only Search

For docs-only hybrid retrieval, `B hybrid` was better than `A hybrid`.

Why:

- the docs-only corpus seems to benefit more from dedicated docs-only ranking
- splitting avoids code terms influencing BM25 statistics
- docs retrieval is more sensitive to clean natural-language ranking than code retrieval is

Practical conclusion for docs-only search:

- if docs search quality is the more important path, `B hybrid` has the edge
- if docs search is secondary and you want one simpler deployment, `A hybrid` is still reasonable

### Decision Rule Without Mixed Search

Use this simpler rule:

- choose **`A hybrid`** if you want one collection, simpler operations, and no mixed retrieval
- choose **`B hybrid`** if docs-only quality matters more, or if you want hard separation between code and docs retrieval systems

In other words:

- with mixed search: `B hybrid` is clearly the right choice
- without mixed search: `A hybrid` becomes much more attractive

That is the real boundary.

## Why `B` Wins

### 1. Mixed queries need both source types

This benchmark is not just about code retrieval or docs retrieval.  
It is about questions that need both.

In `B`:

- code queries search `rag_code`
- docs queries search `rag_docs`
- mixed queries search both and fuse the result lists

That means code and docs both get a fair chance before the final top 10 is built.

In `A`:

- code and docs share one ranking pool
- mixed queries can get dominated by one source type
- the other source can drop out of top 10

That is the main reason `A hybrid` does not win.

The aggregate numbers show it clearly:

- `A hybrid` coverage: `0.8929`
- `B hybrid` coverage: `1.0000`

### 2. BM25 works better when code and docs are separated

BM25 is sensitive to corpus statistics.

Code chunks contain:

- YAML keys
- resource names
- repeated config terms

Docs chunks contain:

- natural language explanations
- headings
- longer descriptive text

When both live in one collection, lexical ranking becomes noisier.  
When they are split, each source gets cleaner BM25 behavior.

That is why sparse recall improves in `B`:

- `A sparse` recall: `0.6679`
- `B sparse` recall: `0.7274`

### 3. Dense retrieval is also cleaner in split collections

`B dense` is stronger than `A dense` on recall and nDCG:

- `A dense` recall: `0.7095`
- `B dense` recall: `0.7691`
- `A dense` nDCG: `0.7335`
- `B dense` nDCG: `0.7386`

This does not come from one special query.  
It comes from reducing cross-source interference.

## Why `A` Is Close, But Not the Winner

`A hybrid` is not a bad design.  
It is the better single-collection option.

What `A` does well:

- simpler deployment
- one collection to build and manage
- good results on single-source queries

What it does not do as well:

- mixed code+docs queries
- source balancing
- keeping both code and docs visible in the final top 10

The result is a tradeoff:

- `A hybrid` is faster
- `B hybrid` is more reliable for the actual docs-agent use case

Mixed query examples:

| Case | `A hybrid` | `B hybrid` | Meaning |
|---|---|---|---|
| `mixed_storage` | 0.75 recall, 0.7877 nDCG | 1.0 recall, 1.0 nDCG | `B` returns both storage code and docs |
| `mixed_spark_application` | 0.40 recall, 0.4600 nDCG | 0.60 recall, 0.6740 nDCG | `B` is better balanced |
| `mixed_model_registry` | 0.50 MRR, 0.5307 nDCG | 1.00 MRR, 0.6364 nDCG | `B` finds the manifest directly |
| `mixed_component_definition` | 0.50 MRR, 0.4939 nDCG | 0.25 MRR, 0.1681 nDCG | `A` can win a case, but not the benchmark goal |

The key point:

- `A` can win a specific question
- `B` wins the system-level objective

## Why `C` Loses

`C` replaces MPNet on the code side with base CodeBERT.

That hurts retrieval quality badly:

- `C hybrid` MRR: `0.6155`
- `C hybrid` recall: `0.6095`
- `C hybrid` nDCG: `0.5571`

Observed failure mode:

- CodeBERT often ranks structurally similar YAML above the actual target file.
- Hybrid fusion cannot fully recover if dense ranking is weak.

Examples:

| Case | `B hybrid` top hit | `C hybrid` top hit |
|---|---|---|
| `code_katib_controller` | Katib controller manifest | Istio sidecar patch |
| `code_pipeline_runner_rbac` | pipeline runner RBAC file | unrelated `kustomization.yaml` |
| `mixed_pipeline_dsl` | real pipeline DSL example | unrelated sidecar YAML |

Conclusion:

- Do not use CodeBERT here.
- If a code-specific embedding model is tested later, use a retrieval-tuned code model, not base CodeBERT.

## Why BGE-base Did Not Win

`A_plus` and `B_plus` tested BGE-base instead of MPNet.

It did not beat MPNet on this corpus.

| Setup | Hybrid MRR | Hybrid Recall@10 | Hybrid nDCG@10 |
|---|---:|---:|---:|
| `A` | 0.8810 | 0.7310 | 0.7301 |
| `A_plus` | 0.7881 | 0.6512 | 0.6668 |
| `B` | 0.8988 | 0.7560 | 0.7612 |
| `B_plus` | 0.8571 | 0.6893 | 0.7008 |

Interpretation:

- this corpus favors MPNet
- BGE-base may need different chunking or prompting to outperform
- there is no evidence here to change the dense baseline

## Latency

Latency is the main reason to consider `A`.

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
- the current benchmark embeds the query once per searched collection

That latency can be improved later by reusing the same MPNet query embedding for both collections.

## Important Correction

The older result file, `eval/eval_topology_models_20260423_172644.json`, should not be used for the final decision.

Reason:

- the partition filter was passed to `hybrid_search` in the wrong place
- PyMilvus expects the scalar filter on each `AnnSearchRequest(expr=...)`

The corrected file is:

```text
eval/eval_topology_models_20260423_175954.json
```

## Final Recommendation

Use `B hybrid` for best retrieval quality.

Use `A hybrid` only if deployment simplicity matters more than mixed-query quality.

Do not use `C`.

Best next improvement:

- optimize `B` by reusing one query embedding for both collections
- then test answer quality with an LLM on top of the best retrieval setup
