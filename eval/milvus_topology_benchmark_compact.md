# Milvus Topology Benchmark

Date: 2026-04-23  
Corrected result file: `eval/eval_topology_models_20260423_175954.json`

## 1. Question

This benchmark asks:

Which retrieval design is better for docs-agent?

- one Milvus collection with `source_type` partitioning
- or separate physical collections for code and docs

It also checks whether CodeBERT helps on the code side.

## 2. Corpus And Cases

Corpus:

- 1,380 code chunks
- 1,771 docs chunks
- 3,151 total chunks

Benchmark coverage:

- 4 code-only queries
- 4 docs-only queries
- 6 mixed code+docs queries

Human-readable cases:

- Katib controller manifests
- Scale-to-zero configuration
- KServe ServingRuntime resources
- Pipeline runner RBAC
- Katib configuration docs
- KServe scale-to-zero docs
- Katib early stopping docs
- Pipelines multi-user docs
- Storage configuration across code and docs
- InferenceService setup across code and docs
- SparkApplication code plus user guidance
- Model Registry implementation plus docs
- Pipeline DSL code plus docs
- Pipeline component definition across code and docs

## 3. Setup Names

| Setup | Meaning |
|---|---|
| `A` | One collection, partitioned by `source_type`, MPNet dense model |
| `A_plus` | Same as `A`, but BGE-base dense model |
| `B` | Split `rag_code` and `rag_docs`, MPNet dense model |
| `B_plus` | Split `rag_code` and `rag_docs`, BGE-base dense model |
| `C` | Split collections, CodeBERT for code, BGE-base for docs |

Search modes:

- `dense`: vector search only
- `sparse`: BM25 only
- `hybrid`: dense + BM25 with RRF fusion

## 4. Metrics

| Metric | Interpretation |
|---|---|
| MRR | How high the first relevant result appears |
| Recall@10 | How many relevant chunks appear in the top 10 |
| nDCG@10 | Quality of the full ranking, not just first hit |
| Source coverage | Whether all expected source types appear in the result set |
| Latency | End-to-end retrieval time |

For mixed queries, source coverage is especially important. A mixed question should return both code and docs, not only one side.

## 5. Main Results

| Setup | Search | MRR | Recall@10 | nDCG@10 | Source coverage | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| `A` | hybrid | 0.8810 | 0.7310 | 0.7301 | 0.8929 | 130.07 ms |
| `B` | hybrid | 0.8988 | 0.7560 | 0.7612 | 1.0000 | 189.97 ms |
| `C` | hybrid | 0.6155 | 0.6095 | 0.5571 | 1.0000 | 180.15 ms |
| `A` | sparse | 0.7936 | 0.6679 | 0.6536 | 0.8929 | 7.27 ms |
| `B` | sparse | 0.7936 | 0.7274 | 0.6885 | 1.0000 | 10.46 ms |

Key takeaways:

- `B hybrid` is the best overall retrieval setup.
- `A hybrid` is close, but weaker on mixed-source reliability.
- `C hybrid` underperforms clearly and should not be used.
- `B sparse` is the strongest fast fallback.

## 6. Best Setup By Objective

| Objective | Best choice | Reason |
|---|---|---|
| Best overall quality | `B hybrid` | Highest hybrid MRR, recall, and nDCG |
| Best mixed-source reliability | `B hybrid` | Perfect source coverage (`1.0000`) |
| Best simple production setup | `A hybrid` | One collection, simpler operations |
| Best lowest-latency mode | `A sparse` | Fastest retrieval overall |
| Best low-latency split mode | `B sparse` | Fast and still has perfect coverage |
| Best code-only search | `A hybrid` and `B hybrid` | Effectively tied in this benchmark |
| Best docs-only search | `B hybrid` | Better docs-only recall and nDCG |
| Setup to avoid | `C hybrid` | CodeBERT hurts retrieval quality too much |

## 7. Why Split Collections Win

The real difference is mixed queries.

In `B`:

- code search runs against `rag_code`
- docs search runs against `rag_docs`
- mixed queries search both
- RRF combines both result lists after each source has had a fair search pass

In `A`:

- code and docs compete in one collection
- for mixed queries, one source type can dominate the ranking
- the other source can disappear from the top 10

That is why `B hybrid` wins:

- `A hybrid` source coverage: `0.8929`
- `B hybrid` source coverage: `1.0000`

This matters for RAG. If a mixed question returns only docs, the LLM may miss implementation details. If it returns only code, it may miss explanation and usage context.

There is also a BM25 advantage:

- code and docs have different token distributions
- keeping them separate gives cleaner lexical ranking
- that is why `B sparse` recall is higher than `A sparse`

## 8. Why Single Collection Is Still Strong

`A hybrid` is not a bad setup. It is the strongest single-collection option.

Strengths:

- simpler deployment
- one collection to manage
- lower hybrid latency
- strong performance for code-only and docs-only search

Weakness:

- mixed queries are less reliable because code and docs share one ranking pool

So the right interpretation is:

- `A hybrid` is better if simplicity matters most
- `B hybrid` is better if retrieval quality for mixed questions matters most

## 9. What If Mixed Search Is Not Needed?

This changes the recommendation.

If your product only does:

- code-only search sometimes
- docs-only search sometimes
- but never a combined code+docs retrieval

then the biggest weakness of `A hybrid` mostly disappears.

Why:

- `A` can filter directly by `source_type`
- once the source is isolated, code and docs do not interfere much
- for code-only retrieval, `A hybrid` and `B hybrid` were effectively tied

Practical recommendation without mixed search:

- choose `A hybrid` if you want the simplest system
- choose `B hybrid` if docs-only quality matters more, or if you want strict operational separation between code and docs

So:

- with mixed search: prefer `B hybrid`
- without mixed search: `A hybrid` becomes much more attractive

## 10. Why CodeBERT Lost

`C` uses CodeBERT on the code side.

It underperformed badly:

- `C hybrid` MRR: `0.6155`
- `C hybrid` recall: `0.6095`
- `C hybrid` nDCG: `0.5571`

Observed pattern:

- CodeBERT often ranked structurally similar YAML above the true target file
- hybrid fusion could not fully recover from weak dense rankings

Conclusion:

- do not use base CodeBERT for this retrieval task
- if code-specific embeddings are tested later, use a retrieval-tuned code model instead

## 11. Final Recommendation

If docs-agent needs code+docs RAG, use **`B hybrid`**.

If docs-agent only needs routed code-only or docs-only search and simplicity matters more, **`A hybrid`** is a valid and strong alternative.

Do not use **`C`**.

