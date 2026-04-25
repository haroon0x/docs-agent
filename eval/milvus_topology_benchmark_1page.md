# Milvus Topology Benchmark Summary

Date: 2026-04-23
Corrected result: `eval/eval_topology_models_20260423_175954.json`

## Goal

Test which retrieval design is best for docs-agent:

- one Milvus collection with `source_type` partitioning
- or separate code/docs collections
- and whether CodeBERT helps code retrieval

## What Was Tested

Corpus:

- 1,380 code chunks
- 1,771 docs chunks
- 3,151 total chunks

Queries:

- 4 code-only
- 4 docs-only
- 6 mixed code+docs

Setups:

- `A`: one collection, partitioned by `source_type`, MPNet
- `A_plus`: one collection, partitioned, BGE-base
- `B`: split code/docs collections, MPNet
- `B_plus`: split code/docs collections, BGE-base
- `C`: split collections, CodeBERT for code

## Main Metrics

- `MRR`: first relevant hit rank
- `Recall@10`: how many known relevant chunks appear in top 10
- `nDCG@10`: ranking quality with top-rank reward
- `Source coverage`: whether both expected source types appear
- `Latency`: end-to-end retrieval time

## Bottom Line

| Setup | Hybrid MRR | Hybrid Recall@10 | Hybrid nDCG@10 | Coverage | Mean latency |
|---|---:|---:|---:|---:|---:|
| `A` | 0.8810 | 0.7310 | 0.7301 | 0.8929 | 130.07 ms |
| `B` | 0.8988 | 0.7560 | 0.7612 | 1.0000 | 189.97 ms |
| `C` | 0.6155 | 0.6095 | 0.5571 | 1.0000 | 180.15 ms |

## Conclusion

Best retrieval design is `B hybrid`:

- separate code and docs collections
- MPNet dense embeddings
- Milvus BM25 sparse search
- RRF fusion

Why it wins:

- mixed questions need both code and docs in the result set
- split collections prevent one source from crowding out the other
- BM25 behaves cleaner on separated corpora
- CodeBERT underperforms badly on this data

Single collection with partitioning (`A`) is still useful:

- simpler to run
- faster
- good on single-source queries
- but weaker on mixed code+docs coverage

## Recommendation

- Use `B hybrid` for best retrieval quality
- Use `A hybrid` only if simplicity matters more than quality
- Do not use `C`

