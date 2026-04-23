# Milvus Benchmark Plan — docs-agent

## Context

Branch: `test/milvus-hybrid-search`  
Project: `docs-agent`  
Milvus: running on Kubernetes in `docs-ag`  
Canonical source collection: `docs_rag`

This file was rewritten after reviewing:

- `src/`
- existing eval scripts
- local Milvus docs in `milvus docs/`
- current Milvus documentation on hybrid search, sparse search, BM25, BGE-M3, and SPLADE

## What Was Wrong Before

The earlier comparison plan was not clean enough for the decision you actually want to make.

Problems in the old approach:

- `scripts/eval_comparison.py` did **not** compare separate physical collections. It compared filters on the same collection.
- The earlier metrics were based mostly on `file_name`, which is unreliable for repeated names like `kustomization.yaml`.
- The current `docs_rag` collection may not be a clean benchmark baseline because it can drift from the latest schema/index code.
- Partition-key performance was being discussed without fully encoding the Milvus rule that partition-key isolation only helps when the filter contains one specific partition-key value.

## What The Codebase Actually Does

Current production-style path:

- `src/milvus/collection.py`
  - Creates a hybrid collection schema with:
    - `source_type` as partition key
    - `dense_vector`
    - `sparse_vector`
    - optional BM25 built-in function
  - Now supports:
    - partitioned vs non-partitioned collections
    - optional BM25 built-in sparse generation

- `src/ingestion/ingest.py`
  - Parses code/docs
  - computes one dense embedding per chunk
  - upserts into Milvus

- `src/mcp/hybrid_server.py`
  - runs dense + BM25 hybrid retrieval with `RRFRanker`
  - uses metadata filtering for `source_type`

Important repo clarification:

- `scripts/eval_comparison.py` is now marked **legacy**.
- The new benchmark entrypoint is `scripts/eval_topology_models.py`.

## Milvus Findings That Affect The Benchmark

From local docs in `milvus docs/use_partition_key.md`:

- Partition key routes entities by hashing the partition-key value into auto-created partitions.
- Default partition count for partition-key collections is 16 unless configured otherwise.
- Search speedup only happens when the query includes a partition-key filter.
- Partition-key isolation is stronger than normal partition-key routing, but:
  - it requires `properties={"partitionkey.isolation": True}`
  - it applies only with `HNSW`
  - the filter must contain **one specific partition-key value**

Implication:

- For your `source_type in {code, docs}` setup, partition-key isolation can help:
  - code-only queries
  - docs-only queries
- It does **not** fully help mixed queries that need both code and docs.

## Web-Research Conclusions

### Better general dense model than MPNet

Recommended upgrade for a modest, low-risk general retrieval baseline:

- `BAAI/bge-base-en-v1.5`

Reason:

- It is a retrieval-oriented embedding model.
- It is still a practical base-size model.
- It is a cleaner “slightly better than MPNet” step than jumping to a large or multi-function model.

### Sparse search beyond BM25

BM25 remains the correct default baseline in Milvus for this project.

Reason:

- Milvus documentation still recommends BM25 first for simplicity and server-side operational ease.
- BM25 built-in does not require client-side corpus management.
- It is a strong lexical baseline for identifiers, CRD fields, YAML keys, and exact configuration strings.

Sparse candidates that were considered but are now out of immediate scope:

1. `BAAI/bge-m3` sparse output
2. `SPLADE`

Decision:

- Do not test these now.
- The current goal is topology/model choice: single partitioned collection vs separate collections vs CodeBERT.
- BM25 is enough as the sparse baseline for that decision.
- SPLADE is supported, but Milvus docs explicitly suggest BM25 over SPLADE for simplicity unless quality evaluation justifies the extra cost.

### Can Milvus do hybrid search without BM25?

Yes.

Milvus hybrid search is not tied to BM25. You can do hybrid search with:

- dense + BM25
- dense + external sparse embeddings
- dense + dense
- multiple vector fields with one ranker

What Milvus does **not** do by default:

- It does not auto-generate non-BM25 sparse vectors on the server for you.

So if you want non-BM25 sparse retrieval, you need to generate sparse vectors client-side and store them in a sparse vector field.

## Revised Benchmark Matrix

All benchmark collections are rebuilt from the same canonical export of `docs_rag`.

This avoids:

- data drift
- schema drift
- unfair comparisons between legacy and rebuilt collections

## Current Decision

Main benchmark question:

- Is one Milvus collection with `source_type` partitioning better?
- Or are separate physical collections better?
- Does a separate CodeBERT code collection help?

Answer from `eval/eval_topology_models_20260423_172644.json`:

- Superseded by corrected result file: `eval/eval_topology_models_20260423_175954.json`
- The older `172644` file had an `A`-setup filter bug:
  - scalar filter was passed as top-level `filter` to `hybrid_search`
  - PyMilvus expects scalar expression on each `AnnSearchRequest` as `expr`
  - `scripts/eval_topology_models.py` is now fixed

Answer from corrected result file:

- Best tested setup remains `B hybrid`
  - separate `rag_code` and `rag_docs`
  - MPNet dense embeddings for both
  - Milvus BM25 sparse
  - client-side RRF fusion for mixed queries
- Best single-collection setup: `A hybrid`
  - one `rag_all` collection
  - `source_type` partition key with partition-key isolation
  - MPNet dense + Milvus BM25
- CodeBERT setup `C` is not recommended.
  - `C hybrid` was much worse than `B hybrid`
  - `C dense` was the weakest dense setup
  - CodeBERT did not improve code retrieval in this corpus/eval

Production recommendation:

- Use `B hybrid` if retrieval quality/source purity matters most.
- Use `A hybrid` only if operational simplicity matters more than quality.
- Do not choose CodeBERT split (`C`) based on current evidence.
- Full report: `eval/milvus_topology_benchmark_report.md`

### Topology + Dense Model Track

- `A`
  - collection: `rag_all`
  - topology: single collection with partition key
  - dense model: `sentence-transformers/all-mpnet-base-v2`
  - sparse model: Milvus BM25 built-in

- `A_plus`
  - collection: `rag_all_bge`
  - topology: single collection with partition key
  - dense model: `BAAI/bge-base-en-v1.5`
  - sparse model: Milvus BM25 built-in

- `B`
  - collections: `rag_code`, `rag_docs`
  - topology: split physical collections
  - dense model: MPNet for both
  - sparse model: Milvus BM25 built-in

- `B_plus`
  - collections: `rag_code_bge`, `rag_docs_bge`
  - topology: split physical collections
  - dense model: BGE-base for both
  - sparse model: Milvus BM25 built-in

- `C`
  - collections: `rag_code_codebert`, `rag_docs_bge`
  - topology: split physical collections
  - dense model:
    - code: `microsoft/codebert-base`
    - docs: `BAAI/bge-base-en-v1.5`
  - sparse model: Milvus BM25 built-in

### Sparse Track

This is now out of scope for the immediate decision.

Reason:

- The main question is whether a single Milvus collection with `source_type` partitioning is better than separate code/docs collections, especially with CodeBERT for code.
- The existing benchmark already answers that question.
- Adding `BGE-M3` sparse or other non-BM25 sparse variants would answer a different question: BM25 vs learned sparse retrieval.

Keep BM25 as the sparse baseline for this decision. Do not add `BGE-M3` sparse unless the benchmark goal changes.

## Benchmark Script

Primary script:

- `scripts/eval_topology_models.py`

What it does:

- exports canonical records from `docs_rag`
- rebuilds benchmark collections from those records
- benchmarks:
  - `dense`
  - `sparse`
  - `hybrid`
- reports:
  - MRR
  - Recall
  - nDCG
  - source purity
  - source coverage
  - embedding latency
  - DB latency
  - end-to-end latency

Important benchmark behavior:

- Partitioned setups use `source_type == "code"` or `source_type == "docs"` filters for routed queries.
- Mixed queries do not get a single partition-key-value optimization.
- Split setups search one or both physical collections and fuse cross-collection results with client-side RRF.
- Split setups now search a larger candidate pool before fusion so mixed-query comparisons are not artificially starved.
- Sparse-only timing no longer pays dense query-embedding cost.
- The script now fails fast if a benchmark case resolves to zero relevant documents in the canonical export.

## Files Changed

- `src/milvus/collection.py`
  - generalized collection creation
  - supports partitioned/non-partitioned collections
  - supports optional BM25 built-in function

- `scripts/eval_topology_models.py`
  - new benchmark harness for `A`, `A_plus`, `B`, `B_plus`, `C`

- `scripts/eval_comparison.py`
  - marked as legacy so it is not confused with the real topology benchmark

## TODO

### Done

- [x] Review `src` and current Milvus integration
- [x] Verify what the old eval actually measured
- [x] Read local Milvus partition docs
- [x] Research Milvus hybrid and sparse options
- [x] Add `A_plus` to the benchmark design
- [x] Add reusable collection support for benchmark-specific schemas
- [x] Implement the new topology benchmark script
- [x] Separate embedding latency from DB latency
- [x] Fix sparse-only latency accounting
- [x] Add strict empty-ground-truth validation
- [x] Increase split-collection candidate pool before fusion
- [x] Add benchmark embedding cache and visible build/upsert progress
- [x] Run full topology benchmark against live Milvus
- [x] Expand benchmark from 7 cases to 14 cases and rerun query-only benchmark

### Next

- [ ] Review result quality for false positives in the current relevance rules
- [ ] Manually inspect representative false positives/false negatives from `A hybrid`, `B hybrid`, and `C hybrid`
- [ ] Decide production retrieval topology from current benchmark: `B hybrid` vs `A hybrid`
- [ ] If needed, add reranker experiments after retrieval benchmarking is stable

## Live Execution Status

Current operational state during live benchmark execution:

- Milvus was not initially benchmark-ready.
- The `my-release-milvus-standalone` pod was crashing because it could not initialize etcd from the DNS-based endpoint in the ConfigMap.
- The Milvus ConfigMap was updated to use direct etcd and MinIO endpoint IPs so the service could start despite cluster DNS issues.
- Milvus is now running and reachable through local `kubectl port-forward`.
- The first benchmark attempt was intentionally stopped because the canonical source collection only contained `code` chunks, which would have produced an incomplete benchmark.
- The Kubeflow website docs corpus was fetched to `/tmp/kubeflow-website-src`.
- Docs ingestion into `docs_rag` completed successfully.
- The full topology benchmark was started against the rebuilt canonical `code + docs` corpus.
- Last confirmed benchmark milestone before shutdown planning:
  - canonical export succeeded with `3151` records
  - setup `A` completed: `[BUILD] A: {'rag_all': 3151}`
  - setup `A_plus` started
  - later output confirmed BGE load and collection creation: `[SETUP] Created collection: rag_all_bge with partition key isolation`
- At the last live check on `Thu Apr 23 00:44:38 IST 2026`:
  - Milvus port-forward process was still active
  - benchmark process PID `354989` was still running
  - process CPU usage was ~`98.6%`
  - benchmark had not yet reached printed case-evaluation output, so it was still in the collection build / embedding phase
- After machine restart on `Thu Apr 23 2026`, the old benchmark process and port-forward were gone.
- Cluster recovery was needed before benchmark rerun:
  - kind node hit `CreateContainerError` from stale containerd name reservations
  - kubelet also failed with `inotify_init: too many open files`
  - fixed by raising inotify limits inside `docs-agent-control-plane`
  - restarted `containerd` and `kubelet`
  - stopped stale runtime containers without deleting Milvus pods first
  - patched Milvus ConfigMap again because MinIO pod IP changed from `10.244.0.5` to `10.244.0.3`
  - restarted only `my-release-milvus-standalone`
- Milvus verified healthy after recovery:
  - `docs_rag` total `3151`
  - `code` count `1380`
  - `docs` count `1771`
- Full benchmark completed after fixing bad ground-truth rule:
  - original `code_scale_to_zero` and `docs_scale_to_zero` rules used `scaletozerograceperiod`, which does not exist in the canonical corpus
  - rules were corrected to `scale to zero`, which exists in both code and docs
  - `scripts/eval_topology_models.py` now validates all ground-truth cases before expensive collection builds
  - final result file: `eval/eval_topology_models_20260423_170730.json`
- Expanded benchmark completed with 14 scenarios:
  - added code-only KServe ServingRuntime and pipeline runner RBAC scenarios
  - added docs-only Katib early stopping and Pipelines multi-user scenarios
  - added mixed InferenceService, SparkApplication, and Model Registry scenarios
  - final expanded result file: `eval/eval_topology_models_20260423_172644.json`
- Corrected benchmark rerun completed after fixing partition filter placement:
  - fixed `scripts/eval_topology_models.py` to put filter expressions on each `AnnSearchRequest(expr=...)`
  - corrected result file: `eval/eval_topology_models_20260423_175954.json`
  - full written report: `eval/milvus_topology_benchmark_report.md`
  - `B hybrid`: MRR `0.8988`, recall `0.7560`, nDCG `0.7612`, source coverage `1.0`, mean latency `189.97ms`
  - `A hybrid`: MRR `0.8810`, recall `0.7310`, nDCG `0.7301`, source coverage `0.8929`, mean latency `130.07ms`
  - `C hybrid`: MRR `0.6155`, recall `0.6095`, nDCG `0.5571`, source coverage `1.0`, mean latency `180.15ms`
  - conclusion unchanged: choose split MPNet+BM25 hybrid for best retrieval quality; avoid CodeBERT
- Scope correction after user clarification:
  - Do not spend time on `BGE-M3` sparse/non-BM25 benchmark now.
  - The benchmark objective is the topology/model decision:
    - single collection + `source_type` partition key
    - separate physical collections
    - CodeBERT code collection vs general dense model
  - Current result already answers this objective.
  - Next work should be interpretation, false-positive review, and production recommendation, not new sparse-model implementation.

## Immediate TODO

These are the next concrete tasks in execution order:

- [x] Finish docs ingestion into `docs_rag`
- [x] Verify canonical source counts show both `code` and `docs`
- [x] If the machine is shut down, rerun `scripts/eval_topology_models.py` from scratch against the rebuilt full source collection
- [x] Inspect the benchmark output for:
  - [x] setup-level metrics sanity
  - [x] case-level relevance sanity
  - [x] latency sanity across `embed_latency_ms`, `db_latency_ms`, and `total_latency_ms`
- [x] Record the final benchmark outcome in this file
- [ ] Manually review relevance rules where relevant doc count is only `1`
- [ ] Decide whether production default should be `B hybrid` or `A hybrid` after reviewing operational complexity

## Operational Notes

- There is an active Milvus workaround in the cluster config right now:
  - etcd endpoint uses direct pod IP
  - MinIO endpoint uses direct pod IP
- This workaround was necessary to unblock benchmarking because cluster DNS resolution is failing in `docs-ag`.
- After the benchmark is complete, the Milvus deployment should be cleaned up properly by replacing this workaround with a durable fix for cluster DNS or service routing.
- Source verification after docs ingestion:
  - `docs_rag` iterator-derived counts are `1380 code` and `1771 docs`, total `3151`
  - `Collection.num_entities` agrees with `3151` after explicit flush/load
  - `get_collection_stats().row_count` still reported `2880`, so benchmark logic should trust exported records rather than stats
- The live canonical collection schema currently describes `dense_vector` as `BAAI/bge-base-en-v1.5 embedding`.
  - This does not invalidate the benchmark because `scripts/eval_topology_models.py` exports raw records and rebuilds every benchmark collection from scratch with the target model for each setup.
- If the in-progress benchmark run is interrupted by shutdown:
  - do not assume partial benchmark collections are complete
  - prefer restarting the benchmark cleanly
  - avoid `--reuse-bench-collections` unless each benchmark collection is explicitly verified first
- `scripts/eval_topology_models.py` now writes document embedding caches under `eval/embedding_cache` by default.
  - Cache key includes model name, ordered `chunk_id`, and content hash.
  - If a future run is interrupted after embeddings are cached, rerun can reuse embeddings instead of recomputing that model/corpus pair.
  - Use `--no-embedding-cache` only when intentionally measuring cold embedding rebuild cost.
- Benchmark result summary from `eval/eval_topology_models_20260423_170730.json`:
  - best overall retrieval quality: `B hybrid`
  - `B hybrid`: avg MRR `0.8929`, avg recall `0.6643`, avg nDCG `0.7206`, source purity `1.0`, source coverage `1.0`, mean total latency `158.8ms`
  - best single-collection result: `A hybrid`
  - `A hybrid`: avg MRR `0.8143`, avg recall `0.6571`, avg nDCG `0.6438`, source purity `0.8189`, source coverage `0.8571`, mean total latency `107.38ms`
  - stronger general model did not clearly beat MPNet in this corpus:
    - `A_plus hybrid`: avg MRR `0.6476`, avg recall `0.5476`, avg nDCG `0.5274`
    - `B_plus hybrid`: avg MRR `0.7143`, avg recall `0.5595`, avg nDCG `0.5769`
  - `CodeBERT` split setup underperformed for dense/hybrid:
    - `C dense`: avg MRR `0.3776`, avg recall `0.3571`, avg nDCG `0.3332`
    - `C hybrid`: avg MRR `0.5048`, avg recall `0.4952`, avg nDCG `0.474`
  - BM25 sparse remained strong and very fast:
    - split sparse variants: avg MRR `0.6587`, avg recall `0.5595`, avg nDCG `0.5674`, mean total latency about `8-10ms`
  - Interpretation: split collections improved routing purity and mixed-source coverage, while MPNet + BM25 hybrid was the strongest tested retrieval configuration.
- Expanded 14-case benchmark summary from `eval/eval_topology_models_20260423_172644.json`:
  - best overall retrieval quality remains `B hybrid`
  - `B hybrid`: avg MRR `0.8988`, avg recall `0.756`, avg nDCG `0.7612`, source purity `1.0`, source coverage `1.0`, mean total latency `177.75ms`
  - best single-collection result remains `A hybrid`
  - `A hybrid`: avg MRR `0.8238`, avg recall `0.7024`, avg nDCG `0.6798`, source purity `0.8827`, source coverage `0.8929`, mean total latency `117.25ms`
  - split BM25 sparse is strong and much faster:
    - `B sparse`: avg MRR `0.7936`, avg recall `0.7274`, avg nDCG `0.6885`, source purity `1.0`, source coverage `1.0`, mean total latency `10.74ms`
  - BGE-base still does not beat MPNet overall:
    - `B_plus hybrid`: avg MRR `0.8571`, avg recall `0.6893`, avg nDCG `0.7008`
    - `B hybrid`: avg MRR `0.8988`, avg recall `0.756`, avg nDCG `0.7612`
  - CodeBERT still underperforms for this benchmark:
    - `C dense`: avg MRR `0.3912`, avg recall `0.4155`, avg nDCG `0.3588`
    - `C hybrid`: avg MRR `0.6155`, avg recall `0.6095`, avg nDCG `0.5571`
  - Updated interpretation: choose `B hybrid` for max retrieval quality, consider `B sparse` for low-latency mode, and keep `A hybrid` only if single-collection operational simplicity matters more than source purity/coverage.

## Commands

### Port-forward Milvus

```bash
kubectl port-forward -n docs-ag svc/my-release-milvus 19530:19530
```

### Verify the canonical source collection

```bash
MILVUS_HOST=127.0.0.1 PYTHONPATH=. uv run python scripts/verify_collection.py
```

### Run the topology benchmark

```bash
MILVUS_HOST=127.0.0.1 PYTHONPATH=. uv run python scripts/eval_topology_models.py
```

### Run with explicit candidate expansion

```bash
MILVUS_HOST=127.0.0.1 PYTHONPATH=. uv run python scripts/eval_topology_models.py --candidate-multiplier 3
```

### Reuse already-built benchmark collections

```bash
MILVUS_HOST=127.0.0.1 PYTHONPATH=. uv run python scripts/eval_topology_models.py --reuse-bench-collections
```

### Limit to specific setups

```bash
MILVUS_HOST=127.0.0.1 PYTHONPATH=. uv run python scripts/eval_topology_models.py --setups A A_plus B B_plus C
```

## Notes

- `A_plus` and `B_plus` use `BAAI/bge-base-en-v1.5` and should prepend the retrieval instruction on queries.
- `C` uses `CodeBERT` only for the code side.
- The current benchmark keeps BM25 as the sparse baseline on purpose.
- The sparse-track decision should be made only after running a proper BM25 vs `BGE-M3` comparison on the same canonical corpus.
- After user clarification, skip `BGE-M3` sparse for now. It is not needed to answer single partitioned collection vs separate collections vs CodeBERT.
- The current output now separates `embed_latency_ms`, `db_latency_ms`, and `total_latency_ms`.
