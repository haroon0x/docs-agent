"""
Benchmark Milvus retrieval topologies and embedding-model variants.

This script uses the current `docs_rag` collection as the canonical corpus, then
rebuilds benchmark collections from the same records so the comparison isolates:

  A       - single collection with partition key + MPNet
  A_plus  - single collection with partition key + BGE-base
  B       - split collections (code/docs) + MPNet
  B_plus  - split collections (code/docs) + BGE-base
  C       - split collections with BGE-base for docs and CodeBERT for code

The sparse baseline remains Milvus built-in BM25. The collection utility now
supports non-BM25 sparse generation for future sparse-track extensions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import AnnSearchRequest, RRFRanker

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.milvus.collection import COLLECTION_NAME, EMBEDDING_MODEL, get_client, setup_collection

EVAL_DIR = Path(__file__).parent.parent / "eval"
DEFAULT_EMBEDDING_CACHE_DIR = EVAL_DIR / "embedding_cache"
EXPORT_FIELDS = [
    "chunk_id",
    "file_unique_id",
    "repo_name",
    "file_path",
    "file_name",
    "citation_url",
    "chunk_index",
    "content_text",
    "source_type",
    "last_updated",
]
SEARCH_OUTPUT_FIELDS = ["chunk_id", "file_path", "file_name", "source_type", "citation_url", "content_text"]
UPSERT_BATCH_SIZE = 256
RRF_K = 60


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    family: str
    query_prefix: str = ""


@dataclass(frozen=True)
class RelevanceRule:
    source_type: str
    file_names: tuple[str, ...] = ()
    path_contains: tuple[str, ...] = ()
    content_all: tuple[str, ...] = ()
    content_any: tuple[str, ...] = ()

    def matches(self, record: dict[str, Any]) -> bool:
        if record.get("source_type") != self.source_type:
            return False

        file_name = str(record.get("file_name", ""))
        file_path = str(record.get("file_path", "")).lower()
        content = str(record.get("content_text", "")).lower()

        if self.file_names and file_name not in self.file_names:
            return False
        if self.path_contains and not all(fragment in file_path for fragment in self.path_contains):
            return False
        if self.content_all and not all(term in content for term in self.content_all):
            return False
        if self.content_any and not any(term in content for term in self.content_any):
            return False
        return True


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    query: str
    description: str
    route_sources: tuple[str, ...]
    rules: tuple[RelevanceRule, ...]


@dataclass(frozen=True)
class SearchExecution:
    hits: list[dict[str, Any]]
    embed_latency_ms: float
    db_latency_ms: float
    total_latency_ms: float


@dataclass(frozen=True)
class SetupSpec:
    key: str
    description: str
    topology: str
    collections: dict[str, str]
    models: dict[str, ModelSpec]


class SentenceTransformerEmbedder:
    def __init__(self, spec: ModelSpec, *, device: str = "cpu"):
        self.spec = spec
        self.device = device
        self._model = None
        self.dimension = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.spec.model_name, device=self.device)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _prepare(self, texts: list[str], *, is_query: bool) -> list[str]:
        if is_query and self.spec.query_prefix:
            return [self.spec.query_prefix + text for text in texts]
        return texts

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        values = model.encode(
            self._prepare(texts, is_query=False),
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return values.tolist()

    def encode_query(self, text: str) -> list[float]:
        model = self._load()
        value = model.encode(
            self._prepare([text], is_query=True),
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return value[0].tolist()


class CodeBertEmbedder:
    def __init__(self, spec: ModelSpec, *, device: str = "cpu"):
        self.spec = spec
        self.device = device
        self.dimension = None
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "CodeBERT requires `transformers` and `torch` in the environment."
                ) from exc

            actual_device = self.device
            if actual_device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)
            self._model = AutoModel.from_pretrained(self.spec.model_name)
            self._model.to(actual_device)
            self._model.eval()
            self.device = actual_device
            self.dimension = int(self._model.config.hidden_size)
        return self._model, self._tokenizer, self._torch

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model, tokenizer, torch = self._load()
        outputs: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = texts[start:start + 16]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                last_hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.extend(pooled.cpu().tolist())
        return outputs

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


MODELS = {
    "mpnet": ModelSpec(
        key="mpnet",
        model_name=EMBEDDING_MODEL,
        family="sentence_transformer",
    ),
    "bge_base": ModelSpec(
        key="bge_base",
        model_name="BAAI/bge-base-en-v1.5",
        family="sentence_transformer",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "codebert": ModelSpec(
        key="codebert",
        model_name="microsoft/codebert-base",
        family="codebert",
    ),
}

SETUPS = {
    "A": SetupSpec(
        key="A",
        description="Single collection with partition key and shared MPNet dense model",
        topology="partitioned",
        collections={"all": "rag_all"},
        models={"all": MODELS["mpnet"]},
    ),
    "A_plus": SetupSpec(
        key="A_plus",
        description="Single collection with partition key and stronger general dense model",
        topology="partitioned",
        collections={"all": "rag_all_bge"},
        models={"all": MODELS["bge_base"]},
    ),
    "B": SetupSpec(
        key="B",
        description="Split code/docs collections with shared MPNet dense model",
        topology="split",
        collections={"code": "rag_code", "docs": "rag_docs"},
        models={"code": MODELS["mpnet"], "docs": MODELS["mpnet"]},
    ),
    "B_plus": SetupSpec(
        key="B_plus",
        description="Split code/docs collections with stronger shared dense model",
        topology="split",
        collections={"code": "rag_code_bge", "docs": "rag_docs_bge"},
        models={"code": MODELS["bge_base"], "docs": MODELS["bge_base"]},
    ),
    "C": SetupSpec(
        key="C",
        description="Split collections with BGE-base for docs and CodeBERT for code",
        topology="split",
        collections={"code": "rag_code_codebert", "docs": "rag_docs_bge"},
        models={"code": MODELS["codebert"], "docs": MODELS["bge_base"]},
    ),
}

BENCHMARK_CASES = [
    BenchmarkCase(
        case_id="code_katib_controller",
        query="katib controller deployment",
        description="Code-only routing: Katib controller manifests",
        route_sources=("code",),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("controller.yaml", "rbac.yaml", "service.yaml", "trial-templates.yaml"),
                path_contains=("applications/katib/upstream/components/controller",),
            ),
            RelevanceRule(
                source_type="code",
                content_all=("katib-controller",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="code_scale_to_zero",
        query="Knative serving scale to zero configuration",
        description="Code-only routing: Knative/KServe scale-to-zero config",
        route_sources=("code",),
        rules=(
            RelevanceRule(
                source_type="code",
                content_all=("scale to zero",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="code_kserve_servingruntime",
        query="KServe ServingRuntime cluster resources",
        description="Code-only routing: KServe ServingRuntime manifests",
        route_sources=("code",),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("kserve.yaml", "kserve_kubeflow.yaml", "kserve-cluster-resources.yaml"),
                content_all=("servingruntime",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="code_pipeline_runner_rbac",
        query="Kubeflow Pipelines pipeline runner RBAC service account",
        description="Code-only routing: pipeline runner RBAC manifests",
        route_sources=("code",),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("pipeline-runner-role.yaml", "pipeline-runner-rolebinding.yaml", "pipeline-runner-sa.yaml"),
                content_all=("pipeline", "runner"),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="docs_katib_config",
        query="What is Katib and how do you configure it?",
        description="Docs-only routing: Katib conceptual and config docs",
        route_sources=("docs",),
        rules=(
            RelevanceRule(
                source_type="docs",
                file_names=("katib-config.md", "architecture.md", "installation.md"),
                content_any=("katib", "experiment", "early stopping"),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="docs_scale_to_zero",
        query="How does scale to zero work in KServe?",
        description="Docs-only routing: KServe scale-to-zero docs",
        route_sources=("docs",),
        rules=(
            RelevanceRule(
                source_type="docs",
                content_all=("scale to zero",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="docs_katib_early_stopping",
        query="How does Katib early stopping work for experiments?",
        description="Docs-only routing: Katib early stopping guidance",
        route_sources=("docs",),
        rules=(
            RelevanceRule(
                source_type="docs",
                file_names=("architecture.md", "configure-experiment.md", "early-stopping.md", "katib-ui.md"),
                content_all=("experiment", "early stopping"),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="docs_pipelines_multi_user",
        query="How does Kubeflow Pipelines multi-user isolation work?",
        description="Docs-only routing: KFP multi-user docs",
        route_sources=("docs",),
        rules=(
            RelevanceRule(
                source_type="docs",
                file_names=("multi-user.md", "connect-api.md"),
                content_all=("multi-user", "pipeline"),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="mixed_storage",
        query="PVC volume mount storage config",
        description="Mixed routing: storage configs across code and docs",
        route_sources=("code", "docs"),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("seaweedfs-pvc.yaml", "metrics-server_resource_table.py"),
                content_any=("persistentvolumeclaim", "volumemounts", "claimname"),
            ),
            RelevanceRule(
                source_type="docs",
                file_names=("manipulate-resources.md", "platform-specific-features.md"),
                content_any=("persistent volume claim", "volume mount", "storage"),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="mixed_kserve_inferenceservice",
        query="KServe InferenceService predictor configuration",
        description="Mixed routing: InferenceService manifests and docs",
        route_sources=("code", "docs"),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("kserve.yaml", "kserve_kubeflow.yaml"),
                content_all=("inferenceservice", "predictor"),
            ),
            RelevanceRule(
                source_type="docs",
                file_names=("webapp.md", "getting-started.md"),
                content_all=("inferenceservice",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="mixed_spark_application",
        query="SparkApplication resources and user guide",
        description="Mixed routing: SparkApplication CRD/manifests and docs",
        route_sources=("code", "docs"),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("resources.yaml", "aggregated-roles.yaml"),
                content_all=("sparkapplication",),
            ),
            RelevanceRule(
                source_type="docs",
                file_names=("writing-sparkapplication.md", "getting-started.md", "notebooks-spark-operator.md"),
                content_all=("sparkapplication",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="mixed_model_registry",
        query="Kubeflow Model Registry service and docs",
        description="Mixed routing: Model Registry service manifest and docs",
        route_sources=("code", "docs"),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("model-registry-service.yaml",),
                content_all=("model registry",),
            ),
            RelevanceRule(
                source_type="docs",
                file_names=("overview.md", "getting-started.md"),
                content_all=("model registry",),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="mixed_pipeline_dsl",
        query="kubeflow pipeline DSL syntax",
        description="Mixed routing: DSL examples in code and docs",
        route_sources=("code", "docs"),
        rules=(
            RelevanceRule(
                source_type="code",
                file_names=("pipeline_run_and_wait_kubeflow.py",),
                content_any=("@dsl.pipeline", "kfp.dsl", "dsl.pipeline"),
            ),
            RelevanceRule(
                source_type="docs",
                file_names=("pipeline.md", "artifacts.md", "getting-started.md", "component-spec.md"),
                content_any=("dsl.pipeline", "kubeflow pipelines", "component"),
            ),
        ),
    ),
    BenchmarkCase(
        case_id="mixed_component_definition",
        query="KFP pipeline component definition",
        description="Mixed routing: component implementation and docs",
        route_sources=("code", "docs"),
        rules=(
            RelevanceRule(
                source_type="code",
                content_any=("@component", "dsl.component"),
            ),
            RelevanceRule(
                source_type="docs",
                file_names=("component-development.md", "component-spec.md", "pipeline.md"),
                content_any=("component", "pipeline"),
            ),
        ),
    ),
]


_EMBEDDER_CACHE: dict[str, Any] = {}


def get_embedder(model: ModelSpec, *, device: str) -> Any:
    cache_key = f"{model.key}:{device}"
    if cache_key not in _EMBEDDER_CACHE:
        if model.family == "sentence_transformer":
            _EMBEDDER_CACHE[cache_key] = SentenceTransformerEmbedder(model, device=device)
        elif model.family == "codebert":
            _EMBEDDER_CACHE[cache_key] = CodeBertEmbedder(model, device=device)
        else:
            raise ValueError(f"Unsupported model family: {model.family}")
    return _EMBEDDER_CACHE[cache_key]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def canonical_doc_id(record: dict[str, Any]) -> str:
    return f"{record.get('source_type', '')}:{record.get('file_path', '')}"


def hit_entity(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        return dict(hit.get("entity", {}))
    entity = getattr(hit, "entity", {})
    if entity is None:
        return {}
    return dict(entity)


def hit_distance(hit: Any) -> float:
    if isinstance(hit, dict):
        return float(hit.get("distance", hit.get("score", 0.0)))
    return float(getattr(hit, "distance", getattr(hit, "score", 0.0)))


def normalize_hits(raw_hits: list[Any], *, label: str) -> list[dict[str, Any]]:
    normalized = []
    for rank, raw in enumerate(raw_hits, start=1):
        entity = hit_entity(raw)
        if "chunk_id" not in entity and "id" in entity:
            entity["chunk_id"] = entity["id"]
        normalized.append(
            {
                "rank": rank,
                "distance": hit_distance(raw),
                "label": label,
                "entity": entity,
            }
        )
    return normalized


def dedupe_hits_by_doc(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for hit in hits:
        doc_id = canonical_doc_id(hit["entity"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(hit)
    return deduped


def dcg_at_k(hits: list[dict[str, Any]], relevant_doc_ids: set[str], k: int) -> float:
    score = 0.0
    for idx, hit in enumerate(hits[:k], start=1):
        rel = 1.0 if canonical_doc_id(hit["entity"]) in relevant_doc_ids else 0.0
        if rel:
            score += rel / math.log2(idx + 1)
    return score


def evaluate_hits(
    hits: list[dict[str, Any]],
    *,
    relevant_doc_ids: set[str],
    route_sources: tuple[str, ...],
    k: int,
) -> dict[str, Any]:
    docs = dedupe_hits_by_doc(hits)
    top_k = docs[:k]

    reciprocal_rank = 0.0
    for idx, hit in enumerate(top_k, start=1):
        if canonical_doc_id(hit["entity"]) in relevant_doc_ids:
            reciprocal_rank = 1.0 / idx
            break

    retrieved_relevant = {
        canonical_doc_id(hit["entity"])
        for hit in top_k
        if canonical_doc_id(hit["entity"]) in relevant_doc_ids
    }
    recall = None
    ndcg = None
    if relevant_doc_ids:
        recall = len(retrieved_relevant) / len(relevant_doc_ids)
        ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(len(relevant_doc_ids), k) + 1))
        ndcg = dcg_at_k(top_k, relevant_doc_ids, k) / ideal if ideal else None

    observed_sources = [hit["entity"].get("source_type", "") for hit in top_k]
    expected_sources = set(route_sources)
    purity = None
    if top_k and expected_sources:
        purity = sum(1 for source in observed_sources if source in expected_sources) / len(top_k)

    coverage = None
    if expected_sources:
        coverage = len(set(observed_sources) & expected_sources) / len(expected_sources)

    return {
        "mrr": round(reciprocal_rank, 4),
        "recall": round(recall, 4) if recall is not None else None,
        "ndcg": round(ndcg, 4) if ndcg is not None else None,
        "source_purity": round(purity, 4) if purity is not None else None,
        "source_coverage": round(coverage, 4) if coverage is not None else None,
        "top_hits": [
            {
                "rank": hit["rank"],
                "source_type": hit["entity"].get("source_type"),
                "file_name": hit["entity"].get("file_name"),
                "file_path": hit["entity"].get("file_path"),
                "distance": round(hit["distance"], 5),
            }
            for hit in top_k[:5]
        ],
    }


def summarize_latency(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(statistics.mean(values), 2) if values else None,
        "p50": round(percentile(values, 0.50), 2) if values else None,
        "p95": round(percentile(values, 0.95), 2) if values else None,
    }


def preview_relevant_docs(relevant_doc_ids: set[str], *, limit: int = 5) -> list[str]:
    return sorted(relevant_doc_ids)[:limit]


def export_canonical_records(client, collection_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    iterator = None
    try:
        iterator = client.query_iterator(
            collection_name=collection_name,
            batch_size=1000,
            limit=-1,
            filter='chunk_id != ""',
            output_fields=EXPORT_FIELDS,
        )
        while True:
            batch = iterator.next()
            if not batch:
                break
            for record in batch:
                records.append({field: record.get(field) for field in EXPORT_FIELDS})
    except Exception:
        if iterator is not None:
            try:
                iterator.close()
            except Exception:
                pass
        raw = client.query(
            collection_name=collection_name,
            filter='chunk_id != ""',
            output_fields=EXPORT_FIELDS,
        )
        records = [{field: record.get(field) for field in EXPORT_FIELDS} for record in raw]
        return records
    else:
        if iterator is not None:
            iterator.close()
    return records


def embedding_cache_key(records: list[dict[str, Any]], embedder: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(embedder.spec.model_name.encode("utf-8"))
    hasher.update(b"\0")
    for record in records:
        hasher.update(str(record["chunk_id"]).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(str(record["content_text"]).encode("utf-8")).digest())
        hasher.update(b"\0")
    return hasher.hexdigest()[:24]


def load_cached_embeddings(cache_dir: Path, records: list[dict[str, Any]], embedder: Any) -> list[list[float]] | None:
    cache_path = cache_dir / f"{embedder.spec.key}_{embedding_cache_key(records, embedder)}.npz"
    if not cache_path.exists():
        return None

    import numpy as np

    cached = np.load(cache_path, allow_pickle=False)
    expected_ids = [str(record["chunk_id"]) for record in records]
    cached_ids = [str(item) for item in cached["chunk_ids"].tolist()]
    if cached_ids != expected_ids:
        print(f"[CACHE] Ignoring stale embedding cache: {cache_path}")
        return None

    print(f"[CACHE] Hit: {cache_path}")
    return cached["vectors"].tolist()


def save_cached_embeddings(cache_dir: Path, records: list[dict[str, Any]], embedder: Any, vectors: list[list[float]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{embedder.spec.key}_{embedding_cache_key(records, embedder)}.npz"
    temp_path = cache_path.with_suffix(".tmp.npz")

    import numpy as np

    chunk_ids = np.array([str(record["chunk_id"]) for record in records])
    vector_array = np.array(vectors, dtype="float32")
    np.savez(temp_path, chunk_ids=chunk_ids, vectors=vector_array)
    temp_path.replace(cache_path)
    print(f"[CACHE] Saved: {cache_path}")


def get_document_embeddings(
    records: list[dict[str, Any]],
    embedder: Any,
    *,
    cache_dir: Path | None,
) -> list[list[float]]:
    if cache_dir is not None:
        cached = load_cached_embeddings(cache_dir, records, embedder)
        if cached is not None:
            return cached

    print(f"[EMBED] Encoding {len(records)} records with {embedder.spec.model_name}")
    vectors = embedder.encode_documents([record["content_text"] for record in records])
    if cache_dir is not None:
        save_cached_embeddings(cache_dir, records, embedder, vectors)
    return vectors


def build_records(records: list[dict[str, Any]], embedder: Any, *, cache_dir: Path | None) -> list[dict[str, Any]]:
    payload = [dict(record) for record in records]
    vectors = get_document_embeddings(payload, embedder, cache_dir=cache_dir)
    for record, vector in zip(payload, vectors):
        record["dense_vector"] = vector
        if len(record["content_text"]) > 7800:
            record["content_text"] = record["content_text"][:7800]
    return payload


def upsert_records(client, collection_name: str, payload: list[dict[str, Any]]) -> int:
    total = 0
    for start in range(0, len(payload), UPSERT_BATCH_SIZE):
        batch = payload[start:start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, data=batch)
        total += len(batch)
        print(f"[UPSERT] {collection_name}: {total}/{len(payload)}", flush=True)
    return total


def prepare_collection(
    records: list[dict[str, Any]],
    collection_name: str,
    embedder: Any,
    *,
    partition_key: bool,
    recreate: bool,
    cache_dir: Path | None,
):
    if embedder.dimension is None:
        embedder._load()

    client = setup_collection(
        recreate=recreate,
        collection_name=collection_name,
        partition_key=partition_key,
        dense_dim=int(embedder.dimension),
        enable_bm25_function=True,
    )
    payload = build_records(records, embedder, cache_dir=cache_dir)
    upserted = upsert_records(client, collection_name, payload)
    try:
        client.load_collection(collection_name)
    except Exception:
        pass
    return upserted


def build_setup_collections(
    setup: SetupSpec,
    canonical_records: list[dict[str, Any]],
    *,
    device: str,
    recreate: bool,
    cache_dir: Path | None,
):
    if setup.topology == "partitioned":
        embedder = get_embedder(setup.models["all"], device=device)
        collection_name = setup.collections["all"]
        return {
            collection_name: prepare_collection(
                canonical_records,
                collection_name,
                embedder,
                partition_key=True,
                recreate=recreate,
                cache_dir=cache_dir,
            )
        }

    stats = {}
    for source in ("code", "docs"):
        records = [record for record in canonical_records if record["source_type"] == source]
        embedder = get_embedder(setup.models[source], device=device)
        collection_name = setup.collections[source]
        stats[collection_name] = prepare_collection(
            records,
            collection_name,
            embedder,
            partition_key=False,
            recreate=recreate,
            cache_dir=cache_dir,
        )
    return stats


def load_setup_collections(client, setup: SetupSpec):
    for collection_name in setup.collections.values():
        try:
            client.load_collection(collection_name)
        except Exception:
            pass


def build_ann_requests(
    query_text: str,
    embedder: Any,
    search_type: str,
    *,
    candidate_limit: int,
    filter_expr: str | None = None,
) -> tuple[list[AnnSearchRequest], float]:
    encode_ms = 0.0
    vector = None
    if search_type in {"dense", "hybrid"}:
        start = time.perf_counter()
        vector = embedder.encode_query(query_text)
        encode_ms = (time.perf_counter() - start) * 1000

    if search_type == "dense":
        reqs = [
            AnnSearchRequest(
                data=[vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=candidate_limit,
                expr=filter_expr,
            )
        ]
    elif search_type == "sparse":
        reqs = [
            AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=candidate_limit,
                expr=filter_expr,
            )
        ]
    elif search_type == "hybrid":
        reqs = [
            AnnSearchRequest(
                data=[vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=candidate_limit,
                expr=filter_expr,
            ),
            AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=candidate_limit,
                expr=filter_expr,
            ),
        ]
    else:
        raise ValueError(f"Unsupported search type: {search_type}")

    return reqs, encode_ms


def run_milvus_search(client, collection_name: str, reqs: list[AnnSearchRequest], *, top_k: int):
    kwargs = {
        "collection_name": collection_name,
        "reqs": reqs,
        "ranker": RRFRanker(k=1 if len(reqs) == 1 else RRF_K),
        "limit": top_k,
        "output_fields": SEARCH_OUTPUT_FIELDS,
    }

    start = time.perf_counter()
    raw = client.hybrid_search(**kwargs)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return normalize_hits(raw, label=collection_name), elapsed_ms


def rrf_fuse(hit_lists: list[list[dict[str, Any]]], *, limit: int, k: int = RRF_K) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = str(hit["entity"].get("chunk_id", ""))
            if not chunk_id:
                continue
            score = 1.0 / (k + rank)
            current = merged.get(chunk_id)
            if current is None:
                merged[chunk_id] = {
                    "rank": rank,
                    "distance": score,
                    "label": "rrf",
                    "entity": hit["entity"],
                }
            else:
                current["distance"] += score
                if rank < current["rank"]:
                    current["rank"] = rank
                    current["entity"] = hit["entity"]
    fused = sorted(merged.values(), key=lambda item: item["distance"], reverse=True)
    for idx, item in enumerate(fused, start=1):
        item["rank"] = idx
    return fused[:limit]


def execute_case_run(
    client,
    setup: SetupSpec,
    case: BenchmarkCase,
    search_type: str,
    *,
    top_k: int,
    candidate_multiplier: int,
    device: str,
) -> SearchExecution:
    total_start = time.perf_counter()
    embed_latency_ms = 0.0
    db_latency_ms = 0.0
    hit_lists = []
    candidate_limit = max(top_k, top_k * candidate_multiplier)
    split_candidate_limit = max(candidate_limit, top_k * max(2, len(case.route_sources)))

    if setup.topology == "partitioned":
        collection_name = setup.collections["all"]
        embedder = get_embedder(setup.models["all"], device=device)
        filter_expr = None
        if len(case.route_sources) == 1:
            filter_expr = f'source_type == "{case.route_sources[0]}"'
        reqs, encode_ms = build_ann_requests(
            case.query,
            embedder,
            search_type,
            candidate_limit=candidate_limit,
            filter_expr=filter_expr,
        )
        embed_latency_ms += encode_ms
        hits, db_ms = run_milvus_search(
            client,
            collection_name,
            reqs,
            top_k=top_k,
        )
        db_latency_ms += db_ms
        hit_lists.append(hits)
    else:
        for source in case.route_sources:
            collection_name = setup.collections[source]
            embedder = get_embedder(setup.models[source], device=device)
            reqs, encode_ms = build_ann_requests(
                case.query,
                embedder,
                search_type,
                candidate_limit=split_candidate_limit,
            )
            embed_latency_ms += encode_ms
            hits, db_ms = run_milvus_search(
                client,
                collection_name,
                reqs,
                top_k=split_candidate_limit,
            )
            db_latency_ms += db_ms
            hit_lists.append(hits)

    total_latency_ms = (time.perf_counter() - total_start) * 1000
    if len(hit_lists) == 1:
        final_hits = hit_lists[0]
    else:
        final_hits = rrf_fuse(hit_lists, limit=top_k)

    return SearchExecution(
        hits=final_hits,
        embed_latency_ms=embed_latency_ms,
        db_latency_ms=db_latency_ms,
        total_latency_ms=total_latency_ms,
    )


def resolve_relevant_doc_ids(case: BenchmarkCase, canonical_records: list[dict[str, Any]]) -> set[str]:
    relevant = set()
    for record in canonical_records:
        if any(rule.matches(record) for rule in case.rules):
            relevant.add(canonical_doc_id(record))
    return relevant


def validate_relevance(canonical_records: list[dict[str, Any]]) -> dict[str, set[str]]:
    relevance_by_case = {}
    for case in BENCHMARK_CASES:
        relevant_doc_ids = resolve_relevant_doc_ids(case, canonical_records)
        if not relevant_doc_ids:
            raise SystemExit(f"No relevant documents resolved for case: {case.case_id}")
        relevance_by_case[case.case_id] = relevant_doc_ids
    return relevance_by_case


def summarize_setup(setup_key: str, setup_results: list[dict[str, Any]], search_types: list[str]) -> dict[str, Any]:
    summary = {"setup": setup_key, "search_types": {}}
    for search_type in search_types:
        rows = [row for row in setup_results if row["search_type"] == search_type]
        if not rows:
            continue
        summary["search_types"][search_type] = {
            "avg_mrr": round(statistics.mean(row["metrics"]["mrr"] for row in rows), 4),
            "avg_recall": round(statistics.mean(row["metrics"]["recall"] for row in rows if row["metrics"]["recall"] is not None), 4)
            if any(row["metrics"]["recall"] is not None for row in rows) else None,
            "avg_ndcg": round(statistics.mean(row["metrics"]["ndcg"] for row in rows if row["metrics"]["ndcg"] is not None), 4)
            if any(row["metrics"]["ndcg"] is not None for row in rows) else None,
            "avg_source_purity": round(statistics.mean(row["metrics"]["source_purity"] for row in rows if row["metrics"]["source_purity"] is not None), 4)
            if any(row["metrics"]["source_purity"] is not None for row in rows) else None,
            "avg_source_coverage": round(statistics.mean(row["metrics"]["source_coverage"] for row in rows if row["metrics"]["source_coverage"] is not None), 4)
            if any(row["metrics"]["source_coverage"] is not None for row in rows) else None,
            "embed_latency_ms": summarize_latency([row["embed_latency_ms"]["mean"] for row in rows if row["embed_latency_ms"]["mean"] is not None]),
            "db_latency_ms": summarize_latency([row["db_latency_ms"]["mean"] for row in rows if row["db_latency_ms"]["mean"] is not None]),
            "total_latency_ms": summarize_latency([row["total_latency_ms"]["mean"] for row in rows if row["total_latency_ms"]["mean"] is not None]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Milvus topologies and embedding-model variants")
    parser.add_argument("--source-collection", default=COLLECTION_NAME)
    parser.add_argument("--setups", nargs="+", default=["A", "A_plus", "B", "B_plus", "C"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--reuse-bench-collections", action="store_true")
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR)
    parser.add_argument("--no-embedding-cache", action="store_true")
    args = parser.parse_args()

    invalid = [setup for setup in args.setups if setup not in SETUPS]
    if invalid:
        raise SystemExit(f"Unknown setups: {invalid}")

    client = get_client()
    source_collection = args.source_collection
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = EVAL_DIR / f"eval_topology_models_{timestamp}.json"
    EVAL_DIR.mkdir(exist_ok=True)
    embedding_cache_dir = None if args.no_embedding_cache else args.embedding_cache_dir
    if embedding_cache_dir is not None:
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        client.load_collection(source_collection)
    except Exception:
        pass

    canonical_records = export_canonical_records(client, source_collection)
    if not canonical_records:
        raise SystemExit(f"No canonical records exported from {source_collection}")

    counts_by_source: dict[str, int] = {}
    for record in canonical_records:
        counts_by_source[record["source_type"]] = counts_by_source.get(record["source_type"], 0) + 1

    print(f"[EVAL] Canonical source: {source_collection}")
    print(f"[EVAL] Canonical counts: {counts_by_source}")

    relevance_by_case = validate_relevance(canonical_records)
    for case in BENCHMARK_CASES:
        print(
            f"[GROUNDTRUTH] {case.case_id}: {len(relevance_by_case[case.case_id])} relevant docs",
            flush=True,
        )

    if not args.skip_build:
        for setup_key in args.setups:
            setup = SETUPS[setup_key]
            if args.reuse_bench_collections:
                print(f"[BUILD] Reusing existing setup {setup_key}: {setup.description}")
                load_setup_collections(client, setup)
                continue
            print(f"[BUILD] Preparing setup {setup_key}: {setup.description}")
            build_stats = build_setup_collections(
                setup,
                canonical_records,
                device=args.device,
                recreate=not args.reuse_bench_collections,
                cache_dir=embedding_cache_dir,
            )
            print(f"[BUILD] {setup_key}: {build_stats}")
            load_setup_collections(client, setup)
    else:
        for setup_key in args.setups:
            load_setup_collections(client, SETUPS[setup_key])

    all_results = {
        "timestamp": timestamp,
        "source_collection": source_collection,
        "canonical_counts": counts_by_source,
        "setups": {
            setup_key: {
                "description": SETUPS[setup_key].description,
                "topology": SETUPS[setup_key].topology,
                "collections": SETUPS[setup_key].collections,
                "models": {
                    slot: spec.model_name for slot, spec in SETUPS[setup_key].models.items()
                },
            }
            for setup_key in args.setups
        },
        "cases": [],
        "summary": {},
        "notes": [
            "All benchmark collections are rebuilt from the same exported canonical corpus.",
            "Partition-key acceleration only applies when the filter uses one specific partition-key value.",
            "BM25 remains the sparse baseline in this script.",
        ],
    }

    search_types = ["dense", "sparse", "hybrid"]
    per_setup_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in args.setups}

    for case in BENCHMARK_CASES:
        relevant_doc_ids = relevance_by_case[case.case_id]
        case_entry = {
            "case_id": case.case_id,
            "query": case.query,
            "description": case.description,
            "route_sources": list(case.route_sources),
            "relevant_doc_count": len(relevant_doc_ids),
            "relevant_doc_preview": preview_relevant_docs(relevant_doc_ids),
            "results": {},
        }
        print(f"[CASE] {case.case_id} | relevant_docs={len(relevant_doc_ids)} | route={case.route_sources}")

        for setup_key in args.setups:
            setup = SETUPS[setup_key]
            setup_entry = {"description": setup.description, "searches": {}}

            for search_type in search_types:
                for _ in range(args.warmup):
                    execute_case_run(
                        client,
                        setup,
                        case,
                        search_type,
                        top_k=args.top_k,
                        candidate_multiplier=args.candidate_multiplier,
                        device=args.device,
                    )

                embed_samples = []
                db_samples = []
                total_samples = []
                final_hits = []
                for _ in range(args.repeats):
                    execution = execute_case_run(
                        client,
                        setup,
                        case,
                        search_type,
                        top_k=args.top_k,
                        candidate_multiplier=args.candidate_multiplier,
                        device=args.device,
                    )
                    final_hits = execution.hits
                    embed_samples.append(execution.embed_latency_ms)
                    db_samples.append(execution.db_latency_ms)
                    total_samples.append(execution.total_latency_ms)

                metrics = evaluate_hits(
                    final_hits,
                    relevant_doc_ids=relevant_doc_ids,
                    route_sources=case.route_sources,
                    k=args.top_k,
                )
                row = {
                    "case_id": case.case_id,
                    "search_type": search_type,
                    "metrics": metrics,
                    "embed_latency_ms": summarize_latency(embed_samples),
                    "db_latency_ms": summarize_latency(db_samples),
                    "total_latency_ms": summarize_latency(total_samples),
                }
                per_setup_rows[setup_key].append(row)
                setup_entry["searches"][search_type] = row

            case_entry["results"][setup_key] = setup_entry
        all_results["cases"].append(case_entry)

    for setup_key in args.setups:
        all_results["summary"][setup_key] = summarize_setup(setup_key, per_setup_rows[setup_key], search_types)

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    print(f"[EVAL] Saved benchmark results to {output_file}")


if __name__ == "__main__":
    main()
