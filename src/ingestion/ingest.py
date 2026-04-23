"""
Hybrid search ingestion pipeline.

Usage:
    PYTHONPATH=. uv run python -m src.ingestion.ingest --manifests-repo ./kubeflow-manifests --source-type code
    PYTHONPATH=. uv run python -m src.ingestion.ingest --docs-repo ./kubeflow-website --source-type docs
"""
import argparse
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.milvus.collection import setup_collection, get_client, EMBEDDING_MODEL
from src.parsers.yaml_parser import parse_directory as parse_yaml_dir
from src.parsers.python_parser import parse_directory as parse_py_dir
from src.parsers.markdown_parser import parse_directory as parse_md_dir

BATCH_SIZE = 500


def _load_model():
    print(f"[INGEST] Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("[INGEST] Model loaded")
    return model


def _compute_dense_vectors(model: SentenceTransformer, texts: list[str]):
    if not texts:
        return []
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()


def ingest_code(repo_path: Path, model, client):
    yaml_roots = [
        repo_path / "common",
        repo_path / "applications",
        repo_path / "example",
    ]
    releases = repo_path / "releases"
    if releases.exists():
        yaml_roots += list(releases.glob("*"))

    all_chunks = []

    for root in yaml_roots:
        if not root.exists():
            print(f"[INGEST] Skipping nonexistent root: {root}")
            continue
        print(f"[INGEST] Parsing YAML under {root.relative_to(repo_path)} ...")
        chunks = list(parse_yaml_dir(root, source_type="code"))
        print(f"[INGEST]   -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    py_roots = [repo_path / "tests", repo_path / "applications"]
    for root in py_roots:
        if not root.exists():
            continue
        print(f"[INGEST] Parsing Python under {root.relative_to(repo_path)} ...")
        chunks = list(parse_py_dir(root, source_type="code"))
        print(f"[INGEST]   -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


def ingest_docs(repo_path: Path, model, client):
    content_dir = repo_path / "content"
    if not content_dir.exists():
        content_dir = repo_path
        print(f"[INGEST] Using repo root as content dir: {content_dir}")

    print(f"[INGEST] Parsing markdown under {content_dir} ...")
    chunks = list(parse_md_dir(content_dir, source_type="docs", repo_name="kubeflow/website"))
    print(f"[INGEST]   -> {len(chunks)} chunks")
    return chunks


def _upsert_chunks(chunks: list[dict], model, client, label: str):
    if not chunks:
        print(f"[INGEST] No {label} chunks to upsert")
        return 0

    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["content_text"] for c in batch]

        print(f"[INGEST] {label} batch {i // BATCH_SIZE + 1}: encoding {len(texts)} texts...")
        vectors = _compute_dense_vectors(model, texts)
        for chunk, vec in zip(batch, vectors):
            chunk["dense_vector"] = vec
            if len(chunk["content_text"]) > 7800:
                chunk["content_text"] = chunk["content_text"][:7800]

        print(f"[INGEST] {label} batch {i // BATCH_SIZE + 1}: upserting {len(batch)} records...")
        client.upsert(collection_name="docs_rag", data=batch)
        total += len(batch)
        print(f"[INGEST] {label} total upserted: {total}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest chunks into Milvus hybrid search collection")
    parser.add_argument("--manifests-repo", type=Path, default=None)
    parser.add_argument("--docs-repo", type=Path, default=None)
    parser.add_argument("--source-type", default="code", choices=["code", "docs"])
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    if not args.manifests_repo and not args.docs_repo:
        print("[INGEST] Error: specify --manifests-repo and/or --docs-repo")
        sys.exit(1)

    client = setup_collection(recreate=args.recreate)
    model = _load_model()

    start = time.time()
    total_chunks = 0

    if args.manifests_repo:
        if not args.manifests_repo.exists():
            print(f"[INGEST] Repo not found at {args.manifests_repo} — clone it first")
            sys.exit(1)
        chunks = ingest_code(args.manifests_repo, model, client)
        total_chunks += _upsert_chunks(chunks, model, client, "code")

    if args.docs_repo:
        if not args.docs_repo.exists():
            print(f"[INGEST] Docs repo not found at {args.docs_repo} — clone it first")
            sys.exit(1)
        chunks = ingest_docs(args.docs_repo, model, client)
        total_chunks += _upsert_chunks(chunks, model, client, "docs")

    print(f"[INGEST] DONE — ingested {total_chunks} total chunks into docs_rag")
    print(f"[INGEST] Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
