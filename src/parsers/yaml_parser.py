"""
YAML manifest parser for Kubernetes resource files.
Splits multi-document YAML files on --- boundaries and prepends metadata headers.
"""
import hashlib
import yaml
from pathlib import Path
from typing import Iterator


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _extract_metadata(doc: dict) -> str:
    parts = []
    kind = doc.get("kind", "")
    name = doc.get("metadata", {}).get("name", "")
    namespace = doc.get("metadata", {}).get("namespace", "")

    if kind:
        parts.append(f"[Kind: {kind}]")
    if name:
        parts.append(f"[Name: {name}]")
    if namespace:
        parts.append(f"[Namespace: {namespace}]")

    return " ".join(parts)


def _parse_single_document(raw: str):
    try:
        docs = list(yaml.safe_load_all(raw))
    except yaml.YAMLError:
        return

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if not doc.get("kind"):
            continue

        header = _extract_metadata(doc)
        body = yaml.dump(doc, default_flow_style=False, sort_keys=False)
        yield header, body


def parse_file(
    path: Path,
    *,
    repo_name: str = "kubeflow/manifests",
    repo_branch: str = "main",
    source_type: str = "code",
) -> Iterator[dict]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    stat = path.stat()
    last_updated = int(stat.st_mtime)

    for chunk_index, (header, body) in enumerate(_parse_single_document(raw)):
        content = f"{header}\n{body}" if header else body
        if len(content) < 20:
            continue

        rel_path = str(path)
        file_unique_id = f"{repo_name}:{rel_path}:chunk-{chunk_index}"
        chunk_id = _sha256(f"{file_unique_id}:{chunk_index}")

        yield {
            "chunk_id": chunk_id,
            "file_unique_id": file_unique_id,
            "repo_name": repo_name,
            "file_path": str(path),
            "file_name": path.name,
            "citation_url": f"https://github.com/{repo_name}/blob/{repo_branch}/{rel_path}",
            "chunk_index": chunk_index,
            "content_text": content,
            "source_type": source_type,
            "last_updated": last_updated,
        }


def parse_directory(
    root: Path,
    *,
    repo_name: str = "kubeflow/manifests",
    repo_branch: str = "main",
    source_type: str = "code",
    extensions: tuple[str, ...] = (".yaml", ".yml"),
) -> Iterator[dict]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if any(x in path.name.lower() for x in ("cert", "key", "secret", ".pem", ".crt")):
            continue
        yield from parse_file(
            path,
            repo_name=repo_name,
            repo_branch=repo_branch,
            source_type=source_type,
        )
