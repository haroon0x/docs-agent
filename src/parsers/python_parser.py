"""
Python source parser using the ast module.
Splits on function/class definitions and prepends signature metadata headers.
"""
import ast
import hashlib
import textwrap
from pathlib import Path
from typing import Iterator


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


class CodeExtractor(ast.NodeVisitor):
    """Walk an AST and extract function/class definitions as chunks."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks: list[dict] = []
        self._module_name = Path(file_path).stem
        self._func_index = 0
        self._class_index = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._emit_class(node)
        self._class_index += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._emit_function(node, prefix="Function")
        self._func_index += 1
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _emit_class(self, node: ast.ClassDef) -> None:
        header = f"[Class: {node.name}] [Module: {self._module_name}]"

        doc = ast.get_docstring(node) or ""
        if doc:
            doc = textwrap.dedent(doc).strip()
            doc = "\n".join(f"  {l}" for l in doc.splitlines()[:10])

        try:
            body_source = textwrap.dedent(ast.unparse(node))
        except Exception:
            body_source = ast.get_source_segment(self._get_source(), node) or ""

        parts = [header]
        if doc:
            parts.append(f"Doc: {doc}")
        parts.append(body_source)
        content = "\n".join(parts)

        if len(content) < 30:
            return

        item_id = f"class:{self._class_index}:{node.name}"
        self.chunks.append({
            "item_id": item_id,
            "content": content,
        })

    def _emit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str) -> None:
        header = f"[{prefix}: {node.name}] [Module: {self._module_name}]"

        try:
            sig = str(ast.unparse(node.args)).strip()
        except Exception:
            sig = ""

        doc = ast.get_docstring(node) or ""
        if doc:
            doc = textwrap.dedent(doc).strip()
            doc = "\n".join(f"  {l}" for l in doc.splitlines()[:8])

        try:
            body_source = textwrap.dedent(ast.unparse(node))
        except Exception:
            body_source = ast.get_source_segment(self._get_source(), node) or ""

        parts = [header]
        if sig:
            parts.append(f"Signature: {sig}")
        if doc:
            parts.append(f"Doc: {doc}")
        parts.append(body_source)
        content = "\n".join(parts)

        if len(content) < 30:
            return

        item_id = f"func:{self._func_index}:{node.name}"
        self.chunks.append({
            "item_id": item_id,
            "content": content,
        })

    def _get_source(self) -> str:
        try:
            return Path(self.file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


def parse_file(
    path: Path,
    *,
    repo_name: str = "kubeflow/manifests",
    repo_branch: str = "main",
    source_type: str = "code",
) -> Iterator[dict]:
    """
    Parse a Python file and yield chunk records ready for Milvus upsert.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return

    extractor = CodeExtractor(str(path))
    extractor.visit(tree)

    rel_path = str(path)
    stat = path.stat()
    last_updated = int(stat.st_mtime)

    for chunk_index, raw in enumerate(extractor.chunks):
        file_unique_id = f"{repo_name}:{rel_path}:{raw['item_id']}"
        chunk_id = _sha256(f"{file_unique_id}:{chunk_index}")

        yield {
            "chunk_id": chunk_id,
            "file_unique_id": file_unique_id,
            "repo_name": repo_name,
            "file_path": str(path),
            "file_name": path.name,
            "citation_url": f"https://github.com/{repo_name}/blob/{repo_branch}/{rel_path}",
            "chunk_index": chunk_index,
            "content_text": raw["content"],
            "source_type": source_type,
            "last_updated": last_updated,
        }


def parse_directory(
    root: Path,
    *,
    repo_name: str = "kubeflow/manifests",
    repo_branch: str = "main",
    source_type: str = "code",
) -> Iterator[dict]:
    """Walk a directory and parse every .py file found."""
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        if any(x in path.name.lower() for x in ("setup", "__init__", "__main__")):
            continue
        yield from parse_file(path, repo_name=repo_name, repo_branch=repo_branch, source_type=source_type)
