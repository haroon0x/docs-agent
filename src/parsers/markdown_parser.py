"""
Markdown document parser for documentation files.
Splits on headings and prepends section metadata headers.
"""
import hashlib
import re
from pathlib import Path
from typing import Iterator


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    match = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.DOTALL)
    if not match:
        return {}, text
    fm = {}
    for line in match.group(1).splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            fm[key.strip()] = val.strip()
    return fm, match.group(2)


def _split_sections(content: str) -> list[tuple[str, str]]:
    pattern = re.compile(r'^#{1,3}\s+.+$', re.MULTILINE)
    parts = []
    pos = 0
    for m in pattern.finditer(content):
        if pos < m.start():
            parts.append(("", content[pos:m.start()]))
        pos = m.start()
    if pos < len(content):
        parts.append(("", content[pos:]))
    return parts


def parse_file(
    path: Path,
    *,
    repo_name: str = "kubeflow/website",
    repo_branch: str = "main",
    source_type: str = "docs",
) -> Iterator[dict]:
    """Parse a markdown file and yield chunk records ready for Milvus upsert."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    fm, body = _extract_frontmatter(raw)
    if not body.strip():
        return

    stat = path.stat()
    last_updated = int(stat.st_mtime)

    sections = _split_sections(body)
    if not sections:
        sections = [("", body)]

    for chunk_index, (header, section_body) in enumerate(sections):
        content = section_body.strip()
        if len(content) < 50:
            continue

        meta_parts = [f"[File: {path.name}]"]
        if header:
            clean_header = header.strip('# ').strip()
            meta_parts.append(f"[Section: {clean_header}]")
        if fm.get("title"):
            meta_parts.append(f"[Title: {fm['title']}]")

        header_block = " ".join(meta_parts)
        full_content = f"{header_block}\n\n{content}"

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
            "content_text": full_content,
            "source_type": source_type,
            "last_updated": last_updated,
        }


def parse_directory(
    root: Path,
    *,
    repo_name: str = "kubeflow/website",
    repo_branch: str = "main",
    source_type: str = "docs",
    extensions: tuple[str, ...] = (".md", ".markdown"),
) -> Iterator[dict]:
    """Walk a directory tree and parse every markdown file found."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in extensions:
            yield from parse_file(
                path,
                repo_name=repo_name,
                repo_branch=repo_branch,
                source_type=source_type,
            )
