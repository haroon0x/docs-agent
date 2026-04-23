import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "beautifulsoup4"]
)
def download_github_directory(
    repo_owner: str,
    repo_name: str,
    directory_path: str,
    github_token: str,
    github_data: dsl.Output[dsl.Dataset]
):
    import requests
    import json
    import base64
    from bs4 import BeautifulSoup

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}"

    def get_files_recursive(url):
        files = []
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            items = response.json()

            for item in items:
                if item['type'] == 'file' and (item['name'].endswith('.md') or item['name'].endswith('.html')):
                    file_response = requests.get(item['url'], headers=headers)
                    file_response.raise_for_status()
                    file_data = file_response.json()
                    content = base64.b64decode(file_data['content']).decode('utf-8')

                    if item['name'].endswith('.html'):
                        soup = BeautifulSoup(content, 'html.parser')
                        content = soup.get_text(separator=' ', strip=True)

                    files.append({
                        'path': item['path'],
                        'content': content,
                        'file_name': item['name']
                    })
                elif item['type'] == 'dir':
                    files.extend(get_files_recursive(item['url']))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return files

    files = get_files_recursive(api_url)
    print(f"Downloaded {len(files)} files")

    with open(github_data.path, 'w', encoding='utf-8') as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests"]
)
def download_github_issues(
    repos: str,
    labels: str,
    state: str,
    max_issues_per_repo: int,
    github_token: str,
    issues_data: dsl.Output[dsl.Dataset]
):
    import requests
    import json
    import time

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    all_issues = []

    def api_request(url, params=None):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers)
                if resp.status_code == 403:
                    remaining = resp.headers.get("X-RateLimit-Remaining", "0")
                    if remaining == "0":
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                        wait_time = max(reset_time - int(time.time()), 60)
                        print(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(min(wait_time, 300))
                        continue
                if resp.status_code == 200:
                    return resp.json()
                else:
                    print(f"API error: HTTP {resp.status_code}")
                    return None
            except Exception as e:
                print(f"Request failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        return None

    def fetch_comments(owner, name, issue_number):
        comments_url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}/comments"
        comments_text = ""
        page = 1
        while True:
            comments = api_request(comments_url, {"per_page": 100, "page": page})
            if not comments:
                break
            for comment in comments:
                author = comment.get("user", {}).get("login", "unknown")
                created = comment.get("created_at", "")[:10]
                body = comment.get("body", "") or ""
                comments_text += f"\n\n---\n**Comment by @{author}** ({created}):\n{body}"
            if len(comments) < 100:
                break
            page += 1
        return comments_text

    for repo in repos.split(","):
        repo = repo.strip()
        if "/" not in repo:
            print(f"Skipping invalid repo format: {repo}")
            continue
        owner, name = repo.split("/", 1)
        print(f"Fetching issues from {owner}/{name}...")
        page = 1
        repo_issues = []
        while len(repo_issues) < max_issues_per_repo:
            url = f"https://api.github.com/repos/{owner}/{name}/issues"
            params = {"state": state, "labels": labels, "per_page": 100, "page": page}
            issues = api_request(url, params)
            if not issues:
                break
            for issue in issues:
                if "pull_request" in issue:
                    continue
                labels_str = ", ".join([l["name"] for l in issue.get("labels", [])])
                issue_url = issue.get("html_url", "")
                created_at = issue.get("created_at", "")[:10]
                updated_at = issue.get("updated_at", "")[:10]
                content = f"# {issue['title']}\n\n"
                content += f"**Repository:** {repo}\n"
                content += f"**Issue:** #{issue['number']}\n"
                content += f"**URL:** {issue_url}\n"
                content += f"**Labels:** {labels_str}\n"
                content += f"**State:** {issue['state']}\n"
                content += f"**Created:** {created_at}\n"
                content += f"**Updated:** {updated_at}\n\n"
                content += issue.get("body", "") or ""
                if issue.get("comments", 0) > 0:
                    comments = fetch_comments(owner, name, issue["number"])
                    content += comments
                repo_issues.append({
                    "path": f"issues/{name}/{issue['number']}",
                    "content": content,
                    "file_name": f"issue-{name}-{issue['number']}.md",
                    "url": issue_url
                })
                if len(repo_issues) >= max_issues_per_repo:
                    break
            page += 1
        all_issues.extend(repo_issues)
        print(f"  Fetched {len(repo_issues)} issues from {repo}")

    print(f"Total issues fetched: {len(all_issues)}")
    with open(issues_data.path, 'w', encoding='utf-8') as f:
        for issue_data in all_issues:
            f.write(json.dumps(issue_data, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["sentence-transformers", "langchain"]
)
def chunk_and_embed(
    github_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    base_url: str,
    chunk_size: int,
    chunk_overlap: int,
    embedded_data: dsl.Output[dsl.Dataset]
):
    import json
    import os
    import re
    import torch
    import hashlib
    from sentence_transformers import SentenceTransformer
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    print(f"Model loaded on {device}")

    records = []

    with open(github_data.path, 'r', encoding='utf-8') as f:
        for line in f:
            file_data = json.loads(line)
            content = file_data['content']

            content = re.sub(r'^\s*[+\-]{3,}.*?[+\-]{3,}\s*', '', content, flags=re.DOTALL | re.MULTILINE)
            content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)
            content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\b(Get Started|Contribute|GenAI|Home|Menu|Navigation)\b', '', content, flags=re.IGNORECASE)
            content = re.sub(r'https?://[^\s]+', '', content)
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            content = content.strip()

            if len(content) < 50:
                print(f"Skipping file after cleaning: {file_data['path']} ({len(content)} chars)")
                continue

            path_parts = file_data['path'].split('/')
            if 'content/en/docs' in file_data['path']:
                docs_index = path_parts.index('docs')
                url_path = '/'.join(path_parts[docs_index+1:])
                url_path = os.path.splitext(url_path)[0]
                citation_url = f"{base_url}/{url_path}"
            else:
                citation_url = f"{base_url}/{file_data['path']}"

            file_unique_id = f"{repo_name}:{file_data['path']}"

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(content)

            print(f"File: {file_data['path']} -> {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                chunk_id_str = f"{file_unique_id}:{chunk_idx}"
                chunk_id = hashlib.sha256(chunk_id_str.encode()).hexdigest()[:32]
                records.append({
                    'chunk_id': chunk_id,
                    'file_unique_id': file_unique_id,
                    'repo_name': repo_name,
                    'file_path': file_data['path'],
                    'file_name': file_data['file_name'],
                    'citation_url': citation_url[:1024],
                    'chunk_index': chunk_idx,
                    'content_text': chunk[:8000],
                    'dense_vector': embedding,
                    'source_type': 'docs'
                })

    print(f"Created {len(records)} total chunks")
    with open(embedded_data.path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    milvus_user: str,
    milvus_password: str,
    collection_name: str
):
    from pymilvus import MilvusClient, DataType, Function, FunctionType
    from datetime import datetime
    import json

    client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}", user=milvus_user, password=milvus_password)

    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists — using existing schema")
    else:
        print(f"Creating collection '{collection_name}' with hybrid schema")
        schema = client.create_schema(
            enable_dynamic_fields=True,
            description="Unified docs RAG: docs + issues + code with hybrid search",
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field(field_name="file_unique_id", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="repo_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="citation_url", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(
            field_name="content_text",
            datatype=DataType.VARCHAR,
            max_length=8000,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"tokenizer": "standard", "filter": ["lowercase"]},
        )
        schema.add_field(
            field_name="source_type",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_partition_key=True,
        )
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="last_updated", datatype=DataType.INT64)
        schema.add_function(Function(
            name="bm25_embedding",
            function_type=FunctionType.BM25,
            input_field_names=["content_text"],
            output_field_names=["sparse_vector"],
        ))

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"Collection '{collection_name}' created with hybrid schema")

    records = []
    timestamp = int(datetime.now().timestamp())

    with open(embedded_data.path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "chunk_id": record["chunk_id"],
                "file_unique_id": record["file_unique_id"],
                "repo_name": record["repo_name"],
                "file_path": record["file_path"],
                "file_name": record["file_name"],
                "citation_url": record["citation_url"],
                "chunk_index": record["chunk_index"],
                "content_text": record["content_text"],
                "dense_vector": record["dense_vector"],
                "source_type": record.get("source_type", "docs"),
                "last_updated": timestamp,
            })

    if records:
        batch_size = 500
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            client.upsert(collection_name=collection_name, data=batch)
            total += len(batch)
        print(f"Upserted {total} records. Total entities: {client.query(collection_name=collection_name, output_fields=['count(*)'])}")

    client.close()


@dsl.pipeline(
    name="github-rag",
    description="RAG pipeline for processing GitHub documentation"
)
def github_rag_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website",
    directory_path: str = "content/en",
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    milvus_host: str = "my-release-milvus.docs-ag.svc.cluster.local",
    milvus_port: str = "19530",
    milvus_user: str = "root",
    milvus_password: str = "Milvus",
    collection_name: str = "docs_rag"
):
    download_task = download_github_directory(
        repo_owner=repo_owner,
        repo_name=repo_name,
        directory_path=directory_path,
        github_token=github_token
    )

    chunk_task = chunk_and_embed(
        github_data=download_task.outputs["github_data"],
        repo_name=repo_name,
        base_url=base_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    store_task = store_milvus(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        milvus_user=milvus_user,
        milvus_password=milvus_password,
        collection_name=collection_name
    )


if __name__ == "__main__":
    import os
    os.environ['KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT'] = 'true'
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_pipeline,
        package_path="github_rag_pipeline.yaml"
    )