import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "beautifulsoup4"]
)
def download_specific_files(
    repo_owner: str,
    repo_name: str,
    file_paths: str,
    github_token: str,
    github_data: dsl.Output[dsl.Dataset]
):
    import requests
    import json
    import base64
    from bs4 import BeautifulSoup

    headers = {"Authorization": f"token {github_token}"} if github_token else {}

    try:
        file_paths_list = json.loads(file_paths)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file_paths: {file_paths}")
        file_paths_list = []

    print(f"Processing {len(file_paths_list)} changed files")
    files = []

    for file_path in file_paths_list:
        if not (file_path.endswith('.md') or file_path.endswith('.html')):
            print(f"Skipping non-doc file: {file_path}")
            continue
        try:
            api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            file_data = response.json()
            content = base64.b64decode(file_data['content']).decode('utf-8')
            if file_path.endswith('.html'):
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text(separator=' ', strip=True)
            files.append({'path': file_path, 'content': content, 'file_name': file_data['name']})
            print(f"Downloaded: {file_path}")
        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
            continue

    print(f"Successfully downloaded {len(files)} files")
    with open(github_data.path, 'w', encoding='utf-8') as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus"]
)
def delete_old_vectors(
    file_paths: str,
    repo_name: str,
    milvus_host: str,
    milvus_port: str,
    milvus_user: str,
    milvus_password: str,
    collection_name: str
):
    from pymilvus import MilvusClient
    import json

    client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}", user=milvus_user, password=milvus_password)

    try:
        if not client.has_collection(collection_name):
            print(f"Collection '{collection_name}' doesn't exist yet — nothing to delete")
            client.close()
            return

        try:
            file_paths_list = json.loads(file_paths)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file_paths: {file_paths}")
            client.close()
            return

        deleted_count = 0
        for file_path in file_paths_list:
            file_unique_id = f"{repo_name}:{file_path}"
            expr = f'file_unique_id == "{file_unique_id}"'
            try:
                results = client.query(collection_name=collection_name, filter=expr, output_fields=["chunk_id"], limit=10000)
                if results:
                    pks = [{'chunk_id': r['chunk_id']} for r in results]
                    client.delete(collection_name=collection_name, pks=pks)
                    deleted_count += len(pks)
                    print(f"Deleted {len(pks)} vectors for file: {file_path}")
                else:
                    print(f"No existing vectors found for file: {file_path}")
            except Exception as e:
                print(f"Error deleting vectors for {file_path}: {e}")
                continue

        print(f"Total deleted vectors: {deleted_count}")
    finally:
        client.close()


@dsl.component(
    base_image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["sentence-transformers", "langchain"]
)
def chunk_and_embed_incremental(
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

    print(f"Created {len(records)} total chunks for incremental update")
    with open(embedded_data.path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_milvus_incremental(
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

    if not client.has_collection(collection_name):
        print(f"Collection '{collection_name}' doesn't exist — creating with hybrid schema")
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
    else:
        print(f"Collection '{collection_name}' already exists — upserting records")

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
        print(f"Upserted {total} records")

    client.close()


@dsl.pipeline(
    name="github-rag-incremental-build",
    description="Incremental RAG pipeline for processing only changed GitHub files"
)
def github_rag_incremental_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website",
    changed_files: str = '[]',
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    milvus_host: str = "my-release-milvus.docs-ag.svc.cluster.local",
    milvus_port: str = "19530",
    milvus_user: str = "root",
    milvus_password: str = "Milvus",
    collection_name: str = "docs_rag"
):
    delete_task = delete_old_vectors(
        file_paths=changed_files,
        repo_name=repo_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        milvus_user=milvus_user,
        milvus_password=milvus_password,
        collection_name=collection_name
    )

    download_task = download_specific_files(
        repo_owner=repo_owner,
        repo_name=repo_name,
        file_paths=changed_files,
        github_token=github_token
    )

    chunk_task = chunk_and_embed_incremental(
        github_data=download_task.outputs["github_data"],
        repo_name=repo_name,
        base_url=base_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    store_task = store_milvus_incremental(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        milvus_user=milvus_user,
        milvus_password=milvus_password,
        collection_name=collection_name
    )
    store_task.after(delete_task)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_incremental_pipeline,
        package_path="github_rag_incremental_pipeline.yaml"
    )