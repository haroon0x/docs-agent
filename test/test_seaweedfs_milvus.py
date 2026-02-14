#!/usr/bin/env python3
"""
End-to-end test: Milvus backed by SeaweedFS (replacing MinIO).

Usage:
  1. docker compose -f docker-compose.seaweedfs.yml up -d
  2. python test_seaweedfs_milvus.py
  3. docker compose -f docker-compose.seaweedfs.yml down -v
"""

import sys
import time
import random

SEAWEEDFS_ENDPOINT = "http://localhost:8333"
SEAWEEDFS_ACCESS_KEY = "admin"
SEAWEEDFS_SECRET_KEY = "adminadmin"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

TEST_COLLECTION = "seaweedfs_test"
VECTOR_DIM = 768

results = []


def report(name: str, passed: bool, detail: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, passed))


def test_seaweedfs_s3():
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=SEAWEEDFS_ENDPOINT,
        aws_access_key_id=SEAWEEDFS_ACCESS_KEY,
        aws_secret_access_key=SEAWEEDFS_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    resp = s3.list_buckets()
    bucket_names = [b["Name"] for b in resp.get("Buckets", [])]
    report("SeaweedFS S3 — list_buckets", True, f"found {len(bucket_names)} bucket(s)")

    test_bucket = "milvus-test-bucket"
    try:
        s3.create_bucket(Bucket=test_bucket)
        report("SeaweedFS S3 — create_bucket", True, test_bucket)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        report("SeaweedFS S3 — create_bucket", True, f"{test_bucket} already exists")

    s3.put_object(Bucket=test_bucket, Key="hello.txt", Body=b"seaweedfs works")
    obj = s3.get_object(Bucket=test_bucket, Key="hello.txt")
    body = obj["Body"].read().decode()
    report("SeaweedFS S3 — put/get object", body == "seaweedfs works", body)

    s3.delete_object(Bucket=test_bucket, Key="hello.txt")
    s3.delete_bucket(Bucket=test_bucket)
    report("SeaweedFS S3 — cleanup", True)


def test_milvus_connection():
    from pymilvus import connections, utility

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    server_version = utility.get_server_version()
    report("Milvus — connection", True, f"server v{server_version}")

    connections.disconnect(alias="default")


def test_milvus_vector_operations():
    from pymilvus import (
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
    )

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(TEST_COLLECTION):
        Collection(TEST_COLLECTION).drop()
        report("Milvus — drop old test collection", True)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="last_updated", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="SeaweedFS backend test")
    collection = Collection(TEST_COLLECTION, schema)
    report("Milvus — create collection", utility.has_collection(TEST_COLLECTION))

    random.seed(42)
    num_records = 50
    records = []
    for i in range(num_records):
        vec = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        records.append({
            "file_unique_id": f"test-repo:docs/page_{i}.md",
            "repo_name": "test-repo",
            "file_path": f"docs/page_{i}.md",
            "file_name": f"page_{i}.md",
            "citation_url": f"https://example.com/docs/page_{i}",
            "chunk_index": 0,
            "content_text": f"This is test document number {i} about Kubeflow pipelines and KServe.",
            "vector": vec,
            "last_updated": int(time.time()),
        })

    collection.insert(records)
    collection.flush()
    report("Milvus — insert vectors", True, f"{num_records} records")

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 32},
    }
    collection.create_index("vector", index_params)
    collection.load()
    report("Milvus — create index + load", True, "IVF_FLAT / COSINE")

    query_vec = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    search_results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param=search_params,
        limit=5,
        output_fields=["file_path", "content_text", "citation_url"],
    )

    num_hits = len(search_results[0])
    report("Milvus — ANN search", num_hits > 0, f"{num_hits} hits returned")

    first_hit = search_results[0][0]
    has_metadata = (
        first_hit.entity.get("file_path") is not None
        and first_hit.entity.get("content_text") is not None
    )
    report("Milvus — search metadata", has_metadata, first_hit.entity.get("file_path"))

    expr = 'file_unique_id == "test-repo:docs/page_0.md"'
    query_before = collection.query(expr=expr, output_fields=["id"], limit=100)
    count_before = len(query_before)

    collection.delete(expr)
    collection.flush()
    time.sleep(1)
    collection.load()

    query_after = collection.query(expr=expr, output_fields=["id"], limit=100)
    count_after = len(query_after)

    report(
        "Milvus — delete by expression",
        count_before > 0 and count_after == 0,
        f"before={count_before}, after={count_after}",
    )

    new_record = {
        "file_unique_id": "test-repo:docs/page_0.md",
        "repo_name": "test-repo",
        "file_path": "docs/page_0.md",
        "file_name": "page_0.md",
        "citation_url": "https://example.com/docs/page_0",
        "chunk_index": 0,
        "content_text": "UPDATED: This is the new version of page 0.",
        "vector": [random.gauss(0, 1) for _ in range(VECTOR_DIM)],
        "last_updated": int(time.time()),
    }
    collection.insert([new_record])
    collection.flush()
    time.sleep(1)
    collection.load()

    query_reinsert = collection.query(expr=expr, output_fields=["content_text"], limit=10)
    reinserted = len(query_reinsert) > 0 and "UPDATED" in query_reinsert[0].get("content_text", "")
    report("Milvus — re-insert (incremental)", reinserted, query_reinsert[0].get("content_text", "")[:60] if query_reinsert else "empty")

    total = collection.num_entities
    report("Milvus — final entity count", total >= num_records, f"{total} entities")

    collection.drop()
    report("Milvus — drop test collection", not utility.has_collection(TEST_COLLECTION))

    connections.disconnect(alias="default")


def wait_for_milvus(max_wait: int = 120):
    from pymilvus import connections
    print(f"\n⏳ Waiting for Milvus at {MILVUS_HOST}:{MILVUS_PORT} (max {max_wait}s)...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            connections.connect(alias="probe", host=MILVUS_HOST, port=MILVUS_PORT)
            connections.disconnect(alias="probe")
            print("   Milvus is ready.\n")
            return True
        except Exception:
            time.sleep(5)
    print("   ❌ Milvus did not become ready in time.\n")
    return False


def main():
    print("=" * 60)
    print("  Milvus + SeaweedFS End-to-End Test")
    print("=" * 60)

    if not wait_for_milvus():
        print("\nAborting: Milvus is not available.\n")
        sys.exit(1)

    try:
        print("\n── Test 1: SeaweedFS S3 ──")
        test_seaweedfs_s3()
    except Exception as e:
        report("SeaweedFS S3 — EXCEPTION", False, str(e))

    try:
        print("\n── Test 2: Milvus Connection ──")
        test_milvus_connection()
    except Exception as e:
        report("Milvus — connection EXCEPTION", False, str(e))

    try:
        print("\n── Test 3: Milvus Vector Operations ──")
        test_milvus_vector_operations()
    except Exception as e:
        report("Milvus — vector ops EXCEPTION", False, str(e))

    print("\n" + "=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    failed = total - passed

    if failed == 0:
        print(f"  🎉 ALL {total} TESTS PASSED")
    else:
        print(f"  ⚠️  {passed}/{total} passed, {failed} FAILED:")
        for name, p in results:
            if not p:
                print(f"     ❌ {name}")

    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
