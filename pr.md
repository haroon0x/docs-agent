## Replace MinIO with SeaweedFS as Milvus object storage backend

### What

This PR swaps out the MinIO sidecar container in the Milvus deployment with SeaweedFS. Both provide an S3-compatible API, so Milvus doesn't know the difference — it connects the same way via `MINIO_ADDRESS`.

### Why

MinIO changed its license from Apache 2.0 to AGPL v3. SeaweedFS is Apache 2.0 and stays that way. Beyond the license, SeaweedFS has a smaller memory footprint and handles small files (which is what Milvus stores — segments and index chunks) better than MinIO.

### Configuration Changes

One new file: `manifests/milvus-seaweedfs-deployment.yaml`.

This manifest uses the same pod architecture as the original but swaps the `minio/minio` container for `chrislusf/seaweedfs`.

**Note on `MINIO_ADDRESS`:**
Milvus uses the `MINIO_ADDRESS` environment variable to configure its S3 client. This variable name is hardcoded in Milvus but accepts **any S3-compatible endpoint**. We set it to point to SeaweedFS (`localhost:8333`), which provides a fully compatible S3 API. Milvus connects to SeaweedFS exactly as it would to MinIO, unaware of the difference.

```yaml
- name: MINIO_ADDRESS
  value: "localhost:8333"  # Points to SeaweedFS S3 port
- name: MINIO_ACCESS_KEY_ID
  value: "admin"
- name: MINIO_SECRET_ACCESS_KEY
  value: "adminadmin"
- name: MINIO_USE_SSL
  value: "false"
```

### Verification & Testing Report

This implementation has been rigorously verified through an automated end-to-end test suite that simulates a full production lifecycle. The tests were run against a pristine environment (fresh Docker containers and volumes) to ensure reproducibility.

Full end-to-end testing was done locally using Docker Compose to spin up the exact same 3-service stack (SeaweedFS, etcd, Milvus). The test infrastructure and scripts live on the [`feature/seaweedfs-backend_test`](https://github.com/haroon0x/docs-agent/tree/feature/seaweedfs-backend_test) branch:

- [`docker-compose.seaweedfs.yml`](https://github.com/haroon0x/docs-agent/blob/feature/seaweedfs-backend_test/docker-compose.seaweedfs.yml) — Docker Compose stack used for testing
- [`seaweedfs/s3.json`](https://github.com/haroon0x/docs-agent/blob/feature/seaweedfs-backend_test/seaweedfs/s3.json) — SeaweedFS S3 IAM config
- [`test/test_seaweedfs_milvus.py`](https://github.com/haroon0x/docs-agent/blob/feature/seaweedfs-backend_test/test/test_seaweedfs_milvus.py) — end-to-end test script

#### 1. S3 Compatibility Verification
Before integrating with Milvus, the SeaweedFS S3 interface was validated directly to ensure compliance with AWS S3 protocol standards required by the Milvus Go SDK.
- **Bucket Operations**: `ListBuckets`, `CreateBucket` verified successful.
- **Object CRUD**: `PutObject`, `GetObject`, `DeleteObject` verified with byte-for-byte content validation.
- **Result**: **PASS**. SeaweedFS correctly handles S3v4 signatures and standard S3 operations.

#### 2. Milvus Integration Verification
The core integration was tested by configuring Milvus to use SeaweedFS as its sole object storage backend.
- **Startup Health**: Milvus services (RootCoord, DataCoord, IndexCoord) started successfully and registered with etcd.
- **Log Analysis**: No S3 connection errors or timeouts were observed in Milvus logs.
- **Result**: **PASS**. Milvus treats SeaweedFS indistinguishably from MinIO.

#### 3. Functional Correctness (Production Schema)
The test suite executed a full vector database lifecycle using the exact schema used in the `incremental-pipeline.py` production pipeline.
- **Schema**: 768-dim `FLOAT_VECTOR` (all-mpnet-base-v2 compatible) with metadata fields (`content_text`, `citation_url`, `file_unique_id`).
- **Ingestion**: 50 vectors inserted. Data persistence in SeaweedFS was confirmed.
- **Indexing**: `IVF_FLAT` index built with `COSINE` metric. Index files were successfully written to and read from SeaweedFS.
- **Search Accuracy**: ANN search (`nprobe=16`) returned the expected top-k results with correct metadata.
- **Data Consistency**:
  - **Deletion**: Validated expression-based deletion (e.g., `file_unique_id == '...'`).
  - **Upsert**: Validated incremental updates (delete + re-insert pattern).
  - **Compaction**: Validated that `num_entities` reflects logical state (accounting for tombstones).
- **Result**: **PASS**. All 14 automated assertions passed.

### How to Reproduce Verification

To independently verify these results:

1. Checkout the test branch:
   ```bash
   git checkout feature/seaweedfs-backend_test
   ```

2. Spin up the test stack (SeaweedFS + etcd + Milvus):
   ```bash
   docker compose -f docker-compose.seaweedfs.yml up -d
   ```

3. Run the automated verification script:
   ```bash
   # Wait ~60s for Milvus to initialize
   python3 test/test_seaweedfs_milvus.py
   ```

4. Expected Output:
   ```
   ...
   🎉 ALL 14 TESTS PASSED
   ```
