# Test Report — GitHub Issues Pipeline Integration

## Overview

This document details the local component testing performed on the GitHub Issues pipeline integration. All tests run **locally** without requiring a Kubernetes cluster, Milvus database, or GPU.

**Test Script:** `pipelines/test_components.py`  
**Run Command:** `python3 pipelines/test_components.py`  
**Result:** All 5/5 tests passed ✅

---

## Test 1: `download_github_issues` — Live API Fetch

### What it tests
The core logic of the `download_github_issues` component — making real HTTP requests to the GitHub API, fetching actual issues, and writing them as valid JSONL.

### How it works
1. Calls the GitHub API for `kubeflow/website` with `max_issues_per_repo=3`
2. Filters out pull requests (only returns actual issues)
3. Formats each issue with structured metadata (repo, number, URL, labels, state, dates)
4. Writes to a temp file in JSONL format (one JSON object per line)

### Assertions
- At least 1 issue fetched
- No more than `max_issues_per_repo` issues returned
- JSONL line count matches issue count
- Each record has required fields: `path`, `content`, `file_name`, `url`
- Content is at least 50 characters (won't be skipped by `chunk_and_embed`)
- Path starts with `issues/` (correct prefix for issue data)

### Sample Output
```
✅ Fetched 3 issues from kubeflow/website
✅ Output JSONL has correct structure (path, content, file_name, url)
✅ Sample issue title: # feature(Website): copy to clipboard button.
```

---

## Test 2: `chunk_and_embed` I/O Contract

### What it tests
Verifies that the **output format** of `download_github_issues` is compatible with what `chunk_and_embed` expects as input. Also tests the citation URL logic for both `source_type="issue"` and `source_type="doc"`.

### How it works
1. Creates mock issue records (simulating `download_github_issues` output)
2. Creates mock doc records (simulating `download_github_directory` output)
3. Writes both to temp JSONL files
4. Validates field presence and citation URL construction

### Citation URL Logic Tested

**For issues (`source_type="issue"`):**
```python
# Uses the GitHub URL directly from the issue record
citation_url = record.get('url', record.get('html_url', ''))
# Result: "https://github.com/kubeflow/kubeflow/issues/1234"
```

**For docs (`source_type="doc"`):**
```python
# Constructs URL from base_url + path
citation_url = f"{base_url}/{url_path}"
# Result: "https://www.kubeflow.org/docs/pipelines/overview"
```

### Assertions
- Issue records have all required fields (`path`, `content`, `file_name`, `url`)
- Issue citation URLs start with `https://github.com/`
- Doc citation URLs contain `kubeflow.org`
- All content is ≥ 50 chars (won't be skipped by the minumum length check)

### Sample Output
```
✅ Issue records have all required fields (path, content, file_name, url)
✅ Citation URL logic correct for source_type='issue' (uses GitHub URL)
✅ Citation URL logic correct for source_type='doc' (builds from base_url)
✅ All content lengths >= 50 chars (won't be skipped)
```

---

## Test 3: `store_milvus` I/O Contract

### What it tests
Verifies that the **output format** of `chunk_and_embed` matches what `store_milvus` expects. Tests the field renaming (`embedding` → `vector`), vector dimensions, and Milvus schema compliance.

### How it works
1. Creates mock embedded records (simulating `chunk_and_embed` output)
2. Applies the exact transformation that `store_milvus` performs
3. Validates field types, dimensions, and VARCHAR length limits

### Key Transformation Tested
```python
# chunk_and_embed outputs "embedding", store_milvus needs "vector"
records.append({
    "file_unique_id": record["file_unique_id"],    # VARCHAR(512)
    "repo_name": record["repo_name"],              # VARCHAR(256)
    "file_path": record["file_path"],              # VARCHAR(512)
    "file_name": record["file_name"],              # VARCHAR(256)
    "citation_url": record["citation_url"],        # VARCHAR(1024)
    "chunk_index": record["chunk_index"],           # INT64
    "content_text": record["content_text"],        # VARCHAR(2000)
    "vector": record["embedding"],                 # FLOAT_VECTOR(768) ← renamed
    "last_updated": timestamp                      # INT64
})
```

### Assertions
- `embedding` key renamed to `vector`
- Vector dimension = 768 (matches `all-mpnet-base-v2` model)
- `file_unique_id` ≤ 512 chars (VARCHAR limit)
- `citation_url` ≤ 1024 chars (VARCHAR limit)
- `content_text` ≤ 2000 chars (VARCHAR limit)
- `chunk_index` is int, `last_updated` is int

### Sample Output
```
✅ Embedding key correctly renamed to 'vector' for Milvus
✅ Vector dimension = 768 (matches all-mpnet-base-v2)
✅ All field lengths within Milvus schema limits
✅ All field types correct (int for chunk_index, last_updated)
```

---

## Test 4: Compiled YAML `dsl.If` Condition

### What it tests
Parses the compiled `github_rag_pipeline.yaml` to verify the `dsl.If` conditional execution is correctly embedded in the pipeline definition.

### How it works
1. Loads the compiled YAML with PyYAML
2. Checks pipeline input parameters exist
3. Verifies conditional execution structure in the DAG
4. Confirms the `download-github-issues` component is present

### Assertions
- `issue_repos` parameter exists with a default value
- `issue_labels`, `issue_state`, `max_issues_per_repo` parameters present
- Conditional execution (sub-DAG or trigger policy) found in compiled YAML
- `download-github-issues` component exists in the YAML

### Sample Output
```
✅ 'issue_repos' parameter exists with default: 'kubeflow/kubeflow,kubeflow/pipelines'
✅ All issue parameters present (issue_repos, issue_labels, issue_state, max_issues_per_repo)
✅ Conditional execution (dsl.If) found in compiled YAML
✅ download-github-issues component present in YAML
```

---

## Test 5: Empty Repos String Edge Case

### What it tests
Verifies that when `issue_repos=""` (empty string), the `download_github_issues` component logic produces zero issues without crashing.

### How it works
1. Splits empty string by `,` → `[""]`
2. Checks each element for `/` (repo format validation)
3. Empty string has no `/`, so the loop body is skipped entirely

### Guard Logic Tested
```python
for repo in repos.split(","):
    repo = repo.strip()
    if "/" not in repo:    # ← This guard catches empty strings
        continue
    # ... rest of logic never executes
```

### Assertions
- Empty repos string produces exactly 0 issues
- No exceptions thrown

### Sample Output
```
✅ Empty repos string correctly produces 0 issues
✅ The '/' check in the component guards against empty/invalid repos
```

---

## Summary

| # | Test | Status | What it proves |
|---|---|---|---|
| 1 | `download_github_issues` live fetch | ✅ PASS | Component fetches real issues, produces valid JSONL |
| 2 | `chunk_and_embed` I/O contract | ✅ PASS | Issue data format is compatible with chunking pipeline |
| 3 | `store_milvus` I/O contract | ✅ PASS | Embedded data format matches Milvus schema |
| 4 | Compiled YAML condition | ✅ PASS | `dsl.If` correctly controls issue branch execution |
| 5 | Empty repos edge case | ✅ PASS | Empty input handled gracefully, no crash |

### Dependencies Used
- **Python 3.12** — test runner
- **requests** — GitHub API calls (Test 1)
- **PyYAML** — YAML parsing (Test 4)
- **kfp 2.15.2** — pipeline compilation verification

### Not Covered by These Tests
- Actual sentence-transformers embedding generation (needs `torch` + GPU)
- Actual Milvus insertion (needs a running Milvus instance)
- KFP pipeline DAG execution on a Kubernetes cluster
- Visual `dsl.If` branch skip in KFP UI

These require a Kubeflow cluster with Milvus deployed.
