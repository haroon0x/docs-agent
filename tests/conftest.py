"""
Pytest configuration and shared fixtures for docs-agent tests.
"""
import os
import pytest


# Test configuration
TEST_COLLECTION_NAME = "test_docs_rag"
TEST_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Milvus)"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (require Milvus + LLM)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(scope="session")
def milvus_host():
    """Get Milvus host from environment or use localhost."""
    return os.getenv("MILVUS_HOST", "localhost")


@pytest.fixture(scope="session")
def milvus_port():
    """Get Milvus port from environment or use default."""
    return os.getenv("MILVUS_PORT", "19530")


@pytest.fixture(scope="session")
def kserve_url():
    """Get KServe/LLM URL from environment."""
    return os.getenv("KSERVE_URL", "http://localhost:8080/v1/chat/completions")


@pytest.fixture(scope="session")
def embedding_model():
    """Get embedding model name."""
    return TEST_EMBEDDING_MODEL


@pytest.fixture(scope="session")
def test_collection_name():
    """Get test collection name."""
    return TEST_COLLECTION_NAME


@pytest.fixture(scope="session")
def sample_kubeflow_docs():
    """Sample Kubeflow documentation chunks for testing."""
    return [
        {
            "file_path": "content/en/docs/pipelines/install.md",
            "content_text": "Kubeflow Pipelines installation guide. Install using kubectl or Helm.",
            "citation_url": "https://www.kubeflow.org/docs/pipelines/install/",
        },
        {
            "file_path": "content/en/docs/kserve/deploy.md",
            "content_text": "KServe InferenceService deployment. Create a Predictor with model URI.",
            "citation_url": "https://www.kubeflow.org/docs/kserve/deploy/",
        },
        {
            "file_path": "content/en/docs/pipelines/v2/sdk.md",
            "content_text": "KFP v2 SDK compile method. Use @dsl.component decorator.",
            "citation_url": "https://www.kubeflow.org/docs/pipelines/v2/sdk/",
        },
        {
            "file_path": "content/en/docs/katib/hp-search.md",
            "content_text": "Katib hyperparameter search. Define search space in YAML.",
            "citation_url": "https://www.kubeflow.org/docs/katib/hp-search/",
        },
        {
            "file_path": "content/en/docs/notebooks/setup.md",
            "content_text": "Kubeflow Notebooks. Create JupyterHub notebooks.",
            "citation_url": "https://www.kubeflow.org/docs/notebooks/setup/",
        },
    ]


@pytest.fixture(scope="session")
def golden_queries():
    """
    Golden test queries for retrieval benchmarking.
    
    These are representative Kubeflow questions that should retrieve
    specific known documents.
    """
    return [
        {
            "query": "How to install Kubeflow Pipelines",
            "expected_doc_ids": ["content/en/docs/pipelines/install.md"],
            "keywords": ["install", "pipelines"],
        },
        {
            "query": "KServe InferenceService canary deployment",
            "expected_doc_ids": ["content/en/docs/kserve/deploy.md"],
            "keywords": ["kserve", "canary", "inferenceservice"],
        },
        {
            "query": "KFP v2 SDK compile",
            "expected_doc_ids": ["content/en/docs/pipelines/v2/sdk.md"],
            "keywords": ["v2", "sdk", "compile"],
        },
        {
            "query": "Katib hyperparameter search",
            "expected_doc_ids": ["content/en/docs/katib/hp-search.md"],
            "keywords": ["katib", "hyperparameter"],
        },
        {
            "query": "Kubeflow Notebooks setup",
            "expected_doc_ids": ["content/en/docs/notebooks/setup.md"],
            "keywords": ["notebooks", "jupyter"],
        },
    ]
