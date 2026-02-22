"""
Unit tests for text chunking logic.

Tests the RecursiveCharacterTextSplitter configuration used in the pipeline
to ensure chunks are created correctly with proper boundaries.
"""
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Sample Markdown content for testing
SAMPLE_MARKDOWN = """# Kubeflow Pipelines

## Overview

Kubeflow Pipelines (KFP) is a platform for building and deploying 
machine learning workflows. It is built on top of Argo.

## Installation

To install Kubeflow Pipelines:

```bash
pip install kubeflow-pipelines
```

## API Reference

### v2 API

The v2 API provides the following endpoints:

- /apis/v2alpha1/pipelines
- /apis/v2alpha1/runs
- /apis/v2alpha1/experiments

### SDK

The SDK provides:

```python
from kfp import dsl

@dsl.component
def my_component():
    return "hello"
```

## Examples

Here's an example pipeline:

```python
@dsl.pipeline(name="my-pipeline")
def my_pipeline():
    step1 = my_component()
    step2 = another_component(step1.output)
"""


@pytest.fixture
def text_splitter():
    """Create a text splitter matching pipeline config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


@pytest.fixture
def small_text_splitter():
    """Text splitter with smaller chunks for testing boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


class TestBasicChunking:
    """Basic chunking functionality tests."""

    def test_chunks_are_not_empty(self, text_splitter):
        """All chunks should have content."""
        chunks = text_splitter.split_text(SAMPLE_MARKDOWN)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_chunks_respect_max_size(self, small_text_splitter):
        """No chunk should exceed max size (except for single long words)."""
        chunks = small_text_splitter.split_text(SAMPLE_MARKDOWN)
        
        for chunk in chunks:
            # Allow some overflow for edge cases with very long words
            assert len(chunk) <= 160, f"Chunk too long: {len(chunk)} chars"

    def test_overlap_is_applied(self, small_text_splitter):
        """Adjacent chunks should share overlap text."""
        chunks = small_text_splitter.split_text("ABCDEF" * 50)
        
        if len(chunks) > 1:
            # Check that some content overlaps between chunks
            assert any(
                chunks[i][-30:] in chunks[i+1] 
                for i in range(len(chunks) - 1)
            )

    def test_preserves_code_blocks(self, text_splitter):
        """Code blocks should be preserved in chunks."""
        chunks = text_splitter.split_text(SAMPLE_MARKDOWN)
        
        combined = " ".join(chunks)
        
        # Check code-related content is preserved
        assert "pip install kubeflow-pipelines" in combined
        assert "@dsl.pipeline" in combined


class TestChunkBoundaries:
    """Tests for proper chunk boundary detection."""

    def test_headers_create_boundaries(self, text_splitter):
        """Section headers should help define chunk boundaries."""
        chunks = text_splitter.split_text(SAMPLE_MARKDOWN)
        
        # Check that different sections appear in chunks
        chunk_text = " ".join(chunks)
        
        assert "Kubeflow Pipelines" in chunk_text
        assert "Installation" in chunk_text
        assert "API Reference" in chunk_text

    def test_empty_input_returns_empty(self, text_splitter):
        """Empty input should return empty list."""
        chunks = text_splitter.split_text("")
        assert chunks == []

    def test_very_short_input(self, text_splitter):
        """Short input that fits in one chunk."""
        text = "Short text."
        chunks = text_splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text


class TestChunkContent:
    """Tests for chunk content quality."""

    def test_no_complete_content_loss(self, text_splitter):
        """All original content should be present across chunks."""
        chunks = text_splitter.split_text(SAMPLE_MARKDOWN)
        combined = " ".join(chunks)
        
        # Key terms should appear somewhere
        assert "Kubeflow" in combined
        assert "pipelines" in combined.lower()
        assert "API" in combined

    def test_unicode_preserved(self, text_splitter):
        """Unicode characters should be preserved."""
        text = "Kubeflow 支持中文安装"
        chunks = text_splitter.split_text(text)
        
        combined = "".join(chunks)
        assert "中文" in combined




class TestChunkCount:
    """Tests for expected chunk counts."""

    def test_longer_text_creates_more_chunks(self, text_splitter):
        """Longer documents should create more chunks."""
        short_text = "Short document."
        long_text = "Word. " * 500
        
        short_chunks = text_splitter.split_text(short_text)
        long_chunks = text_splitter.split_text(long_text)
        
        assert len(long_chunks) > len(short_chunks)

    def test_chunk_count_reasonable(self, text_splitter):
        """Chunk count should be proportional to document length."""
        # ~1000 char doc
        doc_length = len(SAMPLE_MARKDOWN)
        chunks = text_splitter.split_text(SAMPLE_MARKDOWN)
        
        # With 500 char chunks, expect ~2-4 chunks for ~1000 chars
        assert 1 <= len(chunks) <= 10


class TestRealWorldScenarios:
    """Tests simulating real documentation scenarios."""

    def test_kubeflow_doc_scenario(self, text_splitter):
        """Simulate chunking a typical Kubeflow doc page."""
        kfp_doc = """# InferenceService

## Canary Deployment

To perform a canary deployment:

1. Create an InferenceService with canary traffic split
2. Update the canary percentage gradually

```yaml
apiVersion: serving.kubeflow.org/v1beta1
kind: InferenceService
metadata:
  name: my-model
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
  canaryTrafficPercent: 10
```

## Scale to Zero

KServe supports scale-to-zero for idle models.

### Configuration

Set `minReplicas: 0` to enable:

```yaml
spec:
  predictor:
    autoscaler:
      minReplicas: 0
```
"""
        chunks = text_splitter.split_text(kfp_doc)
        
        # Should have chunks covering both sections
        combined = " ".join(chunks)
        assert "canary" in combined.lower()
        assert "scale" in combined.lower() or "zero" in combined.lower()

    def test_api_reference_chunking(self, text_splitter):
        """API references with many short items."""
        api_doc = """
# API Reference

## Methods

### getPipeline(id)
Returns pipeline details.

### deletePipeline(id)
Deletes a pipeline.

### listPipelines()
Lists all pipelines.

### createPipeline(pipeline)
Creates a new pipeline.
"""
        chunks = text_splitter.split_text(api_doc)
        
        # Should preserve method names
        combined = " ".join(chunks)
        assert "getPipeline" in combined
        assert "deletePipeline" in combined
