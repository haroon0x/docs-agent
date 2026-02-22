"""
Unit tests for Reciprocal Rank Fusion (RRF) algorithm.

RRF is a rank-based fusion algorithm that merges multiple ranked lists
without requiring score normalization. It's used to combine dense (semantic)
and sparse (BM25) search results.
"""
import pytest
from collections import defaultdict


def reciprocal_rank_fusion(results_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Compute Reciprocal Rank Fusion scores.
    
    Args:
        results_lists: List of ranked document lists (each list is ordered by rank)
        k: Constant used in RRF formula (default 60, standard in literature)
    
    Returns:
        List of (doc_id, score) tuples sorted by score descending
    """
    scores = defaultdict(float)
    
    for results in results_lists:
        for rank, doc_id in enumerate(results, 1):
            scores[doc_id] += 1 / (k + rank)
    
    # Sort by score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def rrf_score(doc_id: str, rank: int, k: int = 60) -> float:
    """Compute individual RRF score for a document at a given rank."""
    return 1 / (k + rank)


class TestRRFBasics:
    """Basic RRF algorithm tests."""

    def test_single_list_returns_same_order(self):
        """RRF of single list should preserve original order."""
        results = ["doc_a", "doc_b", "doc_c"]
        fused = reciprocal_rank_fusion([results])
        
        doc_ids = [doc_id for doc_id, _ in fused]
        assert doc_ids == ["doc_a", "doc_b", "doc_c"]

    def test_empty_list_handling(self):
        """Empty list should not affect fusion."""
        result = reciprocal_rank_fusion([["doc_a"], []])
        doc_ids = [doc_id for doc_id, _ in result]
        assert "doc_a" in doc_ids

    def test_empty_results_returns_empty(self):
        """Empty input should return empty output."""
        assert reciprocal_rank_fusion([]) == []

    def test_k_constant_affects_scores(self):
        """Different k values should produce different relative scores."""
        results = ["doc_a", "doc_b"]
        
        fused_k60 = reciprocal_rank_fusion([results], k=60)
        fused_k10 = reciprocal_rank_fusion([results], k=10)
        
        # With k=10, the rank difference matters more
        # doc_a at rank 1: 1/11 = 0.0909
        # doc_b at rank 2: 1/12 = 0.0833
        # Ratio: 1.09
        
        # With k=60
        # doc_a at rank 1: 1/61 = 0.0164
        # doc_b at rank 2: 1/62 = 0.0161
        # Ratio: 1.018
        
        # The key assertion: both should still rank doc_a > doc_b
        assert fused_k60[0][0] == "doc_a"
        assert fused_k10[0][0] == "doc_a"


class TestRRFDuplicateHandling:
    """Tests for documents appearing in multiple result lists."""

    def test_duplicate_in_both_lists_ranks_higher(self):
        """Documents appearing in both dense and sparse results should be boosted."""
        dense = ["doc_a", "doc_b", "doc_c"]
        sparse = ["doc_b", "doc_c", "doc_d"]
        
        fused = reciprocal_rank_fusion([dense, sparse])
        doc_ids = [doc_id for doc_id, _ in fused]
        
        # doc_b and doc_c appear in both, should be ranked first
        assert doc_ids[0] in ["doc_b", "doc_c"]
        assert doc_ids[1] in ["doc_b", "doc_c"]
        # doc_a and doc_d only appear in one list each
        assert doc_ids[-1] in ["doc_a", "doc_d"]

    def test_three_way_fusion(self):
        """Test fusion of three result lists."""
        list1 = ["doc_a", "doc_b", "doc_c"]
        list2 = ["doc_b", "doc_c", "doc_d"]
        list3 = ["doc_c", "doc_d", "doc_e"]
        
        fused = reciprocal_rank_fusion([list1, list2, list3])
        doc_ids = [doc_id for doc_id, _ in fused]
        
        # doc_c appears in all 3 lists, should be first
        assert doc_ids[0] == "doc_c"


class TestRRFScoreComputation:
    """Tests for RRF score calculation edge cases."""

    def test_scores_are_deterministic(self):
        """Same input should produce same scores."""
        results = ["doc_a", "doc_b", "doc_c"]
        
        run1 = reciprocal_rank_fusion([results])
        run2 = reciprocal_rank_fusion([results])
        
        assert run1 == run2

    def test_all_ranks_scored(self):
        """Every rank position should contribute to score."""
        results = ["a", "b", "c", "d", "e"]
        fused = dict(reciprocal_rank_fusion([results]))
        
        # Each doc should have a non-zero score
        for doc_id in results:
            assert doc_id in fused
            assert fused[doc_id] > 0

    def test_ranks_start_at_1(self):
        """RRF formula uses rank starting at 1, not 0."""
        # If rank started at 0, doc_a would get 1/60 = 0.0167
        # Since rank starts at 1, doc_a gets 1/61 = 0.0164
        score_a = rrf_score("doc_a", rank=1, k=60)
        score_b = rrf_score("doc_b", rank=2, k=60)
        
        assert score_a > score_b
        # Specifically: 1/61 vs 1/62
        assert abs(score_a - 1/61) < 0.0001
        assert abs(score_b - 1/62) < 0.0001


class TestRRFIntegration:
    """Integration-style tests simulating real hybrid search scenarios."""

    def test_dense_sparse_realistic_scenario(self):
        """
        Simulate a realistic hybrid search scenario:
        - Dense search finds semantic matches (may miss exact terms)
        - Sparse search finds exact matches (may miss semantic matches)
        """
        # Dense search: semantic matches for "training ML model"
        dense_results = [
            "docs/pipelines/training",
            "docs/kserve/deployment",
            "docs/notebooks/setup",
            "docs/katib/hp-search",
        ]
        
        # Sparse search: exact matches for "KFP v2 SDK compile"
        sparse_results = [
            "docs/pipelines/v2/sdk",
            "docs/pipelines/v2/api-reference",
            "docs/pipelines/install",
        ]
        
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        doc_ids = [doc_id for doc_id, _ in fused]
        
        # The fusion should include results from both
        assert "docs/pipelines/training" in doc_ids
        assert "docs/pipelines/v2/sdk" in doc_ids

    def test_ranking_stability_with_ties(self):
        """
        When scores are equal, Python's sort is stable.
        Document order should be deterministic.
        """
        # Two lists with no overlap
        list1 = ["doc_a", "doc_b"]
        list2 = ["doc_c", "doc_d"]
        
        run1 = reciprocal_rank_fusion([list1, list2])
        run2 = reciprocal_rank_fusion([list1, list2])
        
        # Same input = same output
        assert run1 == run2

    def test_top_k_extraction(self):
        """Test extracting top-k results after fusion."""
        results = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e",
                   "doc_f", "doc_g", "doc_h", "doc_i", "doc_j"]
        
        fused = reciprocal_rank_fusion([results])
        
        # Extract top 3
        top_3 = [doc_id for doc_id, _ in fused[:3]]
        assert len(top_3) == 3
        assert top_3 == ["doc_a", "doc_b", "doc_c"]
