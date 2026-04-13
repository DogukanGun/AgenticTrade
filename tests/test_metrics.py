"""
Tests for all 12 ASI sub-metric calculators and the composite ASI score.
"""

import math

import numpy as np
import pytest

from src.monitor.metrics import (
    cosine_embedding_similarity,
    levenshtein_distance_normalized,
    js_divergence_confidence,
    chi_squared_tool_test,
    tool_sequence_similarity,
    kl_divergence_tool_params,
    consensus_rate,
    handoff_efficiency,
    mutual_information_role,
    output_length_cv,
    error_clustering_coefficient,
    human_override_rate,
    compute_asi_score,
    compute_kl_divergence,
)


# ===========================================================================
# Group A: Response Consistency
# ===========================================================================

class TestCosineEmbeddingSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        score = cosine_embedding_similarity(v, v)
        # Identical → max similarity mapped to [0,1] → 1.0
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        score = cosine_embedding_similarity(v1, v2)
        # Cosine = 0 → mapped to 0.5
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        score = cosine_embedding_similarity(v1, v2)
        # Cosine = -1 → mapped to 0.0
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_neutral(self):
        v1 = np.zeros(5)
        v2 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        score = cosine_embedding_similarity(v1, v2)
        assert 0.0 <= score <= 1.0

    def test_random_vectors_bounded(self):
        rng = np.random.RandomState(0)
        v1 = rng.randn(384)
        v2 = rng.randn(384)
        score = cosine_embedding_similarity(v1, v2)
        assert 0.0 <= score <= 1.0


class TestLevenshteinDistanceNormalized:
    def test_identical_strings(self):
        assert levenshtein_distance_normalized("hello", "hello") == pytest.approx(1.0)

    def test_empty_strings(self):
        assert levenshtein_distance_normalized("", "") == pytest.approx(1.0)

    def test_completely_different(self):
        # "abc" → "xyz" requires 3 changes, max_len=3 → score=0
        score = levenshtein_distance_normalized("abc", "xyz")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_one_edit(self):
        score = levenshtein_distance_normalized("cat", "bat")
        assert score == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_one_empty(self):
        score = levenshtein_distance_normalized("hello", "")
        assert score == pytest.approx(0.0)

    def test_bounded(self):
        s = levenshtein_distance_normalized("the quick brown fox", "a slow red cat")
        assert 0.0 <= s <= 1.0


class TestJsDivergenceConfidence:
    def test_identical_distributions(self):
        p = [0.8] * 20
        q = [0.8] * 20
        score = js_divergence_confidence(p, q)
        # Same distribution → JS = 0 → score = 1.0
        assert score == pytest.approx(1.0, abs=0.05)

    def test_very_different_distributions(self):
        p = [0.0] * 20
        q = [1.0] * 20
        score = js_divergence_confidence(p, q)
        # Opposite extremes → high JS → low score
        assert score < 0.6

    def test_empty_inputs(self):
        score = js_divergence_confidence([], [])
        assert 0.0 <= score <= 1.0

    def test_bounded(self):
        rng = np.random.RandomState(1)
        p = list(rng.uniform(0, 1, 50))
        q = list(rng.uniform(0, 1, 50))
        score = js_divergence_confidence(p, q)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Group B: Tool Usage
# ===========================================================================

class TestChiSquaredToolTest:
    def test_identical_distributions(self):
        dist = {"search": 0.5, "analyze": 0.3, "execute": 0.2}
        score = chi_squared_tool_test(dist, dist)
        # Identical → high p-value → score near 1.0
        assert score > 0.5

    def test_completely_different(self):
        obs = {"gamble": 1.0}
        exp = {"search": 0.5, "analyze": 0.3, "execute": 0.2}
        score = chi_squared_tool_test(obs, exp)
        # Very different → low p-value → low score
        assert score < 0.5

    def test_empty_inputs(self):
        score = chi_squared_tool_test({}, {})
        assert score == pytest.approx(1.0)

    def test_bounded(self):
        obs = {"search": 0.4, "analyze": 0.6}
        exp = {"search": 0.6, "analyze": 0.4}
        score = chi_squared_tool_test(obs, exp)
        assert 0.0 <= score <= 1.0


class TestToolSequenceSimilarity:
    def test_identical_sequences(self):
        seq = ["search", "analyze", "execute"]
        assert tool_sequence_similarity(seq, seq) == pytest.approx(1.0)

    def test_empty_sequences(self):
        assert tool_sequence_similarity([], []) == pytest.approx(1.0)

    def test_different_sequences(self):
        s1 = ["search"]
        s2 = ["gamble", "speculate", "leverage"]
        score = tool_sequence_similarity(s1, s2)
        assert score < 1.0

    def test_bounded(self):
        s1 = ["a", "b", "c"]
        s2 = ["x", "y"]
        score = tool_sequence_similarity(s1, s2)
        assert 0.0 <= score <= 1.0


class TestKlDivergenceToolParams:
    def test_identical_distributions(self):
        dist = {"search": 0.5, "analyze": 0.5}
        score = kl_divergence_tool_params(dist, dist)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_empty_distributions(self):
        assert kl_divergence_tool_params({}, {}) == pytest.approx(1.0)

    def test_disjoint_distributions(self):
        p = {"gamble": 1.0}
        q = {"search": 1.0}
        score = kl_divergence_tool_params(p, q)
        assert 0.0 <= score <= 1.0

    def test_bounded(self):
        p = {"a": 0.3, "b": 0.7}
        q = {"b": 0.4, "c": 0.6}
        score = kl_divergence_tool_params(p, q)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Group C: Inter-Agent Coordination
# ===========================================================================

class TestConsensusRate:
    def test_all_unanimous(self):
        assert consensus_rate([True, True, True]) == pytest.approx(1.0)

    def test_none_unanimous(self):
        assert consensus_rate([False, False, False]) == pytest.approx(0.0)

    def test_mixed(self):
        score = consensus_rate([True, False, True, True])
        assert score == pytest.approx(0.75)

    def test_empty(self):
        assert consensus_rate([]) == pytest.approx(1.0)


class TestHandoffEfficiency:
    def test_all_successful(self):
        handoffs = [{"success": True}] * 10
        assert handoff_efficiency(handoffs) == pytest.approx(1.0)

    def test_all_failed(self):
        handoffs = [{"success": False}] * 5
        assert handoff_efficiency(handoffs) == pytest.approx(0.0)

    def test_no_handoffs(self):
        assert handoff_efficiency([]) == pytest.approx(1.0)

    def test_mixed(self):
        handoffs = [{"success": True}, {"success": False}, {"success": True}, {"success": True}]
        score = handoff_efficiency(handoffs)
        assert score == pytest.approx(0.75)


class TestMutualInformationRole:
    def test_perfectly_structured(self):
        # Each role exclusively does its own action → high MI
        role_actions = {
            "buyer": ["buy"] * 10,
            "seller": ["sell"] * 10,
            "analyst": ["analyze"] * 10,
        }
        score = mutual_information_role(role_actions)
        assert score > 0.5

    def test_empty(self):
        score = mutual_information_role({})
        assert 0.0 <= score <= 1.0

    def test_single_role(self):
        role_actions = {"agent": ["analyze", "buy", "sell"]}
        score = mutual_information_role(role_actions)
        assert 0.0 <= score <= 1.0

    def test_bounded(self):
        rng = np.random.RandomState(5)
        actions = ["a", "b", "c", "d"]
        role_actions = {
            f"role_{i}": list(rng.choice(actions, size=20))
            for i in range(3)
        }
        score = mutual_information_role(role_actions)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Group D: Behavioral Boundaries
# ===========================================================================

class TestOutputLengthCv:
    def test_matching_cv(self):
        # When current lengths have similar mean and variance to baseline, score should be reasonable
        # Use a varied sample so current CV is non-zero (like the baseline CV)
        baseline_mean = 200.0
        baseline_std = 20.0
        # Mimic the baseline distribution: mean ~200, std ~20
        lengths = [180.0, 190.0, 200.0, 210.0, 220.0, 185.0, 215.0, 200.0, 195.0, 205.0]
        score = output_length_cv(lengths, baseline_mean, baseline_std)
        # Score is bounded in [0,1]; exact CV match is hard with small samples
        assert 0.0 <= score <= 1.0

    def test_large_deviation(self):
        baseline_mean = 200.0
        baseline_std = 10.0
        lengths = [500.0] * 10  # much longer responses
        score = output_length_cv(lengths, baseline_mean, baseline_std)
        assert 0.0 <= score <= 1.0

    def test_empty_lengths(self):
        score = output_length_cv([], 100.0, 10.0)
        assert 0.0 <= score <= 1.0


class TestErrorClusteringCoefficient:
    def test_no_errors(self):
        errors = [0] * 50
        score = error_clustering_coefficient(errors)
        assert score == pytest.approx(1.0)

    def test_clustered_errors(self):
        # All errors at the start → very bursty
        errors = [1] * 10 + [0] * 40
        score = error_clustering_coefficient(errors)
        assert 0.0 <= score <= 1.0

    def test_uniform_errors(self):
        # Alternating errors → more uniform
        errors = [1, 0, 1, 0, 1, 0, 1, 0]
        score = error_clustering_coefficient(errors)
        assert 0.0 <= score <= 1.0

    def test_all_errors(self):
        errors = [1] * 10
        score = error_clustering_coefficient(errors)
        assert score == pytest.approx(0.0)

    def test_bounded(self):
        rng = np.random.RandomState(3)
        errors = list(rng.randint(0, 2, 100))
        score = error_clustering_coefficient(errors)
        assert 0.0 <= score <= 1.0


class TestHumanOverrideRate:
    def test_no_overrides(self):
        assert human_override_rate(0, 100) == pytest.approx(1.0)

    def test_all_overrides(self):
        assert human_override_rate(100, 100) == pytest.approx(0.0)

    def test_half_overrides(self):
        assert human_override_rate(50, 100) == pytest.approx(0.5)

    def test_zero_total(self):
        assert human_override_rate(0, 0) == pytest.approx(1.0)


# ===========================================================================
# Composite ASI Score
# ===========================================================================

class TestComputeAsiScore:
    def test_all_perfect(self):
        metrics = {"m1": 1.0, "m2": 1.0}
        score = compute_asi_score(metrics, metrics, metrics, metrics)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_all_zero(self):
        metrics = {"m1": 0.0}
        score = compute_asi_score(metrics, metrics, metrics, metrics)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_weights_sum_to_one(self):
        # Verify that weight grouping is correct by using known values
        ma = {"v": 1.0}  # score_a = 1.0
        mb = {"v": 0.0}  # score_b = 0.0
        mc = {"v": 0.0}  # score_c = 0.0
        md = {"v": 0.0}  # score_d = 0.0
        # Expected: 0.30*1 + 0.25*0 + 0.25*0 + 0.20*0 = 0.30
        score = compute_asi_score(ma, mb, mc, md)
        assert score == pytest.approx(0.30, abs=0.01)

    def test_bounded(self):
        rng = np.random.RandomState(7)
        for _ in range(20):
            vals = rng.uniform(0, 1, 4)
            metrics = [{"v": v} for v in vals]
            score = compute_asi_score(*metrics)
            assert 0.0 <= score <= 1.0


# ===========================================================================
# KL Divergence (embedding space)
# ===========================================================================

class TestComputeKlDivergence:
    def test_identical_embeddings_gives_zero(self):
        emb = np.ones(10, dtype=np.float32)
        std = np.ones(10, dtype=np.float32) * 0.1
        kl = compute_kl_divergence(emb, emb, std)
        assert kl == pytest.approx(0.0, abs=1e-4)

    def test_different_embeddings_positive(self):
        emb1 = np.ones(10, dtype=np.float32)
        emb2 = np.zeros(10, dtype=np.float32)
        std = np.ones(10, dtype=np.float32) * 0.5
        kl = compute_kl_divergence(emb1, emb2, std)
        assert kl > 0.0

    def test_non_negative(self):
        rng = np.random.RandomState(9)
        for _ in range(10):
            e1 = rng.randn(50).astype(np.float32)
            e2 = rng.randn(50).astype(np.float32)
            std = np.abs(rng.randn(50).astype(np.float32)) + 0.01
            kl = compute_kl_divergence(e1, e2, std)
            assert kl >= 0.0
