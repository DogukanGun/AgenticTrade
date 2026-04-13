"""
ASI (Agent Strategy Index) metric calculators.

12 sub-metrics across 4 groups:
  A: Response Consistency  (weight 0.30)
  B: Tool Usage            (weight 0.25)
  C: Inter-Agent Coord.    (weight 0.25)
  D: Behavioral Boundaries (weight 0.20)

All metrics return a float in [0, 1] where higher = more consistent.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import scipy.stats
import scipy.special

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import: Levenshtein
# ---------------------------------------------------------------------------
try:
    import Levenshtein
    _LEV_AVAILABLE = True
except ImportError:
    try:
        # Some versions expose it as python_Levenshtein
        import python_Levenshtein as Levenshtein  # type: ignore
        _LEV_AVAILABLE = True
    except ImportError:
        _LEV_AVAILABLE = False
        logger.warning("python-Levenshtein not available; using simple fallback")


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein distance (fallback)."""
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _lev_distance(s1: str, s2: str) -> int:
    if _LEV_AVAILABLE:
        try:
            return Levenshtein.distance(s1, s2)
        except Exception:
            pass
    return _levenshtein_distance(s1, s2)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ===========================================================================
# Group A: Response Consistency
# ===========================================================================

def cosine_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors.
    Returns value in [0, 1] (shifted from [-1,1] range).
    """
    emb1 = np.asarray(emb1, dtype=np.float64).ravel()
    emb2 = np.asarray(emb2, dtype=np.float64).ravel()
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.5  # neutral
    cos = float(np.dot(emb1, emb2) / (norm1 * norm2))
    # Map [-1, 1] → [0, 1]
    return _clamp((cos + 1.0) / 2.0)


def levenshtein_distance_normalized(s1: str, s2: str) -> float:
    """
    1 - normalised Levenshtein edit distance between two strings.
    Returns 1.0 for identical strings, 0.0 for completely different.
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = _lev_distance(s1, s2)
    return _clamp(1.0 - dist / max_len)


def js_divergence_confidence(p: List[float], q: List[float]) -> float:
    """
    1 - JS divergence between two confidence score distributions.
    Distributions are histogrammed into 10 bins over [0, 1].
    Returns 1.0 when distributions are identical.
    """
    if not p or not q:
        return 0.5

    bins = np.linspace(0, 1, 11)
    p_hist, _ = np.histogram(p, bins=bins, density=False)
    q_hist, _ = np.histogram(q, bins=bins, density=False)

    # Smooth to avoid zero divisions
    p_dist = (p_hist + 1e-9).astype(np.float64)
    q_dist = (q_hist + 1e-9).astype(np.float64)
    p_dist /= p_dist.sum()
    q_dist /= q_dist.sum()

    m = 0.5 * (p_dist + q_dist)
    js = 0.5 * scipy.stats.entropy(p_dist, m) + 0.5 * scipy.stats.entropy(q_dist, m)
    # JS divergence ∈ [0, log(2)]; normalise to [0, 1]
    js_norm = js / math.log(2)
    return _clamp(1.0 - js_norm)


# ===========================================================================
# Group B: Tool Usage
# ===========================================================================

def chi_squared_tool_test(observed: Dict[str, float], expected: Dict[str, float]) -> float:
    """
    Chi-squared goodness-of-fit test comparing tool usage distributions.
    High p-value (= consistent) → score near 1.0.
    """
    all_tools = sorted(set(observed) | set(expected))
    if not all_tools:
        return 1.0

    obs_arr = np.array([observed.get(t, 0.0) for t in all_tools], dtype=np.float64)
    exp_arr = np.array([expected.get(t, 1e-9) for t in all_tools], dtype=np.float64)

    # Normalise to counts (scale to sum=100 to avoid issues with tiny values)
    obs_arr = obs_arr / (obs_arr.sum() + 1e-12) * 100
    exp_arr = exp_arr / (exp_arr.sum() + 1e-12) * 100

    # Avoid zero expected values
    exp_arr = np.maximum(exp_arr, 1e-6)

    try:
        chi2, p_value = scipy.stats.chisquare(obs_arr, f_exp=exp_arr)
        return _clamp(float(p_value))
    except Exception:
        return 0.5


def tool_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    """
    1 - normalised Levenshtein distance between tool sequences (joined as strings).
    """
    s1 = ",".join(seq1)
    s2 = ",".join(seq2)
    return levenshtein_distance_normalized(s1, s2)


def kl_divergence_tool_params(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    1 - KL divergence of tool parameter distributions, clamped to [0, 1].
    p and q map tool names to usage frequencies.
    """
    if not p and not q:
        return 1.0

    all_tools = sorted(set(p) | set(q))
    p_arr = np.array([p.get(t, 1e-9) for t in all_tools], dtype=np.float64)
    q_arr = np.array([q.get(t, 1e-9) for t in all_tools], dtype=np.float64)

    p_arr /= p_arr.sum() + 1e-12
    q_arr /= q_arr.sum() + 1e-12

    # KL(p || q)
    kl = scipy.stats.entropy(p_arr, q_arr)
    if not np.isfinite(kl):
        kl = 10.0
    # Normalise: KL can be arbitrarily large; treat 5.0 as "completely different"
    kl_norm = kl / 5.0
    return _clamp(1.0 - kl_norm)


# ===========================================================================
# Group C: Inter-Agent Coordination
# ===========================================================================

def consensus_rate(votes: List[bool]) -> float:
    """
    Fraction of unanimous (True) decisions in *votes*.
    Returns 1.0 if all decisions are unanimous, 0.0 if none are.
    """
    if not votes:
        return 1.0
    return _clamp(float(sum(votes)) / len(votes))


def handoff_efficiency(handoffs: List[dict]) -> float:
    """
    1 - (failed_handoffs / total_handoffs).
    Each entry in *handoffs* should have a 'success' bool key.
    Returns 1.0 if no handoffs occurred.
    """
    if not handoffs:
        return 1.0
    failed = sum(1 for h in handoffs if not h.get("success", True))
    return _clamp(1.0 - failed / len(handoffs))


def mutual_information_role(role_actions: Dict[str, List[str]]) -> float:
    """
    Normalised mutual information between roles and action types.
    *role_actions* maps role_name → list of action strings.
    Returns a value in [0, 1]; higher means roles are more predictably associated
    with distinct actions (i.e., structured coordination).
    """
    if not role_actions:
        return 0.5

    # Build joint distribution
    roles = sorted(role_actions)
    all_actions: List[str] = []
    for actions in role_actions.values():
        all_actions.extend(actions)

    if not all_actions:
        return 0.5

    unique_actions = sorted(set(all_actions))
    role_idx = {r: i for i, r in enumerate(roles)}
    action_idx = {a: i for i, a in enumerate(unique_actions)}

    joint = np.zeros((len(roles), len(unique_actions)), dtype=np.float64)
    for role, actions in role_actions.items():
        ri = role_idx[role]
        for action in actions:
            joint[ri, action_idx[action]] += 1

    total = joint.sum()
    if total == 0:
        return 0.5

    joint /= total
    p_role = joint.sum(axis=1)
    p_action = joint.sum(axis=0)

    # MI = sum p(r,a) log(p(r,a) / (p(r)*p(a)))
    mi = 0.0
    for ri in range(len(roles)):
        for ai in range(len(unique_actions)):
            if joint[ri, ai] > 0:
                mi += joint[ri, ai] * math.log(joint[ri, ai] / (p_role[ri] * p_action[ai] + 1e-12))

    # Normalise by H(role) to bound in [0, 1]
    h_role = scipy.stats.entropy(p_role + 1e-12)
    if h_role < 1e-9:
        return 1.0
    nmi = mi / h_role
    return _clamp(nmi)


# ===========================================================================
# Group D: Behavioral Boundaries
# ===========================================================================

def output_length_cv(
    lengths: List[float],
    baseline_mean: float,
    baseline_std: float,
) -> float:
    """
    1 - normalised coefficient of variation deviation from baseline.
    Returns 1.0 when current CV matches baseline CV exactly.
    """
    if not lengths:
        return 0.5

    current_mean = float(np.mean(lengths))
    current_std = float(np.std(lengths)) + 1e-6
    current_cv = current_std / (current_mean + 1e-6)

    baseline_cv = (baseline_std + 1e-6) / (baseline_mean + 1e-6)
    cv_diff = abs(current_cv - baseline_cv) / (baseline_cv + 1e-6)
    return _clamp(1.0 - cv_diff)


def error_clustering_coefficient(errors: List[int]) -> float:
    """
    1 - error burstiness score.
    *errors* is a list of 0/1 values (1 = error occurred that turn).
    Burstiness is measured via runs test deviation.
    Returns 1.0 for uniformly distributed errors, 0.0 for highly clustered.
    """
    if not errors or sum(errors) == 0:
        return 1.0  # no errors = perfectly consistent

    n = len(errors)
    n1 = sum(errors)          # error turns
    n0 = n - n1               # clean turns

    if n0 == 0:
        return 0.0  # all errors

    # Count runs
    runs = 1
    for i in range(1, n):
        if errors[i] != errors[i - 1]:
            runs += 1

    # Expected runs and variance under H0 (random arrangement)
    expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
    variance_runs = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1) ** 2 * (n0 + n1 - 1) + 1e-9)

    if variance_runs <= 0:
        return 0.5

    z = (runs - expected_runs) / math.sqrt(variance_runs)
    # Negative z = fewer runs = more clustering
    # Map to [0, 1]: z in [-3, +3] → [0, 1]
    return _clamp((z + 3.0) / 6.0)


def human_override_rate(overrides: int, total: int) -> float:
    """
    1 - (overrides / total). Returns 1.0 when no overrides occurred.
    """
    if total <= 0:
        return 1.0
    return _clamp(1.0 - overrides / total)


# ===========================================================================
# Composite ASI Score
# ===========================================================================

WEIGHTS = {
    "a": 0.30,
    "b": 0.25,
    "c": 0.25,
    "d": 0.20,
}


def _group_mean(metrics: Dict[str, float]) -> float:
    values = [v for v in metrics.values() if v is not None]
    if not values:
        return 0.5
    return float(np.mean(values))


def compute_asi_score(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    metrics_c: Dict[str, float],
    metrics_d: Dict[str, float],
) -> float:
    """
    Weighted average of 4 metric groups.

    Returns ASI score in [0, 1] — higher is better / more consistent.
    """
    score_a = _group_mean(metrics_a)
    score_b = _group_mean(metrics_b)
    score_c = _group_mean(metrics_c)
    score_d = _group_mean(metrics_d)

    asi = (
        WEIGHTS["a"] * score_a
        + WEIGHTS["b"] * score_b
        + WEIGHTS["c"] * score_c
        + WEIGHTS["d"] * score_d
    )
    return _clamp(asi)


# ===========================================================================
# KL divergence for drift distance (embedding space)
# ===========================================================================

def compute_kl_divergence(
    current_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    reference_std: np.ndarray,
) -> float:
    """
    Treat embeddings as diagonal Gaussian distributions and compute
    KL divergence D_KL(current || reference).

    current_embedding  : mean of current distribution (shape [D])
    reference_embedding: mean of reference (baseline) distribution (shape [D])
    reference_std      : per-dimension std of reference (shape [D])

    Returns KL divergence D_t ≥ 0.
    """
    mu1 = np.asarray(current_embedding, dtype=np.float64).ravel()
    mu0 = np.asarray(reference_embedding, dtype=np.float64).ravel()
    sigma0 = np.asarray(reference_std, dtype=np.float64).ravel() + 1e-8
    # Assume current distribution has same std as reference for simplicity
    sigma1 = sigma0.copy()

    D = len(mu0)
    # KL(N(mu1,sigma1) || N(mu0,sigma0)) for diagonal Gaussians
    kl = 0.5 * np.sum(
        (sigma1 / sigma0) ** 2
        + ((mu1 - mu0) ** 2) / (sigma0 ** 2)
        - 1.0
        + 2.0 * np.log(sigma0 / sigma1)
    )
    return float(max(0.0, kl))
