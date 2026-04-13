"""
BaselineProfiler: Records agent turns during warm-up phase and builds
a reference behavioral profile using sentence-transformer embeddings.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not available; using random embeddings")

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not available; using numpy cosine similarity")

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class BaselineProfiler:
    """
    Collects agent turns during the warm-up window and builds a reference
    behavioral profile (embeddings, tool distributions, confidence stats).
    """

    def __init__(self, window_size: int = 50, model_name: str = "all-MiniLM-L6-v2"):
        self.window_size = window_size
        self.model_name = model_name
        self._turns: List[dict] = []
        self._embeddings: List[np.ndarray] = []
        self._model: Optional[object] = None
        self._faiss_index: Optional[object] = None

        # Lazy-load the model
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if _ST_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded SentenceTransformer model: %s", self.model_name)
            except Exception as exc:
                logger.warning("Failed to load SentenceTransformer (%s); falling back to random embeddings", exc)
                self._model = None
        else:
            self._model = None

    def _embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised embedding for *text*."""
        if self._model is not None:
            try:
                vec = self._model.encode(text, normalize_embeddings=True)
                return vec.astype(np.float32)
            except Exception as exc:
                logger.debug("Embedding error (%s); using random fallback", exc)

        # Deterministic random fallback based on text hash
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    def record_turn(self, turn_data: dict) -> None:
        """
        Record one turn during the warm-up phase.

        Expected keys in *turn_data*:
            response  (str)     – agent's textual response
            tools_used (list)   – tool names called this turn
            reasoning  (str)    – chain-of-thought / rationale text
            confidence (float)  – scalar confidence in [0, 1]
        """
        if self.is_ready():
            logger.debug("Baseline already full; ignoring extra turn")
            return

        response = turn_data.get("response", "")
        reasoning = turn_data.get("reasoning", "")
        combined_text = f"{response} {reasoning}".strip()

        embedding = self._embed(combined_text)
        self._embeddings.append(embedding)
        self._turns.append(turn_data)

        # Build / update FAISS index once we have at least one vector
        if _FAISS_AVAILABLE and len(self._embeddings) == 1:
            self._faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)  # inner-product (cosine on normed vecs)

        if _FAISS_AVAILABLE and self._faiss_index is not None:
            self._faiss_index.add(np.array([embedding]))

        if self.is_ready():
            logger.info("Baseline warm-up complete (%d turns)", self.window_size)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True when the warm-up window is full."""
        return len(self._turns) >= self.window_size

    # ------------------------------------------------------------------
    # Reference statistics
    # ------------------------------------------------------------------

    def get_reference_embedding(self) -> np.ndarray:
        """Mean embedding vector across all warm-up turns."""
        if not self._embeddings:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        mean_vec = np.mean(self._embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        return mean_vec

    def get_reference_embedding_std(self) -> np.ndarray:
        """Per-dimension std of warm-up embeddings (used for KL divergence)."""
        if len(self._embeddings) < 2:
            return np.ones(EMBEDDING_DIM, dtype=np.float32) * 0.1
        return np.std(self._embeddings, axis=0).astype(np.float32) + 1e-6

    def get_tool_distribution(self) -> dict:
        """
        Return tool name → normalised frequency mapping from warm-up turns.
        """
        counter: Counter = Counter()
        for turn in self._turns:
            for tool in turn.get("tools_used", []):
                counter[tool] += 1
        total = sum(counter.values()) or 1
        return {tool: count / total for tool, count in counter.items()}

    def get_confidence_distribution(self) -> List[float]:
        """List of confidence scores from all warm-up turns."""
        return [t.get("confidence", 0.5) for t in self._turns]

    def get_response_length_stats(self) -> dict:
        """Mean and std of response character lengths."""
        lengths = [len(t.get("response", "")) for t in self._turns]
        if not lengths:
            return {"mean": 0.0, "std": 1.0}
        arr = np.array(lengths, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std()) + 1e-6}

    # ------------------------------------------------------------------
    # Exemplar retrieval
    # ------------------------------------------------------------------

    def get_baseline_exemplars(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Return up to *top_k* warm-up turns most similar to *query_embedding*.
        Uses FAISS when available, otherwise numpy cosine similarity.
        """
        if not self._turns:
            return []

        top_k = min(top_k, len(self._turns))
        query = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        if _FAISS_AVAILABLE and self._faiss_index is not None:
            try:
                distances, indices = self._faiss_index.search(query.reshape(1, -1), top_k)
                return [self._turns[i] for i in indices[0] if 0 <= i < len(self._turns)]
            except Exception as exc:
                logger.debug("FAISS search failed (%s); falling back to numpy", exc)

        # numpy fallback
        emb_matrix = np.array(self._embeddings, dtype=np.float32)
        similarities = emb_matrix @ query
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self._turns[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return (
            f"BaselineProfiler(window_size={self.window_size}, "
            f"recorded={len(self._turns)}, ready={self.is_ready()})"
        )
