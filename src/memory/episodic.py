"""
EpisodicMemory: Manages compressed episodic summaries and recent raw turns
to give the agent efficient access to behavioral context.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Stores raw agent turns and periodically consolidates them into summary
    episodes. Provides context-window retrieval combining recent raw turns
    with the latest consolidated episode.
    """

    def __init__(self, consolidation_interval: int = 50, raw_context_window: int = 10):
        self.consolidation_interval = consolidation_interval
        self.raw_context_window = raw_context_window

        self._raw_turns: List[dict] = []
        self._episodes: List[dict] = []   # list of consolidated summaries

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------

    def add_turn(self, turn_data: dict) -> None:
        """Add a raw turn. Triggers consolidation every K turns."""
        self._raw_turns.append(turn_data)
        total = len(self._raw_turns)

        if total % self.consolidation_interval == 0:
            start = total - self.consolidation_interval
            end = total
            episode = self.consolidate(start=start, end=end)
            self._episodes.append(episode)
            logger.debug(
                "Consolidated turns %d-%d into episode %d",
                start, end, len(self._episodes),
            )

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, k: int = 50, start: int = 0, end: Optional[int] = None) -> dict:
        """
        Create a summary dict over a slice of raw turns.

        If *end* is provided, use turns[start:end]; otherwise use last K turns.
        """
        if end is None:
            window = self._raw_turns[-k:] if len(self._raw_turns) >= k else self._raw_turns[:]
            turn_range = (
                len(self._raw_turns) - len(window),
                len(self._raw_turns),
            )
        else:
            window = self._raw_turns[start:end]
            turn_range = (start, end)

        if not window:
            return {
                "goals": [],
                "key_decisions": [],
                "open_positions": {},
                "risk_summary": {},
                "turn_range": turn_range,
            }

        # Aggregate goals
        goals = []
        goal_counter: Counter = Counter()
        for t in window:
            for g in t.get("goals", []):
                goal_counter[g] += 1
        goals = [g for g, _ in goal_counter.most_common(5)]

        # Key decisions: strategy / action fields
        decisions: List[str] = []
        decision_counter: Counter = Counter()
        for t in window:
            strategy = t.get("strategy", "")
            action = t.get("action", "")
            if strategy:
                decision_counter[f"strategy:{strategy}"] += 1
            if action:
                decision_counter[f"action:{action}"] += 1
        decisions = [d for d, _ in decision_counter.most_common(5)]

        # Open positions summary (aggregate position sizes)
        position_sizes = [t.get("position_size", 0) for t in window if "position_size" in t]
        open_positions = {}
        if position_sizes:
            open_positions = {
                "mean_position_size": float(sum(position_sizes)) / len(position_sizes),
                "max_position_size": float(max(position_sizes)),
                "min_position_size": float(min(position_sizes)),
                "count": len(position_sizes),
            }

        # Risk summary
        drawdowns = [t.get("drawdown_pct", 0.0) for t in window if "drawdown_pct" in t]
        overrides = sum(1 for t in window if t.get("human_override", False))
        errors = sum(1 for t in window if t.get("error", False))
        risk_summary = {
            "mean_drawdown_pct": float(sum(drawdowns) / len(drawdowns)) if drawdowns else 0.0,
            "max_drawdown_pct": float(max(drawdowns)) if drawdowns else 0.0,
            "human_overrides": overrides,
            "errors": errors,
            "override_rate": overrides / len(window),
        }

        # Tool usage
        tool_counter: Counter = Counter()
        for t in window:
            for tool in t.get("tools_used", []):
                tool_counter[tool] += 1
        top_tools = [tool for tool, _ in tool_counter.most_common(5)]

        # Confidence stats
        confidences = [t.get("confidence", 0.5) for t in window]
        mean_conf = sum(confidences) / len(confidences)

        return {
            "goals": goals,
            "key_decisions": decisions,
            "open_positions": open_positions,
            "risk_summary": risk_summary,
            "turn_range": turn_range,
            "turn_count": len(window),
            "top_tools": top_tools,
            "mean_confidence": mean_conf,
        }

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def get_context(self, current_turn: int) -> dict:
        """
        Return the last episode summary + last *raw_context_window* raw turns.
        """
        last_episode = self._episodes[-1] if self._episodes else None
        recent_raw = self._raw_turns[-self.raw_context_window:]

        return {
            "last_episode": last_episode,
            "recent_turns": recent_raw,
            "total_turns": len(self._raw_turns),
            "total_episodes": len(self._episodes),
            "current_turn": current_turn,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compression_ratio(self) -> float:
        """
        (total_turns - context_turns) / total_turns

        context_turns = one episode block + recent raw window
        """
        total = len(self._raw_turns)
        if total == 0:
            return 0.0
        context_turns = min(self.raw_context_window, total)
        compressed = max(0, total - context_turns)
        return compressed / total

    def __len__(self) -> int:
        return len(self._raw_turns)

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(turns={len(self._raw_turns)}, "
            f"episodes={len(self._episodes)}, "
            f"compression={self.compression_ratio():.2%})"
        )
