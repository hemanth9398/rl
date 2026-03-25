"""Consolidation: periodic graph update from episode traces."""
import logging
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from memory.graph import MemoryGraph, make_skill_node
from memory.episode_store import EpisodeStore

logger = logging.getLogger(__name__)


class Consolidator:
    """Runs periodically to extract skills, update edges, and apply decay."""

    def __init__(
        self,
        graph: MemoryGraph,
        episode_store: EpisodeStore,
        consolidate_every: int = 20,
        min_chunk_freq: int = 3,
        merge_coactivation_threshold: float = 0.8,
        decay_factor: float = 0.95,
    ) -> None:
        self.graph = graph
        self.episode_store = episode_store
        self.consolidate_every = consolidate_every
        self.min_chunk_freq = min_chunk_freq
        self.merge_threshold = merge_coactivation_threshold
        self.decay_factor = decay_factor
        self._episode_count_since_last = 0
        self._actions_log: List[str] = []

    def notify_episode(self, episode_data: Dict[str, Any]) -> None:
        """Call after each episode. Triggers consolidation when due."""
        self._episode_count_since_last += 1
        if self._episode_count_since_last >= self.consolidate_every:
            self.run()
            self._episode_count_since_last = 0

    def run(self) -> List[str]:
        """Run a full consolidation pass. Returns list of action strings."""
        self._actions_log = []
        recent = self.episode_store.get_recent(n=self.consolidate_every * 2)
        if not recent:
            return self._actions_log

        verified = [ep for ep in recent if ep.verified]

        self._promote_frequent_chunks(verified)
        self._update_transition_edges(recent)
        self._decay_graph()
        self._check_splits()

        logger.info("Consolidation done: %s", self._actions_log)
        return self._actions_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _promote_frequent_chunks(self, episodes: List) -> None:
        """Promote frequently co-occurring skill pairs to new combined nodes."""
        chunk_counter: Counter = Counter()
        for ep in episodes:
            skills = ep.skills_used
            if len(skills) >= 2:
                for i in range(len(skills) - 1):
                    pair = (skills[i], skills[i + 1])
                    chunk_counter[pair] += 1

        for (s1, s2), freq in chunk_counter.items():
            if freq < self.min_chunk_freq:
                continue
            combined_id = f"skill_combined_{s1}_{s2}"
            if self.graph.has_node(combined_id):
                continue
            n1 = self.graph.get_node(s1)
            n2 = self.graph.get_node(s2)
            if not n1 or not n2:
                continue
            label = f"{n1.get('label', s1)} + {n2.get('label', s2)}"
            self.graph.add_skill_node(
                node_id=combined_id,
                label=label,
                domain=n1.get("domain", "math"),
                topic=n1.get("topic", "general"),
                trigger={
                    "keywords": (
                        n1.get("trigger", {}).get("keywords", [])
                        + n2.get("trigger", {}).get("keywords", [])
                    )
                },
                procedure=(
                    n1.get("procedure", []) + n2.get("procedure", [])
                ),
            )
            action = f"Promoted combined skill: {combined_id} (freq={freq})"
            self._actions_log.append(action)
            logger.debug(action)

    def _update_transition_edges(self, episodes: List) -> None:
        """Update skill→skill transition edge weights from episode traces."""
        transition_counts: Counter = Counter()
        for ep in episodes:
            skills = ep.skills_used
            for i in range(len(skills) - 1):
                transition_counts[(skills[i], skills[i + 1])] += 1

        for (src, dst), count in transition_counts.items():
            if src in self.graph.graph.nodes and dst in self.graph.graph.nodes:
                self.graph.update_edge_weight(src, dst, delta=count * 0.1)
                action = f"Updated edge {src}→{dst} (+{count * 0.1:.2f})"
                self._actions_log.append(action)

    def _decay_graph(self) -> None:
        """Apply recency decay to all edge weights."""
        self.graph.decay_all(factor=self.decay_factor)
        self._actions_log.append(f"Decayed all edges by factor {self.decay_factor}")

    def _check_splits(self) -> None:
        """Check if any skill should be split by topic (bimodal success)."""
        skills = self.graph.get_all_skills()
        for skill in skills:
            node_id = skill["id"]
            recent_uses = skill.get("recent_uses", [])
            if len(recent_uses) < 10:
                continue
            # Count success by topic (we approximate with timestamps)
            total = len(recent_uses)
            successes = sum(1 for _, s in recent_uses if s)
            sr = successes / total
            # bimodal detection: very low or very high success rate
            if 0.2 < sr < 0.8:
                # not clearly bimodal; skip
                continue
            # nothing to split for simple detection; log
            if sr < 0.2:
                action = f"Skill {node_id} has low success rate {sr:.2f}; consider review"
                self._actions_log.append(action)
