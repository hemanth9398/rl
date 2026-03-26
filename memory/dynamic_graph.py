"""DynamicMemoryGraph: extends MemoryGraph to build nodes/edges from any task."""
from typing import Any, Dict, List, Optional

from memory.graph import MemoryGraph, make_skill_node, make_concept_node, make_error_node


class DynamicMemoryGraph(MemoryGraph):
    """Extends MemoryGraph to build nodes/edges dynamically from any task.

    Unlike the base MemoryGraph with fixed seed skills for math,
    this graph grows organically as the system solves new types of problems.
    """

    def __init__(self) -> None:
        super().__init__()
        self._auto_id_counter = 0

    # ------------------------------------------------------------------
    # Auto-ID helper
    # ------------------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._auto_id_counter += 1
        return f"{prefix}_{self._auto_id_counter:06d}"

    # ------------------------------------------------------------------
    # Dynamic node creation
    # ------------------------------------------------------------------

    def add_learned_skill(
        self,
        label: str,
        domain: str,
        topic: str,
        trigger: Dict,
        procedure: List[str],
        source_episode: Optional[str] = None,
    ) -> str:
        """Create a new skill node from a solved subtask.

        Returns the auto-generated node_id.
        """
        node_id = self._next_id("dyn_skill")
        node = make_skill_node(
            node_id,
            label=label,
            domain=domain,
            topic=topic,
            trigger=trigger,
            procedure=procedure,
        )
        if source_episode:
            node["source_episode"] = source_episode
        self.graph.add_node(node_id, **node)
        return node_id

    def add_learned_concept(
        self,
        label: str,
        domain: str,
        keywords: List[str],
    ) -> str:
        """Create a new concept node from observed patterns."""
        node_id = self._next_id("dyn_concept")
        node = make_concept_node(node_id, label=label, domain=domain, keywords=keywords)
        self.graph.add_node(node_id, **node)
        return node_id

    def add_learned_error(
        self,
        label: str,
        diagnostics: str,
        repair_hint: str,
        related_skill: Optional[str] = None,
    ) -> str:
        """Create a new error node from a failure."""
        node_id = self._next_id("dyn_error")
        node = make_error_node(node_id, label=label, diagnostics=diagnostics, repair_hint=repair_hint)
        self.graph.add_node(node_id, **node)
        if related_skill and self.has_node(related_skill):
            self.add_edge(related_skill, node_id, "causes_error", weight=0.5)
        return node_id

    # ------------------------------------------------------------------
    # Subtask completion tracking
    # ------------------------------------------------------------------

    def record_subtask_completion(
        self,
        subtask_id: str,
        skill_used: str,
        success: bool,
        predecessor_skill: Optional[str] = None,
    ) -> None:
        """Record a subtask completion and update graph accordingly.

        - Updates skill stats
        - Adds/strengthens transition edge from predecessor
        - If failure, adds causes_error edge
        """
        if self.has_node(skill_used):
            self.update_node_stats(skill_used, success=success)

        if predecessor_skill and self.has_node(predecessor_skill) and self.has_node(skill_used):
            self.update_edge_weight(predecessor_skill, skill_used, delta=0.1 if success else -0.05)

        if not success and self.has_node(skill_used):
            # Ensure at least a generic error node exists
            error_label = f"failure_in_{skill_used}"
            # Check if we already have an error linked from this skill
            existing_errors = self.get_error_nodes(skill_used)
            if not existing_errors:
                err_id = self.add_learned_error(
                    label=error_label,
                    diagnostics=f"subtask_{subtask_id}_failed",
                    repair_hint="Retry with golden thought hint or different approach",
                    related_skill=skill_used,
                )

    # ------------------------------------------------------------------
    # Similarity and chain queries
    # ------------------------------------------------------------------

    def find_similar_skills(
        self, description: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find skills similar to a description using keyword matching."""
        query_words = set(description.lower().split())
        scored: List[tuple] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "skill":
                continue
            label_words = set(data.get("label", "").lower().split())
            trigger_kws = set(kw.lower() for kw in data.get("trigger", {}).get("keywords", []))
            proc_words = set(
                word.lower()
                for step in data.get("procedure", [])
                for word in step.split()
            )
            all_words = label_words | trigger_kws | proc_words
            overlap = len(query_words & all_words)
            if overlap > 0:
                scored.append((overlap, dict(data)))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:top_k]]

    def get_skill_chain(
        self, start_skill: str, max_length: int = 5
    ) -> List[Dict[str, Any]]:
        """Follow highest-weight transition edges to build a skill chain."""
        chain: List[Dict[str, Any]] = []
        visited: set = set()
        current = start_skill
        while current and len(chain) < max_length and current not in visited:
            node = self.get_node(current)
            if node is None:
                break
            chain.append(node)
            visited.add(current)
            transitions = self.get_transitions(current)
            if not transitions:
                break
            current = transitions[0][0]  # highest-weight next node
        return chain

    def merge_duplicate_skills(self, threshold: float = 0.9) -> List[str]:
        """Find and merge near-duplicate skill nodes.

        Two skill nodes are considered duplicates if the overlap between their
        combined keyword sets divided by the larger set exceeds *threshold*.
        The node with lower use_count is merged into the other.

        Returns a list of merged (removed) node_ids.
        """
        skills = self.get_all_skills()
        merged: List[str] = []
        merge_map: Dict[str, str] = {}  # removed_id → kept_id

        def _keywords(node: Dict) -> set:
            label_words = set(node.get("label", "").lower().split())
            trigger_kws = set(kw.lower() for kw in node.get("trigger", {}).get("keywords", []))
            return label_words | trigger_kws

        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                a = skills[i]
                b = skills[j]
                if a["id"] in merge_map or b["id"] in merge_map:
                    continue
                kws_a = _keywords(a)
                kws_b = _keywords(b)
                if not kws_a or not kws_b:
                    continue
                union = kws_a | kws_b
                inter = kws_a & kws_b
                jaccard_similarity = len(inter) / len(union)
                if jaccard_similarity >= threshold:
                    # Keep the node with more uses
                    keep, remove = (a, b) if a.get("use_count", 0) >= b.get("use_count", 0) else (b, a)
                    keep_id = keep["id"]
                    remove_id = remove["id"]
                    # Redirect all edges from/to remove_id to keep_id
                    for src, dst, data in list(self.graph.in_edges(remove_id, data=True)):
                        if src != keep_id:
                            self.graph.add_edge(src, keep_id, **data)
                    for src, dst, data in list(self.graph.out_edges(remove_id, data=True)):
                        if dst != keep_id:
                            self.graph.add_edge(keep_id, dst, **data)
                    self.graph.remove_node(remove_id)
                    merge_map[remove_id] = keep_id
                    merged.append(remove_id)

        return merged
