"""Two-stage retrieval: BM25 over episodes + graph activation over skills."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

from memory.graph import MemoryGraph
from memory.episode_store import EpisodeStore, Episode


@dataclass
class RetrievedItem:
    skill_node: Dict[str, Any]
    example_episodes: List[Episode] = field(default_factory=list)
    suggested_plan: List[str] = field(default_factory=list)
    confidence: float = 0.0


class Retriever:
    """BM25 + graph activation retriever."""

    def __init__(self, graph: MemoryGraph, episode_store: EpisodeStore) -> None:
        self.graph = graph
        self.episode_store = episode_store

    def retrieve(
        self,
        problem_text: str,
        topic: str = "",
        domain: str = "math",
        top_k: int = 5,
    ) -> List[RetrievedItem]:
        """Main retrieval entry point."""
        # Stage 1: BM25 over recent episodes
        similar_eps = self.episode_store.get_similar_episodes(
            problem_text, topic, limit=10
        )

        # Stage 2: collect skill sequences from similar episodes
        activated_skills: Dict[str, float] = {}
        for ep in similar_eps:
            for skill_id in ep.skills_used:
                activated_skills[skill_id] = activated_skills.get(skill_id, 0) + 1.0

        # Stage 3: graph-based candidate skills (keyword / topic matching)
        features = {
            "topic": topic,
            "domain": domain,
            "keywords": problem_text.lower().split(),
        }
        graph_candidates = self.graph.get_candidate_skills(features)

        # Stage 4: spread activation through transition edges
        for skill in graph_candidates[:5]:
            sid = skill["id"]
            base = activated_skills.get(sid, 0.0) + skill.get("use_count", 0) * 0.1
            activated_skills[sid] = base + 1.0
            for next_id, w in self.graph.get_transitions(sid):
                activated_skills[next_id] = activated_skills.get(next_id, 0) + w * 0.5

        # Sort and build result
        sorted_skills = sorted(
            activated_skills.items(), key=lambda x: -x[1]
        )[:top_k]

        results: List[RetrievedItem] = []
        for skill_id, score in sorted_skills:
            node = self.graph.get_node(skill_id)
            if node is None or node.get("type") != "skill":
                continue
            # gather example episodes that used this skill
            eps = [e for e in similar_eps if skill_id in e.skills_used][:3]
            # suggested plan: procedure from skill node
            plan = node.get("procedure", [])
            # check successor transitions
            transitions = self.graph.get_transitions(skill_id)
            for next_id, _ in transitions[:2]:
                next_node = self.graph.get_node(next_id)
                if next_node and next_node.get("type") == "skill":
                    plan = plan + [f"then: {next_node['label']}"]

            # compute success rate
            use_count = node.get("use_count", 0)
            success_count = node.get("success_count", 0)
            sr = success_count / use_count if use_count > 0 else 0.5
            confidence = (score / (sorted_skills[0][1] + 1e-9)) * sr

            results.append(
                RetrievedItem(
                    skill_node=node,
                    example_episodes=eps,
                    suggested_plan=plan,
                    confidence=confidence,
                )
            )
        return results

    def bm25_search(self, query: str, corpus: List[str], top_k: int = 5) -> List[int]:
        """Return top-k indices from corpus using BM25 (if available)."""
        if not corpus:
            return []
        if _HAS_BM25:
            tokenized = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query.lower().split())
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            return ranked[:top_k]
        # fallback: token overlap
        q_tokens = set(query.lower().split())
        scored = []
        for i, doc in enumerate(corpus):
            overlap = len(q_tokens & set(doc.lower().split()))
            scored.append((overlap, i))
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:top_k]]
