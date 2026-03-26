"""Tree-of-Thought engine for SubAgent reasoning."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThoughtBranch:
    """A leaf branch in the thought tree."""

    branch_id: str
    reasoning: str
    answer: str
    confidence: float
    depth: int


@dataclass
class ThoughtNode:
    """A node in the thought tree."""

    node_id: str
    depth: int
    reasoning: str
    score: float
    children: List["ThoughtNode"] = field(default_factory=list)
    is_leaf: bool = False


class TreeOfThought:
    """Binary tree exploration for reasoning branches.

    At each depth level:
    1. Generate ``branch_factor`` candidate reasoning paths.
    2. Score each using the optional GIGPO scorer or a heuristic.
    3. Expand the best branches to the next depth.
    4. Return all leaf branches with scores.
    """

    # Default confidence used for newly generated branches
    DEFAULT_CONFIDENCE: float = 0.5

    def __init__(
        self,
        max_depth: int = 3,
        branch_factor: int = 2,
        scorer: Optional[Any] = None,
    ) -> None:
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.scorer = scorer  # GIGPOScorer or None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explore(
        self,
        problem_text: str,
        context: Dict[str, Any],
        solver: Any,
        graph: Any,
    ) -> List[ThoughtBranch]:
        """Generate and evaluate thought branches.

        Returns all leaf branches ordered by score (descending).
        """
        root = ThoughtNode(
            node_id=str(uuid.uuid4()),
            depth=0,
            reasoning=f"Root: {problem_text[:120]}",
            score=1.0,
        )
        leaves: List[ThoughtNode] = self._expand(root, context, solver, max_depth=self.max_depth)

        branches: List[ThoughtBranch] = []
        for i, leaf in enumerate(leaves):
            if i == 0:
                branch_id = "left"
            elif i == 1:
                branch_id = "right"
            else:
                branch_id = f"branch_{i}"
            # Derive answer from reasoning heuristic
            answer = self._extract_answer(leaf.reasoning, problem_text, context, solver)
            branches.append(
                ThoughtBranch(
                    branch_id=branch_id,
                    reasoning=leaf.reasoning,
                    answer=answer,
                    confidence=max(0.0, min(1.0, leaf.score)),
                    depth=leaf.depth,
                )
            )
        branches.sort(key=lambda b: b.confidence, reverse=True)
        return branches

    def select_best(self, branches: List[ThoughtBranch]) -> ThoughtBranch:
        """Select the branch with highest confidence."""
        if not branches:
            return ThoughtBranch(
                branch_id="empty",
                reasoning="No branches available",
                answer="",
                confidence=0.0,
                depth=0,
            )
        return max(branches, key=lambda b: b.confidence)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _expand(
        self,
        node: ThoughtNode,
        context: Dict[str, Any],
        solver: Any,
        max_depth: int,
    ) -> List[ThoughtNode]:
        """Recursively expand the tree and return leaf nodes."""
        if node.depth >= max_depth or node.is_leaf:
            node.is_leaf = True
            return [node]

        children = self._generate_branches(node, context, solver)
        for child in children:
            child.score = self._score_branch(child, context)

        # Keep top-2 children to bound the search
        children.sort(key=lambda n: n.score, reverse=True)
        children = children[: self.branch_factor]

        leaves: List[ThoughtNode] = []
        for child in children:
            leaves.extend(self._expand(child, context, solver, max_depth))
        return leaves if leaves else [node]

    def _generate_branches(
        self,
        parent: ThoughtNode,
        context: Dict[str, Any],
        solver: Any,
    ) -> List[ThoughtNode]:
        """Generate child branches from a thought node.

        Uses retrieved skills and heuristic reasoning steps to diversify
        the exploration without requiring an external LLM.
        """
        skills = context.get("retrieved_skills", [])
        skill_labels = [s.get("label", "unknown") for s in skills[:3]]

        strategies = [
            f"Direct approach: apply {skill_labels[0] if skill_labels else 'general method'}",
            (
                f"Alternative: decompose problem and use "
                f"{skill_labels[1] if len(skill_labels) > 1 else 'step-by-step reasoning'}"
            ),
        ]

        branches: List[ThoughtNode] = []
        for strategy in strategies:
            reasoning = f"{parent.reasoning} → [{strategy}]"
            branches.append(
                ThoughtNode(
                    node_id=str(uuid.uuid4()),
                    depth=parent.depth + 1,
                    reasoning=reasoning,
                    score=self.DEFAULT_CONFIDENCE,
                )
            )
        return branches

    def _score_branch(self, node: ThoughtNode, context: Dict[str, Any]) -> float:
        """Score a branch using GIGPO scorer or heuristic."""
        if self.scorer is not None:
            branch = ThoughtBranch(
                branch_id=node.node_id,
                reasoning=node.reasoning,
                answer="",
                confidence=self.DEFAULT_CONFIDENCE,
                depth=node.depth,
            )
            scores = self.scorer.score_branches([branch], context)
            return scores[0] if scores else self.DEFAULT_CONFIDENCE

        # Heuristic: prefer deeper nodes and those referencing known skills
        depth_bonus = node.depth * 0.05
        skill_bonus = 0.1 if any(kw in node.reasoning.lower() for kw in ["apply", "use", "solve"]) else 0.0
        return self.DEFAULT_CONFIDENCE + depth_bonus + skill_bonus

    def _extract_answer(
        self,
        reasoning: str,
        problem_text: str,
        context: Dict[str, Any],
        solver: Any,
    ) -> str:
        """Attempt to extract/generate an answer using the solver."""
        if solver is not None:
            problem = context.get("problem", {})
            if not problem:
                problem = {
                    "statement": problem_text,
                    "topic": context.get("topic", "general"),
                    "domain": context.get("domain", "math"),
                    "answer_spec": context.get("answer_spec", {}),
                }

            # Prefer .solve() if available (LLMSolver provides this)
            if hasattr(solver, "solve"):
                try:
                    result = solver.solve(problem)
                    if result.success and result.answer:
                        return result.answer
                    else:
                        logger.debug(
                            "_extract_answer: solver.solve() returned success=%s answer=%r",
                            result.success,
                            result.answer,
                        )
                except Exception as exc:
                    logger.warning("_extract_answer: solver.solve() raised: %s", exc)

            # Fallback: SymPy Solver exposes attempt_solve()
            elif hasattr(solver, "attempt_solve"):
                try:
                    topic = problem.get("topic", "general")
                    default_skill = (
                        solver.default_skill_for_topic(topic)
                        if hasattr(solver, "default_skill_for_topic")
                        else None
                    )
                    skills = [default_skill] if default_skill else []
                    if skills:
                        result = solver.attempt_solve(problem, skills)
                        if result.success and result.answer:
                            return result.answer
                        else:
                            logger.debug(
                                "_extract_answer: attempt_solve returned success=%s answer=%r",
                                result.success,
                                result.answer,
                            )
                except Exception as exc:
                    logger.warning("_extract_answer: solver.attempt_solve() raised: %s", exc)

        return f"[reasoning: {reasoning[-80:]}]"
