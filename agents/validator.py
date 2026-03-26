"""Validator agent (T2): merge SubAgent outputs, validate, update graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memory.dynamic_graph import DynamicMemoryGraph
from memory.episode_store import EpisodeStore
from verifier.verifier import Verifier
from agents.teacher import SubTask, TaskPlan
from agents.sub_agent import SubAgentResult


@dataclass
class SubTaskResult:
    """Per-subtask validation outcome."""

    task_id: str
    answer: str
    passed: bool
    diagnostics: List[str]


@dataclass
class ValidationResult:
    """Full validation outcome for one episode."""

    merged_answer: str
    passed: bool
    diagnostics: List[str]
    subtask_results: List[SubTaskResult]
    graph_updates: List[Dict[str, Any]]  # nodes/edges to add to graph
    reward_signals: Dict[str, float]     # per-agent rewards


class Validator:
    """Merges SubAgent outputs, validates, and updates memory graph (T2).

    Steps:
    1. Check each subtask result independently.
    2. Merge results respecting dependency order.
    3. Validate merged answer against problem.
    4. Compute reward signals for all agents.
    5. Generate graph updates (new nodes/edges).
    """

    # Reward constants
    REWARD_MISSING_RESULT: float = -0.5
    REWARD_TASK_SUCCESS: float = 1.0
    REWARD_TASK_FAILURE: float = -0.3
    REWARD_FINAL_PASS: float = 1.0
    REWARD_FINAL_FAIL: float = -0.5
    REWARD_VALIDATOR_PASS: float = 0.5
    REWARD_VALIDATOR_FAIL: float = -0.5

    def __init__(
        self,
        graph: DynamicMemoryGraph,
        verifier: Verifier,
        episode_store: EpisodeStore,
    ) -> None:
        self.graph = graph
        self.verifier = verifier
        self.episode_store = episode_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        plan: TaskPlan,
        agent_results: Dict[str, SubAgentResult],
        problem: Dict[str, Any],
    ) -> ValidationResult:
        """Run the full validation pipeline."""
        # Step 1: validate each subtask
        subtask_results: List[SubTaskResult] = []
        for subtask in plan.subtasks:
            agent_result = agent_results.get(subtask.task_id)
            if agent_result is None:
                subtask_results.append(
                    SubTaskResult(
                        task_id=subtask.task_id,
                        answer="",
                        passed=False,
                        diagnostics=["no_result"],
                    )
                )
            else:
                subtask_results.append(
                    self._validate_subtask(subtask, agent_result, problem)
                )

        # Step 2: merge following dependency order
        merged_answer = self._merge_results(plan, subtask_results)

        # Step 3: full-problem validation
        verify_result = self.verifier.verify(problem, merged_answer)
        final_passed = verify_result.passed

        # Step 4: rewards
        reward_signals = self._compute_rewards(plan, subtask_results, final_passed)

        # Step 5: graph updates
        graph_updates = self._generate_graph_updates(plan, subtask_results, problem)

        diagnostics = list(verify_result.diagnostics)
        if verify_result.detail:
            diagnostics.append(verify_result.detail)

        return ValidationResult(
            merged_answer=merged_answer,
            passed=final_passed,
            diagnostics=diagnostics,
            subtask_results=subtask_results,
            graph_updates=graph_updates,
            reward_signals=reward_signals,
        )

    def apply_graph_updates(self, updates: List[Dict]) -> None:
        """Apply generated updates to the dynamic memory graph."""
        for update in updates:
            update_type = update.get("type")
            if update_type == "skill":
                self.graph.add_learned_skill(
                    label=update.get("label", "learned_skill"),
                    domain=update.get("domain", "general"),
                    topic=update.get("topic", "general"),
                    trigger=update.get("trigger", {}),
                    procedure=update.get("procedure", []),
                    source_episode=update.get("source_episode"),
                )
            elif update_type == "concept":
                self.graph.add_learned_concept(
                    label=update.get("label", "learned_concept"),
                    domain=update.get("domain", "general"),
                    keywords=update.get("keywords", []),
                )
            elif update_type == "error":
                self.graph.add_learned_error(
                    label=update.get("label", "learned_error"),
                    diagnostics=update.get("diagnostics", ""),
                    repair_hint=update.get("repair_hint", ""),
                    related_skill=update.get("related_skill"),
                )
            elif update_type == "edge":
                src = update.get("src", "")
                dst = update.get("dst", "")
                if src and dst and self.graph.has_node(src) and self.graph.has_node(dst):
                    self.graph.add_edge(
                        src,
                        dst,
                        edge_type=update.get("edge_type", "transition"),
                        weight=update.get("weight", 1.0),
                    )
            elif update_type == "subtask_completion":
                self.graph.record_subtask_completion(
                    subtask_id=update.get("subtask_id", ""),
                    skill_used=update.get("skill_used", ""),
                    success=update.get("success", False),
                    predecessor_skill=update.get("predecessor_skill"),
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_subtask(
        self,
        subtask: SubTask,
        result: SubAgentResult,
        problem: Dict[str, Any],
    ) -> SubTaskResult:
        """Validate a single subtask result."""
        answer = result.selected_branch.answer if result.selected_branch else ""
        if not answer:
            return SubTaskResult(
                task_id=subtask.task_id,
                answer=answer,
                passed=False,
                diagnostics=["empty_answer"],
            )

        # Use verifier if the subtask problem is verifiable
        subtask_problem = subtask.context.get("problem", problem)
        try:
            vr = self.verifier.verify(subtask_problem, answer)
            return SubTaskResult(
                task_id=subtask.task_id,
                answer=answer,
                passed=vr.passed,
                diagnostics=list(vr.diagnostics),
            )
        except Exception as exc:
            # Fallback: trust the SubAgent's own confidence
            passed = result.success and result.selected_branch.confidence > 0.5
            return SubTaskResult(
                task_id=subtask.task_id,
                answer=answer,
                passed=passed,
                diagnostics=[f"verifier_exception: {exc}"],
            )

    def _merge_results(
        self,
        plan: TaskPlan,
        subtask_results: List[SubTaskResult],
    ) -> str:
        """Merge subtask answers into final answer following dependency order."""
        # Build lookup
        result_map: Dict[str, SubTaskResult] = {r.task_id: r for r in subtask_results}

        # Follow topological order (subtasks are already in dependency order)
        parts: List[str] = []
        last_answer = ""
        for subtask in plan.subtasks:
            sr = result_map.get(subtask.task_id)
            if sr and sr.answer:
                last_answer = sr.answer
                parts.append(sr.answer)

        # The final subtask's answer is the best candidate for the merged result
        if last_answer:
            return last_answer
        return " | ".join(parts) if parts else ""

    def _compute_rewards(
        self,
        plan: TaskPlan,
        subtask_results: List[SubTaskResult],
        final_passed: bool,
    ) -> Dict[str, float]:
        """Compute reward signals for teacher, each sub-agent, and global."""
        rewards: Dict[str, float] = {}

        # Per-agent rewards
        result_map = {r.task_id: r for r in subtask_results}
        for subtask in plan.subtasks:
            sr = result_map.get(subtask.task_id)
            agent_key = f"agent_{subtask.assigned_agent}"
            if sr is None:
                rewards[agent_key] = rewards.get(agent_key, 0.0) + self.REWARD_MISSING_RESULT
            elif sr.passed:
                rewards[agent_key] = rewards.get(agent_key, 0.0) + self.REWARD_TASK_SUCCESS
            else:
                rewards[agent_key] = rewards.get(agent_key, 0.0) + self.REWARD_TASK_FAILURE

        # Teacher reward: quality of decomposition
        pass_rate = sum(1 for r in subtask_results if r.passed) / max(len(subtask_results), 1)
        teacher_reward = pass_rate * 0.5 + (self.REWARD_FINAL_PASS if final_passed else self.REWARD_FINAL_FAIL)
        rewards["teacher"] = teacher_reward

        # Global reward
        rewards["global"] = self.REWARD_FINAL_PASS if final_passed else -self.REWARD_FINAL_PASS

        # Validator penalty for approving wrong work
        rewards["validator"] = self.REWARD_VALIDATOR_PASS if final_passed else self.REWARD_VALIDATOR_FAIL

        return rewards

    def _generate_graph_updates(
        self,
        plan: TaskPlan,
        subtask_results: List[SubTaskResult],
        problem: Dict[str, Any],
    ) -> List[Dict]:
        """Generate new skill/concept/error nodes and edges for the graph."""
        updates: List[Dict] = []
        topic = problem.get("topic", "general")
        domain = problem.get("domain", "general")
        statement = problem.get("statement", "")
        keywords = [w.lower() for w in statement.split() if len(w) > 3][:10]

        result_map = {r.task_id: r for r in subtask_results}
        prev_skill_id: Optional[str] = None

        for subtask in plan.subtasks:
            sr = result_map.get(subtask.task_id)
            if sr is None:
                continue

            if sr.passed:
                # Add a new learned skill node
                updates.append({
                    "type": "skill",
                    "label": f"learned: {subtask.description[:50]}",
                    "domain": domain,
                    "topic": topic,
                    "trigger": {"keywords": keywords},
                    "procedure": [subtask.description[:80]],
                    "source_episode": plan.plan_id,
                })
                updates.append({
                    "type": "subtask_completion",
                    "subtask_id": subtask.task_id,
                    "skill_used": subtask.context.get("retrieved_skills", [{}])[0].get("id", "")
                    if subtask.context.get("retrieved_skills") else "",
                    "success": True,
                    "predecessor_skill": prev_skill_id,
                })
                prev_skill_id = None  # will be updated after apply
            else:
                # Add an error node for this failure
                for diag in sr.diagnostics[:2]:
                    updates.append({
                        "type": "error",
                        "label": f"error in: {subtask.description[:40]}",
                        "diagnostics": diag,
                        "repair_hint": "Retry with different approach or golden thought",
                    })

        # Add a concept node for the problem topic if it's new
        if topic and not self.graph.has_node(f"concept_{topic}"):
            updates.append({
                "type": "concept",
                "label": topic.replace("_", " ").title(),
                "domain": domain,
                "keywords": keywords,
            })

        return updates
