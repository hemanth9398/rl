"""Teacher agent (T1): task decomposition and orchestration."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memory.graph import MemoryGraph
from memory.retrieval import Retriever
from agents.tree_of_thought import ThoughtBranch

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """A single sub-problem created by the Teacher."""

    task_id: str
    description: str
    assigned_agent: int
    context: Dict[str, Any]       # from memory graph
    depends_on: List[str]         # task_ids this depends on


@dataclass
class TaskPlan:
    """The Teacher's decomposition of a problem into subtasks."""

    plan_id: str
    problem: Dict[str, Any]
    subtasks: List[SubTask]
    dependencies: Dict[str, List[str]]  # task_id → depends_on task_ids
    strategy: str                        # reasoning for the split


class Teacher:
    """LLM-based task decomposition and orchestration agent (T1).

    Reads the problem, queries the MemoryGraph for past solutions, splits the
    problem into N subtasks with dependencies, assigns each to a SubAgent, can
    generate "Golden Thought" hints when SubAgents are stuck, and synthesizes
    the final answer once the Validator returns.

    When a :class:`~models.model_registry.ModelRegistry` is provided the
    decomposition, golden-thought generation, and synthesis all use the
    Teacher's 7B LLM.  Without a registry the class falls back to the
    original heuristic-based approach so that existing tests continue to pass.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        retriever: Retriever,
        policy: Optional[Any] = None,
        num_agents: int = 3,
        registry: Optional[Any] = None,
    ) -> None:
        self.graph = graph
        self.retriever = retriever
        self.policy = policy
        self.num_agents = num_agents
        self.registry = registry

        # Lazily-created LLM module (only when registry is provided)
        self._llm_teacher: Optional[Any] = None
        if registry is not None:
            try:
                from models.llm_teacher import LLMTeacherModule
                self._llm_teacher = LLMTeacherModule(registry)
            except Exception as exc:
                logger.warning("Teacher: could not init LLMTeacherModule: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, problem: Dict[str, Any]) -> TaskPlan:
        """Split problem into subtasks with dependencies.

        When a registry is available, uses the Teacher's LLM to generate a
        structured decomposition.  Falls back to heuristic rules when the LLM
        is unavailable or returns an unparseable response.
        """
        features = self._analyze_problem(problem)
        topic = features.get("topic", "general")
        domain = features.get("domain", "math")
        retrieved = self.retriever.retrieve(
            problem_text=problem.get("statement", ""),
            topic=topic,
            domain=domain,
            top_k=5,
        )

        # --- LLM path ---
        if self._llm_teacher is not None:
            try:
                complexity = features.get("complexity", "low")
                difficulty = features.get("difficulty", 1)
                if complexity == "high" or difficulty >= 4:
                    n_subtasks = min(self.num_agents, 3)
                elif complexity == "medium" or difficulty >= 2:
                    n_subtasks = min(self.num_agents, 2)
                else:
                    n_subtasks = 1

                skill_context = {
                    "retrieved_skills": [item.skill_node for item in retrieved],
                    "topic": topic,
                    "domain": domain,
                }
                subtask_dicts = self._llm_teacher.decompose(
                    problem, n_subtasks=n_subtasks, context=skill_context
                )
                return self._build_plan_from_dicts(problem, subtask_dicts, retrieved, features)
            except Exception as exc:
                logger.warning("Teacher.decompose LLM path failed, using heuristic: %s", exc)

        # --- Heuristic fallback ---
        return self._plan_from_features(problem, features, retrieved)

    def generate_golden_thought(
        self,
        subtask: SubTask,
        failed_branches: List[ThoughtBranch],
    ) -> str:
        """Generate correct reasoning hint when SubAgent is stuck.

        Uses the Teacher's LLM when a registry is available; otherwise
        falls back to heuristic hints derived from the memory graph.
        """
        description = subtask.description

        # --- LLM path ---
        if self._llm_teacher is not None:
            try:
                return self._llm_teacher.golden_thought(
                    subtask_description=description,
                    failed_branches=failed_branches,
                )
            except Exception as exc:
                logger.warning(
                    "Teacher.generate_golden_thought LLM failed, using heuristic: %s", exc
                )

        # --- Heuristic fallback ---
        retrieved = self.retriever.retrieve(
            problem_text=description,
            topic=subtask.context.get("topic", "general"),
            domain=subtask.context.get("domain", "math"),
            top_k=3,
        )
        hints: List[str] = []
        for item in retrieved:
            skill = item.skill_node
            plan_steps = item.suggested_plan
            hints.append(
                f"Use skill '{skill.get('label', '?')}': "
                + " → ".join(plan_steps[:3])
            )

        failed_summary = "; ".join(
            f"branch '{b.branch_id}' (conf={b.confidence:.2f})" for b in failed_branches
        )

        golden = (
            f"[Golden Thought for subtask '{description[:60]}']\n"
            + (f"Retrieved hints: {' | '.join(hints)}\n" if hints else "")
            + f"Failed branches: {failed_summary}\n"
            + "Approach: break the problem into smaller steps, "
            "verify each intermediate result, and apply the most relevant skill."
        )
        return golden

    def synthesize_final(
        self,
        plan: TaskPlan,
        subtask_results: Dict[str, Any],
    ) -> str:
        """Combine subtask results into final answer.

        Uses the Teacher's LLM when a registry is available; otherwise
        concatenates subtask answers with a simple joiner.
        """
        # --- LLM path ---
        if self._llm_teacher is not None:
            try:
                return self._llm_teacher.synthesize(plan.problem, subtask_results)
            except Exception as exc:
                logger.warning("Teacher.synthesize_final LLM failed, using fallback: %s", exc)

        # --- Heuristic fallback ---
        parts: List[str] = []
        for subtask in plan.subtasks:
            result = subtask_results.get(subtask.task_id)
            if result is None:
                continue
            if hasattr(result, "selected_branch"):
                answer = result.selected_branch.answer
            elif isinstance(result, dict):
                answer = result.get("answer", "")
            else:
                answer = str(result)
            parts.append(f"[{subtask.description[:40]}]: {answer}")

        return " | ".join(parts) if parts else "No results"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features: topic, keywords, complexity estimate, etc."""
        statement = problem.get("statement", "")
        topic = problem.get("topic", "general")
        domain = problem.get("domain", "math")
        difficulty = problem.get("difficulty", 1)

        keywords = [w.lower() for w in statement.split() if len(w) > 3]
        complexity = "high" if difficulty >= 4 else ("medium" if difficulty >= 2 else "low")

        return {
            "topic": topic,
            "domain": domain,
            "keywords": keywords[:20],
            "complexity": complexity,
            "difficulty": difficulty,
            "statement_length": len(statement),
        }

    def _plan_from_features(
        self,
        problem: Dict[str, Any],
        features: Dict[str, Any],
        retrieved: List,
    ) -> TaskPlan:
        """Build task plan using features and retrieved skills."""
        complexity = features.get("complexity", "low")
        difficulty = features.get("difficulty", 1)
        statement = problem.get("statement", "")

        # Determine number of subtasks based on complexity
        if complexity == "high" or difficulty >= 4:
            n_subtasks = min(self.num_agents, 3)
        elif complexity == "medium" or difficulty >= 2:
            n_subtasks = min(self.num_agents, 2)
        else:
            n_subtasks = 1

        # Build context from retrieved skills
        skill_context = {
            "retrieved_skills": [item.skill_node for item in retrieved],
            "suggested_plans": [item.suggested_plan for item in retrieved],
            "topic": features.get("topic", "general"),
            "domain": features.get("domain", "math"),
            "problem": problem,
            "answer_spec": problem.get("answer_spec", {}),
        }

        subtasks: List[SubTask] = []
        dependencies: Dict[str, List[str]] = {}

        if n_subtasks == 1:
            t = SubTask(
                task_id=str(uuid.uuid4()),
                description=statement[:200],
                assigned_agent=0,
                context=dict(skill_context),
                depends_on=[],
            )
            subtasks.append(t)
            dependencies[t.task_id] = []

        elif n_subtasks == 2:
            # Split: understand/setup + solve/verify
            t0 = SubTask(
                task_id=str(uuid.uuid4()),
                description=f"Understand and set up: {statement[:100]}",
                assigned_agent=0,
                context=dict(skill_context),
                depends_on=[],
            )
            t1 = SubTask(
                task_id=str(uuid.uuid4()),
                description=f"Solve and verify: {statement[:100]}",
                assigned_agent=1 % self.num_agents,
                context=dict(skill_context),
                depends_on=[t0.task_id],
            )
            subtasks = [t0, t1]
            dependencies[t0.task_id] = []
            dependencies[t1.task_id] = [t0.task_id]

        else:
            # 3-way split: parse + solve + verify
            t0 = SubTask(
                task_id=str(uuid.uuid4()),
                description=f"Parse and retrieve relevant skills: {statement[:80]}",
                assigned_agent=0,
                context=dict(skill_context),
                depends_on=[],
            )
            t1 = SubTask(
                task_id=str(uuid.uuid4()),
                description=f"Solve main computation: {statement[:80]}",
                assigned_agent=1 % self.num_agents,
                context=dict(skill_context),
                depends_on=[t0.task_id],
            )
            t2 = SubTask(
                task_id=str(uuid.uuid4()),
                description=f"Verify and synthesize final answer: {statement[:80]}",
                assigned_agent=2 % self.num_agents,
                context=dict(skill_context),
                depends_on=[t1.task_id],
            )
            subtasks = [t0, t1, t2]
            dependencies[t0.task_id] = []
            dependencies[t1.task_id] = [t0.task_id]
            dependencies[t2.task_id] = [t1.task_id]

        strategy = (
            f"Split into {n_subtasks} subtask(s) based on complexity='{complexity}'. "
            f"Retrieved {len(retrieved)} skill(s)."
        )

        return TaskPlan(
            plan_id=str(uuid.uuid4()),
            problem=problem,
            subtasks=subtasks,
            dependencies=dependencies,
            strategy=strategy,
        )

    def _build_plan_from_dicts(
        self,
        problem: Dict[str, Any],
        subtask_dicts: List[Dict[str, Any]],
        retrieved: List,
        features: Dict[str, Any],
    ) -> TaskPlan:
        """Convert a list of subtask dicts (from LLM) into a TaskPlan."""
        skill_context = {
            "retrieved_skills": [item.skill_node for item in retrieved],
            "suggested_plans": [item.suggested_plan for item in retrieved],
            "topic": features.get("topic", "general"),
            "domain": features.get("domain", "math"),
            "problem": problem,
            "answer_spec": problem.get("answer_spec", {}),
        }

        subtasks: List[SubTask] = []
        task_id_map: Dict[int, str] = {}  # index → uuid
        dependencies: Dict[str, List[str]] = {}

        for idx, d in enumerate(subtask_dicts):
            tid = str(uuid.uuid4())
            task_id_map[idx] = tid

            # Resolve depends_on (list of prior indices)
            raw_deps = d.get("depends_on", [])
            resolved_deps: List[str] = []
            for dep_idx in raw_deps:
                if isinstance(dep_idx, int) and dep_idx in task_id_map:
                    resolved_deps.append(task_id_map[dep_idx])

            agent_idx = idx % self.num_agents
            subtasks.append(SubTask(
                task_id=tid,
                description=d.get("description", problem.get("statement", "")[:200]),
                assigned_agent=agent_idx,
                context=dict(skill_context),
                depends_on=resolved_deps,
            ))
            dependencies[tid] = resolved_deps

        strategy = (
            f"LLM-generated split into {len(subtasks)} subtask(s). "
            f"Retrieved {len(retrieved)} skill(s)."
        )

        return TaskPlan(
            plan_id=str(uuid.uuid4()),
            problem=problem,
            subtasks=subtasks,
            dependencies=dependencies,
            strategy=strategy,
        )
