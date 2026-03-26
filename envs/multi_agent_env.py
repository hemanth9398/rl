"""Multi-Agent Environment: orchestrates Teacher → SubAgents → Validator loop."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from memory.dynamic_graph import DynamicMemoryGraph
from memory.episode_store import EpisodeStore
from memory.retrieval import Retriever
from solver.solver import Solver
from verifier.verifier import Verifier
from agents.teacher import Teacher, TaskPlan
from agents.sub_agent import SubAgent, SubAgentResult
from agents.validator import Validator, ValidationResult
from agents.knowledge_transfer import KnowledgeTransferManager


# Observation vector dimension for Teacher and SubAgents
OBS_DIM = 32


class MultiAgentEnv:
    """Orchestrates the Teacher → SubAgents → Validator loop.

    Gymnasium-style environment for the full multi-agent system.  Manages
    shared state, memory graph access, and reward distribution.
    """

    def __init__(
        self,
        graph: DynamicMemoryGraph,
        solver: Solver,
        verifier: Verifier,
        retriever: Retriever,
        episode_store: EpisodeStore,
        num_sub_agents: int = 3,
        max_steps: int = 20,
        kt_manager: Optional[KnowledgeTransferManager] = None,
    ) -> None:
        self.graph = graph
        self.solver = solver
        self.verifier = verifier
        self.retriever = retriever
        self.episode_store = episode_store
        self.num_sub_agents = num_sub_agents
        self.max_steps = max_steps
        self.kt_manager = kt_manager or KnowledgeTransferManager()

        # Episode-level state (reset on each call to reset())
        self._problem: Dict[str, Any] = {}
        self._plan: Optional[TaskPlan] = None
        self._agent_results: Dict[str, SubAgentResult] = {}
        self._validation: Optional[ValidationResult] = None
        self._step_count: int = 0
        self._episode_id: str = ""
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium-style API
    # ------------------------------------------------------------------

    def reset(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Reset environment with a new problem.

        Returns initial observation for Teacher.
        """
        self._problem = problem
        self._plan = None
        self._agent_results = {}
        self._validation = None
        self._step_count = 0
        self._episode_id = str(uuid.uuid4())
        self._start_time = time.time()
        return {"obs": self._build_teacher_obs(), "episode_id": self._episode_id}

    def step_teacher(self, plan: TaskPlan) -> Dict[str, Any]:
        """Teacher submits a task plan.

        Returns observations for each SubAgent indexed by subtask.
        """
        self._plan = plan
        self._step_count += 1
        obs_per_agent: Dict[int, np.ndarray] = {}
        for subtask in plan.subtasks:
            obs_per_agent[subtask.assigned_agent] = self._build_sub_agent_obs(
                subtask.assigned_agent
            )
        return {
            "subtasks": plan.subtasks,
            "obs_per_agent": obs_per_agent,
            "plan_id": plan.plan_id,
        }

    def step_sub_agent(
        self, agent_id: int, result: SubAgentResult
    ) -> Dict[str, Any]:
        """SubAgent submits result for its subtask.

        Updates shared state, may trigger next SubAgent in dependency chain.
        """
        self._agent_results[result.task_id] = result
        self._step_count += 1
        return {
            "task_id": result.task_id,
            "success": result.success,
            "obs": self._build_sub_agent_obs(agent_id),
        }

    def step_validator(
        self, validation: ValidationResult
    ) -> Tuple[Dict, float, bool, Dict]:
        """Validator submits merged+validated result.

        Returns (obs, reward, done, info).
        """
        self._validation = validation
        self._step_count += 1
        reward = validation.reward_signals.get("global", 0.0)
        done = True
        obs = self._build_teacher_obs()
        info = {
            "passed": validation.passed,
            "merged_answer": validation.merged_answer,
            "reward_signals": validation.reward_signals,
            "diagnostics": validation.diagnostics,
        }
        return {"obs": obs}, reward, done, info

    # ------------------------------------------------------------------
    # Full episode runner
    # ------------------------------------------------------------------

    def run_episode(
        self,
        teacher: Teacher,
        sub_agents: List[SubAgent],
        validator: Validator,
        problem: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a complete episode: decompose → solve → validate.

        1. Teacher decomposes problem.
        2. SubAgents solve subtasks (respecting dependencies).
        3. If SubAgent stuck → Teacher provides golden thought → retry.
        4. Validator merges, validates, updates graph.
        5. Return episode record with all traces and rewards.
        """
        self.reset(problem)

        # Step 1: Teacher decomposes
        plan = teacher.decompose(problem)
        self.step_teacher(plan)

        # Step 2: Solve subtasks in dependency order
        completed: Dict[str, Any] = {}  # task_id → answer/result
        for subtask in plan.subtasks:
            # Wait for dependencies
            predecessor_results = {
                dep_id: completed.get(dep_id) for dep_id in subtask.depends_on
            }

            agent_idx = subtask.assigned_agent % len(sub_agents)
            agent = sub_agents[agent_idx]

            result = agent.solve(subtask, predecessor_results=predecessor_results)
            self.step_sub_agent(agent_idx, result)

            # Step 3: Knowledge transfer if stuck
            if self.kt_manager.should_transfer(result.all_branches):
                golden_hint = teacher.generate_golden_thought(
                    subtask, result.all_branches
                )
                result = agent.retry_with_hint(subtask, golden_hint)
                self.step_sub_agent(agent_idx, result)

            completed[subtask.task_id] = result

        # Step 4: Validator
        validation = validator.validate(plan, self._agent_results, problem)
        validator.apply_graph_updates(validation.graph_updates)
        obs, reward, done, info = self.step_validator(validation)

        return self.build_episode_record()

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _build_teacher_obs(self) -> np.ndarray:
        """Build observation vector for Teacher."""
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        # Problem features
        p = self._problem
        vec[0] = float(p.get("difficulty", 1)) / 5.0
        topic = p.get("topic", "")
        vec[1] = float(len(topic)) / 30.0
        # Graph stats
        vec[2] = min(self.graph.num_nodes, 100) / 100.0
        vec[3] = min(self.graph.num_edges, 200) / 200.0
        # Episode progress
        vec[4] = self._step_count / self.max_steps
        # Number of completed tasks
        vec[5] = len(self._agent_results) / max(self.num_sub_agents, 1)
        return vec

    def _build_sub_agent_obs(self, agent_id: int) -> np.ndarray:
        """Build observation vector for a SubAgent."""
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        vec[0] = float(agent_id) / max(self.num_sub_agents, 1)
        p = self._problem
        vec[1] = float(p.get("difficulty", 1)) / 5.0
        vec[2] = min(self.graph.num_nodes, 100) / 100.0
        vec[3] = self._step_count / self.max_steps
        return vec

    # ------------------------------------------------------------------
    # Episode record
    # ------------------------------------------------------------------

    def build_episode_record(self) -> Dict[str, Any]:
        """Build complete episode record for storage."""
        duration = time.time() - self._start_time
        validation = self._validation
        return {
            "episode_id": self._episode_id,
            "problem": self._problem,
            "plan": {
                "plan_id": self._plan.plan_id if self._plan else "",
                "strategy": self._plan.strategy if self._plan else "",
                "num_subtasks": len(self._plan.subtasks) if self._plan else 0,
            },
            "agent_results": {
                tid: {
                    "success": r.success,
                    "answer": r.selected_branch.answer,
                    "confidence": r.selected_branch.confidence,
                    "duration": r.duration,
                    "trace": r.reasoning_trace,
                }
                for tid, r in self._agent_results.items()
            },
            "validation": {
                "passed": validation.passed if validation else False,
                "merged_answer": validation.merged_answer if validation else "",
                "diagnostics": validation.diagnostics if validation else [],
                "reward_signals": validation.reward_signals if validation else {},
            },
            "duration": duration,
            "steps": self._step_count,
        }
