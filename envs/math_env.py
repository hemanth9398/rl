"""Gymnasium-style REPL environment for the math agent."""
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Discrete, Box
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False

from memory.graph import MemoryGraph
from memory.episode_store import EpisodeStore
from memory.retrieval import Retriever
from solver.solver import Solver
from verifier.verifier import Verifier
from policy.policy_nn import (
    ACTION_RETRIEVE, ACTION_SOLVE, ACTION_VERIFY, ACTION_REPAIR, ACTION_GENERATE,
    NUM_ACTIONS,
)


STATE_DIM = 16
MAX_STEPS = 15

TOPIC_MAP = {
    "algebra_linear": 0,
    "algebra_quadratic": 1,
    "algebra_factor": 2,
    "ode_separable": 3,
    "ode_linear_first": 4,
    "ode_ivp": 4,
    "general": 0,
}


class MathREPLEnv:
    """Gymnasium-style environment that orchestrates the closed-loop agent."""

    def __init__(
        self,
        graph: MemoryGraph,
        solver: Solver,
        verifier: Verifier,
        retriever: Retriever,
        episode_store: EpisodeStore,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.graph = graph
        self.solver = solver
        self.verifier = verifier
        self.retriever = retriever
        self.episode_store = episode_store
        self.max_steps = max_steps

        if _HAS_GYM:
            self.action_space = Discrete(NUM_ACTIONS)
            self.observation_space = Box(
                low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
            )

        # Episode state (reset each episode)
        self._problem: Optional[Dict[str, Any]] = None
        self._state: Dict[str, Any] = {}
        self._step_count: int = 0
        self._start_time: float = 0.0
        self._trace: List[Dict[str, Any]] = []
        self._retrieved_skills: List[Dict[str, Any]] = []
        self._verify_result = None
        self._repair_attempts: int = 0
        self._done: bool = False
        self._skills_used: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, problem: Dict[str, Any]) -> np.ndarray:
        """Reset environment with a new problem."""
        self._problem = problem
        self._state = {
            "problem_text": problem.get("statement", ""),
            "topic": problem.get("topic", ""),
            "problem_id": problem.get("id", ""),
            "candidate_answer": None,
            "general_solution": None,
        }
        self._step_count = 0
        self._start_time = time.time()
        self._trace = []
        self._retrieved_skills = []
        self._verify_result = None
        self._repair_attempts = 0
        self._done = False
        self._skills_used = []
        return self._build_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute an action and return (obs, reward, done, info)."""
        if self._done:
            return self._build_obs(), 0.0, True, {}

        self._step_count += 1
        reward = -0.01  # per-step cost
        info: Dict[str, Any] = {"action": action, "step": self._step_count}

        if action == ACTION_RETRIEVE:
            reward += self._do_retrieve(info)
        elif action == ACTION_SOLVE:
            reward += self._do_solve(info)
        elif action == ACTION_VERIFY:
            reward += self._do_verify(info)
        elif action == ACTION_REPAIR:
            reward += self._do_repair(info)
        elif action == ACTION_GENERATE:
            reward += self._do_generate(info)

        # Terminal conditions
        done = False
        if self._verify_result and getattr(self._verify_result, "passed", False):
            reward += 1.0
            done = True
            info["outcome"] = "verified_correct"
        elif self._step_count >= self.max_steps:
            reward -= 1.0
            done = True
            info["outcome"] = "max_steps"

        self._done = done
        self._log_step(action, info)
        return self._build_obs(), reward, done, info

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_retrieve(self, info: Dict) -> float:
        topic = self._problem.get("topic", "") if self._problem else ""
        problem_text = self._state.get("problem_text", "")
        results = self.retriever.retrieve(
            problem_text, topic=topic, domain="math", top_k=3
        )
        self._retrieved_skills = [r.skill_node for r in results]
        info["retrieved"] = [s.get("id") for s in self._retrieved_skills]
        return 0.0

    def _do_solve(self, info: Dict) -> float:
        if not self._retrieved_skills:
            # fallback to default skill for topic
            topic = self._state.get("topic", "")
            default = self.solver.default_skill_for_topic(topic)
            skills_to_try = [default] if default else []
        else:
            skills_to_try = self._retrieved_skills

        best_result = None
        for skill_node in skills_to_try:
            if skill_node is None:
                continue
            result = self.solver.execute_skill(skill_node, self._state)
            if result.success:
                best_result = result
                break
            best_result = result

        if best_result and best_result.success:
            self._state = best_result.new_state
            sid = best_result.skill_id or ""
            if sid:
                self._skills_used.append(sid)
            info["solved"] = True
            info["answer"] = best_result.answer
            return 0.05
        info["solved"] = False
        return 0.0

    def _do_verify(self, info: Dict) -> float:
        candidate = self._state.get("candidate_answer")
        if candidate is None:
            info["verify"] = "no_candidate"
            return 0.0
        result = self.verifier.verify(self._problem or {}, str(candidate))
        self._verify_result = result
        info["verify_passed"] = result.passed
        info["diagnostics"] = result.diagnostics
        return 0.0

    def _do_repair(self, info: Dict) -> float:
        if self._verify_result is None:
            return 0.0
        diagnostics = getattr(self._verify_result, "diagnostics", [])
        hints = getattr(self._verify_result, "repair_hints", [])
        self._repair_attempts += 1
        result = self.solver.repair(self._state, diagnostics, hints)
        if result.success:
            self._state = result.new_state
            sid = result.skill_id or ""
            if sid:
                self._skills_used.append(sid)
            info["repair_success"] = True
            return 0.05
        info["repair_success"] = False
        return 0.0

    def _do_generate(self, info: Dict) -> float:
        # Generate action is handled externally; just note it here
        info["generate"] = True
        return 0.0

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(STATE_DIM, dtype=np.float32)
        if self._problem is None:
            return obs

        topic = self._problem.get("topic", "general")
        topic_id = TOPIC_MAP.get(topic, 0)
        # One-hot encode topic (5 slots, indices 0-4)
        if topic_id < 5:
            obs[topic_id] = 1.0

        # Difficulty (normalised to [0,1] over range 1-5)
        difficulty = self._problem.get("difficulty", 1)
        obs[5] = min(1.0, (difficulty - 1) / 4.0)

        # Step count normalised
        obs[6] = min(1.0, self._step_count / self.max_steps)

        # Has candidate answer
        obs[7] = 1.0 if self._state.get("candidate_answer") else 0.0

        # Number of retrieved skills (normalised)
        obs[8] = min(1.0, len(self._retrieved_skills) / 5.0)

        # Top skill success rate
        if self._retrieved_skills:
            top = self._retrieved_skills[0]
            use_count = top.get("use_count", 0)
            success_count = top.get("success_count", 0)
            obs[9] = success_count / use_count if use_count > 0 else 0.5
        else:
            obs[9] = 0.0

        # Top skill recency (1 = just used, 0 = never)
        if self._retrieved_skills:
            top = self._retrieved_skills[0]
            last_used = top.get("last_used")
            if last_used:
                elapsed = time.time() - last_used
                obs[10] = max(0.0, 1.0 - elapsed / 3600.0)
            else:
                obs[10] = 0.0

        # Verify status one-hot: [unknown, pass, fail] → indices 11,12,13
        if self._verify_result is None:
            obs[11] = 1.0
        elif getattr(self._verify_result, "passed", False):
            obs[12] = 1.0
        else:
            obs[13] = 1.0

        # Repair attempts normalised
        obs[14] = min(1.0, self._repair_attempts / 3.0)

        # Time elapsed normalised (max 5 minutes per episode)
        elapsed = time.time() - self._start_time
        obs[15] = min(1.0, elapsed / 300.0)

        return obs

    # ------------------------------------------------------------------
    # Trace logging
    # ------------------------------------------------------------------

    def _log_step(self, action: int, info: Dict) -> None:
        step_record = {
            "step_id": f"{id(self)}_{self._step_count}",
            "step_number": self._step_count,
            "action_type": _action_name(action),
            "skill_id": info.get("skill_id"),
            "input_summary": self._state.get("problem_text", "")[:100],
            "output_text": str(info)[:200],
            "timestamp": time.time(),
        }
        self._trace.append(step_record)

    def build_episode_record(
        self, episode_id: Optional[str] = None, final_reward: float = 0.0
    ) -> Dict[str, Any]:
        """Build a dict suitable for EpisodeStore.store_episode."""
        verified = bool(
            self._verify_result and getattr(self._verify_result, "passed", False)
        )
        failure_mode = None
        if self._verify_result and not verified:
            diags = getattr(self._verify_result, "diagnostics", [])
            failure_mode = ",".join(diags) if diags else "unknown"

        return {
            "episode_id": episode_id,
            "problem_id": self._problem.get("id", "") if self._problem else "",
            "problem_text": self._state.get("problem_text", ""),
            "topic": self._problem.get("topic", "") if self._problem else "",
            "difficulty": self._problem.get("difficulty", 1) if self._problem else 1,
            "trace": self._trace,
            "final_answer": str(self._state.get("candidate_answer", "")),
            "verified": verified,
            "failure_mode": failure_mode,
            "skills_used": self._skills_used,
            "duration_seconds": time.time() - self._start_time,
            "num_steps": self._step_count,
            "timestamp": time.time(),
        }


def _action_name(action: int) -> str:
    names = {
        ACTION_RETRIEVE: "RETRIEVE",
        ACTION_SOLVE: "SOLVE",
        ACTION_VERIFY: "VERIFY",
        ACTION_REPAIR: "REPAIR",
        ACTION_GENERATE: "GENERATE",
    }
    return names.get(action, "UNKNOWN")
