"""SubAgent: worker agent with Tree-of-Thought reasoning and LoRA slot."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memory.graph import MemoryGraph
from memory.retrieval import Retriever
from solver.solver import Solver
from agents.tree_of_thought import TreeOfThought, ThoughtBranch
from agents.knowledge_transfer import LoRAAdapter
from agents.teacher import SubTask


@dataclass
class SubAgentResult:
    """Result returned by a SubAgent after solving a subtask."""

    task_id: str
    success: bool
    selected_branch: ThoughtBranch
    all_branches: List[ThoughtBranch]
    reasoning_trace: List[str]
    duration: float


class SubAgent:
    """Worker agent with Tree-of-Thought reasoning and LoRA adapter slot.

    Each SubAgent owns a slot for a LoRA adapter that can be hot-injected by
    the KnowledgeTransferManager when the agent is stuck.
    """

    def __init__(
        self,
        agent_id: int,
        solver: Solver,
        graph: MemoryGraph,
        retriever: Retriever,
        tot_engine: TreeOfThought,
        lora_adapter: Optional[LoRAAdapter] = None,
    ) -> None:
        self.agent_id = agent_id
        self.solver = solver
        self.graph = graph
        self.retriever = retriever
        self.tot_engine = tot_engine
        self.lora_adapter = lora_adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        subtask: SubTask,
        predecessor_results: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        """Solve a subtask using Tree-of-Thought exploration.

        1. Retrieve relevant skills from memory graph.
        2. Generate thought branches (binary tree).
        3. Score branches using GIGPO (if available in ToT engine).
        4. Select best branch at each level.
        5. If both branches bad → signal for knowledge transfer via low confidence.
        """
        t0 = time.time()
        trace: List[str] = []

        # Build context from subtask + predecessor results
        context = dict(subtask.context)
        if predecessor_results:
            context["predecessor_results"] = predecessor_results
        context["problem"] = subtask.context.get("problem", {
            "statement": subtask.description,
            "topic": context.get("topic", "general"),
            "domain": context.get("domain", "math"),
            "answer_spec": context.get("answer_spec", {}),
        })

        # Stage 1: retrieve relevant skills
        retrieved = self.retriever.retrieve(
            problem_text=subtask.description,
            topic=context.get("topic", "general"),
            domain=context.get("domain", "math"),
            top_k=5,
        )
        context["retrieved_skills"] = [item.skill_node for item in retrieved]
        trace.append(f"Retrieved {len(retrieved)} skill(s)")

        # Stage 2: Tree-of-Thought exploration
        branches = self.tot_engine.explore(
            problem_text=subtask.description,
            context=context,
            solver=self.solver,
            graph=self.graph,
        )
        trace.append(f"ToT generated {len(branches)} branch(es)")

        # Stage 3: select best branch
        best = self.tot_engine.select_best(branches)
        trace.append(f"Selected branch '{best.branch_id}' (confidence={best.confidence:.3f})")

        success = bool(best.answer and best.confidence > 0.2)
        duration = time.time() - t0

        return SubAgentResult(
            task_id=subtask.task_id,
            success=success,
            selected_branch=best,
            all_branches=branches,
            reasoning_trace=trace,
            duration=duration,
        )

    def inject_lora(self, adapter: LoRAAdapter) -> None:
        """Hot-inject a LoRA adapter into this agent's model."""
        self.lora_adapter = adapter

    def retry_with_hint(
        self,
        subtask: SubTask,
        hint: str,
    ) -> SubAgentResult:
        """Retry solving with a golden thought hint from the Teacher.

        Injects the hint into the subtask context and re-runs ToT with a
        slightly more forgiving confidence threshold.
        """
        t0 = time.time()
        trace: List[str] = [f"Retrying with golden thought hint (len={len(hint)})"]

        context = dict(subtask.context)
        context["golden_hint"] = hint
        context["problem"] = subtask.context.get("problem", {
            "statement": subtask.description,
            "topic": context.get("topic", "general"),
            "domain": context.get("domain", "math"),
            "answer_spec": context.get("answer_spec", {}),
        })

        retrieved = self.retriever.retrieve(
            problem_text=subtask.description + " " + hint,
            topic=context.get("topic", "general"),
            domain=context.get("domain", "math"),
            top_k=5,
        )
        context["retrieved_skills"] = [item.skill_node for item in retrieved]
        trace.append(f"Retrieved {len(retrieved)} skill(s) with hint context")

        branches = self.tot_engine.explore(
            problem_text=subtask.description,
            context=context,
            solver=self.solver,
            graph=self.graph,
        )
        best = self.tot_engine.select_best(branches)
        # Boost confidence slightly when a hint was provided
        boosted_confidence = min(1.0, best.confidence + 0.15)
        boosted_branch = ThoughtBranch(
            branch_id=best.branch_id,
            reasoning=best.reasoning,
            answer=best.answer,
            confidence=boosted_confidence,
            depth=best.depth,
        )
        trace.append(f"Retry best branch conf={boosted_confidence:.3f}")

        success = bool(boosted_branch.answer and boosted_confidence > 0.2)
        return SubAgentResult(
            task_id=subtask.task_id,
            success=success,
            selected_branch=boosted_branch,
            all_branches=branches,
            reasoning_trace=trace,
            duration=time.time() - t0,
        )
