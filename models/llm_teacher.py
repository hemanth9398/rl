"""LLM-based teacher decomposition module.

Provides :class:`LLMTeacherModule` which wraps the registry's *teacher* role
model and exposes three capabilities:

1. ``decompose(problem, n_subtasks, context)`` — split a problem into subtasks
   via structured JSON generation.
2. ``golden_thought(subtask, failures, context)`` — generate a correct
   reasoning hint for a stuck student agent.
3. ``synthesize(problem, results)`` — merge subtask results into a final answer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract the first JSON list from *text*."""
    # Try to find [ ... ] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    # Try fixing common issues (trailing commas)
    cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _build_subtask_fallback(
    problem: str, n_subtasks: int
) -> List[Dict[str, Any]]:
    """Return a simple linear decomposition when the LLM output can't be parsed."""
    if n_subtasks == 1:
        return [{"description": problem[:200], "depends_on": []}]
    if n_subtasks == 2:
        return [
            {"description": f"Understand and set up: {problem[:100]}", "depends_on": []},
            {"description": f"Solve and verify: {problem[:100]}", "depends_on": [0]},
        ]
    return [
        {"description": f"Parse and retrieve relevant skills: {problem[:80]}", "depends_on": []},
        {"description": f"Solve main computation: {problem[:80]}", "depends_on": [0]},
        {"description": f"Verify and synthesize final answer: {problem[:80]}", "depends_on": [1]},
    ]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMTeacherModule:
    """Uses the registry *teacher* model for decomposition and synthesis.

    Parameters
    ----------
    registry:
        A :class:`~models.model_registry.ModelRegistry` instance.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def decompose(
        self,
        problem: Dict[str, Any],
        n_subtasks: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Decompose *problem* into *n_subtasks* subtask dicts.

        Each returned dict has keys:
        - ``description`` (str)
        - ``depends_on`` (list of int subtask indices)

        Falls back to a heuristic decomposition if the LLM output cannot be
        parsed as JSON.
        """
        from models.prompts import TEACHER_DECOMPOSE_WITH_CONTEXT_PROMPT

        statement = problem.get("statement", str(problem))
        ctx_str = ""
        if context:
            skills = context.get("retrieved_skills", [])
            if skills:
                skill_labels = [s.get("label", "?") for s in skills[:3]]
                ctx_str = "Available skills: " + ", ".join(skill_labels)

        prompt = TEACHER_DECOMPOSE_WITH_CONTEXT_PROMPT.format(
            problem=statement,
            n_subtasks=n_subtasks,
            context=ctx_str or "None",
        )

        try:
            raw = self.registry.generate(
                "teacher",
                prompt,
                max_new_tokens=512,
                temperature=0.3,
            )
            subtasks = _extract_json_list(raw)
            if subtasks and len(subtasks) > 0:
                # Normalise each entry
                result = []
                for item in subtasks[:n_subtasks]:
                    if isinstance(item, dict):
                        result.append({
                            "description": str(item.get("description", statement[:100])),
                            "depends_on": list(item.get("depends_on", [])),
                        })
                    elif isinstance(item, str):
                        result.append({"description": item, "depends_on": []})
                if result:
                    return result
        except Exception as exc:
            logger.warning("LLMTeacherModule.decompose LLM call failed: %s", exc)

        # Fallback to heuristic
        return _build_subtask_fallback(statement, n_subtasks)

    def golden_thought(
        self,
        subtask_description: str,
        failed_branches: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a golden-thought hint for a stuck student agent."""
        from models.prompts import TEACHER_GOLDEN_THOUGHT_PROMPT

        failures_str = "; ".join(
            getattr(b, "reasoning", str(b))[:80] for b in failed_branches[:3]
        )

        prompt = TEACHER_GOLDEN_THOUGHT_PROMPT.format(
            subtask=subtask_description[:200],
            failures=failures_str or "none",
        )

        try:
            hint = self.registry.generate(
                "teacher",
                prompt,
                max_new_tokens=256,
                temperature=0.5,
            )
            return hint.strip() or subtask_description
        except Exception as exc:
            logger.warning("LLMTeacherModule.golden_thought failed: %s", exc)
            return (
                f"Approach '{subtask_description[:80]}' by breaking it into smaller "
                "steps and verifying each intermediate result."
            )

    def synthesize(
        self,
        problem: Dict[str, Any],
        subtask_results: Dict[str, Any],
    ) -> str:
        """Combine subtask results into a final answer using the LLM."""
        from models.prompts import TEACHER_SYNTHESIZE_PROMPT

        statement = problem.get("statement", str(problem))
        results_lines = []
        for task_id, result in subtask_results.items():
            if hasattr(result, "selected_branch"):
                answer = result.selected_branch.answer
            elif isinstance(result, dict):
                answer = result.get("answer", "")
            else:
                answer = str(result)
            results_lines.append(f"- {task_id[:20]}: {answer}")

        results_str = "\n".join(results_lines) if results_lines else "No results"

        prompt = TEACHER_SYNTHESIZE_PROMPT.format(
            problem=statement,
            results=results_str,
        )

        try:
            output = self.registry.generate(
                "teacher",
                prompt,
                max_new_tokens=256,
                temperature=0.3,
            )
            return output.strip() or results_str
        except Exception as exc:
            logger.warning("LLMTeacherModule.synthesize failed: %s", exc)
            return results_str
