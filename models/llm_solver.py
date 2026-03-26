"""LLM-based solver module for the models/ package.

Provides :class:`LLMSolverModule` — a thin wrapper around the
:class:`~models.model_registry.ModelRegistry` that generates solutions for
math problems using the *solver* role model.

This is separate from ``solver/llm_solver.py`` (which implements the full
``Solver`` interface).  The two can be used independently or together.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSWER_PATTERNS = [
    re.compile(r"(?:final answer|answer)[^\d\w]*[:=]\s*([\d\w\.\-\+\*/\(\) ]+)", re.I),
    re.compile(r"x\s*=\s*([\d\w\.\-\+\*/\(\) ]+)"),
    re.compile(r"=\s*([\d\.\-\+]+)\s*$"),
]


def _extract_answer(text: str) -> Optional[str]:
    """Try to extract a short answer from generated text."""
    for pattern in _ANSWER_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    # Last non-empty line as fallback
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMSolverModule:
    """Generates solutions using the registry's *solver* role model.

    Parameters
    ----------
    registry:
        A :class:`~models.model_registry.ModelRegistry` instance.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def solve(
        self,
        problem_text: str,
        topic: str = "general",
        context: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Tuple[bool, str, Optional[str]]:
        """Generate a solution for *problem_text*.

        Returns
        -------
        (success, raw_output, extracted_answer)
        """
        from models.prompts import SOLVER_WITH_CONTEXT_PROMPT, SOLVER_PROMPT

        ctx_str = ""
        if context:
            skills = context.get("retrieved_skills", [])
            if skills:
                skill_labels = [s.get("label", "?") for s in skills[:3]]
                ctx_str = "Skills: " + ", ".join(skill_labels)

        if ctx_str:
            prompt = SOLVER_WITH_CONTEXT_PROMPT.format(
                problem=problem_text, topic=topic, context=ctx_str
            )
        else:
            prompt = SOLVER_PROMPT.format(problem=problem_text, topic=topic)

        try:
            raw = self.registry.generate(
                "solver",
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            answer = _extract_answer(raw)
            return True, raw, answer
        except Exception as exc:
            logger.warning("LLMSolverModule.solve failed: %s", exc)
            return False, str(exc), None

    def log_probs(self, problem_text: str, completion: str) -> float:
        """Return log probability of *completion* under the solver model."""
        from models.prompts import SOLVER_PROMPT

        prompt = SOLVER_PROMPT.format(problem=problem_text, topic="general")
        try:
            return self.registry.log_probs("solver", prompt, completion)
        except Exception as exc:
            logger.warning("LLMSolverModule.log_probs failed: %s", exc)
            return 0.0
