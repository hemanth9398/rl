"""LLM-based verifier module.

Provides :class:`LLMVerifierModule` which wraps the registry's *verifier* role
model.  The LLM checks whether a candidate answer is correct given the problem
statement and (optionally) the expected answer format.  SymPy is used as a
secondary cross-check for algebraic answers.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORRECT_RE = re.compile(r"\bCORRECT\b", re.I)
_INCORRECT_RE = re.compile(r"\bINCORRECT\b", re.I)


def _parse_verdict(text: str) -> Optional[bool]:
    """Return True for CORRECT, False for INCORRECT, None if unparseable."""
    if _CORRECT_RE.search(text) and not _INCORRECT_RE.search(text):
        return True
    if _INCORRECT_RE.search(text):
        return False
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMVerifierModule:
    """Uses the registry *verifier* model to judge answer correctness.

    Parameters
    ----------
    registry:
        A :class:`~models.model_registry.ModelRegistry` instance.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def verify(
        self,
        problem: Dict[str, Any],
        candidate_answer: str,
        expected_format: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Verify *candidate_answer* against *problem*.

        Returns
        -------
        (passed: bool, explanation: str)

        The *passed* value defaults to ``False`` when the LLM output is
        ambiguous.
        """
        from models.prompts import VERIFIER_WITH_EXPECTED_PROMPT, VERIFIER_PROMPT

        statement = problem.get("statement", str(problem))
        exp_fmt = expected_format or problem.get("answer_spec", {}).get("type", "value")

        if exp_fmt:
            prompt = VERIFIER_WITH_EXPECTED_PROMPT.format(
                problem=statement,
                answer=candidate_answer,
                expected_format=exp_fmt,
            )
        else:
            prompt = VERIFIER_PROMPT.format(
                problem=statement,
                answer=candidate_answer,
            )

        try:
            raw = self.registry.generate(
                "verifier",
                prompt,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
            )
            verdict = _parse_verdict(raw)
            if verdict is None:
                # Couldn't parse → conservative: treat as incorrect
                logger.debug(
                    "LLMVerifierModule: unparseable verdict for '%s'. Raw: %s",
                    candidate_answer[:40],
                    raw[:120],
                )
                return False, f"Unparseable verdict: {raw[:120]}"
            return verdict, raw.strip()
        except Exception as exc:
            logger.warning("LLMVerifierModule.verify failed: %s", exc)
            return False, f"LLM verification error: {exc}"

    def verify_with_sympy_crosscheck(
        self,
        problem: Dict[str, Any],
        candidate_answer: str,
    ) -> Tuple[bool, str]:
        """LLM primary verdict cross-checked with SymPy when available.

        If the LLM says CORRECT but SymPy disagrees (and vice-versa), the more
        conservative result (INCORRECT) is returned with a note.
        """
        llm_passed, llm_explanation = self.verify(problem, candidate_answer)

        # Try SymPy cross-check
        try:
            from verifier.verifier import Verifier as SymPyVerifier

            sv = SymPyVerifier()
            sp_result = sv.verify(problem, candidate_answer)
            sympy_passed = sp_result.passed

            if llm_passed != sympy_passed:
                # Disagree — return the more conservative result
                explanation = (
                    f"LLM: {'CORRECT' if llm_passed else 'INCORRECT'}, "
                    f"SymPy: {'CORRECT' if sympy_passed else 'INCORRECT'}. "
                    "Using conservative (INCORRECT) result."
                )
                logger.debug("LLM/SymPy disagreement: %s", explanation)
                return False, explanation

            return llm_passed, llm_explanation
        except Exception as exc:
            logger.debug("SymPy cross-check failed (using LLM only): %s", exc)
            return llm_passed, llm_explanation
