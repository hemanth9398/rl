"""LLM-based solver: uses a HuggingFace language model as the primary reasoning engine.

This module provides ``LLMSolver``, a drop-in replacement for the SymPy-based
``Solver`` class.  It accepts the same ``execute_skill()`` / ``repair()`` /
``default_skill_for_topic()`` interface so the rest of the system (env, PPO
loop, graph updates) works unchanged.

Falls back to the SymPy ``Solver`` when the ``transformers`` package is not
installed or the model cannot be loaded.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from solver.solver import SolveResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
_MAX_NEW_TOKENS = 512
_TEMPERATURE = 0.7

# Mapping from topic string → skill-node id used by default_skill_for_topic
_TOPIC_DEFAULT_SKILL = {
    "algebra_linear": "skill_solve_linear",
    "algebra_quadratic": "skill_solve_quadratic",
    "algebra_factor": "skill_factor_polynomial",
    "ode_separable": "skill_separate_variables",
    "ode_linear_first": "skill_integrating_factor",
    "ode_ivp": "skill_ode_sympy",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_solve_prompt(
    problem_text: str,
    skill_node: Optional[Dict[str, Any]] = None,
    episodes: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a chain-of-thought solve prompt for the LLM."""
    lines = [
        "You are a math problem solver. Solve the following problem step by step.",
        "",
        f"Problem: {problem_text}",
    ]

    # Add procedure steps from skill node if available
    if skill_node:
        procedure = skill_node.get("procedure", [])
        if procedure:
            lines += ["", "Suggested approach:"]
            for i, step in enumerate(procedure, 1):
                lines.append(f"{i}. {step}")

    # Add similar example episodes if available
    if episodes:
        lines += ["", "Similar problems solved before:"]
        for ep in episodes[:3]:  # cap at 3 examples
            ep_text = ep.get("problem_text", "")
            ep_answer = ep.get("final_answer", "")
            if ep_text and ep_answer:
                lines.append(f"- Problem: {ep_text} → Answer: {ep_answer}")

    lines += [
        "",
        "Show your work step by step, then provide your final answer on the last line "
        "in the format:",
        "ANSWER: <your answer>",
    ]
    return "\n".join(lines)


def _build_repair_prompt(
    problem_text: str,
    candidate_answer: str,
    diagnostics: List[str],
    repair_hints: List[str],
) -> str:
    """Build a repair prompt incorporating verifier diagnostics."""
    lines = [
        "Your previous answer was incorrect.",
        "",
        f"Problem: {problem_text}",
        f"Your previous answer: {candidate_answer}",
        "",
        f"Error diagnostics: {', '.join(diagnostics) if diagnostics else 'unknown error'}",
        f"Hints for fixing: {', '.join(repair_hints) if repair_hints else 'none'}",
        "",
        "Please re-solve the problem carefully, addressing the identified errors.",
        "Show your work step by step, then provide your final answer on the last line "
        "in the format:",
        "ANSWER: <your answer>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer parser
# ---------------------------------------------------------------------------

def _parse_answer(text: str) -> str:
    """Extract the final answer from LLM output.

    Priority:
    1. ``ANSWER: <value>`` tag (explicitly requested format)
    2. ``\\boxed{...}`` LaTeX notation
    3. ``Answer: …`` / ``The answer is …`` phrase
    4. Last ``x = …`` / ``y = …`` assignment
    5. Last non-empty line as fallback
    """
    text = text.strip()

    # ANSWER: tag (our requested format)
    tag_match = re.search(r"ANSWER\s*:\s*(.+)", text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip()

    # \boxed{...}
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # "The answer is …" / "Answer: …"
    phrase = re.search(
        r"(?:the answer is|answer\s*[:=])\s*([^\n]+)", text, re.IGNORECASE
    )
    if phrase:
        return phrase.group(1).strip().rstrip(".")

    # Last variable assignment (x = …, y = …)
    assignments = list(re.finditer(r"\b([xy])\s*=\s*([^\n,]+)", text))
    if assignments:
        last = assignments[-1]
        return f"{last.group(1)} = {last.group(2).strip()}"

    # Last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return text


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

class _ModelBackend:
    """Lazily-loaded HuggingFace model backend."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None
        self._load_error: Optional[str] = None
        self._loaded = False

    def load(self) -> bool:
        """Load model and tokenizer on first call.  Returns True on success."""
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore

            logger.info("LLMSolver: loading model %s …", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
            # Cache the device so generate() doesn't need to rediscover it
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
            logger.info("LLMSolver: model loaded on device %s.", self._device)
            return True
        except ImportError:
            self._load_error = (
                "transformers not installed. "
                "Install with: pip install transformers>=4.40.0 accelerate>=0.30.0"
            )
            logger.warning("LLMSolver: %s", self._load_error)
            return False
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("LLMSolver: failed to load model '%s': %s", self.model_name, exc)
            return False

    def generate(self, prompt: str) -> str:
        """Run inference and return the newly-generated text."""
        import torch  # type: ignore

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=_MAX_NEW_TOKENS,
                temperature=_TEMPERATURE,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error


# ---------------------------------------------------------------------------
# LLMSolver
# ---------------------------------------------------------------------------

class LLMSolver:
    """LLM-based drop-in replacement for the SymPy ``Solver``.

    Uses a HuggingFace instruction-tuned model (default:
    ``Qwen/Qwen2.5-1.5B-Instruct``) to generate chain-of-thought solutions.
    Falls back to the SymPy ``Solver`` when the model is unavailable.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (or local path).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._backend = _ModelBackend(model_name)
        self._sympy_fallback: Optional[Any] = None  # lazy import to avoid circular

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sympy(self):
        """Return SymPy Solver fallback (lazy import)."""
        if self._sympy_fallback is None:
            from solver.solver import Solver  # noqa: PLC0415
            self._sympy_fallback = Solver()
        return self._sympy_fallback

    def _llm_solve(
        self,
        prompt: str,
        state: Dict[str, Any],
        skill_id: str,
    ) -> SolveResult:
        """Call the LLM with *prompt* and wrap result in a ``SolveResult``."""
        t0 = time.time()
        if not self._backend.load():
            # Graceful degradation — fall back to SymPy
            logger.info("LLMSolver: falling back to SymPy (model unavailable)")
            return self._sympy().execute_skill({"id": skill_id}, state)

        try:
            raw = self._backend.generate(prompt)
            answer = _parse_answer(raw)
            new_state = {**state, "candidate_answer": answer}
            return SolveResult(
                success=True,
                new_state=new_state,
                reasoning_text=raw,
                answer=answer,
                skill_id=skill_id,
                duration=time.time() - t0,
            )
        except (RuntimeError, ValueError, MemoryError) as exc:  # noqa: BLE001
            logger.warning("LLMSolver: inference error: %s", exc)
            return SolveResult(
                success=False,
                new_state=state,
                reasoning_text=f"LLM inference error: {exc}",
                skill_id=skill_id,
                duration=time.time() - t0,
            )

    # ------------------------------------------------------------------
    # Public interface (mirrors Solver)
    # ------------------------------------------------------------------

    def execute_skill(
        self,
        skill_node: Dict[str, Any],
        state: Dict[str, Any],
        episodes: Optional[List[Dict[str, Any]]] = None,
    ) -> SolveResult:
        """Solve using the LLM, guided by *skill_node* procedure steps.

        Parameters
        ----------
        skill_node:
            Memory-graph skill node dict (must have at least ``"id"``).
        state:
            Current episode state dict (must contain ``"problem_text"``).
        episodes:
            Optional list of similar past episodes for few-shot context.
        """
        problem_text = state.get("problem_text", "")
        skill_id = skill_node.get("id", "skill_generic")
        prompt = _build_solve_prompt(problem_text, skill_node, episodes)
        return self._llm_solve(prompt, state, skill_id)

    def repair(
        self,
        state: Dict[str, Any],
        error_diagnostics: List[str],
        repair_hints: List[str],
    ) -> SolveResult:
        """Re-prompt the LLM with verifier diagnostics and repair hints.

        Parameters
        ----------
        state:
            Current episode state (must include ``"problem_text"`` and ideally
            ``"candidate_answer"``).
        error_diagnostics:
            List of diagnostic strings from the verifier.
        repair_hints:
            List of hint strings for how to fix the error.
        """
        problem_text = state.get("problem_text", "")
        candidate = str(state.get("candidate_answer", ""))
        prompt = _build_repair_prompt(problem_text, candidate, error_diagnostics, repair_hints)
        return self._llm_solve(prompt, state, skill_id="repair")

    def default_skill_for_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        """Return a generic skill node for *topic*, or ``None`` if unknown."""
        skill_id = _TOPIC_DEFAULT_SKILL.get(topic)
        if skill_id:
            return {
                "id": skill_id,
                "procedure": [],  # LLM doesn't need hard-coded procedure steps
            }
        return None
