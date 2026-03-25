"""LLM inference backend using Qwen2.5-Math-1.5B-Instruct."""
import logging
import os
import re
from functools import lru_cache
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

SOLVER_BACKEND = os.environ.get("SOLVER_BACKEND", "llm")
SOLVER_MODEL_NAME = os.environ.get("SOLVER_MODEL_NAME", "Qwen/Qwen2.5-Math-1.5B-Instruct")
SOLVER_DEVICE = os.environ.get("SOLVER_DEVICE", "auto")
SOLVER_MAX_NEW_TOKENS = int(os.environ.get("SOLVER_MAX_NEW_TOKENS", "512"))

# ---------------------------------------------------------------------------
# Prompt templates (skill-specific)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES = {
    "linear": (
        "Solve the following linear equation step by step. "
        "Give the final answer as a single number.\n"
        "Problem: {problem}\nSolution:"
    ),
    "quadratic": (
        "Solve the following quadratic equation step by step. "
        "List all solutions.\n"
        "Problem: {problem}\nSolution:"
    ),
    "factor": (
        "Factor the following expression completely. "
        "Show the factored form.\n"
        "Problem: {problem}\nSolution:"
    ),
    "ode": (
        "Solve the following ordinary differential equation. "
        "Give the general solution y = f(x) with arbitrary constant C1.\n"
        "Problem: {problem}\nSolution:"
    ),
    "ivp": (
        "Solve the following initial value problem step by step. "
        "Apply the initial condition to find the particular solution.\n"
        "Problem: {problem}\nSolution:"
    ),
    "generic": (
        "Solve the following math problem step by step.\n"
        "Problem: {problem}\nSolution:"
    ),
}

# Mapping from skill_hint strings to template keys
_SKILL_TO_TEMPLATE = {
    "skill_solve_linear": "linear",
    "skill_solve_quadratic": "quadratic",
    "skill_factor_polynomial": "factor",
    "skill_separate_variables": "ode",
    "skill_integrating_factor": "ode",
    "skill_ode_sympy": "ode",
    "skill_apply_initial_condition": "ivp",
    "skill_algebra_sympy": "generic",
}

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_load_error: Optional[str] = None


def _load_model() -> bool:
    """Lazily load the LLM model and tokenizer. Returns True if successful."""
    global _model, _tokenizer, _load_error

    if _model is not None:
        return True
    if _load_error is not None:
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        model_name = SOLVER_MODEL_NAME
        logger.info("Loading LLM model %s ...", model_name)

        device_map: object
        if SOLVER_DEVICE == "auto":
            device_map = "auto"
        elif SOLVER_DEVICE == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": SOLVER_DEVICE}

        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if SOLVER_DEVICE != "cpu" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
        _model.eval()
        logger.info("LLM model loaded successfully.")
        return True
    except ImportError:
        _load_error = (
            "transformers package is not installed. "
            "Install it with: pip install transformers accelerate sentencepiece"
        )
        logger.warning("LLM backend unavailable: %s", _load_error)
        return False
    except Exception as exc:
        _load_error = str(exc)
        logger.warning("Failed to load LLM model '%s': %s", SOLVER_MODEL_NAME, exc)
        return False


# ---------------------------------------------------------------------------
# Answer parser
# ---------------------------------------------------------------------------

def _parse_answer(text: str) -> str:
    r"""Extract the final answer from LLM output.

    Handles:
    - ``\boxed{5}`` format
    - ``x = 5`` or ``y = C1*exp(x)`` format
    - ``The answer is 5`` format
    - Multi-line reasoning — falls back to last non-empty line
    - List outputs like ``x = 2 or x = 3``
    """
    text = text.strip()

    # \boxed{...}
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # "The answer is …" / "Answer: …"
    ans_phrase = re.search(
        r"(?:the answer is|answer\s*[:=])\s*([^\n]+)", text, re.IGNORECASE
    )
    if ans_phrase:
        return ans_phrase.group(1).strip().rstrip(".")

    # variable = expression (last occurrence wins so reasoning doesn't interfere)
    assignments = list(re.finditer(r"\b([xy])\s*=\s*([^\n,]+)", text))
    if assignments:
        last = assignments[-1]
        return f"{last.group(1)} = {last.group(2).strip()}"

    # y = … ODE style
    ode_match = re.search(r"\by\s*=\s*([^\n]+)", text)
    if ode_match:
        return f"y = {ode_match.group(1).strip()}"

    # Fall back to last meaningful line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return text


# ---------------------------------------------------------------------------
# LRU-cached inference (keyed on problem text + skill hint)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cached_generate(problem_text: str, skill_hint: str) -> str:
    """Run LLM inference (cached to avoid redundant calls)."""
    import torch  # type: ignore

    template_key = _SKILL_TO_TEMPLATE.get(skill_hint, "generic")
    prompt = _PROMPT_TEMPLATES[template_key].format(problem=problem_text)

    inputs = _tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Move to same device as model
    try:
        device = next(_model.parameters()).device
        input_ids = input_ids.to(device)
    except StopIteration:
        pass

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=SOLVER_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_solution(
    problem_text: str,
    skill_hint: str = "generic",
    topic: str = "",
) -> Tuple[bool, str, str]:
    """Generate a solution using the LLM.

    Parameters
    ----------
    problem_text:
        The full problem statement.
    skill_hint:
        The skill ID string (e.g. ``"skill_solve_linear"``), used to select
        the appropriate prompt template.
    topic:
        Problem topic string (informational, not currently used for routing).

    Returns
    -------
    (success, raw_output, parsed_answer)
        *success* is ``False`` when the model is unavailable or raises an
        exception.  *raw_output* is the full model response.
        *parsed_answer* is the extracted final answer.
    """
    if not _load_model():
        return False, f"LLM unavailable: {_load_error}", ""

    try:
        raw = _cached_generate(problem_text, skill_hint)
        answer = _parse_answer(raw)
        return True, raw, answer
    except MemoryError as exc:
        msg = f"Out of memory during LLM inference: {exc}"
        logger.error(msg)
        return False, msg, ""
    except Exception as exc:
        msg = f"LLM inference error: {exc}"
        logger.warning(msg)
        return False, msg, ""


def backend_name() -> str:
    """Return the active backend name for logging."""
    return f"llm:{SOLVER_MODEL_NAME}" if SOLVER_BACKEND == "llm" else "sympy"
