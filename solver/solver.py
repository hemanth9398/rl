"""Solver: LLM primary engine with SymPy fallback."""
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sympy as sp
from sympy import symbols, Function, Eq, solve, dsolve, simplify, expand, factor
from sympy.abc import x, y
from sympy import symbols as _symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
C1 = _symbols("C1")

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# ---------------------------------------------------------------------------
# LLM backend import (lazy — only fails at call time if unavailable)
# ---------------------------------------------------------------------------

try:
    from solver.llm_backend import generate_solution as _llm_generate, SOLVER_BACKEND
except ImportError:  # pragma: no cover - should always succeed in-repo
    def _llm_generate(*_args, **_kwargs):  # type: ignore[misc]
        return False, "llm_backend not found", ""
    SOLVER_BACKEND = "sympy"


def _sympify_expr(s: str, locals_dict: Optional[Dict] = None) -> sp.Expr:
    """Parse a math expression string with implicit multiplication support."""
    s = s.strip().replace("^", "**")
    return parse_expr(s, local_dict=locals_dict or {}, transformations=_TRANSFORMS)


@dataclass
class SolveResult:
    success: bool
    new_state: Dict[str, Any]
    reasoning_text: str
    answer: Optional[str] = None
    skill_id: Optional[str] = None
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Individual skill implementations
# ---------------------------------------------------------------------------

def _skill_solve_linear(state: Dict[str, Any]) -> SolveResult:
    """Solve a linear equation — LLM primary, SymPy fallback."""
    problem = state.get("problem_text", "")
    t0 = time.time()
    skill_id = "skill_solve_linear"

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer, "solved_symbol": "x"}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    # SymPy fallback
    try:
        eq_str = _extract_equation(problem)
        lhs_str, rhs_str = eq_str.split("=", 1)
        lhs = _sympify_expr(lhs_str)
        rhs = _sympify_expr(rhs_str)
        eq = Eq(lhs, rhs)
        sol = solve(eq, x)
        if not sol:
            return SolveResult(False, state, "No solution found for linear equation.")
        answer = str(sol[0])
        new_state = {**state, "candidate_answer": answer, "solved_symbol": "x"}
        reasoning = f"Solved linear equation: {eq} → x = {answer}"
        return SolveResult(True, new_state, reasoning, answer=answer,
                           skill_id=skill_id, duration=time.time() - t0)
    except Exception as exc:
        return SolveResult(False, state, f"solve_linear failed: {exc}",
                           skill_id=skill_id, duration=time.time() - t0)


def _skill_solve_quadratic(state: Dict[str, Any]) -> SolveResult:
    """Solve quadratic equation — LLM primary, SymPy fallback."""
    problem = state.get("problem_text", "")
    t0 = time.time()
    skill_id = "skill_solve_quadratic"

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer, "solved_symbol": "x"}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    # SymPy fallback
    try:
        eq_str = _extract_equation(problem)
        lhs_str, rhs_str = eq_str.split("=", 1)
        lhs = _sympify_expr(lhs_str)
        rhs = _sympify_expr(rhs_str)
        eq = Eq(lhs, rhs)
        sol = solve(eq, x)
        if sol is None or len(sol) == 0:
            return SolveResult(False, state, "No solution found.")
        answer = str(sorted([str(s) for s in sol]))
        new_state = {**state, "candidate_answer": str(sol), "solved_symbol": "x",
                     "solution_list": [str(s) for s in sol]}
        reasoning = f"Solved quadratic: {eq} → x ∈ {sol}"
        return SolveResult(True, new_state, reasoning, answer=answer,
                           skill_id=skill_id, duration=time.time() - t0)
    except Exception as exc:
        return SolveResult(False, state, f"solve_quadratic failed: {exc}",
                           skill_id=skill_id, duration=time.time() - t0)


def _skill_factor_polynomial(state: Dict[str, Any]) -> SolveResult:
    """Factor a polynomial — LLM primary, SymPy fallback."""
    problem = state.get("problem_text", "")
    t0 = time.time()
    skill_id = "skill_factor_polynomial"

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    # SymPy fallback
    try:
        expr_str = _extract_expression(problem)
        expr = _sympify_expr(expr_str)
        factored = factor(expr)
        answer = str(factored)
        new_state = {**state, "candidate_answer": answer}
        reasoning = f"Factored {expr} → {factored}"
        return SolveResult(True, new_state, reasoning, answer=answer,
                           skill_id=skill_id, duration=time.time() - t0)
    except Exception as exc:
        return SolveResult(False, state, f"factor_polynomial failed: {exc}",
                           skill_id=skill_id, duration=time.time() - t0)


def _skill_separate_variables(state: Dict[str, Any]) -> SolveResult:
    """Solve a separable ODE — LLM primary, SymPy fallback."""
    problem = state.get("problem_text", "")
    t0 = time.time()
    skill_id = "skill_separate_variables"

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer, "general_solution": answer,
                         "solved_symbol": "y"}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    return _skill_ode_sympy(state, skill_id=skill_id)


def _skill_integrating_factor(state: Dict[str, Any]) -> SolveResult:
    """Solve a linear first-order ODE — LLM primary, SymPy fallback."""
    problem = state.get("problem_text", "")
    t0 = time.time()
    skill_id = "skill_integrating_factor"

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer, "general_solution": answer,
                         "solved_symbol": "y"}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    return _skill_ode_sympy(state, skill_id=skill_id)


def _skill_ode_sympy(state: Dict[str, Any], skill_id: str = "skill_ode_sympy") -> SolveResult:
    """Generic ODE solver — LLM primary, SymPy fallback via dsolve."""
    problem = state.get("problem_text", "")
    t0 = time.time()

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer, "general_solution": answer,
                         "solved_symbol": "y"}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    # SymPy fallback
    try:
        ode_eq = _parse_ode(problem)
        if ode_eq is None:
            return SolveResult(False, state, "Could not parse ODE.",
                               skill_id=skill_id, duration=time.time() - t0)
        sol = dsolve(ode_eq)
        answer = str(sol.rhs)
        new_state = {**state, "candidate_answer": answer, "general_solution": answer,
                     "solved_symbol": "y"}
        reasoning = f"Solved ODE via dsolve: {ode_eq} → y = {answer}"
        return SolveResult(True, new_state, reasoning, answer=answer,
                           skill_id=skill_id, duration=time.time() - t0)
    except Exception as exc:
        return SolveResult(False, state, f"ode_sympy failed: {exc}",
                           skill_id=skill_id, duration=time.time() - t0)


def _skill_apply_initial_condition(state: Dict[str, Any]) -> SolveResult:
    """Apply initial condition to general solution — LLM primary, SymPy fallback."""
    problem = state.get("problem_text", "")
    gen_sol = state.get("general_solution") or state.get("candidate_answer", "")
    t0 = time.time()
    skill_id = "skill_apply_initial_condition"

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer, "particular_solution": answer}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    # SymPy fallback
    try:
        ic = _extract_initial_condition(problem)
        if ic is None:
            return SolveResult(False, state, "No initial condition found.",
                               skill_id=skill_id, duration=time.time() - t0)
        x0_val, y0_val = ic
        xsym = symbols("x")
        c1 = symbols("C1")
        gen_expr = sp.sympify(gen_sol, locals={"C1": c1, "x": xsym, "exp": sp.exp,
                                                "sin": sp.sin, "cos": sp.cos,
                                                "sqrt": sp.sqrt, "tan": sp.tan})
        lhs = gen_expr.subs(xsym, x0_val)
        c_sol = solve(Eq(lhs, y0_val), c1)
        if not c_sol:
            return SolveResult(False, state, "Could not solve for constant.",
                               skill_id=skill_id, duration=time.time() - t0)
        particular = gen_expr.subs(c1, c_sol[0])
        answer = str(particular)
        new_state = {**state, "candidate_answer": answer, "particular_solution": answer}
        reasoning = (f"Applied IC y({x0_val})={y0_val}: C1={c_sol[0]} → y = {answer}")
        return SolveResult(True, new_state, reasoning, answer=answer,
                           skill_id=skill_id, duration=time.time() - t0)
    except Exception as exc:
        return SolveResult(False, state, f"apply_ic failed: {exc}",
                           skill_id=skill_id, duration=time.time() - t0)


def _skill_algebra_sympy(state: Dict[str, Any]) -> SolveResult:
    """Generic algebraic solver — LLM primary, SymPy fallback."""
    problem = state.get("problem_text", "")
    t0 = time.time()
    skill_id = "skill_algebra_sympy"

    if SOLVER_BACKEND == "llm":
        ok, raw, answer = _llm_generate(problem, skill_id)
        if ok and answer:
            new_state = {**state, "candidate_answer": answer, "solved_symbol": "x"}
            return SolveResult(True, new_state, f"LLM: {raw}", answer=answer,
                               skill_id=skill_id, duration=time.time() - t0)

    # SymPy fallback
    try:
        eq_str = _extract_equation(problem)
        lhs_str, rhs_str = eq_str.split("=", 1)
        lhs = _sympify_expr(lhs_str)
        rhs = _sympify_expr(rhs_str)
        eq = Eq(lhs, rhs)
        sol = solve(eq, x)
        if sol is None or len(sol) == 0:
            return SolveResult(False, state, "No solution found.",
                               skill_id=skill_id, duration=time.time() - t0)
        answer = str(sol[0]) if len(sol) == 1 else str(sol)
        new_state = {**state, "candidate_answer": answer, "solved_symbol": "x"}
        reasoning = f"Solved algebra: {eq} → x = {answer}"
        return SolveResult(True, new_state, reasoning, answer=answer,
                           skill_id=skill_id, duration=time.time() - t0)
    except Exception as exc:
        return SolveResult(False, state, f"algebra_sympy failed: {exc}",
                           skill_id=skill_id, duration=time.time() - t0)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_equation(text: str) -> str:
    """Extract the equation part from a problem statement."""
    match = re.search(r"[Ss]olve.*?:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"([^:]+=[^:]+)", text)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Cannot extract equation from: {text!r}")


def _extract_expression(text: str) -> str:
    """Extract an expression to factor/simplify."""
    match = re.search(r"[Ff]actor.*?:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return text.split(":")[-1].strip()


def _ode_locals(xsym: sp.Symbol) -> Dict:
    """Build a locals dict for sympify of ODE expressions."""
    y_func = Function("y")
    return {
        "y": y_func(xsym),
        "x": xsym,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "tan": sp.tan,
        "atan": sp.atan,
        "asin": sp.asin,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
    }


def _parse_ode(text: str) -> Optional[sp.Eq]:
    """Parse a first-order ODE from text and return a sympy Eq."""
    xsym = symbols("x")
    y_func = Function("y")
    loc = _ode_locals(xsym)

    def _parse_rhs(rhs_raw: str) -> Optional[sp.Expr]:
        rhs_str = rhs_raw.strip().replace("^", "**")
        try:
            return parse_expr(rhs_str, local_dict=loc, transformations=_TRANSFORMS)
        except Exception:
            return None

    # Pattern 1: "dy/dx = <rhs>"
    match = re.search(r"d[yY]/d[xX]\s*=\s*(.+?)(?:,|$)", text)
    if match:
        rhs = _parse_rhs(match.group(1))
        if rhs is not None:
            return Eq(y_func(xsym).diff(xsym), rhs)

    # Pattern 2: "dy/dx + P(x)*y = Q(x)"
    match = re.search(r"d[yY]/d[xX]\s*\+\s*(.+?)\s*=\s*(.+?)(?:,|$)", text)
    if match:
        P = _parse_rhs(match.group(1))
        Q = _parse_rhs(match.group(2))
        if P is not None and Q is not None:
            return Eq(y_func(xsym).diff(xsym), Q - P)

    # Pattern 3: "(g(y))dy = (f(x))dx"
    match = re.search(r"\((.+?)\)\s*dy\s*=\s*\((.+?)\)\s*dx", text)
    if match:
        g = _parse_rhs(match.group(1))
        f = _parse_rhs(match.group(2))
        if g is not None and f is not None:
            return Eq(y_func(xsym).diff(xsym), f / g)

    return None


def _extract_initial_condition(text: str):
    """Extract (x0, y0) from text like 'y(0) = 3' or 'y(1) = 2'."""
    match = re.search(r"y\(([^)]+)\)\s*=\s*([\d\./-]+)", text)
    if match:
        try:
            x0 = sp.sympify(match.group(1).strip())
            y0 = sp.sympify(match.group(2).strip())
            return x0, y0
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

_SKILL_DISPATCH = {
    "skill_solve_linear": _skill_solve_linear,
    "skill_solve_quadratic": _skill_solve_quadratic,
    "skill_factor_polynomial": _skill_factor_polynomial,
    "skill_separate_variables": _skill_separate_variables,
    "skill_integrating_factor": _skill_integrating_factor,
    "skill_ode_sympy": _skill_ode_sympy,
    "skill_apply_initial_condition": _skill_apply_initial_condition,
    "skill_algebra_sympy": _skill_algebra_sympy,
    "skill_direct_integration": _skill_ode_sympy,
}

# Mapping of topic → default skill
_TOPIC_DEFAULT_SKILL = {
    "algebra_linear": "skill_solve_linear",
    "algebra_quadratic": "skill_solve_quadratic",
    "algebra_factor": "skill_factor_polynomial",
    "ode_separable": "skill_separate_variables",
    "ode_linear_first": "skill_integrating_factor",
    "ode_ivp": "skill_ode_sympy",
}


class Solver:
    """SymPy-based template solver."""

    def execute_skill(
        self, skill_node: Dict[str, Any], state: Dict[str, Any]
    ) -> SolveResult:
        """Execute a single skill on the current state."""
        skill_id = skill_node.get("id", "")
        fn = _SKILL_DISPATCH.get(skill_id)
        if fn is not None:
            return fn(state)
        # Fall back to combined skill execution
        if skill_id.startswith("skill_combined_"):
            parts = skill_id.replace("skill_combined_", "").split("_skill_")
            s1 = parts[0] if parts else ""
            s2 = "skill_" + parts[1] if len(parts) > 1 else ""
            result = self.execute_skill({"id": s1}, state)
            if result.success:
                result2 = self.execute_skill({"id": s2}, result.new_state)
                if result2.success:
                    return result2
                return result
            return result
        # Generic fallback
        return SolveResult(
            False, state, f"No implementation for skill: {skill_id}",
            skill_id=skill_id
        )

    def attempt_solve(
        self, problem: Dict[str, Any], skills_sequence: List[Dict[str, Any]]
    ) -> SolveResult:
        """Try solving using a sequence of skills."""
        state = {
            "problem_text": problem.get("statement", ""),
            "topic": problem.get("topic", ""),
            "problem_id": problem.get("id", ""),
        }
        last_result = SolveResult(False, state, "No skills attempted.")
        for skill_node in skills_sequence:
            result = self.execute_skill(skill_node, state)
            last_result = result
            if result.success:
                state = result.new_state
                # If we got a general solution and IC is needed, continue
                if "general_solution" in state and _has_ic(problem.get("statement", "")):
                    continue
                if state.get("candidate_answer"):
                    break
        return last_result

    def repair(
        self,
        state: Dict[str, Any],
        error_diagnostics: List[str],
        repair_hints: List[str],
    ) -> SolveResult:
        """Attempt repair based on verifier feedback."""
        for diag in error_diagnostics:
            if diag == "integration_constant_missing":
                return _skill_apply_initial_condition(state)
            if diag in ("ic_not_satisfied",):
                return _skill_apply_initial_condition(state)
        # Generic: re-attempt with default algebra solver
        topic = state.get("topic", "")
        skill_id = _TOPIC_DEFAULT_SKILL.get(topic, "skill_algebra_sympy")
        fn = _SKILL_DISPATCH.get(skill_id, _skill_algebra_sympy)
        return fn(state)

    def default_skill_for_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        skill_id = _TOPIC_DEFAULT_SKILL.get(topic)
        if skill_id:
            return {"id": skill_id}
        return None


def _has_ic(problem_text: str) -> bool:
    return bool(re.search(r"y\([^)]+\)\s*=", problem_text))
