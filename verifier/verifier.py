"""SymPy-based verifier with error diagnostics."""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sympy as sp
from sympy import symbols, Function, Eq, simplify, solve, dsolve
from sympy.abc import x
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse_sympy(s: str, local_dict: Optional[Dict] = None) -> sp.Expr:
    """Parse a math string with implicit multiplication support."""
    s = s.strip().replace("^", "**")
    return parse_expr(s, local_dict=local_dict or {}, transformations=_TRANSFORMS)


@dataclass
class VerifyResult:
    passed: bool
    diagnostics: List[str] = field(default_factory=list)
    error_location: str = ""
    repair_hints: List[str] = field(default_factory=list)
    detail: str = ""


class Verifier:
    """Verify correctness of candidate answers using SymPy."""

    def verify(
        self, problem: Dict[str, Any], candidate_answer: str
    ) -> VerifyResult:
        topic = problem.get("topic", "")
        answer_spec = problem.get("answer_spec", {})
        spec_type = answer_spec.get("type", "value")

        if topic.startswith("algebra"):
            return self._verify_algebra(problem, candidate_answer, answer_spec)
        elif topic.startswith("ode"):
            return self._verify_ode(problem, candidate_answer, answer_spec)
        else:
            # Generic: compare strings
            return self._verify_string(candidate_answer, answer_spec)

    # ------------------------------------------------------------------
    # Algebra verification
    # ------------------------------------------------------------------

    def _verify_algebra(
        self,
        problem: Dict[str, Any],
        candidate: str,
        answer_spec: Dict[str, Any],
    ) -> VerifyResult:
        spec_type = answer_spec.get("type", "value")
        try:
            if spec_type == "value":
                return self._verify_single_value(problem, candidate, answer_spec)
            elif spec_type == "set":
                return self._verify_solution_set(problem, candidate, answer_spec)
            elif spec_type == "expression":
                return self._verify_expression(problem, candidate, answer_spec)
            else:
                return self._verify_string(candidate, answer_spec)
        except Exception as exc:
            return VerifyResult(
                passed=False,
                diagnostics=["verification_exception"],
                error_location="verify_algebra",
                repair_hints=["Check parsing of candidate answer"],
                detail=str(exc),
            )

    def _verify_single_value(
        self, problem: Dict[str, Any], candidate: str, answer_spec: Dict[str, Any]
    ) -> VerifyResult:
        """Verify by substituting value back into equation."""
        expected = answer_spec.get("value", "")
        try:
            cand_val = sp.sympify(candidate)
            exp_val = sp.sympify(str(expected))
            if simplify(cand_val - exp_val) == 0:
                return VerifyResult(passed=True, detail="Exact match with expected value")
        except Exception:
            pass

        # Substitute back into original equation
        statement = problem.get("statement", "")
        try:
            eq_str = _extract_equation(statement)
            lhs_str, rhs_str = eq_str.split("=", 1)
            lhs = _parse_sympy(lhs_str)
            rhs = _parse_sympy(rhs_str)
            cand_val = sp.sympify(candidate)
            diff = simplify(lhs.subs(x, cand_val) - rhs.subs(x, cand_val))
            if diff == 0:
                return VerifyResult(passed=True, detail="Substitution check passed")
            return VerifyResult(
                passed=False,
                diagnostics=["substitution_failed"],
                error_location="substitute_answer",
                repair_hints=["Re-solve the equation", "Check sign errors"],
                detail=f"LHS - RHS = {diff} ≠ 0",
            )
        except Exception as exc:
            return VerifyResult(
                passed=False,
                diagnostics=["simplification_error"],
                error_location="verify_single_value",
                repair_hints=["Check equation parsing"],
                detail=str(exc),
            )

    def _verify_solution_set(
        self, problem: Dict[str, Any], candidate: str, answer_spec: Dict[str, Any]
    ) -> VerifyResult:
        """Verify a set of roots."""
        expected_str = answer_spec.get("value", "[]")
        statement = problem.get("statement", "")
        try:
            eq_str = _extract_equation(statement)
            lhs_str, rhs_str = eq_str.split("=", 1)
            lhs = _parse_sympy(lhs_str)
            rhs = _parse_sympy(rhs_str)

            # Parse candidate (may be a list repr)
            cand_list = _parse_list(candidate)
            if not cand_list:
                return VerifyResult(
                    passed=False,
                    diagnostics=["empty_solution"],
                    error_location="solution_set",
                    repair_hints=["Return at least one solution"],
                )
            failed_vals = []
            for val_str in cand_list:
                val = sp.sympify(val_str)
                diff = simplify(lhs.subs(x, val) - rhs.subs(x, val))
                if diff != 0:
                    failed_vals.append(str(val))
            if failed_vals:
                return VerifyResult(
                    passed=False,
                    diagnostics=["substitution_failed"],
                    error_location="solution_set",
                    repair_hints=["Check factoring or quadratic formula"],
                    detail=f"Values that failed check: {failed_vals}",
                )
            return VerifyResult(passed=True, detail="All roots satisfy equation")
        except Exception as exc:
            return VerifyResult(
                passed=False,
                diagnostics=["simplification_error"],
                error_location="verify_solution_set",
                detail=str(exc),
            )

    def _verify_expression(
        self, problem: Dict[str, Any], candidate: str, answer_spec: Dict[str, Any]
    ) -> VerifyResult:
        """Verify factored/simplified expression by expanding both and comparing."""
        expected = answer_spec.get("value", "")
        try:
            cand_expr = sp.sympify(candidate.replace("^", "**"))
            exp_expr = sp.sympify(str(expected).replace("^", "**"))
            diff = sp.expand(cand_expr - exp_expr)
            if diff == 0:
                return VerifyResult(passed=True, detail="Expression match confirmed")
            return VerifyResult(
                passed=False,
                diagnostics=["expression_mismatch"],
                error_location="verify_expression",
                repair_hints=["Check factoring steps"],
                detail=f"Difference: {diff}",
            )
        except Exception as exc:
            return VerifyResult(
                passed=False,
                diagnostics=["simplification_error"],
                detail=str(exc),
            )

    # ------------------------------------------------------------------
    # ODE verification
    # ------------------------------------------------------------------

    def _verify_ode(
        self,
        problem: Dict[str, Any],
        candidate: str,
        answer_spec: Dict[str, Any],
    ) -> VerifyResult:
        spec_type = answer_spec.get("type", "ode_general")
        statement = problem.get("statement", "")
        try:
            xsym = symbols("x")
            y_func = Function("y")
            c1 = symbols("C1")

            # Parse candidate solution
            cand_str = candidate.replace("^", "**")
            cand_expr = sp.sympify(cand_str, locals={"C1": c1, "x": xsym})

            # Check initial condition if IVP
            if spec_type == "ode_particular":
                ic = _extract_ic(statement)
                if ic is not None:
                    x0, y0 = ic
                    val_at_x0 = cand_expr.subs(xsym, x0)
                    diff = simplify(val_at_x0 - y0)
                    if diff != 0:
                        return VerifyResult(
                            passed=False,
                            diagnostics=["ic_not_satisfied"],
                            error_location="verify_ode_ic",
                            repair_hints=["Re-apply initial condition and solve for C"],
                            detail=f"y({x0})={val_at_x0} ≠ {y0}",
                        )

            # Differentiate candidate and check it satisfies ODE
            ode_eq = _parse_ode_for_verify(statement)
            if ode_eq is not None:
                dy_dx = sp.diff(cand_expr, xsym)
                # Substitute y = cand_expr in the ODE RHS
                rhs = ode_eq.rhs
                # Replace y(x) with cand_expr
                rhs_sub = rhs.subs(y_func(xsym), cand_expr)
                residual = simplify(dy_dx - rhs_sub)
                if residual == 0:
                    return VerifyResult(passed=True, detail="ODE and IC satisfied")
                # Try numeric check
                numeric_ok = _numeric_ode_check(cand_expr, dy_dx, rhs_sub, xsym)
                if numeric_ok:
                    return VerifyResult(passed=True, detail="ODE satisfied (numeric check)")
                return VerifyResult(
                    passed=False,
                    diagnostics=["derivative_mismatch"],
                    error_location="verify_ode_derivative",
                    repair_hints=["Check differentiation of solution", "Recheck ODE form"],
                    detail=f"Residual dy/dx - f(x,y) = {residual}",
                )

            # Fallback: symbolic comparison with expected
            expected = answer_spec.get("value", "")
            if expected:
                try:
                    exp_expr = sp.sympify(
                        str(expected).replace("^", "**"),
                        locals={"C1": c1, "x": xsym}
                    )
                    diff = simplify(sp.expand(cand_expr) - sp.expand(exp_expr))
                    if diff == 0:
                        return VerifyResult(passed=True, detail="ODE solution matches expected")
                except Exception:
                    pass
            return VerifyResult(
                passed=False,
                diagnostics=["verification_inconclusive"],
                detail="Could not fully verify ODE solution",
            )
        except Exception as exc:
            return VerifyResult(
                passed=False,
                diagnostics=["verification_exception"],
                detail=str(exc),
            )

    # ------------------------------------------------------------------
    # String fallback
    # ------------------------------------------------------------------

    def _verify_string(
        self, candidate: str, answer_spec: Dict[str, Any]
    ) -> VerifyResult:
        expected = str(answer_spec.get("value", ""))
        if candidate.strip() == expected.strip():
            return VerifyResult(passed=True)
        # Try symbolic comparison
        try:
            cand = sp.sympify(candidate.replace("^", "**"))
            exp = sp.sympify(expected.replace("^", "**"))
            if simplify(cand - exp) == 0:
                return VerifyResult(passed=True, detail="Symbolic match")
        except Exception:
            pass
        return VerifyResult(
            passed=False,
            diagnostics=["string_mismatch"],
            detail=f"Got {candidate!r}, expected {expected!r}",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_equation(text: str) -> str:
    match = re.search(r"[Ss]olve.*?:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"([^:]+=[^:]+)", text)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Cannot extract equation: {text!r}")


def _parse_list(s: str) -> List[str]:
    """Parse a string like '[1, 2]' or '1' into a list of value strings."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        return [v.strip() for v in inner.split(",") if v.strip()]
    return [s]


def _extract_ic(text: str):
    match = re.search(r"y\(([^)]+)\)\s*=\s*([\d\./-]+)", text)
    if match:
        try:
            x0 = sp.sympify(match.group(1).strip())
            y0 = sp.sympify(match.group(2).strip())
            return x0, y0
        except Exception:
            pass
    return None


def _parse_ode_for_verify(text: str) -> Optional[sp.Eq]:
    """Parse ODE from problem text returning sympy Eq."""
    xsym = symbols("x")
    y_func = Function("y")
    loc = {
        "y": y_func(xsym), "x": xsym,
        "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
        "tan": sp.tan, "atan": sp.atan, "asin": sp.asin, "sqrt": sp.sqrt,
    }

    def _parse(s: str) -> Optional[sp.Expr]:
        try:
            return parse_expr(s.replace("^", "**"), local_dict=loc,
                              transformations=_TRANSFORMS)
        except Exception:
            return None

    match = re.search(r"d[yY]/d[xX]\s*=\s*(.+?)(?:,|$)", text)
    if match:
        rhs = _parse(match.group(1).strip())
        if rhs is not None:
            return Eq(y_func(xsym).diff(xsym), rhs)
    match = re.search(r"d[yY]/d[xX]\s*\+\s*(.+?)\s*=\s*(.+?)(?:,|$)", text)
    if match:
        P = _parse(match.group(1).strip())
        Q = _parse(match.group(2).strip())
        if P is not None and Q is not None:
            return Eq(y_func(xsym).diff(xsym), Q - P)
    return None


def _numeric_ode_check(cand_expr, dy_dx, rhs_sub, xsym, num_points: int = 5) -> bool:
    """Numerically check the ODE at a few test points."""
    import random
    c1 = symbols("C1")
    # Fix C1 = 1 for the check
    cand_num = cand_expr.subs(c1, 1)
    dy_num = dy_dx.subs(c1, 1)
    rhs_num = rhs_sub.subs(c1, 1)
    test_points = [0.1 * i for i in range(1, num_points + 1)]
    for pt in test_points:
        try:
            lhs_val = float(dy_num.subs(xsym, pt).evalf())
            rhs_val = float(rhs_num.subs(xsym, pt).evalf())
            if abs(lhs_val - rhs_val) > 1e-4:
                return False
        except Exception:
            return False
    return True
