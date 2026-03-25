"""Curriculum generator: creates problem variants targeting weak skills."""
import copy
import random
import re
from typing import Any, Dict, List, Optional

from memory.graph import MemoryGraph


class CurriculumGenerator:
    """Generates problem variants via parametric and adversarial transformations."""

    def __init__(self, rng_seed: Optional[int] = None) -> None:
        self._rng = random.Random(rng_seed)
        self._generated_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_variant(
        self, problem: Dict[str, Any], graph: MemoryGraph
    ) -> Dict[str, Any]:
        """Generate a variant of an existing problem."""
        topic = problem.get("topic", "algebra_linear")
        if topic.startswith("algebra_linear"):
            return self._variant_linear(problem)
        elif topic.startswith("algebra_quadratic"):
            return self._variant_quadratic(problem)
        elif topic.startswith("algebra_factor"):
            return self._variant_factor(problem)
        elif topic.startswith("ode"):
            return self._variant_ode(problem)
        return self._parametric_variant(problem)

    def get_weak_skills(
        self, graph: MemoryGraph, min_uses: int = 3
    ) -> List[Dict[str, Any]]:
        """Return skill nodes with lowest success rates (min uses threshold)."""
        skills = graph.get_all_skills()
        rated = []
        for s in skills:
            use_count = s.get("use_count", 0)
            if use_count < min_uses:
                continue
            sr = s.get("success_count", 0) / use_count
            rated.append((sr, s))
        rated.sort(key=lambda x: x[0])
        return [s for _, s in rated]

    def generate_for_weak_skill(
        self, skill: Dict[str, Any], base_problems: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate a problem specifically targeting a weak skill."""
        skill_topic = skill.get("topic", "algebra_linear")
        matching = [p for p in base_problems if p.get("topic", "") == skill_topic]
        if matching:
            base = self._rng.choice(matching)
            return self._parametric_variant(base)
        return self._generate_from_template(skill_topic)

    # ------------------------------------------------------------------
    # Variant strategies
    # ------------------------------------------------------------------

    def _variant_linear(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parametric: change coefficients in linear equation."""
        self._generated_count += 1
        a = self._rng.randint(2, 9)
        b = self._rng.randint(-10, 10)
        c = self._rng.randint(-20, 20)
        answer = (c - b) / a
        # Keep as integer if possible
        if answer == int(answer):
            answer = int(answer)
        new_prob = copy.deepcopy(problem)
        new_prob["id"] = f"gen_linear_{self._generated_count:04d}"
        new_prob["statement"] = f"Solve for x: {a}x + {b} = {c}"
        new_prob["answer_spec"] = {"type": "value", "symbol": "x", "value": str(answer)}
        new_prob["generated"] = True
        return new_prob

    def _variant_quadratic(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parametric: change roots → build quadratic."""
        self._generated_count += 1
        r1 = self._rng.randint(-5, 5)
        r2 = self._rng.randint(-5, 5)
        # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
        b = -(r1 + r2)
        c = r1 * r2
        b_str = f"+ {b}x" if b >= 0 else f"- {-b}x"
        c_str = f"+ {c}" if c >= 0 else f"- {-c}"
        roots = sorted([r1, r2])
        new_prob = copy.deepcopy(problem)
        new_prob["id"] = f"gen_quad_{self._generated_count:04d}"
        new_prob["statement"] = f"Solve for x: x^2 {b_str} {c_str} = 0"
        new_prob["answer_spec"] = {
            "type": "set", "symbol": "x", "value": str(roots)
        }
        new_prob["generated"] = True
        return new_prob

    def _variant_factor(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parametric: change factoring problem."""
        self._generated_count += 1
        r1 = self._rng.randint(-4, 4)
        r2 = self._rng.randint(-4, 4)
        b = -(r1 + r2)
        c = r1 * r2
        b_str = f"+ {b}x" if b >= 0 else f"- {-b}x"
        c_str = f"+ {c}" if c >= 0 else f"- {-c}"
        new_prob = copy.deepcopy(problem)
        new_prob["id"] = f"gen_factor_{self._generated_count:04d}"
        new_prob["statement"] = f"Factor: x^2 {b_str} {c_str}"
        new_prob["answer_spec"] = {
            "type": "expression",
            "symbol": "x",
            "value": f"(x - {r1})*(x - {r2})",
        }
        new_prob["generated"] = True
        return new_prob

    def _variant_ode(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parametric: change ODE coefficient."""
        self._generated_count += 1
        k = self._rng.randint(-4, 4)
        if k == 0:
            k = 1
        new_prob = copy.deepcopy(problem)
        new_prob["id"] = f"gen_ode_{self._generated_count:04d}"
        new_prob["statement"] = f"Solve the ODE: dy/dx = {k}y"
        new_prob["answer_spec"] = {
            "type": "ode_general",
            "symbol": "y",
            "value": f"C1*exp({k}*x)",
        }
        new_prob["topic"] = "ode_separable"
        new_prob["generated"] = True
        return new_prob

    def _parametric_variant(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Generic: perturb any numeric constant in the statement."""
        self._generated_count += 1
        statement = problem.get("statement", "")

        def perturb(m: re.Match) -> str:
            val = int(m.group(0))
            delta = self._rng.choice([-2, -1, 1, 2])
            return str(val + delta)

        new_statement = re.sub(r"\b\d+\b", perturb, statement)
        new_prob = copy.deepcopy(problem)
        new_prob["id"] = f"gen_param_{self._generated_count:04d}"
        new_prob["statement"] = new_statement
        new_prob["answer_spec"] = {"type": "value", "symbol": "x", "value": "unknown"}
        new_prob["generated"] = True
        return new_prob

    def _generate_from_template(self, topic: str) -> Dict[str, Any]:
        """Generate a fresh problem from a template for the given topic."""
        self._generated_count += 1
        if topic.startswith("algebra_linear"):
            return self._variant_linear({"topic": topic, "difficulty": 2})
        elif topic.startswith("algebra_quadratic"):
            return self._variant_quadratic({"topic": topic, "difficulty": 2})
        elif topic.startswith("ode"):
            return self._variant_ode({"topic": topic, "difficulty": 2})
        a = self._rng.randint(1, 5)
        b = self._rng.randint(-10, 10)
        c = self._rng.randint(-20, 20)
        return {
            "id": f"gen_template_{self._generated_count:04d}",
            "topic": topic,
            "difficulty": 2,
            "statement": f"Solve for x: {a}x + {b} = {c}",
            "answer_spec": {"type": "value", "symbol": "x", "value": "unknown"},
            "domain": "math",
            "generated": True,
        }
