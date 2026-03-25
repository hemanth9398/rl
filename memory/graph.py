"""Memory Graph: NetworkX-based skill/concept/error node store."""
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Node factory helpers
# ---------------------------------------------------------------------------

def make_skill_node(
    node_id: str,
    label: str,
    domain: str = "math",
    topic: str = "general",
    trigger: Optional[Dict] = None,
    procedure: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "skill",
        "label": label,
        "domain": domain,
        "topic": topic,
        "trigger": trigger or {},
        "procedure": procedure or [],
        "use_count": 0,
        "success_count": 0,
        "recent_uses": [],
        "avg_cost": 0.0,
        "last_used": None,
        "embedding": None,
    }


def make_concept_node(
    node_id: str,
    label: str,
    domain: str = "math",
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "concept",
        "label": label,
        "domain": domain,
        "keywords": keywords or [],
    }


def make_error_node(
    node_id: str,
    label: str,
    diagnostics: str = "",
    repair_hint: str = "",
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": "error",
        "label": label,
        "diagnostics": diagnostics,
        "repair_hint": repair_hint,
    }


# ---------------------------------------------------------------------------
# MemoryGraph
# ---------------------------------------------------------------------------

class MemoryGraph:
    """NetworkX DiGraph holding skill, concept, and error nodes."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node addition
    # ------------------------------------------------------------------

    def add_skill_node(self, node_id: str, **kwargs: Any) -> None:
        node = make_skill_node(node_id, **kwargs)
        self.graph.add_node(node_id, **node)

    def add_concept_node(self, node_id: str, **kwargs: Any) -> None:
        node = make_concept_node(node_id, **kwargs)
        self.graph.add_node(node_id, **node)

    def add_error_node(self, node_id: str, **kwargs: Any) -> None:
        node = make_error_node(node_id, **kwargs)
        self.graph.add_node(node_id, **node)

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        weight: float = 1.0,
    ) -> None:
        self.graph.add_edge(src, dst, edge_type=edge_type, weight=weight)

    def update_edge_weight(self, src: str, dst: str, delta: float) -> None:
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]["weight"] = max(
                0.0, self.graph[src][dst].get("weight", 1.0) + delta
            )
        else:
            self.graph.add_edge(src, dst, edge_type="transition", weight=max(0.0, delta))

    # ------------------------------------------------------------------
    # Node stats
    # ------------------------------------------------------------------

    def update_node_stats(
        self,
        node_id: str,
        success: bool,
        cost: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        if node_id not in self.graph.nodes:
            return
        ts = timestamp or time.time()
        node = self.graph.nodes[node_id]
        node["use_count"] = node.get("use_count", 0) + 1
        if success:
            node["success_count"] = node.get("success_count", 0) + 1
        recent = node.get("recent_uses", [])
        recent.append((ts, success))
        node["recent_uses"] = recent[-50:]  # keep last 50
        prev_cost = node.get("avg_cost", 0.0)
        use_count = node["use_count"]
        node["avg_cost"] = prev_cost + (cost - prev_cost) / use_count
        node["last_used"] = ts

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None

    def has_node(self, node_id: str) -> bool:
        """Return True if a node with the given ID exists in the graph."""
        return node_id in self.graph.nodes

    def get_all_skills(self) -> List[Dict[str, Any]]:
        return [
            dict(data)
            for _, data in self.graph.nodes(data=True)
            if data.get("type") == "skill"
        ]

    def get_candidate_skills(
        self, problem_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return skill nodes matching topic/keywords from problem_features."""
        topic = problem_features.get("topic", "")
        keywords = problem_features.get("keywords", [])
        kw_lower = [k.lower() for k in keywords]

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "skill":
                continue
            score = 0.0
            # topic match
            if data.get("topic", "") == topic:
                score += 2.0
            elif data.get("domain", "") == problem_features.get("domain", ""):
                score += 0.5
            # keyword match
            trigger = data.get("trigger", {})
            trigger_kws = [k.lower() for k in trigger.get("keywords", [])]
            for kw in kw_lower:
                if kw in trigger_kws:
                    score += 1.0
            # boost recently successful skills
            use_count = data.get("use_count", 0)
            success_count = data.get("success_count", 0)
            if use_count > 0:
                sr = success_count / use_count
                score += sr * 0.5
            if score > 0:
                scored.append((score, dict(data)))

        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored]

    def get_transitions(self, node_id: str) -> List[Tuple[str, float]]:
        """Return list of (next_node_id, weight) for transition edges."""
        result = []
        for _, dst, data in self.graph.out_edges(node_id, data=True):
            if data.get("edge_type") == "transition":
                result.append((dst, data.get("weight", 1.0)))
        result.sort(key=lambda x: -x[1])
        return result

    def get_error_nodes(self, node_id: str) -> List[Tuple[Dict, float]]:
        """Return list of (error_node_data, weight) for causes_error edges."""
        result = []
        for _, dst, data in self.graph.out_edges(node_id, data=True):
            if data.get("edge_type") == "causes_error":
                err_data = self.get_node(dst)
                if err_data:
                    result.append((err_data, data.get("weight", 1.0)))
        result.sort(key=lambda x: -x[1])
        return result

    def get_repair_skills(self, error_node_id: str) -> List[Tuple[Dict, float]]:
        """Return skill nodes that fix this error."""
        result = []
        for _, dst, data in self.graph.out_edges(error_node_id, data=True):
            if data.get("edge_type") == "fixes":
                skill = self.get_node(dst)
                if skill:
                    result.append((skill, data.get("weight", 1.0)))
        result.sort(key=lambda x: -x[1])
        return result

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay_all(self, factor: float = 0.95) -> None:
        for src, dst in self.graph.edges():
            self.graph[src][dst]["weight"] = (
                self.graph[src][dst].get("weight", 1.0) * factor
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.graph = pickle.load(f)

    def save_json(self, path: str) -> None:
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_json(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()


# ---------------------------------------------------------------------------
# Seed graph initialisation
# ---------------------------------------------------------------------------

SEED_SKILLS = [
    {
        "node_id": "skill_solve_linear",
        "label": "Solve Linear Equation",
        "topic": "algebra_linear",
        "trigger": {
            "symbolic_pattern": "ax + b = c",
            "keywords": ["linear", "solve", "equation", "x"],
        },
        "procedure": [
            "isolate variable term",
            "subtract/add constants",
            "divide by coefficient",
            "simplify",
        ],
    },
    {
        "node_id": "skill_solve_quadratic",
        "label": "Solve Quadratic Equation",
        "topic": "algebra_quadratic",
        "trigger": {
            "symbolic_pattern": "ax^2 + bx + c = 0",
            "keywords": ["quadratic", "x^2", "x**2"],
        },
        "procedure": [
            "identify coefficients a, b, c",
            "try factoring",
            "use quadratic formula if needed",
            "simplify roots",
        ],
    },
    {
        "node_id": "skill_factor_polynomial",
        "label": "Factor Polynomial",
        "topic": "algebra_factor",
        "trigger": {
            "symbolic_pattern": "ax^n + ...",
            "keywords": ["factor", "polynomial"],
        },
        "procedure": [
            "look for common factors",
            "try difference of squares",
            "try perfect square trinomial",
            "apply general factoring",
        ],
    },
    {
        "node_id": "skill_complete_square",
        "label": "Complete the Square",
        "topic": "algebra_quadratic",
        "trigger": {
            "symbolic_pattern": "x^2 + bx + ...",
            "keywords": ["complete square", "vertex form"],
        },
        "procedure": [
            "rewrite x^2 + bx as (x + b/2)^2 - (b/2)^2",
            "adjust constant",
            "simplify",
        ],
    },
    {
        "node_id": "skill_separate_variables",
        "label": "Separate Variables",
        "topic": "ode_separable",
        "trigger": {
            "symbolic_pattern": "dy/dx = f(x)*g(y)",
            "keywords": ["separable", "dy/dx", "separate"],
        },
        "procedure": [
            "rewrite as g(y)dy = f(x)dx",
            "integrate both sides",
            "add integration constant",
            "solve for y if possible",
        ],
    },
    {
        "node_id": "skill_integrating_factor",
        "label": "Integrating Factor Method",
        "topic": "ode_linear_first",
        "trigger": {
            "symbolic_pattern": "dy/dx + P(x)y = Q(x)",
            "keywords": ["integrating factor", "linear ode", "first order"],
        },
        "procedure": [
            "identify P(x)",
            "compute mu = exp(integral P(x)dx)",
            "multiply both sides by mu",
            "integrate d/dx(mu*y) = mu*Q(x)",
            "solve for y",
        ],
    },
    {
        "node_id": "skill_direct_integration",
        "label": "Direct Integration",
        "topic": "ode_separable",
        "trigger": {
            "symbolic_pattern": "dy/dx = f(x)",
            "keywords": ["integrate", "direct", "antiderivative"],
        },
        "procedure": [
            "integrate right side with respect to x",
            "add constant C",
        ],
    },
    {
        "node_id": "skill_apply_initial_condition",
        "label": "Apply Initial Condition",
        "topic": "ode_ivp",
        "trigger": {
            "symbolic_pattern": "y(x0) = y0",
            "keywords": ["initial condition", "y(0)", "ivp", "particular solution"],
        },
        "procedure": [
            "substitute x0 into general solution",
            "set equal to y0",
            "solve for constant C",
            "substitute C back",
        ],
    },
    {
        "node_id": "skill_substitute_simplify",
        "label": "Substitute and Simplify",
        "topic": "general",
        "trigger": {
            "symbolic_pattern": "expression with substitution",
            "keywords": ["substitute", "simplify", "plug in"],
        },
        "procedure": [
            "identify substitution target",
            "replace symbols",
            "apply algebraic simplification",
        ],
    },
    {
        "node_id": "skill_check_solution",
        "label": "Check Solution",
        "topic": "general",
        "trigger": {
            "symbolic_pattern": "verify solution",
            "keywords": ["check", "verify", "substitute back"],
        },
        "procedure": [
            "substitute solution into original equation",
            "simplify both sides",
            "check equality",
        ],
    },
    {
        "node_id": "skill_quadratic_formula",
        "label": "Quadratic Formula",
        "topic": "algebra_quadratic",
        "trigger": {
            "symbolic_pattern": "x = (-b ± sqrt(b^2 - 4ac)) / 2a",
            "keywords": ["quadratic formula", "discriminant"],
        },
        "procedure": [
            "compute discriminant D = b^2 - 4ac",
            "apply x = (-b ± sqrt(D)) / 2a",
            "simplify",
        ],
    },
    {
        "node_id": "skill_expand_simplify",
        "label": "Expand and Simplify",
        "topic": "algebra_linear",
        "trigger": {
            "symbolic_pattern": "a(bx + c)",
            "keywords": ["expand", "distribute", "simplify"],
        },
        "procedure": [
            "distribute multiplication",
            "combine like terms",
            "simplify",
        ],
    },
    {
        "node_id": "skill_ode_sympy",
        "label": "Solve ODE with SymPy",
        "topic": "ode_separable",
        "trigger": {
            "symbolic_pattern": "dy/dx = f(x, y)",
            "keywords": ["ode", "differential equation", "dsolve"],
        },
        "procedure": [
            "parse ODE",
            "call sympy.dsolve",
            "extract general solution",
            "simplify",
        ],
    },
    {
        "node_id": "skill_algebra_sympy",
        "label": "Solve Algebra with SymPy",
        "topic": "algebra_linear",
        "trigger": {
            "symbolic_pattern": "f(x) = 0",
            "keywords": ["solve", "algebra", "sympy"],
        },
        "procedure": [
            "parse equation",
            "call sympy.solve",
            "return solutions",
        ],
    },
    {
        "node_id": "skill_unit_check",
        "label": "Unit Check",
        "topic": "general",
        "trigger": {
            "symbolic_pattern": "dimensional analysis",
            "keywords": ["units", "dimensions", "unit check"],
        },
        "procedure": [
            "identify units of each term",
            "verify dimensional consistency",
        ],
    },
]

SEED_CONCEPTS = [
    {
        "node_id": "concept_derivative",
        "label": "Derivative",
        "keywords": ["derivative", "differentiate", "d/dx", "dy/dx"],
    },
    {
        "node_id": "concept_integral",
        "label": "Integral",
        "keywords": ["integral", "integrate", "antiderivative"],
    },
    {
        "node_id": "concept_equation",
        "label": "Equation",
        "keywords": ["equation", "solve", "roots", "solutions"],
    },
    {
        "node_id": "concept_constant",
        "label": "Arbitrary Constant",
        "keywords": ["constant", "C", "C1", "integration constant"],
    },
]

SEED_ERRORS = [
    {
        "node_id": "error_forgot_constant",
        "label": "Forgot Integration Constant",
        "diagnostics": "integration_constant_missing",
        "repair_hint": "Add arbitrary constant C after integration",
    },
    {
        "node_id": "error_wrong_sign",
        "label": "Wrong Sign",
        "diagnostics": "sign_error",
        "repair_hint": "Check sign when moving terms across equals sign",
    },
    {
        "node_id": "error_division_by_zero",
        "label": "Division by Zero",
        "diagnostics": "division_by_zero",
        "repair_hint": "Check denominator is non-zero; consider domain restrictions",
    },
    {
        "node_id": "error_ic_not_satisfied",
        "label": "Initial Condition Not Satisfied",
        "diagnostics": "ic_not_satisfied",
        "repair_hint": "Re-apply initial condition and solve for C",
    },
]


def build_seed_graph() -> MemoryGraph:
    """Construct and return a MemoryGraph pre-loaded with seed nodes."""
    mg = MemoryGraph()

    for s in SEED_SKILLS:
        mg.add_skill_node(**s)

    for c in SEED_CONCEPTS:
        mg.add_concept_node(**c)

    for e in SEED_ERRORS:
        mg.add_error_node(**e)

    # --- Edges ---
    # requires edges (skill → concept)
    mg.add_edge("skill_separate_variables", "concept_integral", "requires")
    mg.add_edge("skill_direct_integration", "concept_integral", "requires")
    mg.add_edge("skill_integrating_factor", "concept_integral", "requires")
    mg.add_edge("skill_integrating_factor", "concept_derivative", "requires")
    mg.add_edge("skill_apply_initial_condition", "concept_constant", "requires")
    mg.add_edge("skill_check_solution", "concept_equation", "requires")

    # transition edges (skill → skill)
    mg.add_edge("skill_separate_variables", "skill_apply_initial_condition", "transition", weight=1.5)
    mg.add_edge("skill_integrating_factor", "skill_apply_initial_condition", "transition", weight=1.5)
    mg.add_edge("skill_solve_quadratic", "skill_check_solution", "transition", weight=1.0)
    mg.add_edge("skill_solve_linear", "skill_check_solution", "transition", weight=1.0)
    mg.add_edge("skill_factor_polynomial", "skill_solve_quadratic", "transition", weight=1.2)
    mg.add_edge("skill_complete_square", "skill_solve_quadratic", "transition", weight=1.0)
    mg.add_edge("skill_expand_simplify", "skill_solve_linear", "transition", weight=1.0)
    mg.add_edge("skill_ode_sympy", "skill_apply_initial_condition", "transition", weight=1.5)
    mg.add_edge("skill_algebra_sympy", "skill_check_solution", "transition", weight=1.0)

    # causes_error edges (skill → error)
    mg.add_edge("skill_separate_variables", "error_forgot_constant", "causes_error", weight=0.3)
    mg.add_edge("skill_direct_integration", "error_forgot_constant", "causes_error", weight=0.3)
    mg.add_edge("skill_integrating_factor", "error_forgot_constant", "causes_error", weight=0.2)
    mg.add_edge("skill_solve_linear", "error_wrong_sign", "causes_error", weight=0.2)
    mg.add_edge("skill_apply_initial_condition", "error_ic_not_satisfied", "causes_error", weight=0.2)

    # fixes edges (error → skill)
    mg.add_edge("error_forgot_constant", "skill_apply_initial_condition", "fixes", weight=1.0)
    mg.add_edge("error_wrong_sign", "skill_substitute_simplify", "fixes", weight=1.0)
    mg.add_edge("error_ic_not_satisfied", "skill_apply_initial_condition", "fixes", weight=1.5)

    return mg
