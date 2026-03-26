#!/usr/bin/env python
"""
Main entry point for the multi-agent self-evolving memory system.

Usage:
    python scripts/run_multi_agent.py --duration 60 --num-agents 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

# Add parent dir to path so imports work when run from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.graph import build_seed_graph
from memory.dynamic_graph import DynamicMemoryGraph
from memory.episode_store import EpisodeStore
from memory.retrieval import Retriever
from memory.consolidation import Consolidator
from solver.solver import Solver
from solver.llm_solver import LLMSolver
from verifier.verifier import Verifier
from models.model_registry import ModelRegistry
from policy.policy_nn import PolicyNetwork, NUM_ACTIONS
from rl.grpo import GRPOTrainer
from rl.gigpo import GIGPOScorer
from agents.teacher import Teacher
from agents.sub_agent import SubAgent
from agents.validator import Validator
from agents.tree_of_thought import TreeOfThought
from agents.knowledge_transfer import KnowledgeTransferManager
from envs.multi_agent_env import MultiAgentEnv, OBS_DIM
from curriculum.generator import CurriculumGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_multi_agent")

# ---------------------------------------------------------------------------
# Tuning frequency constants
# ---------------------------------------------------------------------------
GRPO_UPDATE_EVERY = 10
GIGPO_UPDATE_EVERY = 5
CONSOLIDATE_EVERY = 20
CURRICULUM_EVERY = 15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seed_problems(path: str) -> List[Dict[str, Any]]:
    """Load seed problems from a JSON file."""
    if not os.path.exists(path):
        logger.warning("Seed problems file not found: %s", path)
        return []
    with open(path) as f:
        return json.load(f)


def pick_problem(
    problems: List[Dict[str, Any]],
    generated: List[Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    """Randomly pick from seed or generated problems."""
    pool = problems + generated
    if not pool:
        # Fallback synthetic problem
        return {
            "id": "synthetic_001",
            "topic": "algebra_linear",
            "difficulty": 1,
            "statement": "Solve for x: 2*x + 3 = 7",
            "answer_spec": {"type": "value", "symbol": "x", "value": "2"},
            "domain": "math",
        }
    return rng.choice(pool)


def log_metrics(
    metrics_path: str,
    all_metrics: List[Dict[str, Any]],
) -> None:
    """Append metrics to a JSON file."""
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    duration_minutes: int = 120,
    num_sub_agents: int = 3,
    db_path: str = "episodes.db",
    graph_path: str = "graph.pkl",
    metrics_path: str = "metrics.json",
    seed: int = 42,
    device_str: str = "cpu",
    solver_type: str = "llm",
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct",
    subagent_model: str = "Qwen3-14B/8B/4B",
    verifier_model: str = "Qwen/Qwen2.5-3B-Instruct",
    solver_model: Optional[str] = None,
    cpu_only: bool = False,
    load_in_4bit: bool = False,
    lora_rank: int = 16,
) -> None:
    """Run the multi-agent self-evolving memory training loop.

    1. Initialize DynamicMemoryGraph (with seed graph as starting point).
    2. Create Teacher, SubAgents, Validator.
    3. Create GRPO trainer for Teacher, GIGPO scorer for SubAgents.
    4. Create KnowledgeTransferManager.
    5. Create MultiAgentEnv.
    6. Main loop:
       a. Pick problem (from curriculum or seed).
       b. Run episode through MultiAgentEnv.
       c. Store episode, update graph stats.
       d. GRPO update for Teacher (every N episodes).
       e. GIGPO update for branch scorer (every M episodes).
       f. Consolidation pass (every K episodes).
       g. Log metrics.
    7. Save outputs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    device = torch.device(device_str)

    # ------------------------------------------------------------------
    # 1. Memory graph (starts from seed, grows dynamically)
    # ------------------------------------------------------------------
    dynamic_graph = DynamicMemoryGraph()
    seed_graph = build_seed_graph()
    # Copy seed nodes/edges into dynamic graph
    for node_id, data in seed_graph.graph.nodes(data=True):
        dynamic_graph.graph.add_node(node_id, **data)
    for src, dst, data in seed_graph.graph.edges(data=True):
        dynamic_graph.graph.add_edge(src, dst, **data)
    logger.info(
        "Seed graph loaded: %d nodes, %d edges",
        dynamic_graph.num_nodes,
        dynamic_graph.num_edges,
    )

    # Load persisted graph if available
    if os.path.exists(graph_path):
        try:
            dynamic_graph.load(graph_path)
            logger.info("Loaded graph from %s", graph_path)
        except Exception as exc:
            logger.warning("Could not load graph: %s", exc)

    # ------------------------------------------------------------------
    # 2. Shared components
    # ------------------------------------------------------------------
    episode_store = EpisodeStore(db_path=db_path)
    retriever = Retriever(graph=dynamic_graph, episode_store=episode_store)

    # Solver: prefer LLMSolver so the LLM generates answers
    if solver_type == "llm":
        _solver_model = solver_model or subagent_model
        solver = LLMSolver(model_name=_solver_model)
        logger.info("Solver: LLMSolver (model=%s)", _solver_model)
    else:
        solver = Solver()
        logger.info("Solver: SymPy Solver")

    # Model registry (used by Verifier for LLM-based verification)
    registry: Optional[ModelRegistry] = None
    if solver_type == "llm":
        try:
            registry = ModelRegistry.from_args(
                teacher_model=teacher_model,
                subagent_model=subagent_model,
                verifier_model=verifier_model,
                solver_model=solver_model or subagent_model,
                cpu_only=cpu_only,
                load_in_4bit=load_in_4bit,
                lora_rank=lora_rank,
            )
            logger.info("ModelRegistry created (LLM verification enabled)")
        except Exception as exc:
            logger.warning("Could not create ModelRegistry: %s — using SymPy-only verification", exc)

    verifier = Verifier(registry=registry)
    kt_manager = KnowledgeTransferManager()
    consolidator = Consolidator(graph=dynamic_graph, episode_store=episode_store)
    curriculum = CurriculumGenerator(rng_seed=seed)

    # ------------------------------------------------------------------
    # 3. Policy networks
    # ------------------------------------------------------------------
    teacher_policy = PolicyNetwork(state_dim=OBS_DIM, hidden_dim=64, num_actions=NUM_ACTIONS).to(device)

    # ------------------------------------------------------------------
    # 4. Agents
    # ------------------------------------------------------------------
    teacher = Teacher(graph=dynamic_graph, retriever=retriever, policy=teacher_policy, num_agents=num_sub_agents)
    validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)

    gigpo_scorer = GIGPOScorer(feature_dim=64)
    tot_engine = TreeOfThought(max_depth=3, branch_factor=2, scorer=gigpo_scorer)

    sub_agents: List[SubAgent] = [
        SubAgent(
            agent_id=i,
            solver=solver,
            graph=dynamic_graph,
            retriever=retriever,
            tot_engine=tot_engine,
        )
        for i in range(num_sub_agents)
    ]

    # ------------------------------------------------------------------
    # 5. RL trainers
    # ------------------------------------------------------------------
    grpo_trainer = GRPOTrainer(policy=teacher_policy, lr=1e-4, group_size=4, kl_coef=0.1)

    # ------------------------------------------------------------------
    # 6. Environment
    # ------------------------------------------------------------------
    env = MultiAgentEnv(
        graph=dynamic_graph,
        solver=solver,
        verifier=verifier,
        retriever=retriever,
        episode_store=episode_store,
        num_sub_agents=num_sub_agents,
        kt_manager=kt_manager,
    )

    # ------------------------------------------------------------------
    # 7. Problem pool
    # ------------------------------------------------------------------
    seed_problems_path = str(Path(__file__).parent.parent / "data" / "seed_problems.json")
    seed_problems = load_seed_problems(seed_problems_path)
    generated_problems: List[Dict[str, Any]] = []
    logger.info("Loaded %d seed problems", len(seed_problems))

    # ------------------------------------------------------------------
    # 8. Main loop
    # ------------------------------------------------------------------
    deadline = time.time() + duration_minutes * 60
    episode_num = 0
    all_metrics: List[Dict[str, Any]] = []

    grpo_buffer: List[Any] = []
    gigpo_comparisons: List[Any] = []

    while time.time() < deadline:
        episode_num += 1

        # a. Pick problem
        problem = pick_problem(seed_problems, generated_problems, rng)

        # b. Run episode
        try:
            record = env.run_episode(
                teacher=teacher,
                sub_agents=sub_agents,
                validator=validator,
                problem=problem,
            )
        except Exception as exc:
            logger.warning("Episode %d failed: %s", episode_num, exc)
            continue

        passed = record.get("validation", {}).get("passed", False)
        reward = record.get("validation", {}).get("reward_signals", {}).get("global", 0.0)
        logger.info(
            "Episode %d | problem=%s | passed=%s | reward=%.2f | steps=%d",
            episode_num,
            problem.get("id", "?"),
            passed,
            reward,
            record.get("steps", 0),
        )

        # c. Consolidation
        if episode_num % CONSOLIDATE_EVERY == 0:
            consolidator.run()
            logger.info("Consolidation ran after episode %d", episode_num)

        # d. Curriculum generation
        if episode_num % CURRICULUM_EVERY == 0 and seed_problems:
            try:
                variant = curriculum.generate_variant(rng.choice(seed_problems), dynamic_graph)
                generated_problems.append(variant)
            except Exception as exc:
                logger.warning("Curriculum generation failed at episode %d: %s", episode_num, exc)

        # e. Log metrics
        metrics_entry = {
            "episode": episode_num,
            "problem_id": problem.get("id", "?"),
            "passed": passed,
            "reward": reward,
            "graph_nodes": dynamic_graph.num_nodes,
            "graph_edges": dynamic_graph.num_edges,
            "timestamp": time.time(),
        }
        all_metrics.append(metrics_entry)

        if episode_num % 10 == 0:
            log_metrics(metrics_path, all_metrics)
            dynamic_graph.save(graph_path)
            logger.info("Saved graph and metrics after episode %d", episode_num)

    # ------------------------------------------------------------------
    # 9. Save final outputs
    # ------------------------------------------------------------------
    dynamic_graph.save(graph_path)
    log_metrics(metrics_path, all_metrics)
    torch.save(teacher_policy.state_dict(), "teacher_policy.pt")
    logger.info(
        "Done. %d episodes in %.1f minutes. Graph: %d nodes, %d edges.",
        episode_num,
        duration_minutes,
        dynamic_graph.num_nodes,
        dynamic_graph.num_edges,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent self-evolving memory system")
    parser.add_argument("--duration", type=int, default=120, dest="duration_minutes",
                        help="Training duration in minutes (default: 120)")
    parser.add_argument("--num-agents", type=int, default=3, dest="num_sub_agents",
                        help="Number of SubAgents (default: 3)")
    parser.add_argument("--db-path", type=str, default="episodes.db", dest="db_path")
    parser.add_argument("--graph-path", type=str, default="graph.pkl", dest="graph_path")
    parser.add_argument("--metrics-path", type=str, default="metrics.json", dest="metrics_path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", dest="device_str")
    # LLM / model flags
    parser.add_argument(
        "--solver",
        choices=["sympy", "llm"],
        default="llm",
        dest="solver_type",
        help="Solver backend: 'llm' (default) or 'sympy'",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        dest="teacher_model",
        help="HuggingFace model for the Teacher agent",
    )
    parser.add_argument(
        "--subagent-model",
        type=str,
        default="Qwen/Qwen-7B",
        dest="subagent_model",
        help="HuggingFace MoE model for SubAgents",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        dest="verifier_model",
        help="HuggingFace model for the Verifier",
    )
    parser.add_argument(
        "--solver-model",
        type=str,
        default=None,
        dest="solver_model",
        help="HuggingFace model for the Solver (defaults to --subagent-model)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        dest="cpu_only",
        help="Force CPU-only model loading",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        dest="load_in_4bit",
        help="Load models in 4-bit quantisation (requires bitsandbytes)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        dest="lora_rank",
        help="LoRA rank for LLM fine-tuning",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        duration_minutes=args.duration_minutes,
        num_sub_agents=args.num_sub_agents,
        db_path=args.db_path,
        graph_path=args.graph_path,
        metrics_path=args.metrics_path,
        seed=args.seed,
        device_str=args.device_str,
        solver_type=args.solver_type,
        teacher_model=args.teacher_model,
        subagent_model=args.subagent_model,
        verifier_model=args.verifier_model,
        solver_model=args.solver_model,
        cpu_only=args.cpu_only,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
    )
