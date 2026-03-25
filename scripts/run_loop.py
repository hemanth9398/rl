#!/usr/bin/env python
"""
Main entry point for the self-evolving memory agent.

Usage:
    python scripts/run_loop.py --duration 60 --db-path episodes.db --graph-path graph.pkl
"""
import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

# Add parent dir to path so imports work when run from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.graph import MemoryGraph, build_seed_graph
from memory.episode_store import EpisodeStore
from memory.retrieval import Retriever
from memory.consolidation import Consolidator
from solver.solver import Solver
from solver.llm_backend import backend_name as _solver_backend_name
from verifier.verifier import Verifier
from policy.policy_nn import PolicyNetwork, NUM_ACTIONS
from rl.ppo import PPOTrainer
from envs.math_env import MathREPLEnv, STATE_DIM
from curriculum.generator import CurriculumGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_loop")
logger.info("Solver backend: %s", _solver_backend_name())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seed_problems(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def pick_problem(
    problems: List[Dict[str, Any]],
    generated: List[Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    pool = problems + generated
    return rng.choice(pool)


def run_episode(
    env: MathREPLEnv,
    policy: PolicyNetwork,
    ppo: PPOTrainer,
    problem: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Run one episode and collect experience."""
    obs = env.reset(problem)
    total_reward = 0.0
    done = False

    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action, log_prob, value = policy.select_action(state_tensor)
        obs_next, reward, done, info = env.step(action)
        total_reward += reward
        ppo.store_transition(
            state_tensor, action, reward, log_prob, value, done
        )
        obs = obs_next

    episode_record = env.build_episode_record(episode_id=str(uuid.uuid4()))
    episode_record["total_reward"] = total_reward
    return episode_record


def update_graph_stats(
    graph: MemoryGraph, episode_record: Dict[str, Any]
) -> None:
    """Update skill node stats in the graph based on episode outcome."""
    verified = episode_record.get("verified", False)
    skills = episode_record.get("skills_used", [])
    for skill_id in skills:
        graph.update_node_stats(
            skill_id,
            success=verified,
            cost=episode_record.get("duration_seconds", 0.0),
        )
    # Update transition edges
    for i in range(len(skills) - 1):
        delta = 0.1 if verified else -0.05
        graph.update_edge_weight(skills[i], skills[i + 1], delta)


def evaluate_held_out(
    env: MathREPLEnv,
    policy: PolicyNetwork,
    held_out: List[Dict[str, Any]],
    device: torch.device,
    n_eval: int = 10,
) -> Dict[str, float]:
    """Evaluate policy on a held-out set (no PPO updates)."""
    if not held_out:
        return {"eval_accuracy": 0.0, "eval_avg_steps": 0.0}
    subset = held_out[:n_eval]
    successes = 0
    total_steps = 0
    for problem in subset:
        obs = env.reset(problem)
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            state_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action_probs, _ = policy(state_tensor.unsqueeze(0))
            action = int(action_probs.squeeze(0).argmax().item())
            obs, _, done, info = env.step(action)
            steps += 1
        record = env.build_episode_record()
        if record.get("verified", False):
            successes += 1
        total_steps += steps
    n = len(subset)
    return {
        "eval_accuracy": successes / n,
        "eval_avg_steps": total_steps / n,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(
    duration_minutes: int = 120,
    log_interval: int = 30,
    db_path: str = "episodes.db",
    graph_path: str = "graph.pkl",
    policy_path: str = "policy.pt",
    metrics_path: str = "metrics.json",
    seed: int = 42,
    device_str: str = "cpu",
) -> None:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    # ── Load seed problems ──────────────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data"
    problems = load_seed_problems(str(data_dir / "seed_problems.json"))
    rng.shuffle(problems)
    held_out = problems[-10:]
    train_problems = problems[:-10]
    logger.info("Loaded %d seed problems (%d train, %d held-out)", len(problems), len(train_problems), len(held_out))

    # ── Initialise components ───────────────────────────────────────────
    graph = build_seed_graph()
    episode_store = EpisodeStore(db_path)
    retriever = Retriever(graph, episode_store)
    solver = Solver()
    verifier = Verifier()
    curriculum = CurriculumGenerator(rng_seed=seed)
    consolidator = Consolidator(graph, episode_store)

    policy = PolicyNetwork(state_dim=STATE_DIM, hidden_dim=64, num_actions=NUM_ACTIONS).to(device)
    ppo = PPOTrainer(policy, lr=3e-4, update_every=10, epochs_per_update=4)

    env = MathREPLEnv(graph, solver, verifier, retriever, episode_store)

    # ── Metrics store ───────────────────────────────────────────────────
    metrics_history = []
    episode_num = 0
    generated_problems: List[Dict[str, Any]] = []

    start_time = time.time()
    deadline = start_time + duration_minutes * 60
    last_log_time = start_time

    logger.info("Starting run loop for %d minutes", duration_minutes)

    try:
        while time.time() < deadline:
            # ── Pick problem ──────────────────────────────────────────────
            problem = pick_problem(train_problems, generated_problems, rng)

            # ── Run episode ───────────────────────────────────────────────
            episode_record = run_episode(env, policy, ppo, problem, device)
            episode_num += 1

            # ── Store episode ─────────────────────────────────────────────
            episode_store.store_episode(episode_record)

            # ── Update graph stats ────────────────────────────────────────
            update_graph_stats(graph, episode_record)

            # ── Notify consolidator ───────────────────────────────────────
            consolidator.notify_episode(episode_record)

            # ── PPO update (every 10 episodes) ────────────────────────────
            updated = ppo.notify_episode_end()

            # ── Logging ───────────────────────────────────────────────────
            now = time.time()
            if now - last_log_time >= log_interval or episode_num == 1:
                eval_metrics = evaluate_held_out(env, policy, held_out, device)
                stats = episode_store.get_stats()
                ppo_metrics = ppo.last_metrics
                log_entry = {
                    "wall_time": now - start_time,
                    "episode": episode_num,
                    "graph_nodes": graph.num_nodes,
                    "graph_edges": graph.num_edges,
                    "verified_accuracy": eval_metrics["eval_accuracy"],
                    "eval_avg_steps": eval_metrics["eval_avg_steps"],
                    "overall_success_rate": stats["success_rate"],
                    "total_episodes": stats["total_episodes"],
                    "curriculum_generated": len(generated_problems),
                    **ppo_metrics,
                }
                metrics_history.append(log_entry)
                logger.info(
                    "Ep %d | accuracy=%.2f | success_rate=%.2f | "
                    "graph=%d nodes/%d edges | curriculum=%d",
                    episode_num,
                    eval_metrics["eval_accuracy"],
                    stats["success_rate"],
                    graph.num_nodes,
                    graph.num_edges,
                    len(generated_problems),
                )
                last_log_time = now

            # ── Curriculum generation (every 5 episodes) ─────────────────
            if episode_num % 5 == 0:
                weak_skills = curriculum.get_weak_skills(graph, min_uses=2)
                if weak_skills:
                    new_prob = curriculum.generate_for_weak_skill(
                        weak_skills[0], train_problems
                    )
                    if new_prob:
                        generated_problems.append(new_prob)
                else:
                    base = rng.choice(train_problems)
                    new_prob = curriculum.generate_variant(base, graph)
                    generated_problems.append(new_prob)
                # Keep generated pool bounded
                if len(generated_problems) > 200:
                    generated_problems = generated_problems[-200:]

    except KeyboardInterrupt:
        logger.info("Interrupted by user after %d episodes", episode_num)

    # ── Save outputs ────────────────────────────────────────────────────
    logger.info("Saving graph to %s", graph_path)
    graph.save(graph_path)

    logger.info("Saving policy to %s", policy_path)
    torch.save(policy.state_dict(), policy_path)

    logger.info("Saving metrics to %s", metrics_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

    episode_store.close()

    # ── Print summary ────────────────────────────────────────────────────
    final_stats = metrics_history[-1] if metrics_history else {}
    print("\n" + "=" * 60)
    print("Run Complete")
    print("=" * 60)
    print(f"  Episodes:           {episode_num}")
    print(f"  Duration (min):     {(time.time() - start_time) / 60:.1f}")
    print(f"  Graph nodes/edges:  {graph.num_nodes} / {graph.num_edges}")
    print(f"  Verified accuracy:  {final_stats.get('verified_accuracy', 0):.2%}")
    print(f"  Curriculum probs:   {len(generated_problems)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Evolving Memory Agent")
    parser.add_argument("--duration", type=int, default=120, help="Duration in minutes")
    parser.add_argument("--log-interval", type=int, default=30, help="Log interval in seconds")
    parser.add_argument("--db-path", default="episodes.db")
    parser.add_argument("--graph-path", default="graph.pkl")
    parser.add_argument("--policy-path", default="policy.pt")
    parser.add_argument("--metrics-path", default="metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    main(
        duration_minutes=args.duration,
        log_interval=args.log_interval,
        db_path=args.db_path,
        graph_path=args.graph_path,
        policy_path=args.policy_path,
        metrics_path=args.metrics_path,
        seed=args.seed,
        device_str=args.device,
    )
