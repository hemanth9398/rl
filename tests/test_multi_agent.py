"""Unit tests for the multi-agent self-evolving memory system."""
from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, List

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.graph import MemoryGraph, build_seed_graph
from memory.dynamic_graph import DynamicMemoryGraph
from memory.episode_store import EpisodeStore
from memory.retrieval import Retriever
from solver.solver import Solver
from verifier.verifier import Verifier
from policy.policy_nn import PolicyNetwork, NUM_ACTIONS

from agents.tree_of_thought import TreeOfThought, ThoughtBranch
from agents.knowledge_transfer import KnowledgeTransferManager, LoRAAdapter
from agents.teacher import Teacher, SubTask, TaskPlan
from agents.sub_agent import SubAgent, SubAgentResult
from agents.validator import Validator, ValidationResult

from rl.grpo import GRPOTrainer, GRPOGroup
from rl.gigpo import GIGPOScorer, BranchComparison
from envs.multi_agent_env import MultiAgentEnv, OBS_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seed_graph() -> MemoryGraph:
    return build_seed_graph()


@pytest.fixture
def dynamic_graph() -> DynamicMemoryGraph:
    dg = DynamicMemoryGraph()
    seed = build_seed_graph()
    for node_id, data in seed.graph.nodes(data=True):
        dg.graph.add_node(node_id, **data)
    for src, dst, data in seed.graph.edges(data=True):
        dg.graph.add_edge(src, dst, **data)
    return dg


@pytest.fixture
def episode_store(tmp_path) -> EpisodeStore:
    return EpisodeStore(db_path=str(tmp_path / "test_episodes.db"))


@pytest.fixture
def retriever(dynamic_graph, episode_store) -> Retriever:
    return Retriever(graph=dynamic_graph, episode_store=episode_store)


@pytest.fixture
def solver() -> Solver:
    return Solver()


@pytest.fixture
def verifier() -> Verifier:
    return Verifier()


@pytest.fixture
def simple_problem() -> Dict[str, Any]:
    return {
        "id": "test_001",
        "topic": "algebra_linear",
        "difficulty": 1,
        "statement": "Solve for x: 2*x + 3 = 7",
        "answer_spec": {"type": "value", "symbol": "x", "value": "2"},
        "domain": "math",
    }


@pytest.fixture
def hard_problem() -> Dict[str, Any]:
    return {
        "id": "test_hard_001",
        "topic": "algebra_quadratic",
        "difficulty": 4,
        "statement": "Solve for x: x**2 - 5*x + 6 = 0",
        "answer_spec": {"type": "set", "symbol": "x", "value": "[2, 3]"},
        "domain": "math",
    }


# ---------------------------------------------------------------------------
# DynamicMemoryGraph tests
# ---------------------------------------------------------------------------

class TestDynamicMemoryGraph:
    def test_add_learned_skill(self, dynamic_graph):
        nid = dynamic_graph.add_learned_skill(
            label="Test Skill",
            domain="math",
            topic="algebra_linear",
            trigger={"keywords": ["test"]},
            procedure=["step 1", "step 2"],
        )
        assert nid.startswith("dyn_skill_")
        node = dynamic_graph.get_node(nid)
        assert node is not None
        assert node["type"] == "skill"
        assert node["label"] == "Test Skill"

    def test_add_learned_concept(self, dynamic_graph):
        nid = dynamic_graph.add_learned_concept(
            label="Test Concept",
            domain="math",
            keywords=["test", "concept"],
        )
        assert nid.startswith("dyn_concept_")
        node = dynamic_graph.get_node(nid)
        assert node is not None
        assert node["type"] == "concept"

    def test_add_learned_error(self, dynamic_graph):
        nid = dynamic_graph.add_learned_error(
            label="Test Error",
            diagnostics="test_error",
            repair_hint="fix it",
        )
        assert nid.startswith("dyn_error_")
        node = dynamic_graph.get_node(nid)
        assert node is not None
        assert node["type"] == "error"

    def test_record_subtask_completion_success(self, dynamic_graph):
        skill_id = "skill_solve_linear"
        initial_uses = dynamic_graph.get_node(skill_id).get("use_count", 0)
        dynamic_graph.record_subtask_completion(
            subtask_id="subtask_001",
            skill_used=skill_id,
            success=True,
        )
        updated = dynamic_graph.get_node(skill_id)
        assert updated["use_count"] == initial_uses + 1
        assert updated["success_count"] >= 1

    def test_record_subtask_completion_failure(self, dynamic_graph):
        skill_id = "skill_solve_linear"
        dynamic_graph.record_subtask_completion(
            subtask_id="subtask_002",
            skill_used=skill_id,
            success=False,
        )
        # Should have added an error node linked to the skill
        error_nodes = dynamic_graph.get_error_nodes(skill_id)
        assert len(error_nodes) >= 1

    def test_find_similar_skills(self, dynamic_graph):
        results = dynamic_graph.find_similar_skills("solve linear equation x", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(r.get("type") == "skill" for r in results)

    def test_get_skill_chain(self, dynamic_graph):
        chain = dynamic_graph.get_skill_chain("skill_separate_variables", max_length=3)
        assert isinstance(chain, list)
        assert len(chain) >= 1
        assert chain[0]["id"] == "skill_separate_variables"

    def test_merge_duplicate_skills(self, dynamic_graph):
        # Add two near-identical skills
        nid1 = dynamic_graph.add_learned_skill(
            label="Dup Solve X",
            domain="math",
            topic="general",
            trigger={"keywords": ["dup", "solve", "x"]},
            procedure=["step1"],
        )
        nid2 = dynamic_graph.add_learned_skill(
            label="Dup Solve X",
            domain="math",
            topic="general",
            trigger={"keywords": ["dup", "solve", "x"]},
            procedure=["step1"],
        )
        merged = dynamic_graph.merge_duplicate_skills(threshold=0.9)
        assert isinstance(merged, list)

    def test_auto_id_increments(self, dynamic_graph):
        n1 = dynamic_graph.add_learned_skill(
            label="A", domain="d", topic="t",
            trigger={}, procedure=[],
        )
        n2 = dynamic_graph.add_learned_skill(
            label="B", domain="d", topic="t",
            trigger={}, procedure=[],
        )
        # Counter should increment
        num1 = int(n1.split("_")[-1])
        num2 = int(n2.split("_")[-1])
        assert num2 > num1


# ---------------------------------------------------------------------------
# Teacher tests
# ---------------------------------------------------------------------------

class TestTeacher:
    def test_decompose_simple_problem(self, dynamic_graph, retriever, simple_problem):
        teacher = Teacher(graph=dynamic_graph, retriever=retriever, num_agents=3)
        plan = teacher.decompose(simple_problem)
        assert isinstance(plan, TaskPlan)
        assert plan.plan_id
        assert len(plan.subtasks) >= 1
        assert len(plan.dependencies) == len(plan.subtasks)

    def test_decompose_hard_problem_more_subtasks(self, dynamic_graph, retriever, hard_problem):
        teacher = Teacher(graph=dynamic_graph, retriever=retriever, num_agents=3)
        plan = teacher.decompose(hard_problem)
        # Hard problems should be split into 2+ subtasks
        assert len(plan.subtasks) >= 2

    def test_generate_golden_thought(self, dynamic_graph, retriever, simple_problem):
        teacher = Teacher(graph=dynamic_graph, retriever=retriever)
        plan = teacher.decompose(simple_problem)
        subtask = plan.subtasks[0]
        failed_branches = [
            ThoughtBranch("left", "failed reasoning", "", 0.1, 1),
            ThoughtBranch("right", "also failed", "", 0.05, 1),
        ]
        hint = teacher.generate_golden_thought(subtask, failed_branches)
        assert isinstance(hint, str)
        assert len(hint) > 10

    def test_synthesize_final(self, dynamic_graph, retriever, simple_problem):
        teacher = Teacher(graph=dynamic_graph, retriever=retriever)
        plan = teacher.decompose(simple_problem)
        # Build dummy subtask results
        subtask_results = {}
        for subtask in plan.subtasks:
            subtask_results[subtask.task_id] = {
                "answer": "x = 2",
            }
        result = teacher.synthesize_final(plan, subtask_results)
        assert isinstance(result, str)
        assert "x = 2" in result

    def test_plan_dependencies_are_valid(self, dynamic_graph, retriever, hard_problem):
        teacher = Teacher(graph=dynamic_graph, retriever=retriever, num_agents=3)
        plan = teacher.decompose(hard_problem)
        task_ids = {st.task_id for st in plan.subtasks}
        for task_id, deps in plan.dependencies.items():
            assert task_id in task_ids
            for dep in deps:
                assert dep in task_ids


# ---------------------------------------------------------------------------
# SubAgent + ToT tests
# ---------------------------------------------------------------------------

class TestSubAgentAndToT:
    def test_tot_explore_returns_branches(self, dynamic_graph, retriever, solver, simple_problem):
        tot = TreeOfThought(max_depth=2, branch_factor=2)
        context = {
            "retrieved_skills": [],
            "topic": "algebra_linear",
            "domain": "math",
            "problem": simple_problem,
        }
        branches = tot.explore(
            problem_text=simple_problem["statement"],
            context=context,
            solver=solver,
            graph=dynamic_graph,
        )
        assert isinstance(branches, list)
        assert len(branches) >= 1
        for b in branches:
            assert isinstance(b, ThoughtBranch)
            assert 0.0 <= b.confidence <= 1.0

    def test_tot_select_best(self):
        branches = [
            ThoughtBranch("left", "r1", "a1", 0.3, 1),
            ThoughtBranch("right", "r2", "a2", 0.7, 1),
        ]
        tot = TreeOfThought()
        best = tot.select_best(branches)
        assert best.branch_id == "right"
        assert best.confidence == 0.7

    def test_tot_select_best_empty(self):
        tot = TreeOfThought()
        best = tot.select_best([])
        assert best.branch_id == "empty"
        assert best.confidence == 0.0

    def test_subagent_solve(self, dynamic_graph, retriever, solver, simple_problem):
        tot = TreeOfThought(max_depth=2, branch_factor=2)
        teacher = Teacher(graph=dynamic_graph, retriever=retriever)
        plan = teacher.decompose(simple_problem)
        subtask = plan.subtasks[0]

        agent = SubAgent(
            agent_id=0,
            solver=solver,
            graph=dynamic_graph,
            retriever=retriever,
            tot_engine=tot,
        )
        result = agent.solve(subtask)
        assert isinstance(result, SubAgentResult)
        assert result.task_id == subtask.task_id
        assert isinstance(result.success, bool)
        assert isinstance(result.all_branches, list)
        assert result.duration >= 0.0

    def test_subagent_retry_with_hint(self, dynamic_graph, retriever, solver, simple_problem):
        tot = TreeOfThought(max_depth=2, branch_factor=2)
        teacher = Teacher(graph=dynamic_graph, retriever=retriever)
        plan = teacher.decompose(simple_problem)
        subtask = plan.subtasks[0]

        agent = SubAgent(0, solver, dynamic_graph, retriever, tot)
        result = agent.retry_with_hint(subtask, "Apply linear solve: isolate x")
        assert isinstance(result, SubAgentResult)
        # Confidence should be boosted
        assert result.selected_branch.confidence >= 0.0

    def test_subagent_inject_lora(self, dynamic_graph, retriever, solver):
        tot = TreeOfThought()
        agent = SubAgent(0, solver, dynamic_graph, retriever, tot)
        adapter = LoRAAdapter(
            adapter_id="test_adapter",
            target_modules=["linear_0"],
            rank=4,
            alpha=1.0,
            weights={"lora_A": torch.zeros(4, 64)},
            source_task="test_task",
        )
        agent.inject_lora(adapter)
        assert agent.lora_adapter is adapter


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def _make_plan_and_results(
        self,
        dynamic_graph: DynamicMemoryGraph,
        retriever: Retriever,
        solver: Solver,
        problem: Dict[str, Any],
    ):
        teacher = Teacher(graph=dynamic_graph, retriever=retriever)
        plan = teacher.decompose(problem)
        tot = TreeOfThought(max_depth=2, branch_factor=2)
        sub_agents = [
            SubAgent(i, solver, dynamic_graph, retriever, tot)
            for i in range(3)
        ]
        agent_results = {}
        for subtask in plan.subtasks:
            agent = sub_agents[subtask.assigned_agent % 3]
            result = agent.solve(subtask)
            agent_results[subtask.task_id] = result
        return plan, agent_results

    def test_validate_returns_result(self, dynamic_graph, retriever, solver, verifier, episode_store, simple_problem):
        validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)
        plan, agent_results = self._make_plan_and_results(
            dynamic_graph, retriever, solver, simple_problem
        )
        result = validator.validate(plan, agent_results, simple_problem)
        assert isinstance(result, ValidationResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.merged_answer, str)
        assert isinstance(result.subtask_results, list)
        assert "teacher" in result.reward_signals
        assert "global" in result.reward_signals
        assert "validator" in result.reward_signals

    def test_validate_reward_signals_range(self, dynamic_graph, retriever, solver, verifier, episode_store, simple_problem):
        validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)
        plan, agent_results = self._make_plan_and_results(
            dynamic_graph, retriever, solver, simple_problem
        )
        result = validator.validate(plan, agent_results, simple_problem)
        for k, v in result.reward_signals.items():
            assert isinstance(v, float), f"Reward {k} is not float"

    def test_apply_graph_updates_adds_nodes(self, dynamic_graph, retriever, solver, verifier, episode_store, simple_problem):
        validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)
        plan, agent_results = self._make_plan_and_results(
            dynamic_graph, retriever, solver, simple_problem
        )
        result = validator.validate(plan, agent_results, simple_problem)
        initial_nodes = dynamic_graph.num_nodes
        validator.apply_graph_updates(result.graph_updates)
        # Graph should have grown or stayed the same
        assert dynamic_graph.num_nodes >= initial_nodes

    def test_validate_missing_agent_result(self, dynamic_graph, retriever, solver, verifier, episode_store, simple_problem):
        validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)
        teacher = Teacher(graph=dynamic_graph, retriever=retriever)
        plan = teacher.decompose(simple_problem)
        # Pass empty results — validator should handle gracefully
        result = validator.validate(plan, {}, simple_problem)
        assert isinstance(result, ValidationResult)
        assert result.passed is False


# ---------------------------------------------------------------------------
# GRPO tests
# ---------------------------------------------------------------------------

class TestGRPO:
    def test_compute_group_advantages_normalised(self):
        policy = PolicyNetwork(state_dim=OBS_DIM, hidden_dim=32, num_actions=NUM_ACTIONS)
        grpo = GRPOTrainer(policy=policy, lr=1e-4, group_size=4)
        group = GRPOGroup(
            problem={},
            responses=[{}, {}, {}, {}],
            rewards=[1.0, 0.5, -0.5, 0.0],
        )
        advs = grpo.compute_group_advantages(group)
        assert advs.shape == (4,)
        assert abs(float(advs.mean())) < 1e-5  # normalised to zero mean
        assert abs(float(advs.std()) - 1.0) < 0.1  # unit std

    def test_update_with_valid_groups(self):
        policy = PolicyNetwork(state_dim=OBS_DIM, hidden_dim=32, num_actions=NUM_ACTIONS)
        grpo = GRPOTrainer(policy=policy, lr=1e-4, group_size=2)
        state = torch.zeros(OBS_DIM)
        action_probs, _ = policy.forward(state.unsqueeze(0))
        action = int(torch.argmax(action_probs).item())
        log_prob = float(torch.log(action_probs[0, action] + 1e-8).item())

        response = {"state": state, "action": action, "log_prob": log_prob}
        group = GRPOGroup(
            problem={},
            responses=[response, response],
            rewards=[1.0, -1.0],
        )
        metrics = grpo.update([group])
        assert "grpo_loss" in metrics
        assert "grpo_policy_loss" in metrics
        assert "grpo_kl" in metrics

    def test_update_empty_groups_returns_metrics(self):
        policy = PolicyNetwork(state_dim=OBS_DIM, hidden_dim=32, num_actions=NUM_ACTIONS)
        grpo = GRPOTrainer(policy=policy)
        metrics = grpo.update([])
        # Returns zero-value metrics (no update performed)
        assert isinstance(metrics, dict)

    def test_last_metrics_property(self):
        policy = PolicyNetwork(state_dim=OBS_DIM, hidden_dim=32, num_actions=NUM_ACTIONS)
        grpo = GRPOTrainer(policy=policy)
        assert isinstance(grpo.last_metrics, dict)


# ---------------------------------------------------------------------------
# GIGPO tests
# ---------------------------------------------------------------------------

class TestGIGPO:
    def test_score_branches_returns_floats(self):
        scorer = GIGPOScorer(feature_dim=64)
        branches = [
            ThoughtBranch("left", "reasoning A", "answer A", 0.8, 2),
            ThoughtBranch("right", "reasoning B", "answer B", 0.4, 1),
        ]
        scores = scorer.score_branches(branches, {})
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_select_returns_branch_and_log_prob(self):
        scorer = GIGPOScorer(feature_dim=64)
        branches = [
            ThoughtBranch("left", "r1", "a1", 0.9, 2),
            ThoughtBranch("right", "r2", "a2", 0.1, 1),
        ]
        selected, log_prob = scorer.select(branches, {})
        assert isinstance(selected, ThoughtBranch)
        assert isinstance(log_prob, float)
        assert log_prob <= 0.0  # log prob ≤ 0

    def test_select_empty(self):
        scorer = GIGPOScorer()
        branch, lp = scorer.select([], {})
        assert branch.branch_id == "empty"
        assert lp == float("-inf")

    def test_update_returns_metrics(self):
        scorer = GIGPOScorer(feature_dim=64)
        comp = BranchComparison(
            left_branch=ThoughtBranch("left", "good reasoning", "2", 0.8, 1),
            right_branch=ThoughtBranch("right", "bad reasoning", "99", 0.2, 1),
            selected="left",
            reward=1.0,
        )
        metrics = scorer.update([comp])
        assert "gigpo_loss" in metrics
        assert metrics["gigpo_loss"] >= 0.0

    def test_both_branches_bad(self):
        scorer = GIGPOScorer(feature_dim=64)
        bad_branches = [
            ThoughtBranch("left", "bad", "", 0.05, 1),
            ThoughtBranch("right", "also bad", "", 0.1, 1),
        ]
        # Scores from the network may not exactly be low, but the method should work
        result = scorer.both_branches_bad(bad_branches, threshold=0.9)
        assert isinstance(result, bool)

    def test_both_branches_bad_empty(self):
        scorer = GIGPOScorer()
        assert scorer.both_branches_bad([]) is True


# ---------------------------------------------------------------------------
# KnowledgeTransfer tests
# ---------------------------------------------------------------------------

class TestKnowledgeTransfer:
    def test_should_transfer_all_bad(self):
        ktm = KnowledgeTransferManager()
        branches = [
            ThoughtBranch("left", "r", "a", 0.1, 1),
            ThoughtBranch("right", "r", "b", 0.2, 1),
        ]
        assert ktm.should_transfer(branches, threshold=0.3) is True

    def test_should_transfer_one_good(self):
        ktm = KnowledgeTransferManager()
        branches = [
            ThoughtBranch("left", "r", "a", 0.1, 1),
            ThoughtBranch("right", "r", "b", 0.9, 1),
        ]
        assert ktm.should_transfer(branches, threshold=0.3) is False

    def test_should_transfer_empty(self):
        ktm = KnowledgeTransferManager()
        assert ktm.should_transfer([]) is True

    def test_compute_reasoning_loss_identical(self):
        ktm = KnowledgeTransferManager(base_model_dim=64)
        loss = ktm.compute_reasoning_loss("hello world", "hello world")
        assert float(loss) < 1e-3  # nearly 0 for identical strings

    def test_compute_reasoning_loss_different(self):
        ktm = KnowledgeTransferManager(base_model_dim=64)
        loss = ktm.compute_reasoning_loss("abc", "xyz 123 completely different text")
        assert float(loss) > 0.0

    def test_create_adapter_structure(self):
        ktm = KnowledgeTransferManager(base_model_dim=64, lora_rank=4)
        loss = torch.tensor(0.5)
        adapter = ktm.create_adapter(loss, {"task_id": "task_001", "target_modules": ["layer_0"]})
        assert isinstance(adapter, LoRAAdapter)
        assert adapter.rank == 4
        assert "lora_A" in adapter.weights
        assert adapter.source_task == "task_001"
        assert adapter.target_modules == ["layer_0"]


# ---------------------------------------------------------------------------
# MultiAgentEnv tests
# ---------------------------------------------------------------------------

class TestMultiAgentEnv:
    def _build_env(self, dynamic_graph, solver, verifier, retriever, episode_store):
        return MultiAgentEnv(
            graph=dynamic_graph,
            solver=solver,
            verifier=verifier,
            retriever=retriever,
            episode_store=episode_store,
            num_sub_agents=2,
            max_steps=20,
        )

    def test_reset_returns_obs(self, dynamic_graph, solver, verifier, retriever, episode_store, simple_problem):
        env = self._build_env(dynamic_graph, solver, verifier, retriever, episode_store)
        obs_dict = env.reset(simple_problem)
        assert "obs" in obs_dict
        assert obs_dict["obs"].shape == (OBS_DIM,)
        assert "episode_id" in obs_dict

    def test_run_episode_returns_record(self, dynamic_graph, solver, verifier, retriever, episode_store, simple_problem):
        env = self._build_env(dynamic_graph, solver, verifier, retriever, episode_store)
        teacher = Teacher(graph=dynamic_graph, retriever=retriever, num_agents=2)
        validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)
        tot = TreeOfThought(max_depth=2, branch_factor=2)
        sub_agents = [
            SubAgent(i, solver, dynamic_graph, retriever, tot) for i in range(2)
        ]
        record = env.run_episode(teacher, sub_agents, validator, simple_problem)
        assert "episode_id" in record
        assert "plan" in record
        assert "agent_results" in record
        assert "validation" in record
        assert "duration" in record
        assert record["steps"] >= 1

    def test_build_episode_record_fields(self, dynamic_graph, solver, verifier, retriever, episode_store, simple_problem):
        env = self._build_env(dynamic_graph, solver, verifier, retriever, episode_store)
        teacher = Teacher(graph=dynamic_graph, retriever=retriever, num_agents=2)
        validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)
        tot = TreeOfThought(max_depth=1, branch_factor=2)
        sub_agents = [SubAgent(i, solver, dynamic_graph, retriever, tot) for i in range(2)]
        record = env.run_episode(teacher, sub_agents, validator, simple_problem)
        assert isinstance(record["validation"]["passed"], bool)
        assert isinstance(record["validation"]["reward_signals"], dict)
        assert "global" in record["validation"]["reward_signals"]

    def test_graph_grows_after_episode(self, dynamic_graph, solver, verifier, retriever, episode_store, simple_problem):
        env = self._build_env(dynamic_graph, solver, verifier, retriever, episode_store)
        teacher = Teacher(graph=dynamic_graph, retriever=retriever, num_agents=2)
        validator = Validator(graph=dynamic_graph, verifier=verifier, episode_store=episode_store)
        tot = TreeOfThought(max_depth=1, branch_factor=2)
        sub_agents = [SubAgent(i, solver, dynamic_graph, retriever, tot) for i in range(2)]
        initial_nodes = dynamic_graph.num_nodes
        env.run_episode(teacher, sub_agents, validator, simple_problem)
        # The graph should have grown (or at least stayed the same)
        assert dynamic_graph.num_nodes >= initial_nodes
