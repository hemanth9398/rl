"""Group Relative Policy Optimization (GRPO) for Teacher and Global policies."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from policy.policy_nn import PolicyNetwork

logger = logging.getLogger(__name__)


@dataclass
class GRPOGroup:
    """A group of responses generated for the same problem."""

    problem: Dict[str, Any]
    responses: List[Dict[str, Any]]  # different plans/answers for same problem
    rewards: List[float]


class GRPOTrainer:
    """Group Relative Policy Optimization for Teacher and Global policies.

    Key idea: for the same problem, generate multiple responses (task plans),
    compute relative advantages within the group (mean-normalised), then update
    the policy using a clipped surrogate objective + KL penalty.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 1e-4,
        group_size: int = 4,
        kl_coef: float = 0.1,
        clip_epsilon: float = 0.2,
    ) -> None:
        self.policy = policy
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self._metrics: Dict[str, float] = {}

        # Reference policy (frozen copy) for KL penalty
        self._ref_policy: PolicyNetwork = copy.deepcopy(policy)
        for p in self._ref_policy.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_group(
        self,
        problem: Dict[str, Any],
        policy: PolicyNetwork,
        env_fn: Callable,
        group_size: Optional[int] = None,
    ) -> GRPOGroup:
        """Generate multiple responses for the same problem.

        ``env_fn`` is a callable that takes a policy and a problem dict and
        returns ``(response_dict, reward_float)``.  We call it ``group_size``
        times to build a comparison group.
        """
        n = group_size or self.group_size
        responses: List[Dict[str, Any]] = []
        rewards: List[float] = []
        for _ in range(n):
            resp, rew = env_fn(policy, problem)
            responses.append(resp)
            rewards.append(float(rew))
        return GRPOGroup(problem=problem, responses=responses, rewards=rewards)

    def compute_group_advantages(self, group: GRPOGroup) -> torch.Tensor:
        """Compute advantages relative to group mean.

        advantage_i = (reward_i − mean(rewards)) / (std(rewards) + eps)
        """
        rewards = torch.tensor(group.rewards, dtype=torch.float32)
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        return (rewards - mean) / std

    def update(self, groups: List[GRPOGroup]) -> Dict[str, float]:
        """Run GRPO update over collected groups.

        For each response in each group:
        1. Compute new log prob under current policy.
        2. Compute ratio to old log prob.
        3. Apply clipped surrogate with group-relative advantage.
        4. Add KL penalty to reference policy.
        """
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        n_updates = 0

        for group in groups:
            advantages = self.compute_group_advantages(group)

            for i, response in enumerate(group.responses):
                state = response.get("state")
                action = response.get("action")
                old_log_prob = response.get("log_prob")

                if state is None or action is None or old_log_prob is None:
                    continue

                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.long)
                if not isinstance(old_log_prob, torch.Tensor):
                    old_log_prob = torch.tensor(old_log_prob, dtype=torch.float32)

                # New log prob under current policy
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    state.unsqueeze(0), action.unsqueeze(0)
                )
                new_log_prob = new_log_probs[0]

                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                adv = advantages[i]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2)

                # KL penalty vs reference policy
                ref_log_probs, _, _ = self._ref_policy.evaluate_actions(
                    state.unsqueeze(0), action.unsqueeze(0)
                )
                kl = (ref_log_probs[0].detach() - new_log_prob).mean()

                loss = policy_loss + self.kl_coef * kl

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_kl += kl.item()
                n_updates += 1

        n = max(n_updates, 1)
        self._metrics = {
            "grpo_loss": total_loss / n,
            "grpo_policy_loss": total_policy_loss / n,
            "grpo_kl": total_kl / n,
        }
        return self._metrics

    @property
    def last_metrics(self) -> Dict[str, float]:
        """Return metrics from last update."""
        return self._metrics


# ---------------------------------------------------------------------------
# LLM GRPO Trainer
# ---------------------------------------------------------------------------

class LLMGRPOTrainer:
    """GRPO trainer that fine-tunes the Teacher's language model via LoRA.

    Generates multiple task-decomposition responses from the Teacher's LLM for
    the same problem, computes group-relative advantages, and updates the
    Teacher model weights via LoRA.

    The :class:`PolicyNetwork` (MLP) is kept for environment action selection.

    Parameters
    ----------
    policy:
        MLP policy network (action selector).
    registry:
        :class:`~models.model_registry.ModelRegistry` instance.
    group_size:
        Number of responses to generate per problem for the group.
    lr_llm:
        Learning rate for LoRA parameters.
    lora_rank:
        LoRA rank.
    kl_coef:
        KL penalty coefficient (vs reference / frozen model snapshot).
    clip_epsilon:
        PPO clipping parameter for the surrogate objective.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        registry: Optional[Any] = None,
        group_size: int = 4,
        lr_llm: float = 1e-4,
        lora_rank: int = 16,
        kl_coef: float = 0.1,
        clip_epsilon: float = 0.2,
    ) -> None:
        self.mlp_trainer = GRPOTrainer(
            policy, group_size=group_size, kl_coef=kl_coef, clip_epsilon=clip_epsilon
        )
        self.registry = registry
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self._lora_model: Optional[nn.Module] = None
        self._lora_optimizer: Optional[torch.optim.Optimizer] = None
        self._lora_rank = lora_rank
        self._lr_llm = lr_llm
        self._metrics: Dict[str, float] = {}

        if registry is not None:
            self._attach_lora()

    def _attach_lora(self) -> None:
        try:
            from models.lora_utils import attach_lora, get_lora_optimizer, is_peft_available

            if not is_peft_available():
                return

            model, _ = self.registry.get_raw_model("teacher")
            if model is None:
                return

            self._lora_model = attach_lora(model, rank=self._lora_rank)
            self._lora_optimizer = get_lora_optimizer(self._lora_model, lr=self._lr_llm)
            logger.info("LLMGRPOTrainer: LoRA attached for teacher")
        except Exception as exc:
            logger.warning("LLMGRPOTrainer: LoRA attachment failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_group(
        self,
        problem: Dict[str, Any],
        policy: PolicyNetwork,
        env_fn: Callable,
        group_size: Optional[int] = None,
    ) -> GRPOGroup:
        """Generate multiple responses for the same problem (MLP path)."""
        return self.mlp_trainer.collect_group(problem, policy, env_fn, group_size)

    def compute_group_advantages(self, group: GRPOGroup) -> torch.Tensor:
        return self.mlp_trainer.compute_group_advantages(group)

    def update(self, groups: List[GRPOGroup]) -> Dict[str, float]:
        """MLP GRPO update."""
        return self.mlp_trainer.update(groups)

    def update_teacher_llm(
        self,
        problems: List[Dict[str, Any]],
        completions_per_problem: List[List[str]],
        rewards_per_problem: List[List[float]],
    ) -> Dict[str, float]:
        """Fine-tune the Teacher LLM with group-relative advantages.

        For each problem a *group* of decomposition completions is generated.
        Group-relative advantages normalise rewards within each group.

        Parameters
        ----------
        problems:
            List of problem dicts.
        completions_per_problem:
            For each problem, a list of Teacher LLM completions (decompositions).
        rewards_per_problem:
            For each problem, corresponding scalar rewards.

        Returns
        -------
        dict with ``llm_grpo_loss``.
        """
        if self._lora_model is None or self._lora_optimizer is None:
            return {}

        try:
            from models.prompts import TEACHER_DECOMPOSE_PROMPT
            from models.lora_utils import is_peft_available

            _, tokenizer = self.registry.get_raw_model("teacher")
            if tokenizer is None:
                return {}

            device = next(self._lora_model.parameters()).device
            total_loss = 0.0
            n = 0

            for problem, completions, rewards in zip(
                problems, completions_per_problem, rewards_per_problem
            ):
                if not completions:
                    continue

                statement = problem.get("statement", str(problem))
                prompt = TEACHER_DECOMPOSE_PROMPT.format(problem=statement)

                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                mean_r = rewards_t.mean()
                std_r = rewards_t.std() + 1e-8
                advantages = (rewards_t - mean_r) / std_r

                for completion, adv in zip(completions, advantages.tolist()):
                    full_text = prompt + completion
                    full_enc = tokenizer(full_text, return_tensors="pt")
                    prompt_enc = tokenizer(prompt, return_tensors="pt")
                    prompt_len = prompt_enc["input_ids"].shape[-1]

                    input_ids = full_enc["input_ids"].to(device)
                    if input_ids.shape[-1] <= prompt_len:
                        continue

                    outputs = self._lora_model(input_ids)
                    logits = outputs.logits

                    shift_logits = logits[0, prompt_len - 1 : -1, :]
                    shift_labels = input_ids[0, prompt_len:]
                    if shift_labels.numel() == 0:
                        continue

                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    token_lps = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                    mean_lp = token_lps.mean()

                    loss = -float(adv) * mean_lp
                    self._lora_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._lora_model.parameters(), 1.0)
                    self._lora_optimizer.step()

                    total_loss += loss.item()
                    n += 1

            self._metrics["llm_grpo_loss"] = total_loss / max(n, 1)
        except Exception as exc:
            logger.warning("LLMGRPOTrainer.update_teacher_llm failed: %s", exc)

        return self._metrics

    @property
    def last_metrics(self) -> Dict[str, float]:
        metrics = dict(self.mlp_trainer.last_metrics)
        metrics.update(self._metrics)
        return metrics
