"""Group Relative Policy Optimization (GRPO) for Teacher and Global policies."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from policy.policy_nn import PolicyNetwork


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
