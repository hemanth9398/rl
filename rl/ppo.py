"""Simple PPO trainer for the policy network."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from policy.policy_nn import PolicyNetwork


@dataclass
class Trajectory:
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


class PPOTrainer:
    """Simple PPO with GAE advantages."""

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_every: int = 10,
        epochs_per_update: int = 4,
        batch_size: int = 64,
    ) -> None:
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_every = update_every
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self._episode_count = 0
        self.buffer: Trajectory = Trajectory()
        self._metrics: Dict[str, float] = {}

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.add(state, action, reward, log_prob, value, done)

    def notify_episode_end(self) -> bool:
        """Call after each episode. Returns True if update was performed."""
        self._episode_count += 1
        if self._episode_count % self.update_every == 0:
            self.update()
            return True
        return False

    def update(self) -> Dict[str, float]:
        """Run PPO update on buffered transitions."""
        if len(self.buffer.states) < 2:
            return {}
        states = torch.stack(self.buffer.states)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        old_values = torch.stack(self.buffer.values).detach().squeeze(-1)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32)

        # Compute GAE advantages
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) < 2:
                    continue

                b_states = states[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]

                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    b_states, b_actions
                )

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, b_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                n_updates += 1

        self.buffer.clear()
        n = max(n_updates, 1)
        self._metrics = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }
        return self._metrics

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalised Advantage Estimates.

        Args:
            rewards: Per-step rewards tensor.
            values:  Critic value estimates for each step.
            dones:   Done flags (1.0 = episode ended).
            last_value: Bootstrap value for the step after the last one.
                        Pass 0.0 for a terminal step, or the critic's estimate
                        at the next state for a truncated episode.
        """
        n = len(rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)
        last_adv = 0.0

        for t in reversed(range(n)):
            # For the final step, bootstrap from last_value if not done
            if t == n - 1:
                next_val = last_value * (1.0 - float(dones[t]))
            else:
                next_val = float(values[t + 1]) * (1.0 - float(dones[t]))
            delta = float(rewards[t]) + self.gamma * next_val - float(values[t])
            last_adv = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * last_adv
            advantages[t] = last_adv
            returns[t] = last_adv + float(values[t])

        return advantages, returns

    @property
    def last_metrics(self) -> Dict[str, float]:
        return self._metrics
