"""PPO trainers: MLP policy network and LLM fine-tuning via LoRA."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from policy.policy_nn import PolicyNetwork

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# LLM PPO Trainer
# ---------------------------------------------------------------------------

class LLMPPOTrainer:
    """PPO trainer that fine-tunes a language model via LoRA.

    Computes PPO loss on the language model's log probabilities (not just the
    MLP).  Uses the reward from the Verifier to update the solver/subagent
    model with LoRA adapters.

    The :class:`PolicyNetwork` (MLP) is kept as the *action selector* for the
    environment loop.  The LLM is fine-tuned on the text it generated,
    weighted by the PPO advantage.

    Parameters
    ----------
    policy:
        MLP policy network used for environment action selection.
    registry:
        :class:`~models.model_registry.ModelRegistry` instance.  If ``None``,
        LLM fine-tuning is disabled (falls back to MLP-only PPO).
    role:
        Which registry role to fine-tune (``"solver"`` or ``"subagent"``).
    lr_llm:
        Learning rate for the LLM LoRA parameters.
    lora_rank:
        LoRA rank to use when attaching adapters.
    **ppo_kwargs:
        Forwarded to the inner :class:`PPOTrainer`.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        registry: Optional[Any] = None,
        role: str = "solver",
        lr_llm: float = 1e-4,
        lora_rank: int = 16,
        **ppo_kwargs: Any,
    ) -> None:
        self.mlp_trainer = PPOTrainer(policy, **ppo_kwargs)
        self.registry = registry
        self.role = role
        self._lora_model: Optional[nn.Module] = None
        self._lora_optimizer: Optional[torch.optim.Optimizer] = None
        self._lora_rank = lora_rank
        self._lr_llm = lr_llm
        self._llm_metrics: Dict[str, float] = {}

        if registry is not None:
            self._attach_lora()

    def _attach_lora(self) -> None:
        """Attach LoRA adapters to the LLM for the configured role."""
        try:
            from models.lora_utils import attach_lora, get_lora_optimizer, is_peft_available

            if not is_peft_available():
                logger.warning("peft unavailable — LLM PPO fine-tuning disabled")
                return

            model, _ = self.registry.get_raw_model(self.role)
            if model is None:
                logger.warning(
                    "Model for role '%s' not loaded — LLM PPO disabled", self.role
                )
                return

            self._lora_model = attach_lora(model, rank=self._lora_rank)
            self._lora_optimizer = get_lora_optimizer(self._lora_model, lr=self._lr_llm)
            logger.info("LLMPPOTrainer: LoRA attached for role '%s'", self.role)
        except Exception as exc:
            logger.warning("LLMPPOTrainer: failed to attach LoRA: %s", exc)

    # ------------------------------------------------------------------
    # Delegation to MLP PPO
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        self.mlp_trainer.store_transition(state, action, reward, log_prob, value, done)

    def notify_episode_end(self) -> bool:
        return self.mlp_trainer.notify_episode_end()

    # ------------------------------------------------------------------
    # LLM PPO update
    # ------------------------------------------------------------------

    def update_llm(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
        clip_epsilon: float = 0.2,
    ) -> Dict[str, float]:
        """Run a policy-gradient update on the LLM using token log-probabilities.

        This is a *simplified* variant of PPO for language models: it computes
        advantage-weighted log-probability losses without ratio clipping (since
        storing reference log-probs per token would require caching the full
        generation).  Ratio clipping (standard PPO) can be enabled in a future
        version once reference log-probs are stored.  The ``clip_epsilon``
        parameter is accepted for API compatibility but is not applied here.

        Parameters
        ----------
        prompts:
            Input prompts that were fed to the LLM.
        completions:
            Corresponding outputs generated by the LLM.
        rewards:
            Scalar rewards for each (prompt, completion) pair.
        clip_epsilon:
            Reserved for future PPO ratio clipping (not yet applied).

        Returns
        -------
        dict with ``llm_ppo_loss`` and ``llm_ppo_n_updates``.
        """
        if self._lora_model is None or self._lora_optimizer is None:
            return {}
        if not prompts:
            return {}

        try:
            _, tokenizer = self.registry.get_raw_model(self.role)
            if tokenizer is None:
                return {}

            device = next(self._lora_model.parameters()).device
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            mean_r = rewards_t.mean()
            std_r = rewards_t.std() + 1e-8
            advantages = (rewards_t - mean_r) / std_r

            total_loss = 0.0
            n = 0

            for prompt, completion, adv in zip(prompts, completions, advantages.tolist()):
                full_text = prompt + completion
                full_enc = tokenizer(full_text, return_tensors="pt")
                prompt_enc = tokenizer(prompt, return_tensors="pt")
                prompt_len = prompt_enc["input_ids"].shape[-1]

                input_ids = full_enc["input_ids"].to(device)
                if input_ids.shape[-1] <= prompt_len:
                    continue

                # Compute log-probs under current (LoRA) model
                outputs = self._lora_model(input_ids)
                logits = outputs.logits  # (1, seq_len, vocab)

                shift_logits = logits[0, prompt_len - 1 : -1, :]
                shift_labels = input_ids[0, prompt_len:]
                if shift_labels.numel() == 0:
                    continue

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_lps = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                mean_lp = token_lps.mean()

                # Advantage-weighted policy gradient: -adv * mean_log_prob
                # (simplified PPO without ratio clipping — see update_llm docstring)
                loss = -float(adv) * mean_lp

                self._lora_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._lora_model.parameters(), 1.0)
                self._lora_optimizer.step()

                total_loss += loss.item()
                n += 1

            self._llm_metrics = {
                "llm_ppo_loss": total_loss / max(n, 1),
                "llm_ppo_n_updates": float(n),
            }
        except Exception as exc:
            logger.warning("LLMPPOTrainer.update_llm failed: %s", exc)

        return self._llm_metrics

    @property
    def last_metrics(self) -> Dict[str, float]:
        metrics = dict(self.mlp_trainer.last_metrics)
        metrics.update(self._llm_metrics)
        return metrics
