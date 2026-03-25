"""Small actor-critic policy network (PyTorch)."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Actions
ACTION_RETRIEVE = 0
ACTION_SOLVE = 1
ACTION_VERIFY = 2
ACTION_REPAIR = 3
ACTION_GENERATE = 4
NUM_ACTIONS = 5

ACTION_NAMES = {
    ACTION_RETRIEVE: "RETRIEVE",
    ACTION_SOLVE: "SOLVE",
    ACTION_VERIFY: "VERIFY",
    ACTION_REPAIR: "REPAIR",
    ACTION_GENERATE: "GENERATE",
}


class PolicyNetwork(nn.Module):
    """2-layer MLP actor-critic policy.

    Input:  state feature vector (state_dim)
    Output: action_probs (num_actions), value_estimate (scalar)
    """

    def __init__(
        self,
        state_dim: int = 16,
        hidden_dim: int = 64,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Actor head
        self.actor = nn.Linear(hidden_dim, num_actions)
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,)
        Returns:
            action_probs: Tensor of shape (batch, num_actions)
            value:        Tensor of shape (batch, 1)
        """
        x = self.backbone(state)
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(x)
        return action_probs, value

    def select_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, value)."""
        action_probs, value = self.forward(state.unsqueeze(0))
        action_probs = action_probs.squeeze(0)
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob, value.squeeze(0)

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_probs, values, and entropy for given (states, actions)."""
        action_probs, values = self.forward(states)
        dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy
