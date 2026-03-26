"""GIGPO: Group-In-Group Policy Optimization for ToT branch selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from agents.tree_of_thought import ThoughtBranch


@dataclass
class BranchComparison:
    """A labelled comparison between two thought branches."""

    left_branch: ThoughtBranch
    right_branch: ThoughtBranch
    selected: str    # "left" or "right"
    reward: float    # outcome reward


class _BranchScorerNet(nn.Module):
    """Simple MLP that scores a thought branch given its features."""

    def __init__(self, input_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GIGPOScorer:
    """Group-In-Group Policy Optimization for ToT branch selection.

    Scores and selects between thought branches at each ToT node.
    Uses pairwise comparison within the same thought tree.
    """

    def __init__(
        self,
        scorer_network: Optional[nn.Module] = None,
        lr: float = 1e-4,
        margin: float = 0.1,
        feature_dim: int = 64,
    ) -> None:
        self.scorer_network = scorer_network or _BranchScorerNet(input_dim=feature_dim)
        self.feature_dim = feature_dim
        self.margin = margin
        self.optimizer = optim.Adam(self.scorer_network.parameters(), lr=lr)
        self._metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_branches(
        self,
        branches: List[ThoughtBranch],
        context: Dict[str, Any],
    ) -> List[float]:
        """Score all branches in a comparison group."""
        if not branches:
            return []
        features = torch.stack([self._branch_to_features(b, context) for b in branches])
        with torch.no_grad():
            scores = self.scorer_network(features)
        # Convert to [0, 1] via sigmoid
        return torch.sigmoid(scores).tolist()

    def select(
        self,
        branches: List[ThoughtBranch],
        context: Dict[str, Any],
    ) -> Tuple[ThoughtBranch, float]:
        """Select best branch and return with log probability."""
        if not branches:
            empty = ThoughtBranch(
                branch_id="empty", reasoning="", answer="", confidence=0.0, depth=0
            )
            return empty, float("-inf")

        features = torch.stack([self._branch_to_features(b, context) for b in branches])
        with torch.no_grad():
            logits = self.scorer_network(features)
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.argmax(probs).item())
        log_prob = float(torch.log(probs[idx] + 1e-8).item())
        return branches[idx], log_prob

    def update(self, comparisons: List[BranchComparison]) -> Dict[str, float]:
        """Update scorer from outcome-labeled comparisons.

        For each comparison:
        1. Score both branches.
        2. If selected branch got positive reward → reinforce selection.
        3. If selected branch got negative reward → penalise.
        4. Both bad → signal for knowledge transfer (no update).
        """
        if not comparisons:
            return {}

        total_loss = 0.0
        n = 0
        for comp in comparisons:
            left_feat = self._branch_to_features(comp.left_branch, {})
            right_feat = self._branch_to_features(comp.right_branch, {})

            left_score = self.scorer_network(left_feat.unsqueeze(0)).squeeze()
            right_score = self.scorer_network(right_feat.unsqueeze(0)).squeeze()

            # Margin-based ranking loss: selected should score higher
            if comp.reward > 0:
                # selected branch should outscore the other
                if comp.selected == "left":
                    loss = torch.clamp(self.margin - (left_score - right_score), min=0.0)
                else:
                    loss = torch.clamp(self.margin - (right_score - left_score), min=0.0)
            else:
                # penalise — the other branch should score higher
                if comp.selected == "left":
                    loss = torch.clamp(self.margin - (right_score - left_score), min=0.0)
                else:
                    loss = torch.clamp(self.margin - (left_score - right_score), min=0.0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n += 1

        self._metrics = {"gigpo_loss": total_loss / max(n, 1)}
        return self._metrics

    def both_branches_bad(
        self,
        branches: List[ThoughtBranch],
        threshold: float = 0.3,
    ) -> bool:
        """Check if all branches are below quality threshold."""
        if not branches:
            return True
        scores = self.score_branches(branches, {})
        return all(s < threshold for s in scores)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _branch_to_features(
        self, branch: ThoughtBranch, context: Dict[str, Any]
    ) -> torch.Tensor:
        """Convert a ThoughtBranch to a fixed-size feature vector."""
        vec = torch.zeros(self.feature_dim)

        # confidence in first slot
        vec[0] = branch.confidence
        # depth normalised
        vec[1] = branch.depth / 10.0
        # answer length normalised
        vec[2] = min(len(branch.answer), 200) / 200.0
        # reasoning length normalised
        vec[3] = min(len(branch.reasoning), 500) / 500.0

        # character-hash of reasoning into remaining slots
        for i, ch in enumerate(branch.reasoning[: self.feature_dim - 4]):
            idx = 4 + (ord(ch) % (self.feature_dim - 4))
            vec[idx] += 0.1

        # Clip to [0, 1]
        vec = torch.clamp(vec, 0.0, 1.0)
        return vec
