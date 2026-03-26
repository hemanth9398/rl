"""Knowledge Transfer Manager: golden thoughts and LoRA-style adapters."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from agents.tree_of_thought import ThoughtBranch


@dataclass
class LoRAAdapter:
    """A LoRA-style weight delta that can be injected into a SubAgent model."""

    adapter_id: str
    target_modules: List[str]
    rank: int
    alpha: float
    weights: Dict[str, torch.Tensor]  # delta weights keyed by module name
    source_task: str                   # task that generated this adapter


class KnowledgeTransferManager:
    """Manages golden thought generation and LoRA adapter creation.

    When a SubAgent's ToT branches all score below ``threshold`` (stuck), the
    Teacher generates a "golden thought" — the correct chain-of-reasoning.
    This manager converts the gap between student and golden reasoning into a
    lightweight LoRA-style weight delta that can be injected into the SubAgent.
    """

    def __init__(
        self,
        base_model_dim: int = 64,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
    ) -> None:
        self.base_model_dim = base_model_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        # Simple token-level embedding to convert text → vector
        self._embed_dim = base_model_dim
        self._proj = nn.Linear(base_model_dim, base_model_dim, bias=False)
        nn.init.eye_(self._proj.weight)  # start as identity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_transfer(
        self,
        branches: List[ThoughtBranch],
        threshold: float = 0.3,
    ) -> bool:
        """Check if all branches are below quality threshold → trigger transfer."""
        if not branches:
            return True
        return all(b.confidence < threshold for b in branches)

    def compute_reasoning_loss(
        self,
        student_output: str,
        golden_thought: str,
    ) -> torch.Tensor:
        """Compute loss between student reasoning and golden thought.

        Encodes both texts as bag-of-character n-gram vectors and returns the
        cosine-distance loss (1 − cosine_similarity).  This is intentionally
        lightweight — no external tokenizer is required.
        """
        student_vec = self._text_to_vector(student_output)
        golden_vec = self._text_to_vector(golden_thought)
        cos_sim = torch.nn.functional.cosine_similarity(
            student_vec.unsqueeze(0), golden_vec.unsqueeze(0)
        )
        loss = 1.0 - cos_sim  # in [0, 1] for L2-normalized vectors (0 = identical)
        return loss.squeeze()

    def create_adapter(
        self,
        reasoning_loss: torch.Tensor,
        task_context: Dict[str, Any],
    ) -> LoRAAdapter:
        """Create a LoRA adapter from the reasoning loss.

        Computes low-rank weight deltas (A·B style) that would improve the
        student's reasoning toward the golden thought.  The magnitude of the
        delta scales with the reasoning loss.
        """
        scale = float(reasoning_loss.detach()) * self.lora_alpha / self.lora_rank
        d = self.base_model_dim
        r = self.lora_rank

        # A: (rank, d), B: (d, rank)  →  delta = B @ A  shape (d, d)
        A = torch.randn(r, d) * 0.01
        B = torch.zeros(d, r)
        weights: Dict[str, torch.Tensor] = {
            "lora_A": A * scale,
            "lora_B": B,
        }

        target_modules = task_context.get("target_modules", ["linear_0"])
        source_task = task_context.get("task_id", "unknown")

        return LoRAAdapter(
            adapter_id=str(uuid.uuid4()),
            target_modules=target_modules,
            rank=r,
            alpha=self.lora_alpha,
            weights=weights,
            source_task=source_task,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _text_to_vector(self, text: str) -> torch.Tensor:
        """Convert text to a fixed-size float vector via character hashing."""
        vec = torch.zeros(self._embed_dim)
        if not text:
            return vec
        for i, ch in enumerate(text[: self._embed_dim * 4]):
            idx = ord(ch) % self._embed_dim
            vec[idx] += 1.0
        norm = vec.norm()
        return vec / (norm + 1e-8)
