"""LoRA setup utilities for RL training of language models.

Provides helper functions to attach LoRA adapters to HuggingFace models
using the ``peft`` library, and to merge / detach adapters.

If ``peft`` is not installed the helpers degrade gracefully: they return the
original model unchanged and log a warning.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# peft import (optional)
# ---------------------------------------------------------------------------
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
    )
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False
    logger.warning(
        "peft is not installed. LoRA fine-tuning will be disabled. "
        "Install it with: pip install peft>=0.11"
    )


# Default target modules for causal LM LoRA
_DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


def attach_lora(
    model: nn.Module,
    rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> nn.Module:
    """Attach a LoRA adapter to *model* and return the wrapped model.

    Parameters
    ----------
    model:
        A HuggingFace ``AutoModelForCausalLM`` instance.
    rank:
        LoRA rank (number of low-rank dimensions).
    lora_alpha:
        Scaling factor for LoRA weights.
    lora_dropout:
        Dropout probability on LoRA layers.
    target_modules:
        Which attention modules to inject LoRA into.
        Defaults to ``["q_proj", "v_proj"]``.
    bias:
        Whether to train bias parameters (``"none"``, ``"all"``, or
        ``"lora_only"``).

    Returns
    -------
    nn.Module
        The model wrapped with LoRA adapters (or the original model if
        ``peft`` is unavailable).
    """
    if not _HAS_PEFT:
        logger.warning("peft unavailable — returning model without LoRA")
        return model

    modules = target_modules or _DEFAULT_TARGET_MODULES
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias=bias,
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return lora_model


def merge_and_unload(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base model and return a plain ``nn.Module``.

    This is useful after training when you want to save/serve the model
    without the peft overhead.  If peft is unavailable the model is returned
    unchanged.
    """
    if not _HAS_PEFT:
        return model
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model


def get_lora_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
) -> torch.optim.Optimizer:
    """Return an AdamW optimizer that only updates LoRA parameters."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        # Fall back to all parameters if nothing is marked trainable
        trainable = list(model.parameters())
    return torch.optim.AdamW(trainable, lr=lr)


def is_peft_available() -> bool:
    """Return True if peft is installed."""
    return _HAS_PEFT
