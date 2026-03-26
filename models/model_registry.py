"""Central model registry: loads and manages HuggingFace transformer models.

Each agent role (Teacher, SubAgent, Verifier, Solver) gets its own model
configuration.  Models are loaded lazily on first use to avoid OOM on
machines without a GPU.

Usage::

    from models.model_registry import ModelRegistry, ModelConfig

    registry = ModelRegistry.from_cli_args(args)
    teacher_gen = registry.generate("teacher", prompt, max_new_tokens=256)
    verifier_gen = registry.generate("verifier", prompt, max_new_tokens=128)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model names per role
# ---------------------------------------------------------------------------

DEFAULT_SUBAGENT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_SUBAGENT_FALLBACK = "Qwen/Qwen2.5-1.5B"
DEFAULT_VERIFIER_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SOLVER_MODEL = DEFAULT_SUBAGENT_MODEL  # same as subagent by default

# Max new tokens per role
_DEFAULT_MAX_TOKENS: Dict[str, int] = {
    "teacher": 512,
    "subagent": 512,
    "verifier": 256,
    "solver": 512,
}


@dataclass
class ModelConfig:
    """Configuration for a single role's language model."""

    role: str
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    cpu_only: bool = False
    load_in_4bit: bool = False
    torch_dtype: Any = torch.bfloat16

    def device_map(self) -> str:
        return "cpu" if self.cpu_only else "auto"


class _ModelHandle:
    """Lazy-loaded model + tokenizer pair."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    def _try_load(self, model_name: str) -> bool:
        """Attempt to load a specific model name. Returns True on success."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "Loading model '%s' for role '%s' …", model_name, self.config.role
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            load_kwargs: Dict[str, Any] = {
                "device_map": self.config.device_map(),
                "torch_dtype": self.config.torch_dtype,
                "trust_remote_code": True,
            }

            if self.config.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    load_kwargs.pop("torch_dtype", None)
                except ImportError:
                    logger.warning("bitsandbytes not available; skipping 4-bit quant")

            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model.eval()

            self._model = model
            self._tokenizer = tokenizer
            return True

        except Exception as exc:
            logger.warning("Failed to load '%s': %s", model_name, exc)
            return False

    def ensure_loaded(self) -> None:
        """Load the model lazily (thread-safe).  Falls back to lighter model."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            primary = self.config.model_name
            ok = self._try_load(primary)

            # If the primary is the large MoE model, fall back to a lighter one
            if not ok and primary == DEFAULT_SUBAGENT_MODEL:
                logger.info(
                    "Falling back to '%s' for role '%s'",
                    DEFAULT_SUBAGENT_FALLBACK,
                    self.config.role,
                )
                ok = self._try_load(DEFAULT_SUBAGENT_FALLBACK)

            if not ok:
                self._load_error = (
                    f"Could not load any model for role '{self.config.role}'"
                )
                logger.error(self._load_error)

            self._loaded = True

    @property
    def available(self) -> bool:
        self.ensure_loaded()
        return self._model is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        """Run inference and return the generated text (excluding the prompt)."""
        self.ensure_loaded()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                f"Model for role '{self.config.role}' is not available. "
                f"Error: {self._load_error}"
            )

        tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature

        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample and temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        # Decode only the newly generated tokens
        new_ids = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def log_probs(self, prompt: str, completion: str) -> float:
        """Return the mean log-probability of ``completion`` given ``prompt``."""
        self.ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return 0.0

        full_text = prompt + completion
        inputs = self._tokenizer(full_text, return_tensors="pt")
        prompt_inputs = self._tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[-1]

        device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(device)

        with torch.inference_mode():
            outputs = self._model(input_ids, labels=input_ids)
            # outputs.loss is mean NLL over all tokens; we want completion only
            logits = outputs.logits  # (1, seq_len, vocab_size)

        shift_logits = logits[0, prompt_len - 1 : -1, :]  # (comp_len, vocab)
        shift_labels = input_ids[0, prompt_len:]           # (comp_len,)
        if shift_labels.numel() == 0:
            return 0.0

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        return float(token_lps.mean())


class ModelRegistry:
    """Central registry that holds one :class:`_ModelHandle` per role."""

    def __init__(self, configs: Dict[str, ModelConfig]) -> None:
        self._handles: Dict[str, _ModelHandle] = {
            role: _ModelHandle(cfg) for role, cfg in configs.items()
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_args(
        cls,
        teacher_model: str = DEFAULT_TEACHER_MODEL,
        subagent_model: str = DEFAULT_SUBAGENT_MODEL,
        verifier_model: str = DEFAULT_VERIFIER_MODEL,
        solver_model: Optional[str] = None,
        cpu_only: bool = False,
        load_in_4bit: bool = False,
        lora_rank: int = 16,
    ) -> "ModelRegistry":
        """Create a registry from explicit model-name arguments."""
        _solver = solver_model or subagent_model
        dtype = torch.float32 if cpu_only else torch.bfloat16
        configs = {
            "teacher": ModelConfig(
                role="teacher",
                model_name=teacher_model,
                max_new_tokens=_DEFAULT_MAX_TOKENS["teacher"],
                cpu_only=cpu_only,
                load_in_4bit=load_in_4bit,
                torch_dtype=dtype,
            ),
            "subagent": ModelConfig(
                role="subagent",
                model_name=subagent_model,
                max_new_tokens=_DEFAULT_MAX_TOKENS["subagent"],
                cpu_only=cpu_only,
                load_in_4bit=load_in_4bit,
                torch_dtype=dtype,
            ),
            "verifier": ModelConfig(
                role="verifier",
                model_name=verifier_model,
                max_new_tokens=_DEFAULT_MAX_TOKENS["verifier"],
                cpu_only=cpu_only,
                load_in_4bit=load_in_4bit,
                torch_dtype=dtype,
            ),
            "solver": ModelConfig(
                role="solver",
                model_name=_solver,
                max_new_tokens=_DEFAULT_MAX_TOKENS["solver"],
                cpu_only=cpu_only,
                load_in_4bit=load_in_4bit,
                torch_dtype=dtype,
            ),
        }
        return cls(configs)

    @classmethod
    def from_cli_args(cls, args: Any) -> "ModelRegistry":
        """Build from an ``argparse.Namespace`` produced by the CLI scripts."""
        return cls.from_args(
            teacher_model=getattr(args, "teacher_model", DEFAULT_TEACHER_MODEL),
            subagent_model=getattr(args, "subagent_model", DEFAULT_SUBAGENT_MODEL),
            verifier_model=getattr(args, "verifier_model", DEFAULT_VERIFIER_MODEL),
            solver_model=getattr(args, "solver_model", None),
            cpu_only=getattr(args, "cpu_only", False),
            load_in_4bit=getattr(args, "load_in_4bit", False),
            lora_rank=getattr(args, "lora_rank", 16),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        role: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        """Generate text for the given role."""
        handle = self._handles.get(role)
        if handle is None:
            raise KeyError(f"Unknown role '{role}'. Available: {list(self._handles)}")
        return handle.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    def log_probs(self, role: str, prompt: str, completion: str) -> float:
        """Return mean log-probability of ``completion`` given ``prompt``."""
        handle = self._handles.get(role)
        if handle is None:
            return 0.0
        return handle.log_probs(prompt, completion)

    def is_available(self, role: str) -> bool:
        """Return True if the model for ``role`` loaded successfully."""
        handle = self._handles.get(role)
        return handle is not None and handle.available

    def preload(self, *roles: str) -> None:
        """Eagerly load models for the given roles (blocking)."""
        for role in roles:
            handle = self._handles.get(role)
            if handle is not None:
                handle.ensure_loaded()

    def get_raw_model(self, role: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Return (model, tokenizer) for a role — needed for LoRA attachment."""
        handle = self._handles.get(role)
        if handle is None:
            return None, None
        handle.ensure_loaded()
        return handle._model, handle._tokenizer
