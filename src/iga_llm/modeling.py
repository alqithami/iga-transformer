from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype


def resolve_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        return torch.float16 if device.type in {"mps", "cuda"} else torch.float32
    key = str(dtype).lower()
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def load_causal_lm(
    model_id: str,
    device: str | None = "auto",
    dtype: str | torch.dtype | None = "auto",
    attn_implementation: str = "eager",
    trust_remote_code: bool = False,
    low_cpu_mem_usage: bool = True,
    local_files_only: bool = False,
    revision: str | None = None,
) -> LoadedModel:
    """Load a decoder-only HF model and tokenizer for IGA experiments.

    IGA injects an additive attention-mask bias, so eager attention is preferred.
    """
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        revision=revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = dict(
        torch_dtype=resolved_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        local_files_only=local_files_only,
        revision=revision,
    )
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.to(resolved_device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, device=resolved_device, dtype=resolved_dtype)
