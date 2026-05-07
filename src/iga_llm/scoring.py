from __future__ import annotations

import math
import re
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from .iga import IGAController
from .metrics import normalize_text
from .uncertainty import normalized_token_entropy


def _maybe_autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def compute_base_phi(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    controller: IGAController | None = None,
) -> torch.Tensor:
    was_enabled = None
    if controller is not None:
        was_enabled = controller.enabled
        controller.enabled = False
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        phi = normalized_token_entropy(out.logits)
    finally:
        if controller is not None and was_enabled is not None:
            controller.enabled = was_enabled
    return phi.detach()


def set_iga_phi_from_base(model: Any, controller: IGAController | None, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
    if controller is None:
        return
    phi = compute_base_phi(model, input_ids=input_ids, attention_mask=attention_mask, controller=controller)
    controller.set_phi(phi)
    controller.clear_penalties()
    controller.clear_stats()
    controller.key_cache.clear()


def encode_prompt_continuation(tokenizer: Any, prompt: str, continuation: str, device: torch.device, max_length: int | None = None):
    text = prompt + " " + continuation.strip()
    enc_full = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=max_length is not None, max_length=max_length)
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=max_length is not None, max_length=max_length)
    input_ids = enc_full.input_ids.to(device)
    attention_mask = enc_full.attention_mask.to(device)
    prompt_len = min(enc_prompt.input_ids.shape[1], input_ids.shape[1])
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return input_ids, attention_mask, labels


def sequence_loss_tensor(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    controller: IGAController | None = None,
    length_norm: bool = True,
    max_length: int | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids, attention_mask, labels = encode_prompt_continuation(tokenizer, prompt, continuation, device, max_length=max_length)
    if controller is not None:
        set_iga_phi_from_base(model, controller, input_ids, attention_mask)
        controller.enabled = True
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    mask = shifted_labels.ne(-100)
    denom = mask.sum().clamp_min(1) if length_norm else torch.ones((), device=mask.device)
    return (token_loss * mask).sum() / denom


def sequence_nll(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    controller: IGAController | None = None,
    length_norm: bool = True,
    max_length: int | None = None,
) -> float:
    with torch.no_grad():
        loss = sequence_loss_tensor(model, tokenizer, prompt, continuation, controller=controller, length_norm=length_norm, max_length=max_length)
    return float(loss.detach().cpu().item())


def choice_distribution(
    model: Any,
    tokenizer: Any,
    prompt: str,
    choices: list[str],
    controller: IGAController | None = None,
    temperature: float = 1.0,
    length_norm: bool = True,
    max_length: int | None = None,
) -> tuple[int, float, list[float], list[float]]:
    if not choices:
        raise ValueError("choices cannot be empty")
    nlls = [sequence_nll(model, tokenizer, prompt, c, controller=controller, length_norm=length_norm, max_length=max_length) for c in choices]
    scores = torch.tensor([-x for x in nlls], dtype=torch.float32)
    temperature = max(float(temperature), 1e-6)
    probs = torch.softmax(scores / temperature, dim=-1).cpu().numpy().tolist()
    pred = int(max(range(len(probs)), key=lambda i: probs[i]))
    conf = float(probs[pred])
    return pred, conf, probs, nlls


def choice_loss(
    model: Any,
    tokenizer: Any,
    prompt: str,
    choices: list[str],
    correct_choice: int,
    controller: IGAController,
    length_norm: bool = True,
    max_length: int | None = None,
) -> torch.Tensor:
    losses = [sequence_loss_tensor(model, tokenizer, prompt, c, controller=controller, length_norm=length_norm, max_length=max_length) for c in choices]
    scores = -torch.stack(losses).view(1, -1)
    target = torch.tensor([int(correct_choice)], device=scores.device)
    return F.cross_entropy(scores.float(), target)


@torch.no_grad()
def generate_vanilla(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    extra_generate_kwargs: dict[str, Any] | None = None,
) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    if do_sample is None:
        do_sample = temperature > 0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-6)
        gen_kwargs["top_p"] = float(top_p)
    if extra_generate_kwargs:
        gen_kwargs.update(extra_generate_kwargs)
    out = model.generate(**enc, **gen_kwargs)
    return tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_iga_greedy(
    model: Any,
    tokenizer: Any,
    controller: IGAController,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    prompt_len = input_ids.shape[1]
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(input_ids, device=device)
        set_iga_phi_from_base(model, controller, input_ids, attention_mask)
        controller.enabled = True
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, -1, :].float()
        if temperature and temperature > 0:
            probs = torch.softmax(logits / float(temperature), dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                keep = cdf <= top_p
                keep[..., 0] = True
                filtered = torch.zeros_like(probs)
                filtered.scatter_(1, sorted_idx, sorted_probs * keep)
                probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True).strip()


def parse_choice_from_generation(text: str, choices: list[str]) -> int | None:
    if not choices:
        return None
    norm = normalize_text(text)
    # Strong aliases for benchmark labels.
    aliases: list[list[str]] = []
    for choice in choices:
        cn = normalize_text(choice)
        group = [cn]
        if cn == "supported":
            group.extend(["supports", "support", "true", "yes"])
        if cn == "refuted":
            group.extend(["refutes", "refute", "false", "no"])
        if cn == "not enough information":
            group.extend(["nei", "unknown", "insufficient information", "cannot verify"])
        if cn == "hallucinated":
            group.extend(["yes hallucinated", "is hallucinated", "hallucination"])
        if cn == "not hallucinated":
            group.extend(["not a hallucination", "faithful", "grounded", "not hallucinated"])
        aliases.append(group)
    # Look near beginning first.
    early = " ".join(norm.split()[:24])
    for idx, group in enumerate(aliases):
        for alias in sorted(set(group), key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", early):
                return idx
    for idx, group in enumerate(aliases):
        for alias in sorted(set(group), key=len, reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", norm):
                return idx
    return None


def vote_distribution(votes: list[int], num_choices: int) -> tuple[int, float, list[float]]:
    counts = torch.zeros(num_choices, dtype=torch.float32)
    for v in votes:
        if 0 <= int(v) < num_choices:
            counts[int(v)] += 1
    if float(counts.sum().item()) <= 0:
        return 0, 0.0, [1.0 / num_choices] * num_choices
    probs = counts / counts.sum().clamp_min(1.0)
    pred = int(probs.argmax().item())
    return pred, float(probs[pred].item()), probs.numpy().tolist()


def entropy_confidence_from_votes(votes: list[int], num_choices: int) -> tuple[int, float, list[float], float]:
    pred, vote_conf, probs = vote_distribution(votes, num_choices)
    p = torch.tensor(probs, dtype=torch.float32).clamp_min(1e-12)
    ent = float((-(p * torch.log(p)).sum() / math.log(max(num_choices, 2))).item())
    return pred, float(max(0.0, min(1.0, 1.0 - ent))), probs, ent
