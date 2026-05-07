from __future__ import annotations

import math
import torch


def normalized_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return H(p)/log(|V|) for logits with shape [..., vocab].

    The computation is performed in float32 for numerical stability and returns
    a tensor with shape logits.shape[:-1].
    """
    if logits.ndim < 2:
        raise ValueError("logits must have a vocabulary dimension")
    vocab = logits.shape[-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return ent / math.log(vocab)


def lag_entropy(entropy: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Causally shift entropy so position t uses uncertainty from t-1."""
    if entropy.ndim != 2:
        raise ValueError("entropy must have shape [batch, seq]")
    fill = torch.full_like(entropy[:, :1], fill_value)
    return torch.cat([fill, entropy[:, :-1]], dim=1)


def align_phi(phi: torch.Tensor | None, batch: int, q_len: int, device, dtype) -> torch.Tensor:
    """Align a cached phi tensor to the current query length."""
    if phi is None:
        return torch.zeros((batch, q_len), device=device, dtype=dtype)
    if phi.ndim == 1:
        phi = phi[:, None]
    if phi.ndim != 2:
        raise ValueError(f"phi must have shape [batch, seq] or [batch], got {tuple(phi.shape)}")
    phi = phi.to(device=device, dtype=dtype)
    if phi.shape[0] != batch:
        if phi.shape[0] == 1:
            phi = phi.expand(batch, -1)
        else:
            raise ValueError(f"phi batch {phi.shape[0]} != hidden batch {batch}")
    if phi.shape[1] == q_len:
        return phi
    if phi.shape[1] > q_len:
        return phi[:, -q_len:]
    pad = torch.zeros((batch, q_len - phi.shape[1]), device=device, dtype=dtype)
    return torch.cat([pad, phi], dim=1)
