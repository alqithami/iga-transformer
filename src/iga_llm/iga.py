from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty import align_phi


@dataclass
class IGAConfig:
    selected_layers: list[int]
    rank: int = 32
    gamma_max: float = 2.0
    beta: float = 8.0
    tau: float = 0.35
    init_head_strength: float = 0.1
    share_pattern_across_heads: bool = True
    train_gamma_threshold: bool = False
    pattern_type: str = "pair_mlp"  # pair_mlp, low_rank, constant, zero
    uncertainty_mode: str = "entropy"  # entropy, constant, zero
    head_selection: str = "all"  # all, even, odd, first_half, second_half
    pair_hidden_mult: float = 2.0
    topk_inhibition: int = 0  # 0 disables; otherwise keep top-k inhibitory edges per query
    risk_hidden_mult: float = 0.25



class IGAController:
    """Runtime state for IGA hooks.

    Call reset_runtime_state() before each independent prompt/batch. Diagnostics
    are accumulated during a forward pass and then can be emitted into JSONL logs.
    """

    def __init__(self) -> None:
        self.enabled: bool = True
        self.phi: torch.Tensor | None = None
        self.key_cache: dict[int, torch.Tensor] = {}
        self.gamma_penalties: list[torch.Tensor] = []
        self.risk_losses: list[torch.Tensor] = []
        self.risk_label: torch.Tensor | None = None
        self.stats: list[dict[str, float]] = []

    def set_phi(self, phi: torch.Tensor | None) -> None:
        self.phi = phi.detach() if phi is not None else None

    def clear_penalties(self) -> None:
        self.gamma_penalties.clear()
        self.risk_losses.clear()

    def set_risk_label(self, label: float | torch.Tensor | None) -> None:
        if label is None:
            self.risk_label = None
        elif isinstance(label, torch.Tensor):
            self.risk_label = label.detach().float().view(-1)
        else:
            self.risk_label = torch.tensor([float(label)], dtype=torch.float32)

    def clear_stats(self) -> None:
        self.stats.clear()

    def reset_runtime_state(self) -> None:
        self.phi = None
        self.risk_label = None
        self.key_cache.clear()
        self.gamma_penalties.clear()
        self.risk_losses.clear()
        self.stats.clear()

    def gamma_regularizer(self) -> torch.Tensor | None:
        if not self.gamma_penalties:
            return None
        return torch.stack([p.float() for p in self.gamma_penalties]).mean()

    def risk_regularizer(self) -> torch.Tensor | None:
        if not self.risk_losses:
            return None
        return torch.stack([p.float() for p in self.risk_losses]).mean()

    def diagnostics(self) -> dict[str, float]:
        if not self.stats:
            return {}
        keys = sorted({k for row in self.stats for k in row})
        out: dict[str, float] = {}
        for key in keys:
            values = [row[key] for row in self.stats if key in row]
            if values:
                out[f"iga_{key}"] = float(sum(values) / len(values))
        return out


def _head_mask(num_heads: int, selection: str) -> torch.Tensor:
    selection = str(selection or "all").lower()
    mask = torch.ones(num_heads, dtype=torch.float32)
    if selection == "all":
        return mask
    mask.zero_()
    if selection == "even":
        mask[0::2] = 1.0
    elif selection == "odd":
        mask[1::2] = 1.0
    elif selection == "first_half":
        mask[: max(1, num_heads // 2)] = 1.0
    elif selection == "second_half":
        mask[num_heads // 2 :] = 1.0
    else:
        # comma-separated head ids
        for piece in selection.split(","):
            piece = piece.strip()
            if piece:
                idx = int(piece)
                if 0 <= idx < num_heads:
                    mask[idx] = 1.0
    return mask


class IGABias(nn.Module):
    """Nonnegative inhibitory attention-logit bias Γ, returned as -Γ.

    The primary `pair_mlp` pattern follows the paper's q/k interaction form, but
    on low-rank learned features for tractability:
        fθ(i,j) = softplus(MLP([q_i ⊙ k_j; |q_i-k_j|; q_iᵀk_j])).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rank: int,
        gamma_max: float,
        beta: float,
        tau: float,
        init_head_strength: float = 0.1,
        train_gamma_threshold: bool = False,
        pattern_type: str = "pair_mlp",
        uncertainty_mode: str = "entropy",
        head_selection: str = "all",
        pair_hidden_mult: float = 2.0,
        topk_inhibition: int = 0,
        risk_hidden_mult: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rank = rank
        self.pattern_type = str(pattern_type).lower()
        self.uncertainty_mode = str(uncertainty_mode).lower()
        self.topk_inhibition = int(topk_inhibition or 0)
        self.q_proj = nn.Linear(hidden_size, rank, bias=False)
        self.k_proj = nn.Linear(hidden_size, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        hidden = max(4, int(rank * pair_hidden_mult))
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * rank + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        risk_hidden = max(8, int(hidden_size * risk_hidden_mult))
        self.risk_probe = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, risk_hidden),
            nn.SiLU(),
            nn.Linear(risk_hidden, 1),
        )
        self.constant_logit = nn.Parameter(torch.zeros(()))
        inv_softplus = math.log(math.exp(init_head_strength) - 1.0)
        self.head_alpha = nn.Parameter(torch.full((num_heads,), inv_softplus))
        self.register_buffer("head_mask", _head_mask(num_heads, head_selection), persistent=True)
        if train_gamma_threshold:
            self.log_gamma_max = nn.Parameter(torch.tensor(math.log(max(gamma_max, 1e-6))))
            self.log_beta = nn.Parameter(torch.tensor(math.log(max(beta, 1e-6))))
            self.tau_param = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer("gamma_max_buf", torch.tensor(float(gamma_max)), persistent=True)
            self.register_buffer("beta_buf", torch.tensor(float(beta)), persistent=True)
            self.register_buffer("tau_buf", torch.tensor(float(tau)), persistent=True)
            self.log_gamma_max = None
            self.log_beta = None
            self.tau_param = None
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        for module in self.pair_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
        for module in self.risk_probe:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    @property
    def gamma_max(self) -> torch.Tensor:
        if self.log_gamma_max is not None:
            return self.log_gamma_max.exp().clamp(max=20.0)
        return self.gamma_max_buf

    @property
    def beta(self) -> torch.Tensor:
        if self.log_beta is not None:
            return self.log_beta.exp().clamp(max=100.0)
        return self.beta_buf

    @property
    def tau(self) -> torch.Tensor:
        if self.tau_param is not None:
            return self.tau_param.sigmoid()
        return self.tau_buf

    def _pattern(self, hidden_states: torch.Tensor, controller: IGAController, layer_idx: int, use_cache: bool) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        h = hidden_states.to(dtype=self.q_proj.weight.dtype)
        q = self.q_proj(h).float()
        k_cur = self.k_proj(h).float()
        if use_cache and q_len == 1 and layer_idx in controller.key_cache:
            k_all = torch.cat([controller.key_cache[layer_idx].to(device=hidden_states.device), k_cur.detach()], dim=1)
            controller.key_cache[layer_idx] = k_all.detach()
        else:
            k_all = k_cur
            controller.key_cache[layer_idx] = k_cur.detach()
        if self.pattern_type in {"zero", "none", "disabled"}:
            return torch.zeros(bsz, q_len, k_all.shape[1], device=hidden_states.device, dtype=torch.float32)
        if self.pattern_type in {"constant", "scalar", "fixed"}:
            return F.softplus(self.constant_logit.float()).expand(bsz, q_len, k_all.shape[1])
        dot = torch.einsum("btr,bsr->bts", q, k_all) / math.sqrt(max(1, self.rank))
        if self.pattern_type in {"low_rank", "dot", "bilinear"}:
            return F.softplus(dot + self.bias.float())
        if self.pattern_type in {"pair_mlp", "mlp", "exact_pair_mlp"}:
            # Build [q*k, |q-k|, dot] features over all i,j. This is O(T^2 r).
            q_exp = q[:, :, None, :]
            k_exp = k_all[:, None, :, :]
            prod = q_exp * k_exp
            diff = (q_exp - k_exp).abs()
            dot_feat = dot[:, :, :, None]
            pair = torch.cat([prod, diff, dot_feat], dim=-1)
            return F.softplus(self.pair_mlp(pair).squeeze(-1).float())
        raise ValueError(f"Unknown IGA pattern_type={self.pattern_type!r}")

    def _entropy_gate(self, phi: torch.Tensor | None, bsz: int, q_len: int, device: torch.device) -> torch.Tensor:
        phi_q = align_phi(phi, bsz, q_len, device, torch.float32)
        return torch.sigmoid(self.beta.float() * (phi_q - self.tau.float()))

    def _risk_gate(
        self,
        hidden_states: torch.Tensor,
        controller: IGAController,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = hidden_states.to(dtype=self.risk_probe[1].weight.dtype)
        risk_logits = self.risk_probe(h).squeeze(-1).float()
        risk_score = torch.sigmoid(risk_logits)
        if controller.risk_label is not None:
            target = controller.risk_label.to(device=hidden_states.device, dtype=torch.float32)
            if target.numel() == 1:
                target = target.expand(risk_logits.shape[0])
            target = target.view(-1, 1).expand_as(risk_logits)
            controller.risk_losses.append(F.binary_cross_entropy_with_logits(risk_logits, target))
        return risk_score, risk_logits

    def _gate(self, phi: torch.Tensor | None, hidden_states: torch.Tensor, controller: IGAController) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        mode = self.uncertainty_mode
        if mode in {"zero", "none", "off"}:
            return torch.zeros(bsz, q_len, device=device, dtype=torch.float32), None
        if mode in {"constant", "flat", "fixed"}:
            return torch.full((bsz, q_len), float(self.gamma_max.detach().float().cpu()), device=device, dtype=torch.float32), None
        entropy_gate = self._entropy_gate(phi, bsz, q_len, device)
        risk_logits: torch.Tensor | None = None
        if mode in {"learned_risk", "risk", "hybrid_risk", "entropy_x_risk"}:
            risk_score, risk_logits = self._risk_gate(hidden_states, controller)
            if mode in {"hybrid_risk", "entropy_x_risk"}:
                base_gate = entropy_gate * risk_score
            else:
                base_gate = risk_score
        else:
            base_gate = entropy_gate
        return self.gamma_max.float() * base_gate, risk_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        phi: torch.Tensor | None,
        controller: IGAController,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be [B,T,D], got {tuple(hidden_states.shape)}")
        bsz, q_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        f = self._pattern(hidden_states, controller, layer_idx, use_cache=use_cache)
        gate, risk_logits = self._gate(phi, hidden_states, controller)
        gamma = f * gate.unsqueeze(-1)
        if self.topk_inhibition > 0 and gamma.shape[-1] > self.topk_inhibition:
            k = min(int(self.topk_inhibition), gamma.shape[-1])
            vals, idx = torch.topk(gamma, k=k, dim=-1)
            sparse = torch.zeros_like(gamma)
            gamma = sparse.scatter(-1, idx, vals)
        head_strength = F.softplus(self.head_alpha).view(1, self.num_heads, 1, 1).float()
        mask = self.head_mask.to(device=device, dtype=torch.float32).view(1, self.num_heads, 1, 1)
        gamma_h = gamma.unsqueeze(1) * head_strength * mask
        if attention_mask is not None and attention_mask.ndim >= 4:
            key_len = attention_mask.shape[-1]
            gamma_h = _align_key_length(gamma_h, key_len)
        controller.gamma_penalties.append(gamma_h.mean())
        with torch.no_grad():
            controller.stats.append(
                {
                    "gamma_mean": float(gamma_h.detach().float().mean().cpu()),
                    "gamma_max": float(gamma_h.detach().float().max().cpu()) if gamma_h.numel() else 0.0,
                    "gate_mean": float(gate.detach().float().mean().cpu()) if gate.numel() else 0.0,
                    "gate_max": float(gate.detach().float().max().cpu()) if gate.numel() else 0.0,
                    "pattern_mean": float(f.detach().float().mean().cpu()) if f.numel() else 0.0,
                    "risk_mean": float(torch.sigmoid(risk_logits.detach().float()).mean().cpu()) if risk_logits is not None else 0.0,
                    "risk_logit_mean": float(risk_logits.detach().float().mean().cpu()) if risk_logits is not None else 0.0,
                    "topk_inhibition": float(self.topk_inhibition),
                    "active_heads": float(mask.sum().detach().cpu()),
                }
            )
        return (-gamma_h).to(dtype=dtype, device=device)


def _align_key_length(bias: torch.Tensor, key_len: int) -> torch.Tensor:
    cur = bias.shape[-1]
    if cur == key_len:
        return bias
    if cur > key_len:
        return bias[..., -key_len:]
    pad_shape = list(bias.shape)
    pad_shape[-1] = key_len - cur
    pad = torch.zeros(pad_shape, device=bias.device, dtype=bias.dtype)
    return torch.cat([pad, bias], dim=-1)


def _attention_mask_to_additive(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.ndim == 2:
        mask = (1.0 - attention_mask[:, None, None, :].to(dtype=dtype))
        return mask * torch.finfo(dtype).min
    return attention_mask


def _make_hook(layer_idx: int, controller: IGAController, bias_module: IGABias) -> Callable[..., Any]:
    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if not controller.enabled:
            return None
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            return None
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        use_cache = bool(kwargs.get("use_cache", False))
        bias = bias_module(
            hidden_states=hidden_states,
            phi=controller.phi,
            controller=controller,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        if attention_mask is None:
            new_mask = bias
        else:
            attn = _attention_mask_to_additive(attention_mask, bias.dtype).to(device=bias.device, dtype=bias.dtype)
            if attn.ndim == 4 and attn.shape[1] == 1 and bias.shape[1] > 1:
                attn = attn.expand(-1, bias.shape[1], -1, -1)
            bias = _align_key_length(bias, attn.shape[-1]) if attn.ndim >= 4 else bias
            new_mask = (attn + bias).to(dtype=hidden_states.dtype)
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = new_mask
            return args, kwargs
        args_list = list(args)
        if len(args_list) > 1:
            args_list[1] = new_mask
        else:
            kwargs["attention_mask"] = new_mask
        return tuple(args_list), kwargs
    return hook


def infer_selected_layers(num_layers: int, spec: str | Iterable[int]) -> list[int]:
    if isinstance(spec, str):
        spec = spec.strip().lower()
        if spec in {"midlate", "mid+late", "default"}:
            start = num_layers // 3
            return list(range(start, num_layers))
        if spec == "late":
            return list(range((2 * num_layers) // 3, num_layers))
        if spec == "middle":
            return list(range(num_layers // 3, (2 * num_layers) // 3))
        if spec == "early":
            return list(range(0, max(1, num_layers // 3)))
        if spec == "all":
            return list(range(num_layers))
        return [int(x) for x in spec.split(",") if x.strip()]
    return [int(x) for x in spec]


def install_iga(model: nn.Module, config: IGAConfig, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> IGAController:
    """Install IGA pre-hooks into Llama/Mistral-like decoder self-attention.

    New modules are moved to the model's device/dtype to avoid CPU/MPS mismatch.
    """
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Could not find model.model.layers; installer targets Llama/Mistral-like decoders")
    hidden_size = int(getattr(model.config, "hidden_size"))
    num_heads = int(getattr(model.config, "num_attention_heads"))
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32
    controller = IGAController()
    modules = nn.ModuleDict()
    handles = []
    for layer_idx in config.selected_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"selected layer {layer_idx} outside [0, {len(layers)-1}]")
        bias_module = IGABias(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rank=config.rank,
            gamma_max=config.gamma_max,
            beta=config.beta,
            tau=config.tau,
            init_head_strength=config.init_head_strength,
            train_gamma_threshold=config.train_gamma_threshold,
            pattern_type=config.pattern_type,
            uncertainty_mode=config.uncertainty_mode,
            head_selection=config.head_selection,
            pair_hidden_mult=config.pair_hidden_mult,
            topk_inhibition=config.topk_inhibition,
            risk_hidden_mult=config.risk_hidden_mult,
        ).to(device=device, dtype=dtype)
        modules[str(layer_idx)] = bias_module
        attn = getattr(layers[layer_idx], "self_attn")
        handles.append(attn.register_forward_pre_hook(_make_hook(layer_idx, controller, bias_module), with_kwargs=True))
    model.iga_modules = modules
    model.iga_controller = controller
    model.iga_hook_handles = handles
    return controller


def freeze_backbone_for_iga(model: nn.Module, train_lora: bool = False) -> None:
    for name, param in model.named_parameters():
        train = name.startswith("iga_modules")
        if train_lora and "lora_" in name:
            train = True
        param.requires_grad_(train)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
