from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import deep_update


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    config_override: dict[str, Any]
    train: bool = True


def get_ablation_suite() -> list[AblationSpec]:
    return [
        AblationSpec("ablate_full_iga", "Full configured IGA method.", {}),
        AblationSpec("ablate_no_uncertainty_gate", "Replace entropy gate with constant gate.", {"iga": {"uncertainty_mode": "constant"}}),
        AblationSpec("ablate_no_inhibition", "Set inhibition to zero; should match vanilla except hook overhead.", {"iga": {"pattern_type": "zero", "uncertainty_mode": "zero"}}, train=False),
        AblationSpec("ablate_constant_inhibition", "Use token-pair-independent scalar inhibition.", {"iga": {"pattern_type": "constant"}}),
        AblationSpec("ablate_low_rank_dot", "Use low-rank dot-product f_theta instead of pair MLP.", {"iga": {"pattern_type": "low_rank"}}),
        AblationSpec("ablate_pair_mlp", "Use pair MLP f_theta from the paper equations.", {"iga": {"pattern_type": "pair_mlp"}}),
        AblationSpec("ablate_early_layers", "Apply IGA to early transformer layers.", {"iga": {"layers": "early"}}),
        AblationSpec("ablate_middle_layers", "Apply IGA to middle transformer layers.", {"iga": {"layers": "middle"}}),
        AblationSpec("ablate_all_layers", "Apply IGA to all transformer layers.", {"iga": {"layers": "all"}}),
        AblationSpec("ablate_selected_heads_even", "Apply IGA only to even-indexed attention heads.", {"iga": {"head_selection": "even"}}),
        AblationSpec("ablate_train_threshold", "Learn gamma_max, beta, and tau.", {"iga": {"train_gamma_threshold": True}}),
        AblationSpec("ablate_rank_8", "Low-rank bottleneck rank 8.", {"iga": {"rank": 8}}),
        AblationSpec("ablate_rank_64", "Low-rank bottleneck rank 64.", {"iga": {"rank": 64}}),
        AblationSpec("ablate_gamma_0p5", "Weak max inhibition gamma_max=0.5.", {"iga": {"gamma_max": 0.5}}),
        AblationSpec("ablate_gamma_4", "Strong max inhibition gamma_max=4.0.", {"iga": {"gamma_max": 4.0}}),
        AblationSpec("ablate_inference_only", "Randomly initialized IGA modules; no training.", {}, train=False),
    ]


def get_ablation_by_name(name: str) -> AblationSpec | None:
    for spec in get_ablation_suite():
        if spec.name == name:
            return spec
    return None


def ablation_config(base_config: dict[str, Any], spec: AblationSpec) -> dict[str, Any]:
    return deep_update(base_config, spec.config_override)


def default_ablation_names() -> list[str]:
    return [
        "ablate_full_iga",
        "ablate_no_uncertainty_gate",
        "ablate_no_inhibition",
        "ablate_constant_inhibition",
        "ablate_low_rank_dot",
        "ablate_early_layers",
        "ablate_middle_layers",
        "ablate_selected_heads_even",
        "ablate_inference_only",
    ]


def describe_ablations() -> str:
    lines = ["IGA ablation suite", "=" * 60]
    for spec in get_ablation_suite():
        lines.append(f"{spec.name}: {spec.description} train={spec.train} overrides={spec.config_override}")
    return "\n".join(lines)
