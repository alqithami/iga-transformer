from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .iga import IGAConfig, infer_selected_layers, install_iga
from .memory import peak_memory_mb
from .modeling import load_causal_lm
from .scoring import generate_iga_greedy, generate_vanilla
from .utils import ensure_dir, load_yaml, package_versions, set_seed, sha256_json, write_jsonl


def _build_iga_config(model: Any, cfg: dict[str, Any]) -> IGAConfig:
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    iga_cfg = cfg.get("iga", {})
    return IGAConfig(
        selected_layers=infer_selected_layers(num_layers, iga_cfg.get("layers", "late")),
        rank=int(iga_cfg.get("rank", 32)),
        gamma_max=float(iga_cfg.get("gamma_max", 2.0)),
        beta=float(iga_cfg.get("beta", 8.0)),
        tau=float(iga_cfg.get("tau", 0.35)),
        init_head_strength=float(iga_cfg.get("init_head_strength", 0.1)),
        train_gamma_threshold=bool(iga_cfg.get("train_gamma_threshold", False)),
        pattern_type=str(iga_cfg.get("pattern_type", "pair_mlp")),
        uncertainty_mode=str(iga_cfg.get("uncertainty_mode", "entropy")),
        head_selection=str(iga_cfg.get("head_selection", "all")),
        pair_hidden_mult=float(iga_cfg.get("pair_hidden_mult", 2.0)),
    )


def _load_iga_state(model: Any, path: str | None) -> None:
    if not path:
        return
    state_path = Path(path)
    if state_path.is_dir():
        state_path = state_path / "iga_modules.pt"
    state = torch.load(state_path, map_location="cpu")
    model.iga_modules.load_state_dict(state["iga_modules"], strict=True)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure generation latency/memory for vanilla vs IGA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["vanilla", "iga"], required=True)
    parser.add_argument("--iga_checkpoint", default=None)
    parser.add_argument("--prompt_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_id", default="manual")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_yaml(args.config)
    loaded = load_causal_lm(
        cfg["model_id"],
        device=cfg.get("device", "auto"),
        dtype=cfg.get("dtype", "auto"),
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        revision=cfg.get("model_revision"),
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    controller = None
    if args.method == "iga":
        controller = install_iga(model, _build_iga_config(model, cfg), device=loaded.device, dtype=torch.float32)
        _load_iga_state(model, args.iga_checkpoint)

    base = "The following factuality benchmark passage is deterministic and used only for measuring throughput. "
    text = base
    while len(tokenizer(text, add_special_tokens=False).input_ids) < args.prompt_tokens:
        text += base
    ids = tokenizer(text, add_special_tokens=False).input_ids[: args.prompt_tokens]
    prompt = tokenizer.decode(ids)

    def run_once() -> tuple[float, int, str]:
        _sync()
        start = time.perf_counter()
        if args.method == "iga":
            assert controller is not None
            out = generate_iga_greedy(model, tokenizer, controller, prompt, max_new_tokens=args.max_new_tokens)
        else:
            out = generate_vanilla(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)
        _sync()
        latency = time.perf_counter() - start
        return latency, len(tokenizer(out, add_special_tokens=False).input_ids), out

    for _ in range(args.warmup):
        run_once()
    rows = []
    for idx in range(args.runs):
        latency, generated, out = run_once()
        rows.append(
            {
                "method": args.method,
                "model_id": cfg["model_id"],
                "seed": args.seed,
                "run_id": args.run_id,
                "prompt_tokens": args.prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "run": idx,
                "latency_s": latency,
                "generated_tokens": generated,
                "tokens_per_second": generated / latency if latency > 0 else None,
                "peak_memory_mb": peak_memory_mb(),
                "config_sha256": sha256_json(cfg),
                "package_versions": package_versions(),
            }
        )
    ensure_dir(Path(args.out).parent)
    write_jsonl(args.out, rows)
    summary = {
        "n": len(rows),
        "mean_latency_s": statistics.mean([r["latency_s"] for r in rows]) if rows else None,
        "mean_tokens_per_second": statistics.mean([r["tokens_per_second"] for r in rows]) if rows else None,
    }
    print(json.dumps({"out": args.out, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
