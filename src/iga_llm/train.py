from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .data import EvalExample, load_examples
from .iga import IGAConfig, count_trainable_parameters, freeze_backbone_for_iga, infer_selected_layers, install_iga
from .modeling import load_causal_lm
from .scoring import choice_distribution, choice_loss, sequence_loss_tensor
from .utils import ensure_dir, json_dump, load_yaml, package_versions, set_seed, sha256_file, sha256_json, write_jsonl, write_yaml


def build_iga_config(model: Any, cfg: dict[str, Any]) -> IGAConfig:
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    iga_cfg = cfg.get("iga", {})
    selected = infer_selected_layers(num_layers, iga_cfg.get("layers", "late"))
    return IGAConfig(
        selected_layers=selected,
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


def _training_loss_for_example(model: Any, tokenizer: Any, controller: Any, ex: EvalExample, max_length: int | None, length_norm: bool) -> torch.Tensor | None:
    if ex.choices is not None and ex.correct_choice is not None:
        return choice_loss(
            model,
            tokenizer,
            ex.prompt,
            ex.choices,
            int(ex.correct_choice),
            controller=controller,
            length_norm=length_norm,
            max_length=max_length,
        )
    target = ex.target
    if target is None:
        return None
    return sequence_loss_tensor(model, tokenizer, ex.prompt, str(target), controller=controller, length_norm=length_norm, max_length=max_length)


@torch.no_grad()
def _eval_dev_choice(model: Any, tokenizer: Any, controller: Any, examples: list[EvalExample], max_examples: int, max_length: int | None) -> dict[str, float]:
    rows = []
    for ex in examples[:max_examples]:
        if ex.choices is None or ex.correct_choice is None:
            continue
        pred, conf, probs, nlls = choice_distribution(model, tokenizer, ex.prompt, ex.choices, controller=controller, max_length=max_length)
        rows.append({"correct": int(pred) == int(ex.correct_choice), "confidence": conf})
    if not rows:
        return {"dev_choice_accuracy": float("nan"), "dev_n": 0.0}
    return {"dev_choice_accuracy": float(sum(r["correct"] for r in rows) / len(rows)), "dev_n": float(len(rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-backbone IGA modules on held-out calibration examples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="Kept for compatibility; training uses grad accumulation over examples.")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--gamma_reg", type=float, default=None)
    parser.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 2e-4))
    max_length = args.max_length if args.max_length is not None else train_cfg.get("max_length", None)
    max_length = int(max_length) if max_length is not None else None
    gamma_reg = float(args.gamma_reg if args.gamma_reg is not None else train_cfg.get("gamma_reg", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    dev_eval_limit = int(train_cfg.get("dev_eval_limit", 64))
    set_seed(seed)

    output_dir = ensure_dir(args.output_dir)
    train_examples = load_examples(args.train_jsonl, limit=args.limit)
    if not train_examples:
        raise ValueError(f"No training examples loaded from {args.train_jsonl}")
    dev_examples = load_examples(args.dev_jsonl) if args.dev_jsonl else []

    loaded = load_causal_lm(
        cfg["model_id"],
        device=cfg.get("device", "auto"),
        dtype=cfg.get("dtype", "auto"),
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        revision=cfg.get("model_revision"),
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    # Keep IGA modules in fp32 for optimizer stability; the hook casts returned bias to attention dtype.
    controller = install_iga(model, build_iga_config(model, cfg), device=loaded.device, dtype=torch.float32)
    freeze_backbone_for_iga(model, train_lora=bool(train_cfg.get("train_lora", False)))
    trainable, total = count_trainable_parameters(model)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    logs: list[dict[str, Any]] = []

    start_time = time.time()
    global_step = 0
    for epoch in range(1, epochs + 1):
        # Keep the frozen backbone deterministic; only IGA modules are optimized.
        model.eval()
        model.iga_modules.train()
        order = torch.randperm(len(train_examples)).tolist()
        pbar = tqdm(order, desc=f"epoch {epoch}/{epochs}")
        for pos, idx in enumerate(pbar, start=1):
            ex = train_examples[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = _training_loss_for_example(model, tokenizer, controller, ex, max_length=max_length, length_norm=args.length_norm)
            if loss is None:
                continue
            reg = controller.gamma_regularizer()
            if reg is not None and gamma_reg > 0:
                loss = loss + gamma_reg * reg
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
            optimizer.step()
            global_step += 1
            row = {
                "event": "train_step",
                "epoch": epoch,
                "step": global_step,
                "position_in_epoch": pos,
                "example_id": ex.id,
                "benchmark": ex.benchmark,
                "loss": float(loss.detach().float().cpu()),
                "gamma_regularizer": float(reg.detach().float().cpu()) if reg is not None else None,
                "grad_norm": float(torch.as_tensor(grad_norm).detach().float().cpu()),
                "lr": lr,
                "seed": seed,
            }
            row.update(controller.diagnostics())
            logs.append(row)
            if args.print_every and global_step % args.print_every == 0:
                pbar.set_postfix(loss=row["loss"], gamma=row.get("iga_gamma_mean"))
        if dev_examples:
            model.eval()
            dev_metrics = _eval_dev_choice(model, tokenizer, controller, dev_examples, max_examples=dev_eval_limit, max_length=max_length)
            dev_row = {"event": "dev_eval", "epoch": epoch, "step": global_step, "seed": seed}
            dev_row.update(dev_metrics)
            logs.append(dev_row)
            print(json.dumps(dev_row, sort_keys=True))

    state_path = output_dir / "iga_modules.pt"
    torch.save(
        {
            "iga_modules": model.iga_modules.state_dict(),
            "config": cfg,
            "seed": seed,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "train_jsonl_sha256": sha256_file(args.train_jsonl),
            "dev_jsonl_sha256": sha256_file(args.dev_jsonl) if args.dev_jsonl else None,
        },
        state_path,
    )
    write_jsonl(output_dir / "train_log.jsonl", logs)
    summary = {
        "output_dir": str(output_dir),
        "checkpoint": str(state_path),
        "seed": seed,
        "epochs": epochs,
        "steps": global_step,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total if total else None,
        "elapsed_s": time.time() - start_time,
        "config_sha256": sha256_json(cfg),
        "package_versions": package_versions(),
    }
    json_dump(output_dir / "train_summary.json", summary)
    write_yaml(output_dir / "resolved_config.yaml", cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
