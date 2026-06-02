from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .data import EvalExample, load_examples
from .iga import IGAConfig, infer_selected_layers, install_iga
from .metrics import is_refusal
from .modeling import load_causal_lm
from .scoring import (
    choice_distribution,
    entropy_confidence_from_votes,
    generate_iga_greedy,
    generate_vanilla,
    parse_choice_from_generation,
    vote_distribution,
)
from .utils import ensure_dir, load_yaml, package_versions, set_seed, sha256_file, sha256_json, sha256_text, write_jsonl

CHOICE_METHODS = {"vanilla_mc", "temperature_mc", "iga_mc", "self_consistency_mc", "semantic_entropy_mc"}
GEN_METHODS = {"vanilla_gen", "contrastive_gen", "dola_gen", "self_refine_gen", "cove_gen", "iga_gen"}
ALIASES = {
    "vanilla": "vanilla_mc",
    "temperature": "temperature_mc",
    "iga": "iga_mc",
    "contrastive_search": "contrastive_gen",
    "dola": "dola_gen",
    "self_refine": "self_refine_gen",
    "cove": "cove_gen",
}


def _canonical_method(method: str) -> str:
    return ALIASES.get(method, method)


def _load_iga_state(model: Any, path: str | None) -> None:
    if not path:
        return
    state_path = Path(path)
    if state_path.is_dir():
        state_path = state_path / "iga_modules.pt"
    state = torch.load(state_path, map_location="cpu")
    model.iga_modules.load_state_dict(state["iga_modules"], strict=True)


def _build_iga_config(model: Any, cfg: dict[str, Any]) -> IGAConfig:
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
        topk_inhibition=int(iga_cfg.get("topk_inhibition", 0) or 0),
        risk_hidden_mult=float(iga_cfg.get("risk_hidden_mult", 0.25)),
    )


def _method_generate_kwargs(method: str) -> dict[str, Any]:
    if method == "contrastive_gen":
        return {"penalty_alpha": 0.6, "top_k": 4, "do_sample": False}
    if method == "dola_gen":
        return {"dola_layers": "high", "do_sample": False}
    return {}


def _method_config(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": args.method,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": args.num_samples,
        "length_norm": not args.no_length_norm,
        "iga_checkpoint": args.iga_checkpoint,
        "lora_adapter_dir": args.lora_adapter_dir,
        "lora_merged_for_eval": bool(args.lora_adapter_dir),
        "iga": cfg.get("iga", {}),
    }


def _base_row(args: argparse.Namespace, ex: EvalExample, method_config: dict[str, Any], dataset_sha: str, model_id: str) -> dict[str, Any]:
    return {
        "id": ex.id,
        "benchmark": ex.benchmark,
        "split": ex.split,
        "source_dataset": ex.source_dataset,
        "source_config": ex.source_config,
        "source_split": ex.source_split,
        "method": getattr(args, "report_method", None) or args.method,
        "actual_method": args.method,
        "model_id": model_id,
        "seed": int(args.seed),
        "run_id": args.run_id,
        "dataset_file_sha256": dataset_sha,
        "method_config_sha256": sha256_json(method_config),
        "method_config": method_config,
        "prompt": ex.prompt,
        "prompt_sha256": sha256_text(ex.prompt),
        "choices": ex.choices,
        "correct_choice": ex.correct_choice,
        "target": ex.target,
        "metadata": ex.metadata or {},
    }


def _classification_prompt(prompt: str, choices: list[str]) -> str:
    return f"{prompt}\n\nRespond with exactly one of the following labels: {', '.join(choices)}.\nLabel:"


def _generate_for_method(args: argparse.Namespace, model: Any, tokenizer: Any, controller: Any | None, prompt: str) -> str:
    method = args.method
    if method == "iga_gen":
        if controller is None:
            raise RuntimeError("IGA controller is missing")
        return generate_iga_greedy(model, tokenizer, controller, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    if method in {"self_refine_gen", "cove_gen"}:
        draft = generate_vanilla(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        if method == "self_refine_gen":
            refine_prompt = (
                f"Original prompt:\n{prompt}\n\nDraft answer:\n{draft}\n\n"
                "Identify unsupported factual claims, avoid overconfident speculation, and rewrite the answer as a concise final label or answer.\nFinal answer:"
            )
        else:
            refine_prompt = (
                f"Original prompt:\n{prompt}\n\nDraft answer:\n{draft}\n\n"
                "Write verification questions, answer them using only the prompt evidence, and then give the final label or answer.\nFinal answer:"
            )
        return generate_vanilla(model, tokenizer, refine_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    return generate_vanilla(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        extra_generate_kwargs=_method_generate_kwargs(method),
    )


def _sample_votes(args: argparse.Namespace, model: Any, tokenizer: Any, ex: EvalExample) -> tuple[list[int], list[str]]:
    assert ex.choices is not None
    votes: list[int] = []
    generations: list[str] = []
    prompt = _classification_prompt(ex.prompt, ex.choices)
    for _ in range(args.num_samples):
        text = generate_vanilla(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=max(args.temperature, 0.7),
            top_p=args.top_p,
            do_sample=True,
        )
        generations.append(text)
        pred = parse_choice_from_generation(text, ex.choices)
        if pred is not None:
            votes.append(pred)
    return votes, generations


def run_choice_eval(args: argparse.Namespace, model: Any, tokenizer: Any, controller: Any | None, cfg: dict[str, Any], model_id: str) -> list[dict[str, Any]]:
    examples = load_examples(args.data, limit=args.limit)
    dataset_sha = sha256_file(args.data)
    method_config = _method_config(args, cfg)
    rows: list[dict[str, Any]] = []
    for ex in examples:
        if not ex.choices or ex.correct_choice is None:
            continue
        start = time.perf_counter()
        pred: int
        conf: float
        probs: list[float]
        nlls: list[float] | None = None
        generations: list[str] | None = None
        semantic_entropy: float | None = None
        parse_success: bool | None = None
        active_controller = controller if args.method == "iga_mc" else None
        try:
            if args.method in {"vanilla_mc", "temperature_mc", "iga_mc"}:
                temp = args.temperature if args.method == "temperature_mc" else 1.0
                pred, conf, probs, nlls = choice_distribution(
                    model,
                    tokenizer,
                    ex.prompt,
                    ex.choices,
                    controller=active_controller,
                    temperature=temp,
                    length_norm=not args.no_length_norm,
                    max_length=args.max_length,
                )
                parse_success = None
            elif args.method in {"self_consistency_mc", "semantic_entropy_mc"}:
                votes, generations = _sample_votes(args, model, tokenizer, ex)
                parse_success = len(votes) > 0
                if votes:
                    if args.method == "semantic_entropy_mc":
                        pred, conf, probs, semantic_entropy = entropy_confidence_from_votes(votes, len(ex.choices))
                    else:
                        pred, conf, probs = vote_distribution(votes, len(ex.choices))
                else:
                    pred, conf, probs, nlls = choice_distribution(
                        model,
                        tokenizer,
                        ex.prompt,
                        ex.choices,
                        controller=None,
                        temperature=1.0,
                        length_norm=not args.no_length_norm,
                        max_length=args.max_length,
                    )
                    conf = min(conf, 0.5)
            else:
                raise ValueError(f"Unsupported choice method: {args.method}")
            error = None
        except Exception as exc:
            pred, conf, probs, nlls = -1, 0.0, [], None
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - start
        row = _base_row(args, ex, method_config, dataset_sha, model_id)
        row.update(
            {
                "evaluation_mode": "choice",
                "pred_choice": pred,
                "confidence": conf,
                "choice_probs": probs,
                "choice_nlls": nlls,
                "correct": pred == int(ex.correct_choice),
                "hallucination_proxy": float(pred != int(ex.correct_choice)),
                "latency_s": latency,
                "error": error,
                "parse_success": parse_success,
                "generations": generations,
                "semantic_entropy": semantic_entropy,
            }
        )
        if controller is not None and args.method == "iga_mc":
            row.update(controller.diagnostics())
        rows.append(row)
        if args.print_every and len(rows) % args.print_every == 0:
            print(f"[{args.method}] {len(rows)}/{len(examples)} examples")
    return rows


def run_generation_eval(args: argparse.Namespace, model: Any, tokenizer: Any, controller: Any | None, cfg: dict[str, Any], model_id: str) -> list[dict[str, Any]]:
    examples = load_examples(args.data, limit=args.limit)
    dataset_sha = sha256_file(args.data)
    method_config = _method_config(args, cfg)
    rows: list[dict[str, Any]] = []
    for ex in examples:
        start = time.perf_counter()
        gen_prompt = _classification_prompt(ex.prompt, ex.choices) if ex.choices else ex.prompt
        try:
            text = _generate_for_method(args, model, tokenizer, controller, gen_prompt)
            error = None
        except Exception as exc:
            text = f"[GENERATION_ERROR] {type(exc).__name__}: {exc}"
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - start
        gen_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
        pred_choice = parse_choice_from_generation(text, ex.choices) if ex.choices else None
        parse_success = pred_choice is not None
        confidence = 1.0 if parse_success else 0.0
        row = _base_row(args, ex, method_config, dataset_sha, model_id)
        row.update(
            {
                "evaluation_mode": "generate_classify" if ex.choices else "generate_free",
                "generation": text,
                "pred_choice": pred_choice,
                "confidence": confidence,
                "confidence_source": "parse_indicator",
                "correct": (pred_choice == int(ex.correct_choice)) if pred_choice is not None and ex.correct_choice is not None else None,
                "hallucination_proxy": float(pred_choice != int(ex.correct_choice)) if pred_choice is not None and ex.correct_choice is not None else None,
                "is_refusal": is_refusal(text),
                "generated_tokens": gen_tokens,
                "latency_s": latency,
                "parse_success": parse_success,
                "error": error,
            }
        )
        if controller is not None and args.method == "iga_gen":
            row.update(controller.diagnostics())
        rows.append(row)
        if args.print_every and len(rows) % args.print_every == 0:
            print(f"[{args.method}] {len(rows)}/{len(examples)} examples")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vanilla, IGA, calibration, uncertainty, and decoding baselines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="vanilla_mc")
    parser.add_argument("--iga_checkpoint", default=None)
    parser.add_argument("--lora_adapter_dir", default=None, help="Optional PEFT LoRA adapter to merge into the backbone before evaluation.")
    parser.add_argument("--mode", default="auto", choices=["auto", "choice", "generate"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_id", default="manual")
    parser.add_argument("--report_method", default=None, help="Optional method name to write into result rows, useful for ablations.")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--no_length_norm", action="store_true")
    parser.add_argument("--print_every", type=int, default=25)
    args = parser.parse_args()
    args.method = _canonical_method(args.method)
    if args.method not in CHOICE_METHODS | GEN_METHODS:
        raise ValueError(f"Unsupported method {args.method}. Supported: {sorted(CHOICE_METHODS | GEN_METHODS)}")

    cfg: dict[str, Any] = load_yaml(args.config) if args.config else {}
    model_id = args.model_id or cfg.get("model_id")
    if not model_id:
        raise ValueError("Provide --model_id or model_id in config")
    device = args.device or cfg.get("device", "auto")
    dtype = args.dtype or cfg.get("dtype", "auto")
    set_seed(args.seed)

    loaded = load_causal_lm(
        model_id=model_id,
        device=device,
        dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        revision=cfg.get("model_revision"),
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    if args.lora_adapter_dir:
        from peft import PeftModel
        print(f"[LoRA+IGA] Loading and merging frozen LoRA adapter for eval: {args.lora_adapter_dir}", flush=True)
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter_dir)
        model = peft_model.merge_and_unload()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    controller = None
    needs_iga = args.method in {"iga_mc", "iga_gen"}
    if needs_iga:
        controller = install_iga(model, _build_iga_config(model, cfg), device=loaded.device, dtype=torch.float32)
        _load_iga_state(model, args.iga_checkpoint)
        model.eval()

    if args.mode == "choice" or (args.mode == "auto" and args.method in CHOICE_METHODS):
        rows = run_choice_eval(args, model, tokenizer, controller, cfg, model_id)
    else:
        rows = run_generation_eval(args, model, tokenizer, controller, cfg, model_id)
    for row in rows:
        row["package_versions"] = package_versions()
    ensure_dir(Path(args.out).parent)
    write_jsonl(args.out, rows)
    print(json.dumps({"out": args.out, "n": len(rows), "method": args.method}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
