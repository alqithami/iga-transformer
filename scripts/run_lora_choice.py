#!/usr/bin/env python
from __future__ import annotations

import argparse, json, math, os, random, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def load_base(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    model_id = cfg["model_id"]
    dtype = get_dtype(cfg.get("dtype", "auto"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=bool(cfg.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        attn_implementation=cfg.get("attn_implementation", None),
    )
    return cfg, model, tokenizer


def encode_choice_batch(tokenizer, prompt: str, choices: list[str], max_length: int, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    seqs, labels = [], []
    for c in choices:
        c_text = " " + str(c)
        choice_ids = tokenizer(c_text, add_special_tokens=False).input_ids
        if not choice_ids:
            choice_ids = [tokenizer.eos_token_id]
        ids = prompt_ids + choice_ids
        lab = [-100] * len(prompt_ids) + choice_ids
        if len(ids) > max_length:
            # Keep the answer tokens and truncate the left side of prompt if needed.
            ids = ids[-max_length:]
            lab = lab[-max_length:]
            # Ensure at least one label remains valid.
            if all(x == -100 for x in lab):
                lab[-1] = ids[-1]
        seqs.append(ids)
        labels.append(lab)
    max_len = max(len(x) for x in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids, attn, lab_pad = [], [], []
    for ids, lab in zip(seqs, labels):
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        lab_pad.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
        "labels": torch.tensor(lab_pad, dtype=torch.long, device=device),
    }


def choice_nlls(model, tokenizer, ex: dict[str, Any], max_length: int, length_norm: bool = True):
    device = next(model.parameters()).device
    batch = encode_choice_batch(tokenizer, ex["prompt"], ex["choices"], max_length, device)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()
    vocab = logits.shape[-1]
    loss_tok = F.cross_entropy(logits.view(-1, vocab).float(), labels.view(-1), reduction="none", ignore_index=-100).view(labels.shape)
    mask = (labels != -100).float()
    sums = (loss_tok * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    if length_norm:
        return sums / counts
    return sums


def train(args):
    set_seed(args.seed)
    cfg, model, tokenizer = load_base(args.config)
    model.train()
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if args.max_trainable and trainable > args.max_trainable:
        raise SystemExit(f"Too many trainable parameters: {trainable} > {args.max_trainable}")
    data = load_jsonl(args.train_jsonl)
    dev = load_jsonl(args.dev_jsonl) if args.dev_jsonl else []
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    start = time.time()
    for epoch in range(args.epochs):
        random.shuffle(data)
        for ex in data:
            step += 1
            nlls = choice_nlls(model, tokenizer, ex, args.max_length, args.length_norm)
            target = torch.tensor([int(ex["correct_choice"])], device=nlls.device)
            loss = F.cross_entropy((-nlls).unsqueeze(0), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step()
            if step % args.log_every == 0:
                print(json.dumps({"event":"train", "step":step, "loss":float(loss.detach().cpu())}), flush=True)
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    summary = {
        "model_id": cfg["model_id"],
        "seed": args.seed,
        "epochs": args.epochs,
        "steps": step,
        "train_examples": len(data),
        "dev_examples": len(dev),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "elapsed_s": time.time() - start,
        "adapter_dir": str(adapter_dir),
        "cuda_available": torch.cuda.is_available(),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(args):
    set_seed(args.seed)
    cfg, model, tokenizer = load_base(args.config)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    data = load_jsonl(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f, torch.no_grad():
        for idx, ex in enumerate(data, start=1):
            t0 = time.time()
            nlls = choice_nlls(model, tokenizer, ex, args.max_length, args.length_norm)
            probs = torch.softmax(-nlls, dim=0)
            pred = int(torch.argmax(probs).item())
            correct_choice = int(ex["correct_choice"])
            row = dict(ex)
            row.update({
                "model_id": cfg["model_id"],
                "method": args.report_method,
                "evaluation_mode": "choice",
                "run_id": args.run_id,
                "seed": args.seed,
                "pred_choice": pred,
                "prediction": ex["choices"][pred],
                "correct": bool(pred == correct_choice),
                "confidence": float(probs[pred].detach().cpu().item()),
                "choice_nlls": [float(x) for x in nlls.detach().cpu().tolist()],
                "choice_probs": [float(x) for x in probs.detach().cpu().tolist()],
                "latency_s": time.time() - t0,
                "refusal": False,
                "parse_success": True,
            })
            f.write(json.dumps(row) + "\n")
            if idx % 25 == 0:
                print(f"[{args.report_method}] {idx}/{len(data)} examples", flush=True)
    print(json.dumps({"method":args.report_method, "n":len(data), "out":str(out_path)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--config", required=True)
    tr.add_argument("--train_jsonl", required=True)
    tr.add_argument("--dev_jsonl")
    tr.add_argument("--output_dir", required=True)
    tr.add_argument("--seed", type=int, default=1)
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight_decay", type=float, default=0.0)
    tr.add_argument("--max_grad_norm", type=float, default=1.0)
    tr.add_argument("--max_length", type=int, default=768)
    tr.add_argument("--lora_r", type=int, default=2)
    tr.add_argument("--lora_alpha", type=int, default=4)
    tr.add_argument("--lora_dropout", type=float, default=0.0)
    tr.add_argument("--target_modules", default="q_proj,v_proj,o_proj")
    tr.add_argument("--length_norm", action="store_true", default=True)
    tr.add_argument("--max_trainable", type=int, default=2_000_000)
    tr.add_argument("--log_every", type=int, default=100)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--config", required=True)
    ev.add_argument("--adapter_dir", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--seed", type=int, default=1)
    ev.add_argument("--run_id", required=True)
    ev.add_argument("--report_method", default="lora_mc")
    ev.add_argument("--max_length", type=int, default=768)
    ev.add_argument("--length_norm", action="store_true", default=True)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        evaluate(args)

if __name__ == "__main__":
    main()
