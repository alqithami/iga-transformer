#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def softmax_neg(vals):
    xs = [-float(v) for v in vals]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def choice_nll(model, tokenizer, prompt: str, choice: str, device: torch.device) -> float:
    answer_text = " " + str(choice).strip()
    full_text = str(prompt) + answer_text
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
    enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = input_ids.clone()
    labels[:, :len(prompt_ids)] = -100
    if (labels != -100).sum().item() == 0:
        labels[:] = -100
        labels[:, -1] = input_ids[:, -1]
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
    val = float(loss.detach().cpu())
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"Bad NLL for choice={choice!r}: {val}")
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--method", default="vanilla_fixed_mc")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="eager")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = get_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=False,
    )
    model.eval()
    device = next(model.parameters()).device
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.data) as fin, open(out_path, "w") as fout:
        for line in fin:
            ex = json.loads(line)
            prompt = ex["prompt"]
            choices = ex["choices"]
            gold = int(ex["correct_choice"])
            t0 = time.perf_counter()
            nlls = [choice_nll(model, tokenizer, prompt, c, device) for c in choices]
            probs = softmax_neg(nlls)
            pred = max(range(len(probs)), key=lambda i: probs[i])
            latency = time.perf_counter() - t0
            row = dict(ex)
            row.update({
                "model_id": args.model_id,
                "method": args.method,
                "evaluation_mode": "choice",
                "run_id": args.run_id,
                "seed": args.seed,
                "pred_choice": pred,
                "prediction": choices[pred],
                "correct": bool(pred == gold),
                "confidence": float(probs[pred]),
                "choice_nlls": [float(x) for x in nlls],
                "choice_probs": [float(x) for x in probs],
                "latency_s": float(latency),
                "refusal": False,
                "parse_success": True,
            })
            fout.write(json.dumps(row) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"[{args.method}] {n} examples", flush=True)
    print(json.dumps({"method": args.method, "n": n, "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
