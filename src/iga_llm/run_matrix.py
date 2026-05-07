from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .ablations import ablation_config, default_ablation_names, get_ablation_by_name
from .utils import ensure_dir, json_dump, load_yaml, package_versions, read_jsonl, sha256_file, sha256_json, write_jsonl, write_yaml


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _cat_jsonl(paths: list[Path], out: Path) -> None:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for row in read_jsonl(path):
            key = (str(row.get("benchmark")), str(row.get("id")))
            if key in seen:
                raise ValueError(f"Duplicate example in mixed file: {key}")
            seen.add(key)
            rows.append(row)
    write_jsonl(out, rows)


def _prepare_all(args: argparse.Namespace, data_dir: Path) -> dict[str, list[Path]]:
    ensure_dir(data_dir)
    split_seed = str(args.split_seed)
    load_limit = args.load_limit
    # If not explicitly provided, load enough examples before hash splitting.
    if load_limit is None and args.limit_train:
        load_limit = max(args.limit_train * 4, args.limit_eval * 4, 100)
    prepared: dict[str, list[Path]] = {"train": [], "dev": [], "eval": []}

    def prep(name: str, benchmark: str, role: str, out: Path, source_split: str | None, limit: int | None, halueval_config: str = "qa") -> None:
        cmd = [
            sys.executable,
            "-m",
            "iga_llm.prepare_data",
            "--benchmark",
            benchmark,
            "--role",
            role,
            "--split_seed",
            split_seed,
            "--out",
            str(out),
        ]
        if source_split:
            cmd += ["--source_split", source_split]
        if benchmark == "halueval":
            cmd += ["--halueval_config", halueval_config]
        if load_limit:
            cmd += ["--load_limit", str(load_limit)]
        if limit:
            cmd += ["--limit", str(limit)]
        if args.dataset_revision:
            cmd += ["--revision", args.dataset_revision]
        _run(cmd)

    # TruthfulQA and HaluEval have a single split, so we hash-partition it.
    prep("truthfulqa_train", "truthfulqa_mc", "train", data_dir / "truthfulqa_train.jsonl", "validation", args.limit_train)
    prep("truthfulqa_dev", "truthfulqa_mc", "dev", data_dir / "truthfulqa_dev.jsonl", "validation", args.limit_dev)
    prep("truthfulqa_eval", "truthfulqa_mc", "test", data_dir / "truthfulqa_eval.jsonl", "validation", args.limit_eval)
    prep("halueval_train", "halueval", "train", data_dir / "halueval_train.jsonl", "data", args.limit_train, args.halueval_config)
    prep("halueval_dev", "halueval", "dev", data_dir / "halueval_dev.jsonl", "data", args.limit_dev, args.halueval_config)
    prep("halueval_eval", "halueval", "test", data_dir / "halueval_eval.jsonl", "data", args.limit_eval, args.halueval_config)
    # FEVER: train/dev come from train, final eval from labelled_dev to avoid leakage.
    prep("fever_train", "fever", "train", data_dir / "fever_train.jsonl", "train", args.limit_train)
    prep("fever_dev", "fever", "dev", data_dir / "fever_dev.jsonl", "train", args.limit_dev)
    prep("fever_eval", "fever", "all", data_dir / "fever_eval.jsonl", "labelled_dev", args.limit_eval)

    prepared["train"] = [data_dir / "truthfulqa_train.jsonl", data_dir / "fever_train.jsonl", data_dir / "halueval_train.jsonl"]
    prepared["dev"] = [data_dir / "truthfulqa_dev.jsonl", data_dir / "fever_dev.jsonl", data_dir / "halueval_dev.jsonl"]
    prepared["eval"] = [data_dir / "truthfulqa_eval.jsonl", data_dir / "fever_eval.jsonl", data_dir / "halueval_eval.jsonl"]
    _cat_jsonl(prepared["train"], data_dir / "calibration_train_mix.jsonl")
    _cat_jsonl(prepared["dev"], data_dir / "calibration_dev_mix.jsonl")
    return prepared


def _config_name(path: Path) -> str:
    return path.stem.replace("/", "_")


def _train_one(config_path: Path, train_jsonl: Path, dev_jsonl: Path, out_dir: Path, seed: int, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "iga_llm.train",
        "--config",
        str(config_path),
        "--train_jsonl",
        str(train_jsonl),
        "--dev_jsonl",
        str(dev_jsonl),
        "--output_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
    ]
    if args.train_limit_override:
        cmd += ["--limit", str(args.train_limit_override)]
    _run(cmd)
    checkpoint = out_dir / "iga_modules.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Training finished but checkpoint is missing: {checkpoint}")


def _evaluate_one(config_path: Path, data_path: Path, out_path: Path, method: str, seed: int, run_id: str, args: argparse.Namespace, checkpoint: Path | None = None, report_method: str | None = None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "iga_llm.evaluate",
        "--config",
        str(config_path),
        "--data",
        str(data_path),
        "--out",
        str(out_path),
        "--method",
        method,
        "--seed",
        str(seed),
        "--run_id",
        run_id,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--num_samples",
        str(args.num_samples),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
    ]
    if args.eval_limit_override:
        cmd += ["--limit", str(args.eval_limit_override)]
    if checkpoint is not None:
        cmd += ["--iga_checkpoint", str(checkpoint)]
    if report_method is not None:
        cmd += ["--report_method", report_method]
    _run(cmd)


def _latency_one(config_path: Path, out_path: Path, method: str, seed: int, run_id: str, args: argparse.Namespace, checkpoint: Path | None = None, prompt_tokens: int = 512) -> None:
    cmd = [
        sys.executable,
        "-m",
        "iga_llm.benchmark_latency",
        "--config",
        str(config_path),
        "--method",
        method,
        "--out",
        str(out_path),
        "--seed",
        str(seed),
        "--run_id",
        run_id,
        "--prompt_tokens",
        str(prompt_tokens),
        "--max_new_tokens",
        str(args.latency_new_tokens),
        "--warmup",
        str(args.latency_warmup),
        "--runs",
        str(args.latency_runs),
    ]
    if checkpoint is not None:
        cmd += ["--iga_checkpoint", str(checkpoint)]
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full reviewer-grade IGA experiment matrix.")
    parser.add_argument("--configs", nargs="+", default=["configs/llama3_8b_iga.yaml"], help="One or more model YAML configs.")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--limit_train", type=int, default=1000)
    parser.add_argument("--limit_dev", type=int, default=300)
    parser.add_argument("--limit_eval", type=int, default=300)
    parser.add_argument("--load_limit", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--halueval_config", default="qa")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_limit_override", type=int, default=None)
    parser.add_argument("--eval_limit_override", type=int, default=None)
    parser.add_argument("--choice_methods", default="vanilla_mc,temperature_mc,self_consistency_mc,semantic_entropy_mc,iga_mc")
    parser.add_argument("--generation_methods", default="vanilla_gen,contrastive_gen,dola_gen,self_refine_gen,cove_gen,iga_gen")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_latency", action="store_true")
    parser.add_argument("--run_ablations", action="store_true")
    parser.add_argument("--ablation_names", default=",".join(default_ablation_names()))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--latency_prompt_tokens", default="256,512,1024")
    parser.add_argument("--latency_new_tokens", type=int, default=32)
    parser.add_argument("--latency_warmup", type=int, default=1)
    parser.add_argument("--latency_runs", type=int, default=3)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(args.out_dir or f"results/full_{timestamp}")
    data_dir = ensure_dir(root / "data")
    run_dir = ensure_dir(root / "runs")
    pred_dir = ensure_dir(root / "predictions")
    latency_dir = ensure_dir(root / "latency")
    cfg_dir = ensure_dir(root / "resolved_configs")
    aggregate_dir = ensure_dir(root / "aggregate")

    if not args.skip_prepare:
        prepared = _prepare_all(args, data_dir)
    else:
        prepared = {
            "train": [data_dir / "truthfulqa_train.jsonl", data_dir / "fever_train.jsonl", data_dir / "halueval_train.jsonl"],
            "dev": [data_dir / "truthfulqa_dev.jsonl", data_dir / "fever_dev.jsonl", data_dir / "halueval_dev.jsonl"],
            "eval": [data_dir / "truthfulqa_eval.jsonl", data_dir / "fever_eval.jsonl", data_dir / "halueval_eval.jsonl"],
        }
    train_mix = data_dir / "calibration_train_mix.jsonl"
    dev_mix = data_dir / "calibration_dev_mix.jsonl"
    if not train_mix.exists() and not args.skip_prepare:
        _cat_jsonl(prepared["train"], train_mix)
    if not dev_mix.exists() and not args.skip_prepare:
        _cat_jsonl(prepared["dev"], dev_mix)

    prediction_files: list[str] = []
    latency_files: list[str] = []
    manifest: dict[str, Any] = {
        "root": str(root),
        "args": vars(args),
        "package_versions": package_versions(),
        "data_files": {},
        "runs": [],
    }
    for p in list(data_dir.glob("*.jsonl")):
        manifest["data_files"][p.name] = {"path": str(p), "sha256": sha256_file(p), "n": len(read_jsonl(p))}

    choice_methods = [m.strip() for m in args.choice_methods.split(",") if m.strip()]
    generation_methods = [m.strip() for m in args.generation_methods.split(",") if m.strip()] if not args.skip_generation else []
    eval_files = prepared["eval"]
    prompt_lengths = [int(x) for x in args.latency_prompt_tokens.split(",") if x.strip()]

    for config_in in args.configs:
        base_config_path = Path(config_in)
        base_cfg = load_yaml(base_config_path)
        model_tag = _config_name(base_config_path)
        base_resolved = cfg_dir / f"{model_tag}.yaml"
        write_yaml(base_resolved, base_cfg)
        for seed in args.seeds:
            run_id = f"{model_tag}_seed{seed}"
            ckpt_dir = run_dir / run_id / "full_iga"
            checkpoint = ckpt_dir / "iga_modules.pt"
            if not args.skip_training:
                _train_one(base_resolved, train_mix, dev_mix, ckpt_dir, seed, args)
            elif not checkpoint.exists():
                checkpoint = None  # allow vanilla-only or inference-only runs
            for data_path in eval_files:
                bench_tag = data_path.stem.replace("_eval", "")
                for method in choice_methods:
                    ckpt = checkpoint if method == "iga_mc" else None
                    out = pred_dir / f"{run_id}_{bench_tag}_{method}.jsonl"
                    _evaluate_one(base_resolved, data_path, out, method, seed, run_id, args, checkpoint=ckpt)
                    prediction_files.append(str(out))
                for method in generation_methods:
                    ckpt = checkpoint if method == "iga_gen" else None
                    out = pred_dir / f"{run_id}_{bench_tag}_{method}.jsonl"
                    _evaluate_one(base_resolved, data_path, out, method, seed, run_id, args, checkpoint=ckpt)
                    prediction_files.append(str(out))
            if not args.skip_latency:
                for prompt_tokens in prompt_lengths:
                    out_v = latency_dir / f"{run_id}_vanilla_{prompt_tokens}.jsonl"
                    _latency_one(base_resolved, out_v, "vanilla", seed, run_id, args, prompt_tokens=prompt_tokens)
                    latency_files.append(str(out_v))
                    out_i = latency_dir / f"{run_id}_iga_{prompt_tokens}.jsonl"
                    _latency_one(base_resolved, out_i, "iga", seed, run_id, args, checkpoint=checkpoint, prompt_tokens=prompt_tokens)
                    latency_files.append(str(out_i))
            manifest["runs"].append({"model_config": str(base_config_path), "seed": seed, "run_id": run_id, "checkpoint": str(checkpoint) if checkpoint else None})

            if args.run_ablations:
                for ab_name in [x.strip() for x in args.ablation_names.split(",") if x.strip()]:
                    spec = get_ablation_by_name(ab_name)
                    if spec is None:
                        raise ValueError(f"Unknown ablation: {ab_name}")
                    ab_cfg = ablation_config(base_cfg, spec)
                    ab_cfg_path = cfg_dir / f"{model_tag}_{ab_name}.yaml"
                    write_yaml(ab_cfg_path, ab_cfg)
                    ab_run_id = f"{model_tag}_seed{seed}_{ab_name}"
                    ab_ckpt_dir = run_dir / run_id / ab_name
                    ab_ckpt = ab_ckpt_dir / "iga_modules.pt" if spec.train else None
                    # Reuse full checkpoint for ablate_full_iga to avoid duplicate training.
                    if ab_name == "ablate_full_iga":
                        ab_ckpt = checkpoint
                    elif spec.train and not args.skip_training:
                        _train_one(ab_cfg_path, train_mix, dev_mix, ab_ckpt_dir, seed, args)
                    for data_path in eval_files:
                        bench_tag = data_path.stem.replace("_eval", "")
                        out = pred_dir / f"{ab_run_id}_{bench_tag}.jsonl"
                        _evaluate_one(ab_cfg_path, data_path, out, "iga_mc", seed, ab_run_id, args, checkpoint=ab_ckpt, report_method=ab_name)
                        prediction_files.append(str(out))

    json_dump(root / "matrix_manifest.json", manifest)
    _run([
        sys.executable,
        "-m",
        "iga_llm.report",
        "--out_dir",
        str(aggregate_dir),
        "--predictions",
        *prediction_files,
        "--latency",
        *latency_files,
    ])
    print(f"\nComplete. Results root: {root}")
    print(f"Aggregate tables: {aggregate_dir}")


if __name__ == "__main__":
    main()
