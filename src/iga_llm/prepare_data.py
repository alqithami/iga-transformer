from __future__ import annotations

import argparse
from pathlib import Path

from .data import (
    dataset_manifest,
    deterministic_partition,
    prepare_fever,
    prepare_halueval,
    prepare_truthfulqa_mc,
    save_examples,
    synthetic_examples,
)
from .utils import json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark JSONL files for IGA evaluation/training.")
    parser.add_argument("--benchmark", required=True, choices=["synthetic", "truthfulqa_mc", "fever", "halueval"])
    parser.add_argument("--source_split", default=None, help="HF split to load, e.g. validation, train, labelled_dev, data")
    parser.add_argument("--halueval_config", default="qa", help="HaluEval subset/config: qa, general, dialogue, summarization, ...")
    parser.add_argument("--revision", default=None, help="Optional HF dataset revision/commit for reproducibility")
    parser.add_argument("--role", default="all", choices=["all", "train", "dev", "test"], help="Hash-partition role after loading")
    parser.add_argument("--split_seed", type=int, default=1)
    parser.add_argument("--train_frac", type=float, default=0.60)
    parser.add_argument("--dev_frac", type=float, default=0.20)
    parser.add_argument("--load_limit", type=int, default=None, help="Limit rows/examples while loading raw HF dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit after partitioning")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest_out", default=None)
    args = parser.parse_args()

    if args.benchmark == "synthetic":
        examples = synthetic_examples()
    elif args.benchmark == "truthfulqa_mc":
        examples = prepare_truthfulqa_mc(split=args.source_split or "validation", limit=args.load_limit, revision=args.revision)
    elif args.benchmark == "fever":
        examples = prepare_fever(split=args.source_split or "labelled_dev", limit=args.load_limit, revision=args.revision)
    elif args.benchmark == "halueval":
        examples = prepare_halueval(config_name=args.halueval_config, split=args.source_split or "data", limit=args.load_limit, revision=args.revision)
    else:  # pragma: no cover
        raise ValueError(args.benchmark)

    if args.benchmark != "synthetic" or args.role != "all":
        examples = deterministic_partition(examples, role=args.role, seed=args.split_seed, train_frac=args.train_frac, dev_frac=args.dev_frac)
    if args.limit is not None:
        examples = examples[: args.limit]

    out = Path(args.out)
    save_examples(out, examples)
    manifest = dataset_manifest(examples, name=args.benchmark, revision=args.revision)
    manifest.update(
        {
            "benchmark": args.benchmark,
            "source_split": args.source_split,
            "halueval_config": args.halueval_config if args.benchmark == "halueval" else None,
            "role": args.role,
            "split_seed": args.split_seed,
            "train_frac": args.train_frac,
            "dev_frac": args.dev_frac,
            "load_limit": args.load_limit,
            "limit": args.limit,
            "out": str(out),
        }
    )
    manifest_path = Path(args.manifest_out) if args.manifest_out else out.with_suffix(out.suffix + ".manifest.json")
    json_dump(manifest_path, manifest)
    print(f"Wrote {len(examples)} examples to {out}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
