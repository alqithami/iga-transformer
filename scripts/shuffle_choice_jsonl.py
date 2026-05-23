#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def stable_seed(example_id: str, seed: int) -> int:
    s = f"{seed}::{example_id}".encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    changed = 0
    with inp.open() as fin, out.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            choices = list(r["choices"])
            old_correct = int(r["correct_choice"])
            old_target = choices[old_correct]
            idxs = list(range(len(choices)))
            rng = random.Random(stable_seed(str(r.get("id", n)), args.seed))
            rng.shuffle(idxs)
            new_choices = [choices[i] for i in idxs]
            new_correct = idxs.index(old_correct)
            if new_correct != old_correct:
                changed += 1
            r["choices"] = new_choices
            r["correct_choice"] = new_correct
            r["target"] = old_target
            r.setdefault("metadata", {})
            r["metadata"]["choice_shuffle_seed"] = args.seed
            r["metadata"]["old_correct_choice"] = old_correct
            r["metadata"]["new_correct_choice"] = new_correct
            fout.write(json.dumps(r) + "\n")
            n += 1
    print(json.dumps({"input": str(inp), "output": str(out), "n": n, "changed_correct_position": changed, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
