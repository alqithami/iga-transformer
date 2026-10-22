"""Read-only replay of IGA clean/background error partitions from saved logits.

No training, model loading, relabeling, network access, or outcome selection.
This portable publication utility is new; it is not the historical analysis
program whose hash appears in the October 21 audit. It recomputes point
estimates, not the historical bootstrap intervals or training/source audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Sequence

ARCHIVE_SHA256 = "1f8b058bf8983451c5d8c5d582d229a0aa9b1c1c0024710eccc3d10fd3a56938"
FAMILIES = {
    f"{method}_{label}": [f"{method}_{part}_s{seed}" for seed in (1, 2, 3)]
    for method in ("iga", "lora")
    for label, part in (("3k", "3k"), ("A", "12k"), ("B", "drawB"))
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def semantic(logits: Sequence[Sequence[float]]) -> list[float]:
    require(len(logits) == 6 and all(len(row) == 3 for row in logits),
            "Expected six semantic-format vectors with three class logits each")
    require(all(math.isfinite(x) for row in logits for x in row), "Nonfinite logit")
    probabilities = []
    for row in logits:
        maximum = max(row)
        weights = [math.exp(x - maximum) for x in row]
        total = sum(weights)
        probabilities.append([x / total for x in weights])
    return [sum(p[c] for p in probabilities) / 6 for c in range(3)]


def partition(gold: Sequence[int], clean: Sequence[int], context: Sequence[int]) -> dict[str, Any]:
    require(len(gold) == len(clean) == len(context) and len(gold) > 0, "Unpaired or empty predictions")
    require(all(type(x) is int and 0 <= x < 3 for seq in (gold, clean, context) for x in seq),
            "Invalid semantic class")
    cc = cw = wc = ww = changed_wrong = 0
    for target, left, right in zip(gold, clean, context):
        a, b = left == target, right == target
        cc += int(a and b)
        cw += int(a and not b)
        wc += int(not a and b)
        ww += int(not a and not b)
        changed_wrong += int(not a and not b and left != right)
    n = len(gold)
    require(cc + cw + wc + ww == n, "Transition partition failure")
    require((cw + ww) - (wc + ww) == cw - wc, "Error identity failure")
    return {
        "examples": n, "clean_correct": cc + cw, "context_correct": cc + wc,
        "stable_correct": cc, "new_errors": cw, "corrected_clean_errors": wc,
        "persistent_errors": ww, "clean_errors": wc + ww, "context_errors": cw + ww,
        "changed_wrong_label": changed_wrong, "net_extra_errors": cw - wc,
    }


def common_clean(gold: Sequence[int], iga_clean: Sequence[int], lora_clean: Sequence[int],
                 iga_context: Sequence[int], lora_context: Sequence[int]) -> dict[str, int]:
    sequences = (gold, iga_clean, lora_clean, iga_context, lora_context)
    require(len(gold) > 0 and all(len(x) == len(gold) for x in sequences), "Unpaired comparison")
    require(all(type(v) is int and 0 <= v < 3 for x in sequences for v in x), "Invalid semantic class")
    eligible = [i for i, y in enumerate(gold) if iga_clean[i] == lora_clean[i] == y]
    return {
        "common_clean_correct": len(eligible),
        "iga_new_errors": sum(iga_context[i] != gold[i] for i in eligible),
        "lora_new_errors": sum(lora_context[i] != gold[i] for i in eligible),
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def verify_payloads(archive: zipfile.ZipFile) -> int:
    names = archive.namelist()
    require(len(names) == len(set(names)), "Duplicate archive member")
    require(sum(item.file_size for item in archive.infolist()) < 2_000_000_000, "Archive too large")
    manifest = json.loads(archive.read("MANIFEST.json"))
    require(set(names) == set(manifest) | {"MANIFEST.json"}, "Archive membership mismatch")
    for name, expected in manifest.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
                "Unsafe archive member")
        value = hashlib.sha256()
        count = 0
        with archive.open(name) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
                count += len(block)
        require(count == expected["bytes"] and value.hexdigest() == expected["sha256"],
                "Corrupt payload: " + name)
    return len(manifest)


def analyze(path: Path) -> dict[str, Any]:
    require(sha256(path) == ARCHIVE_SHA256, "Unexpected archive: use the fixed completed replication export")
    models: dict[str, dict[int, list[int]]] = {}
    transitions: dict[str, dict[str, Any]] = {}
    reconstructed = 0
    with zipfile.ZipFile(path) as archive:
        payloads = verify_payloads(archive)
        source = [json.loads(line) for line in archive.read("inputs/fresh_confirmation.jsonl").splitlines() if line.strip()]
        source_by_id = {row["source_id"]: row for row in source}
        require(len(source) == len(source_by_id) == 800, "Expected 800 unique target examples")
        require(len({row["page_group"] for row in source}) == 400, "Expected 400 source pages")
        require(all(row["role"] == "fresh_confirmation" and row["original_source_split"] == "dev" for row in source),
                "Wrong cohort role or original split")
        ids = sorted(source_by_id)
        gold = [source_by_id[key]["gold"] for key in ids]
        require([gold.count(c) for c in (0, 1, 2)] == [400, 200, 200], "Wrong label counts")
        for name in (n for group in FAMILIES.values() for n in group):
            rows = [json.loads(line) for line in archive.read(f"evaluation/{name}/fresh_confirmation.jsonl").splitlines() if line.strip()]
            lookup: dict[tuple[str, int], int] = {}
            for row in rows:
                key = (row["source_id"], row["k"])
                require(key not in lookup and key[0] in source_by_id and key[1] in (0, 4, 8), "Duplicate/unexpected prediction")
                original = source_by_id[key[0]]
                require(all(row[field] == original[field] for field in ("pair_id", "page_group", "stratum", "gold", "role")),
                        "Prediction/source mismatch")
                require(row["mapping_sequence_evaluations"] == 6 and row["no_model_ensemble"] is True
                        and row["uses_answer_for_intervention"] is False and row["privileged_injection_mask"] is False,
                        "Unexpected scoring intervention")
                probability = semantic(row["canonical_logits"])
                require(len(row["probabilities"]) == 3 and all(math.isfinite(b) and abs(a - b) <= 2e-12
                        for a, b in zip(probability, row["probabilities"])), "Stored probability mismatch")
                prediction = max(range(3), key=probability.__getitem__)
                require(prediction == row["prediction"], "Stored decision mismatch")
                lookup[key] = prediction
            require(len(lookup) == 2400, "Incomplete model scoring")
            models[name] = {k: [lookup[(key, k)] for key in ids] for k in (0, 4, 8)}
            transitions[name] = {f"d{k}": partition(gold, models[name][0], models[name][k]) for k in (4, 8)}
            reconstructed += len(lookup)
    means: dict[str, Any] = {}
    for family, names in FAMILIES.items():
        means[family] = {}
        for k in (4, 8):
            items = [transitions[name][f"d{k}"] for name in names]
            average = {field: mean(item[field] for item in items) for field in items[0]}
            average["clean_accuracy"] = average["clean_correct"] / 800
            average["context_accuracy"] = average["context_correct"] / 800
            average["persistent_error_share"] = ratio(average["persistent_errors"], average["context_errors"])
            means[family][f"d{k}"] = average
    paired = {}
    for draw, part in (("3k", "3k"), ("A", "12k"), ("B", "drawB")):
        paired[draw] = {}
        for k in (4, 8):
            per_seed = [common_clean(gold, models[f"iga_{part}_s{s}"][0], models[f"lora_{part}_s{s}"][0],
                         models[f"iga_{part}_s{s}"][k], models[f"lora_{part}_s{s}"][k]) for s in (1, 2, 3)]
            denominator = sum(row["common_clean_correct"] for row in per_seed)
            paired[draw][f"d{k}"] = {
                "per_seed": per_seed,
                "model_example_denominator_not_unique_targets": denominator,
                "iga_error_rate": ratio(sum(row["iga_new_errors"] for row in per_seed), denominator),
                "lora_error_rate": ratio(sum(row["lora_new_errors"] for row in per_seed), denominator),
            }
    require(sha256(path) == ARCHIVE_SHA256, "Archive changed during reading")
    return {
        "scope": "Portable point-estimate reproduction; not a bootstrap, training, or source-history audit",
        "archive_sha256": ARCHIVE_SHA256, "payloads_verified": payloads,
        "confirmation_predictions_reconstructed": reconstructed,
        "transition_records": 18 * 800 * 2, "new_model_forwards": 0,
        "new_optimizer_steps": 0, "inherited_labels_unchanged": True,
        "models": transitions, "family_means": means, "common_clean": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists; preserve the earlier result")
    result = analyze(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("POINT_ESTIMATES_RECOMPUTED", args.output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print("AUDIT NOT COMPLETE: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
