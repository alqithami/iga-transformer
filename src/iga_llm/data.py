from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import read_jsonl, sha256_json, write_jsonl


@dataclass
class EvalExample:
    id: str
    benchmark: str
    prompt: str
    choices: list[str] | None = None
    correct_choice: int | None = None
    target: str | None = None
    split: str = "unknown"
    source_dataset: str | None = None
    source_config: str | None = None
    source_split: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EvalExample":
        return cls(
            id=str(row.get("id", "")),
            benchmark=str(row.get("benchmark", "unknown")),
            prompt=str(row["prompt"]),
            choices=list(row["choices"]) if row.get("choices") is not None else None,
            correct_choice=int(row["correct_choice"]) if row.get("correct_choice") is not None else None,
            target=str(row["target"]) if row.get("target") is not None else None,
            split=str(row.get("split", "unknown")),
            source_dataset=row.get("source_dataset"),
            source_config=row.get("source_config"),
            source_split=row.get("source_split"),
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if row["choices"] is None:
            row.pop("choices")
        if row["correct_choice"] is None:
            row.pop("correct_choice")
        if row["target"] is None:
            row.pop("target")
        if row["source_dataset"] is None:
            row.pop("source_dataset")
        if row["source_config"] is None:
            row.pop("source_config")
        if row["source_split"] is None:
            row.pop("source_split")
        row["metadata"] = row.get("metadata") or {}
        return row


def load_examples(path: str | Path, limit: int | None = None) -> list[EvalExample]:
    rows = read_jsonl(path)
    examples = [EvalExample.from_dict(row) for row in rows]
    return examples[:limit] if limit else examples


def save_examples(path: str | Path, examples: Iterable[EvalExample]) -> None:
    write_jsonl(path, [ex.to_dict() for ex in examples])


def _role_score(ex: EvalExample, seed: int) -> float:
    key = f"{seed}|{ex.benchmark}|{ex.id}"
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return value / float(16**16 - 1)


def deterministic_partition(
    examples: list[EvalExample],
    role: str,
    seed: int = 1,
    train_frac: float = 0.60,
    dev_frac: float = 0.20,
) -> list[EvalExample]:
    """Hash-based train/dev/test split that does not depend on example order."""
    if role not in {"train", "dev", "test", "all"}:
        raise ValueError(f"Unknown role: {role}")
    out: list[EvalExample] = []
    for ex in examples:
        score = _role_score(ex, seed)
        if score < train_frac:
            split = "train"
        elif score < train_frac + dev_frac:
            split = "dev"
        else:
            split = "test"
        if role == "all" or role == split:
            row = EvalExample.from_dict(ex.to_dict())
            row.split = split
            out.append(row)
    return out


def dataset_manifest(examples: list[EvalExample], *, name: str, revision: str | None = None) -> dict[str, Any]:
    ids = [ex.id for ex in examples]
    return {
        "name": name,
        "revision": revision,
        "n": len(examples),
        "benchmark_counts": {b: sum(1 for ex in examples if ex.benchmark == b) for b in sorted({ex.benchmark for ex in examples})},
        "split_counts": {s: sum(1 for ex in examples if ex.split == s) for s in sorted({ex.split for ex in examples})},
        "ids_sha256": sha256_json(ids),
        "rows_sha256": sha256_json([ex.to_dict() for ex in examples]),
    }


def synthetic_examples() -> list[EvalExample]:
    return [
        EvalExample(
            id="synthetic-0",
            benchmark="SyntheticTruth",
            split="test",
            source_dataset="synthetic",
            prompt="Question: What is the capital of France?\nAnswer:",
            choices=["Paris", "Berlin", "Madrid"],
            correct_choice=0,
            target="Paris",
            metadata={"category": "geography"},
        ),
        EvalExample(
            id="synthetic-1",
            benchmark="SyntheticTruth",
            split="test",
            source_dataset="synthetic",
            prompt="Claim: The Moon is made of green cheese.\nDecide whether the claim is supported, refuted, or not enough information.\nAnswer:",
            choices=["supported", "refuted", "not enough information"],
            correct_choice=1,
            target="refuted",
            metadata={"category": "fact_checking"},
        ),
        EvalExample(
            id="synthetic-2",
            benchmark="SyntheticTruth",
            split="test",
            source_dataset="synthetic",
            prompt="Reference context: Water freezes at 0 degrees Celsius at standard pressure.\nQuestion: At standard pressure, does water freeze at 0 degrees Celsius?\nCandidate answer: Yes.\nDecide whether the answer is hallucinated or not hallucinated.\nDecision:",
            choices=["not hallucinated", "hallucinated"],
            correct_choice=0,
            target="not hallucinated",
            metadata={"category": "halueval_like"},
        ),
        EvalExample(
            id="synthetic-3",
            benchmark="SyntheticTruth",
            split="train",
            source_dataset="synthetic",
            prompt="Question: What planet is known as the Red Planet?\nAnswer:",
            choices=["Mars", "Venus", "Jupiter"],
            correct_choice=0,
            target="Mars",
            metadata={"category": "geography"},
        ),
    ]


def prepare_truthfulqa_mc(split: str = "validation", limit: int | None = None, revision: str | None = None) -> list[EvalExample]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split=split, revision=revision)
    examples: list[EvalExample] = []
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        question = row["question"]
        targets = row["mc1_targets"]
        choices = list(targets["choices"])
        labels = list(targets["labels"])
        try:
            correct_choice = labels.index(1)
        except ValueError:
            continue
        examples.append(
            EvalExample(
                id=f"truthfulqa-mc1-{idx}",
                benchmark="TruthfulQA-MC1",
                split="raw",
                source_dataset="truthful_qa",
                source_config="multiple_choice",
                source_split=split,
                prompt=f"Question: {question}\nAnswer:",
                choices=choices,
                correct_choice=correct_choice,
                target=choices[correct_choice],
                metadata={"category": row.get("category"), "source_index": idx},
            )
        )
    return examples


# FEVER is intentionally loaded from the original JSONL files instead of
# load_dataset("fever", "v1.0", ...). Recent Hugging Face Datasets releases reject
# legacy Hub dataset scripts, which breaks the old FEVER loader with:
# RuntimeError: Dataset scripts are no longer supported, but found fever.py
_FEVER_V1_JSONL_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "labelled_dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "paper_dev": "https://fever.ai/download/fever/paper_dev.jsonl",
    # These are intentionally not used by the default pipeline because they may
    # be unlabelled or not suitable for final held-out scoring. They are kept for
    # explicit user experiments. Rows without labels are skipped below.
    "unlabelled_dev": "https://fever.ai/download/fever/shared_task_dev_public.jsonl",
    "unlabelled_test": "https://fever.ai/download/fever/shared_task_test.jsonl",
    "paper_test": "https://fever.ai/download/fever/paper_test.jsonl",
}


def _load_fever_v1_jsonl(split: str):
    from datasets import load_dataset

    if split not in _FEVER_V1_JSONL_URLS:
        known = ", ".join(sorted(_FEVER_V1_JSONL_URLS))
        raise ValueError(f"Unknown FEVER v1.0 split '{split}'. Known splits: {known}")
    url = _FEVER_V1_JSONL_URLS[split]
    return load_dataset("json", data_files={split: url}, split=split)


def prepare_fever(split: str = "labelled_dev", limit: int | None = None, revision: str | None = None) -> list[EvalExample]:
    # revision is accepted for a uniform CLI, but FEVER v1.0 is loaded directly
    # from fever.ai JSONL files to avoid legacy HF dataset-script execution.
    ds = _load_fever_v1_jsonl(split)
    label_map = {"SUPPORTS": "supported", "REFUTES": "refuted", "NOT ENOUGH INFO": "not enough information"}
    choices = ["supported", "refuted", "not enough information"]
    examples: list[EvalExample] = []
    for idx, row in enumerate(ds):
        if limit is not None and len(examples) >= limit:
            break
        label = label_map.get(str(row.get("label", "")).upper())
        if label is None:
            continue
        claim = str(row.get("claim", "")).strip()
        if not claim:
            continue
        correct_choice = choices.index(label)
        examples.append(
            EvalExample(
                id=f"fever-v1.0-{split}-{row.get('id', idx)}",
                benchmark="FEVER",
                split="raw",
                source_dataset="fever.ai/fever-v1.0-jsonl",
                source_config="v1.0",
                source_split=split,
                prompt=(
                    f"Claim: {claim}\n"
                    "Decide whether the claim is supported, refuted, or not enough information.\nAnswer:"
                ),
                choices=choices,
                correct_choice=correct_choice,
                target=label,
                metadata={
                    "original_id": row.get("id"),
                    "source_index": idx,
                    "source_url": _FEVER_V1_JSONL_URLS[split],
                    "loader_note": "Loaded via datasets json builder to avoid legacy HF dataset scripts.",
                    "revision_arg_ignored": revision,
                },
            )
        )
    return examples


def _truthy_hallucination_label(value: Any) -> int | None:
    label_text = str(value).strip().lower()
    if label_text in {"1", "true", "yes", "hallucinated", "hallucination", "fail"}:
        return 1
    if label_text in {"0", "false", "no", "not hallucinated", "non-hallucinated", "pass"}:
        return 0
    return None


def prepare_halueval(
    config_name: str = "qa",
    split: str = "data",
    limit: int | None = None,
    revision: str | None = None,
) -> list[EvalExample]:
    """Prepare HaluEval as a binary hallucination-classification task.

    The QA subset has right_answer and hallucinated_answer fields. We emit both
    a non-hallucinated and a hallucinated candidate, creating balanced binary
    examples. For the general/dialogue/summarization subsets, we use available
    label fields defensively.
    """
    from datasets import load_dataset

    ds_dict = load_dataset("pminervini/HaluEval", config_name, revision=revision)
    ds = ds_dict[split] if split in ds_dict else ds_dict[next(iter(ds_dict.keys()))]
    examples: list[EvalExample] = []
    choices = ["not hallucinated", "hallucinated"]
    for idx, row in enumerate(ds):
        if limit is not None and len(examples) >= limit:
            break
        rowd = dict(row)
        knowledge = rowd.get("knowledge") or rowd.get("context") or rowd.get("document") or ""
        question = rowd.get("question") or rowd.get("query") or rowd.get("user_query") or rowd.get("dialogue_history") or ""
        # QA/dialogue/summarization paired fields.
        pairs: list[tuple[str, int, str]] = []
        for good_key, bad_key in [
            ("right_answer", "hallucinated_answer"),
            ("right_response", "hallucinated_response"),
            ("right_summary", "hallucinated_summary"),
        ]:
            if rowd.get(good_key):
                pairs.append((str(rowd[good_key]), 0, good_key))
            if rowd.get(bad_key):
                pairs.append((str(rowd[bad_key]), 1, bad_key))
        if not pairs:
            answer = rowd.get("answer") or rowd.get("response") or rowd.get("model_response") or rowd.get("chatgpt_response") or rowd.get("generation")
            raw_label = rowd.get("hallucination") or rowd.get("label") or rowd.get("is_hallucinated") or rowd.get("hallucinated") or rowd.get("hallucination_label")
            mapped = _truthy_hallucination_label(raw_label)
            if answer is not None and mapped is not None:
                pairs.append((str(answer), mapped, "labeled_answer"))
        for pair_idx, (answer, correct_choice, field_name) in enumerate(pairs):
            prompt = ""
            if knowledge:
                prompt += f"Reference context: {knowledge}\n"
            if question:
                prompt += f"Question: {question}\n"
            prompt += f"Candidate answer: {answer}\nDecide whether the answer is hallucinated or not hallucinated.\nDecision:"
            examples.append(
                EvalExample(
                    id=f"halueval-{config_name}-{split}-{idx}-{pair_idx}",
                    benchmark="HaluEval-QA" if config_name == "qa" else f"HaluEval-{config_name}",
                    split="raw",
                    source_dataset="pminervini/HaluEval",
                    source_config=config_name,
                    source_split=split,
                    prompt=prompt,
                    choices=choices,
                    correct_choice=correct_choice,
                    target=choices[correct_choice],
                    metadata={"source_index": idx, "answer_field": field_name},
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples
