from __future__ import annotations


def truthfulqa_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def fever_prompt(claim: str) -> str:
    return f"Claim: {claim}\nDecide whether the claim is supported, refuted, or not enough information.\nAnswer:"


def halueval_prompt(answer: str, question: str = "", context: str = "") -> str:
    parts = []
    if context:
        parts.append(f"Reference context: {context}")
    if question:
        parts.append(f"Question: {question}")
    parts.append(f"Answer to evaluate: {answer}")
    parts.append("Decide whether the answer is hallucinated or not hallucinated.\nDecision:")
    return "\n".join(parts)


def self_refine_prompt(original_prompt: str, draft_answer: str) -> str:
    return (
        f"Original prompt:\n{original_prompt}\n\nDraft answer:\n{draft_answer}\n\n"
        "Identify unsupported factual claims and rewrite the answer to be accurate and concise.\nRevised answer:"
    )


def chain_of_verification_prompt(original_prompt: str, draft_answer: str) -> str:
    return (
        f"Original prompt:\n{original_prompt}\n\nDraft answer:\n{draft_answer}\n\n"
        "Generate verification questions for the factual claims, answer them, and then provide a final verified answer.\nFinal answer:"
    )
