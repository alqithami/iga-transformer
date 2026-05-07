"""Additional baseline methods for IGA evaluation.

Implements:
- Semantic entropy (uncertainty over meanings)
- SelfCheckGPT-style consistency checking
- Self-Refine (iterative refinement)
- Chain-of-Verification
- Inference-Time Intervention (ITI) - simplified version
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from .uncertainty import normalized_token_entropy


class SemanticEntropyUncertainty:
    """Compute semantic entropy (uncertainty over meanings).
    
    Samples multiple responses and clusters by semantic similarity,
    then computes entropy over semantic clusters.
    """

    def __init__(self, num_samples: int = 5, similarity_threshold: float = 0.8):
        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold

    def compute_uncertainty(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
    ) -> float:
        """Compute semantic entropy from multiple samples."""
        samples = []
        for _ in range(self.num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                )
            decoded = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            samples.append(decoded.strip())

        # Simple semantic similarity: character overlap ratio
        similarities = self._compute_pairwise_similarity(samples)

        # Cluster samples: treat as same semantic meaning if similar
        clusters = self._cluster_by_similarity(samples, similarities)

        # Entropy over clusters
        if len(clusters) == 0:
            return 0.0
        cluster_sizes = [len(c) for c in clusters]
        probs = [size / sum(cluster_sizes) for size in cluster_sizes]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        # Normalize by log(num_clusters) to get [0, 1]
        max_entropy = math.log(len(clusters)) if len(clusters) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _compute_pairwise_similarity(self, texts: list[str]) -> list[list[float]]:
        """Compute pairwise similarity using simple character overlap."""
        n = len(texts)
        sims = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                # Jaccard similarity on character n-grams
                si = set(texts[i].lower().split())
                sj = set(texts[j].lower().split())
                if len(si | sj) > 0:
                    sim = len(si & sj) / len(si | sj)
                else:
                    sim = 1.0 if texts[i] == texts[j] else 0.0
                sims[i][j] = sim
                sims[j][i] = sim
        return sims

    def _cluster_by_similarity(self, texts: list[str], similarities: list[list[float]]) -> list[list[int]]:
        """Cluster texts by similarity threshold."""
        n = len(texts)
        assigned = [False] * n
        clusters = []
        for i in range(n):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if not assigned[j] and similarities[i][j] >= self.similarity_threshold:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return clusters


class SelfCheckGPT:
    """SelfCheckGPT: Check consistency via multiple samples."""

    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples

    def check_consistency(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> tuple[float, float]:
        """
        Generate multiple samples and check consistency.
        Returns (consistency_score, disagreement_score) in [0, 1].
        """
        samples = []
        for _ in range(self.num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                )
            decoded = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            samples.append(decoded.strip().lower())

        # Agreement: fraction of pairs that are very similar
        if len(samples) < 2:
            return 1.0, 0.0

        agreements = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                # Exact match or high word overlap
                sim = self._text_similarity(samples[i], samples[j])
                agreements.append(sim)

        consistency = sum(agreements) / len(agreements) if agreements else 0.0
        disagreement = 1.0 - consistency
        return consistency, disagreement

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity [0, 1]."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if len(words1 | words2) == 0:
            return 1.0
        return len(words1 & words2) / len(words1 | words2)


class InferenceTimeIntervention:
    """Simplified ITI: shift activations toward learned truth direction."""

    def __init__(self, intervention_strength: float = 1.0):
        self.intervention_strength = intervention_strength
        self.truth_directions: dict[int, torch.Tensor] = {}

    def learn_directions(
        self,
        model: Any,
        tokenizer: Any,
        examples: list[tuple[str, str]],
        layer_ids: list[int] | None = None,
    ) -> None:
        """Learn truth directions from correct examples."""
        if layer_ids is None:
            num_layers = int(getattr(model.config, "num_hidden_layers", 32))
            layer_ids = list(range(max(0, num_layers - 6), num_layers))  # Last 6 layers

        hidden_states_correct = {lid: [] for lid in layer_ids}
        hidden_states_incorrect = {lid: [] for lid in layer_ids}

        for prompt, correct_answer in examples:
            # Get activations for correct completion
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                )
            for lid in layer_ids:
                h = outputs.hidden_states[lid + 1][:, -1, :]  # Last token
                hidden_states_correct[lid].append(h.cpu())

            # Get activations for incorrect completion (negative example)
            wrong_prompt = prompt.replace(correct_answer, "wrong incorrect false")
            inputs_wrong = tokenizer(wrong_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs_wrong = model(
                    **inputs_wrong,
                    output_hidden_states=True,
                )
            for lid in layer_ids:
                h = outputs_wrong.hidden_states[lid + 1][:, -1, :]
                hidden_states_incorrect[lid].append(h.cpu())

        # Compute direction: mean(correct) - mean(incorrect)
        for lid in layer_ids:
            if hidden_states_correct[lid]:
                mean_correct = torch.stack(hidden_states_correct[lid]).mean(dim=0)
                mean_incorrect = torch.stack(hidden_states_incorrect[lid]).mean(dim=0)
                direction = mean_correct - mean_incorrect
                self.truth_directions[lid] = direction / (direction.norm() + 1e-10)

    def apply_intervention(self, hidden_states: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Apply intervention to shift activations toward truth."""
        if layer_id not in self.truth_directions:
            return hidden_states
        direction = self.truth_directions[layer_id].to(hidden_states.device).to(hidden_states.dtype)
        # Shift last token representation along truth direction
        shift = self.intervention_strength * direction
        return hidden_states + shift.unsqueeze(0).unsqueeze(0)


class SelfRefineGenerator:
    """Generate with iterative self-refinement."""

    def __init__(self, max_refinements: int = 2):
        self.max_refinements = max_refinements

    def generate_with_refinement(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
    ) -> str:
        """Generate, then refine based on explicit self-critique."""
        # First pass
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=False,
            )
        response = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Refinement passes
        for _ in range(self.max_refinements):
            refinement_prompt = (
                f"Original response: {response}\n\n"
                f"Critique the above response. What could be improved?\n"
                f"Improved response:"
            )
            inputs_refine = tokenizer(refinement_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_refine = model.generate(
                    **inputs_refine,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=False,
                )
            response = tokenizer.decode(
                output_refine[0, inputs_refine["input_ids"].shape[1] :], skip_special_tokens=True
            )

        return response


class ChainOfVerification:
    """Chain-of-Verification: explicit verification steps."""

    def generate_with_verification(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 100,
    ) -> str:
        """Generate with explicit verification steps."""
        # Step 1: Initial generation
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=False,
            )
        response = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Step 2: Generate verification questions
        verify_prompt = (
            f"Response: {response}\n\n"
            f"Generate 3 yes/no questions to verify the accuracy of this response.\n"
            f"Questions:"
        )
        inputs_verify = tokenizer(verify_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_verify = model.generate(
                **inputs_verify,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=False,
            )
        questions = tokenizer.decode(
            output_verify[0, inputs_verify["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Step 3: Answer verification questions
        answer_prompt = f"Questions:\n{questions}\n\nAnswers:"
        inputs_answer = tokenizer(answer_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_answer = model.generate(
                **inputs_answer,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=False,
            )
        answers = tokenizer.decode(
            output_answer[0, inputs_answer["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Step 4: Revise if needed
        if "no" in answers.lower():
            revise_prompt = (
                f"Original: {response}\n"
                f"Based on verification failures, provide a corrected response:"
            )
            inputs_revise = tokenizer(revise_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_revise = model.generate(
                    **inputs_revise,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=False,
                )
            response = tokenizer.decode(
                output_revise[0, inputs_revise["input_ids"].shape[1] :], skip_special_tokens=True
            )

        return response


class EntropyCutoff(LogitsProcessor):
    """Suppress tokens when normalized entropy is above threshold."""

    def __init__(self, entropy_threshold: float = 0.5):
        self.entropy_threshold = entropy_threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        entropy = normalized_token_entropy(scores)  # Shape: [batch_size]
        mask = entropy > self.entropy_threshold
        # Set probabilities to near-zero for high-entropy positions
        scores[mask] = torch.finfo(scores.dtype).min
        return scores
