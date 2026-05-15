from __future__ import annotations

import csv
import typing as tp
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class FeatureAttribution:
    """
    Attribution record for one SAE feature and one concept.

    This is a candidate attribution based on activation strength, not causal
    proof. Use the resulting `feature_index` values in LLM interventions to
    test causal selectivity.
    """

    concept_label: str
    feature_index: int
    score: float
    score_method: str
    mean_inside: float
    mean_outside: float
    activation_rate_inside: float
    activation_rate_outside: float
    n_inside: int
    n_outside: int

    def to_dict(self) -> dict[str, tp.Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureExample:
    """Top activating example for a selected SAE feature."""

    feature_index: int
    sample_index: int
    activation: float
    concept_label: str
    text: str | None = None

    def to_dict(self) -> dict[str, tp.Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenFeatureAttribution:
    """
    Token-level attribution record for one SAE feature.

    This preserves the token position that activated a feature, which is needed
    for discovering more complex feature semantics from top activating tokens
    and their local contexts.
    """

    feature_index: int
    sample_index: int
    token_position: int
    activation: float
    raw_activation: float
    concept_label: str
    token_text: str | None = None
    left_context: str | None = None
    right_context: str | None = None

    def to_dict(self) -> dict[str, tp.Any]:
        return asdict(self)


def aggregate_latent_activations(
    latent_activations: torch.Tensor,
    token_aggregation: str = "mean_abs",
) -> torch.Tensor:
    """
    Convert SAE latents `z` to one activation-strength vector per example.

    Args:
        latent_activations: Tensor shaped [samples, latent] or
            [samples, sequence, latent].
        token_aggregation: Token aggregation for 3D tensors. Supported values:
            `mean_abs`, `max_abs`, `mean`, `max`, `last_token_abs`,
            `last_token`.

    Returns:
        Tensor shaped [samples, latent].
    """
    if latent_activations.ndim == 2:
        if token_aggregation in {"mean_abs", "max_abs", "last_token_abs"}:
            return latent_activations.abs()
        return latent_activations

    if latent_activations.ndim != 3:
        raise ValueError(
            "latent_activations must have shape [samples, latent] or [samples, sequence, latent]"
        )

    if token_aggregation == "mean_abs":
        return latent_activations.abs().mean(dim=1)
    if token_aggregation == "max_abs":
        return latent_activations.abs().max(dim=1).values
    if token_aggregation == "mean":
        return latent_activations.mean(dim=1)
    if token_aggregation == "max":
        return latent_activations.max(dim=1).values
    if token_aggregation == "last_token_abs":
        return latent_activations[:, -1, :].abs()
    if token_aggregation == "last_token":
        return latent_activations[:, -1, :]

    raise ValueError(
        "token_aggregation must be one of: mean_abs, max_abs, mean, max, last_token_abs, last_token"
    )


def top_tokens_for_feature(
    latent_activations: torch.Tensor,
    concept_labels: tp.Sequence[str],
    feature_index: int,
    tokens: tp.Sequence[tp.Sequence[str]] | None = None,
    top_k: int = 50,
    concept_label: str | None = None,
    activation_transform: str = "abs",
    context_window: int = 3,
) -> list[TokenFeatureAttribution]:
    """
    Return tokens that most strongly activate one SAE feature.

    Args:
        latent_activations: Token-level SAE latents shaped
            [samples, sequence, latent].
        concept_labels: Concept label for each sample.
        feature_index: SAE feature index to analyze.
        tokens: Optional token strings shaped [samples][sequence]. They can be
            produced with a tokenizer, for example `tokenizer.convert_ids_to_tokens`.
        top_k: Number of token positions to return.
        concept_label: Optional concept filter. If provided, only samples with
            this label are considered.
        activation_transform: `abs`, `raw`, or `positive`.
        context_window: Number of neighboring tokens to include on each side.
    """
    _validate_token_level_latents(latent_activations)
    labels = _normalize_labels(concept_labels)
    _validate_token_samples(latent_activations, labels, tokens)

    if feature_index < 0 or feature_index >= latent_activations.shape[-1]:
        raise IndexError(f"feature_index must be in [0, {latent_activations.shape[-1]})")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    sample_indices = torch.arange(latent_activations.shape[0], device=latent_activations.device)
    if concept_label is not None:
        sample_mask = torch.tensor(
            [label == concept_label for label in labels],
            dtype=torch.bool,
            device=latent_activations.device,
        )
        if not sample_mask.any():
            raise ValueError(f"Concept label not found: {concept_label}")
        selected_latents = latent_activations[sample_mask]
        selected_sample_indices = sample_indices[sample_mask]
    else:
        selected_latents = latent_activations
        selected_sample_indices = sample_indices

    raw_feature_activations = selected_latents[:, :, feature_index].float()
    feature_activations = _transform_token_activations(
        raw_feature_activations,
        activation_transform,
    )

    flat_activations = feature_activations.reshape(-1)
    values, flat_indices = torch.topk(flat_activations, k=min(top_k, flat_activations.numel()))
    sequence_length = selected_latents.shape[1]
    selected_rows = flat_indices // sequence_length
    token_positions = flat_indices % sequence_length

    attributions: list[TokenFeatureAttribution] = []
    for value, selected_row, token_position in zip(values, selected_rows, token_positions):
        sample_index = int(selected_sample_indices[selected_row].item())
        token_position_int = int(token_position.item())
        token_text = _get_token_text(tokens, sample_index, token_position_int)
        left_context, right_context = _get_token_context(
            tokens,
            sample_index,
            token_position_int,
            context_window,
        )
        raw_activation = raw_feature_activations[selected_row, token_position].item()
        attributions.append(
            TokenFeatureAttribution(
                feature_index=feature_index,
                sample_index=sample_index,
                token_position=token_position_int,
                activation=float(value.item()),
                raw_activation=float(raw_activation),
                concept_label=labels[sample_index],
                token_text=token_text,
                left_context=left_context,
                right_context=right_context,
            )
        )
    return attributions


def top_tokens_for_features(
    latent_activations: torch.Tensor,
    concept_labels: tp.Sequence[str],
    feature_indices: tp.Sequence[int],
    tokens: tp.Sequence[tp.Sequence[str]] | None = None,
    top_k: int = 50,
    concept_label: str | None = None,
    activation_transform: str = "abs",
    context_window: int = 3,
) -> list[TokenFeatureAttribution]:
    """Return top token activations for several SAE features."""
    attributions: list[TokenFeatureAttribution] = []
    for feature_index in feature_indices:
        attributions.extend(
            top_tokens_for_feature(
                latent_activations=latent_activations,
                concept_labels=concept_labels,
                feature_index=int(feature_index),
                tokens=tokens,
                top_k=top_k,
                concept_label=concept_label,
                activation_transform=activation_transform,
                context_window=context_window,
            )
        )
    return attributions


def format_feature_token_summary(
    token_attributions: tp.Sequence[TokenFeatureAttribution],
    feature_index: int,
    max_rows: int = 50,
) -> str:
    """
    Format top tokens and contexts for LLM-based feature categorization.

    The returned text is intended to be passed to a stronger LLM with a prompt
    asking it to infer a semantic category for the SAE feature.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    selected = [
        attribution
        for attribution in token_attributions
        if attribution.feature_index == feature_index
    ][:max_rows]
    if not selected:
        raise ValueError(f"No token attributions found for feature_index={feature_index}")

    lines = [
        f"SAE feature: {feature_index}",
        "Top activating tokens and contexts:",
    ]
    for attribution in selected:
        token_text = attribution.token_text if attribution.token_text is not None else "<unknown>"
        left_context = attribution.left_context if attribution.left_context is not None else ""
        right_context = attribution.right_context if attribution.right_context is not None else ""
        lines.append(
            "- "
            f"activation={attribution.activation:.6f}; "
            f"concept={attribution.concept_label}; "
            f"token={token_text!r}; "
            f"context={left_context} [{token_text}] {right_context}"
        )
    return "\n".join(lines)


def compute_feature_scores_for_concept(
    latent_activations: torch.Tensor,
    concept_labels: tp.Sequence[str],
    concept_label: str,
    top_k: int | None = None,
    score_method: str = "mean_diff",
    activation_threshold: float = 0.0,
    token_aggregation: str = "mean_abs",
    eps: float = 1e-8,
) -> list[FeatureAttribution]:
    """
    Rank SAE features by how strongly they activate for one concept.

    The function compares examples with `concept_label` against all other
    examples. It should be used for candidate feature discovery before causal
    intervention tests.
    """
    activations = aggregate_latent_activations(latent_activations, token_aggregation).float()
    labels = _normalize_labels(concept_labels)
    _validate_samples(activations, labels)

    concept_mask = torch.tensor(
        [label == concept_label for label in labels],
        dtype=torch.bool,
        device=activations.device,
    )
    if not concept_mask.any():
        raise ValueError(f"Concept label not found: {concept_label}")
    if concept_mask.all():
        raise ValueError("At least one outside-concept example is required for contrastive scoring")

    inside = activations[concept_mask]
    outside = activations[~concept_mask]

    mean_inside = inside.mean(dim=0)
    mean_outside = outside.mean(dim=0)
    rate_inside = (inside > activation_threshold).float().mean(dim=0)
    rate_outside = (outside > activation_threshold).float().mean(dim=0)
    scores = _compute_scores(
        inside=inside,
        outside=outside,
        mean_inside=mean_inside,
        mean_outside=mean_outside,
        rate_inside=rate_inside,
        rate_outside=rate_outside,
        score_method=score_method,
        eps=eps,
    )

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        sorted_scores = sorted_scores[:top_k]
        sorted_indices = sorted_indices[:top_k]

    return [
        FeatureAttribution(
            concept_label=concept_label,
            feature_index=int(feature_idx.item()),
            score=float(score.item()),
            score_method=score_method,
            mean_inside=float(mean_inside[feature_idx].item()),
            mean_outside=float(mean_outside[feature_idx].item()),
            activation_rate_inside=float(rate_inside[feature_idx].item()),
            activation_rate_outside=float(rate_outside[feature_idx].item()),
            n_inside=int(inside.shape[0]),
            n_outside=int(outside.shape[0]),
        )
        for score, feature_idx in zip(sorted_scores, sorted_indices)
    ]


def attribute_features_to_concepts(
    latent_activations: torch.Tensor,
    concept_labels: tp.Sequence[str],
    concepts: tp.Sequence[str] | None = None,
    top_k: int = 10,
    score_method: str = "mean_diff",
    activation_threshold: float = 0.0,
    token_aggregation: str = "mean_abs",
) -> list[FeatureAttribution]:
    """
    Return top SAE feature candidates for each concept.

    Args:
        latent_activations: SAE latents `z`, shaped [samples, latent] or
            [samples, sequence, latent].
        concept_labels: Concept label for each sample.
        concepts: Optional subset of labels to analyze.
        top_k: Number of features to return per concept.
        score_method: `mean_diff`, `mean_ratio`, `cohen_d`,
            `activation_rate_diff`, or `combined`.
        activation_threshold: Threshold for feature activation-rate metrics.
        token_aggregation: How to aggregate token-level activations.
    """
    labels = _normalize_labels(concept_labels)
    concepts = list(concepts) if concepts is not None else sorted(set(labels))

    attributions: list[FeatureAttribution] = []
    for concept_label in concepts:
        attributions.extend(
            compute_feature_scores_for_concept(
                latent_activations=latent_activations,
                concept_labels=labels,
                concept_label=concept_label,
                top_k=top_k,
                score_method=score_method,
                activation_threshold=activation_threshold,
                token_aggregation=token_aggregation,
            )
        )
    return attributions


def feature_indices_for_concept(
    latent_activations: torch.Tensor,
    concept_labels: tp.Sequence[str],
    concept_label: str,
    top_k: int = 10,
    score_method: str = "mean_diff",
    activation_threshold: float = 0.0,
    token_aggregation: str = "mean_abs",
) -> list[int]:
    """
    Return only feature indices for a concept.

    These indices can be passed to `LLM.add_sae(..., feature_indices=...)`.
    """
    attributions = compute_feature_scores_for_concept(
        latent_activations=latent_activations,
        concept_labels=concept_labels,
        concept_label=concept_label,
        top_k=top_k,
        score_method=score_method,
        activation_threshold=activation_threshold,
        token_aggregation=token_aggregation,
    )
    return [attribution.feature_index for attribution in attributions]


def top_examples_for_feature(
    latent_activations: torch.Tensor,
    concept_labels: tp.Sequence[str],
    feature_index: int,
    texts: tp.Sequence[str] | None = None,
    top_k: int = 10,
    token_aggregation: str = "mean_abs",
) -> list[FeatureExample]:
    """
    Return examples that most strongly activate one SAE feature.

    Use this for manual inspection before treating a feature as concept-related.
    """
    activations = aggregate_latent_activations(latent_activations, token_aggregation).float()
    labels = _normalize_labels(concept_labels)
    _validate_samples(activations, labels)

    if feature_index < 0 or feature_index >= activations.shape[-1]:
        raise IndexError(f"feature_index must be in [0, {activations.shape[-1]})")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if texts is not None and len(texts) != len(labels):
        raise ValueError("texts length must match concept_labels length")

    feature_activations = activations[:, feature_index]
    values, indices = torch.topk(feature_activations, k=min(top_k, len(labels)))

    return [
        FeatureExample(
            feature_index=feature_index,
            sample_index=int(sample_idx.item()),
            activation=float(value.item()),
            concept_label=labels[int(sample_idx.item())],
            text=None if texts is None else texts[int(sample_idx.item())],
        )
        for value, sample_idx in zip(values, indices)
    ]


def write_feature_attributions_csv(
    attributions: tp.Sequence[FeatureAttribution],
    path: str | Path,
) -> Path:
    """Write feature attributions to CSV for later intervention experiments."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(FeatureAttribution.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for attribution in attributions:
            writer.writerow(attribution.to_dict())
    return path


def write_token_attributions_csv(
    token_attributions: tp.Sequence[TokenFeatureAttribution],
    path: str | Path,
) -> Path:
    """Write token-level feature attributions to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(TokenFeatureAttribution.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for attribution in token_attributions:
            writer.writerow(attribution.to_dict())
    return path


def _compute_scores(
    inside: torch.Tensor,
    outside: torch.Tensor,
    mean_inside: torch.Tensor,
    mean_outside: torch.Tensor,
    rate_inside: torch.Tensor,
    rate_outside: torch.Tensor,
    score_method: str,
    eps: float,
) -> torch.Tensor:
    if score_method == "mean_diff":
        return mean_inside - mean_outside
    if score_method == "mean_ratio":
        return mean_inside / mean_outside.clamp_min(eps)
    if score_method == "cohen_d":
        var_inside = inside.var(dim=0, unbiased=False)
        var_outside = outside.var(dim=0, unbiased=False)
        pooled_std = ((var_inside + var_outside) / 2.0).sqrt().clamp_min(eps)
        return (mean_inside - mean_outside) / pooled_std
    if score_method == "activation_rate_diff":
        return rate_inside - rate_outside
    if score_method == "combined":
        return (mean_inside - mean_outside) * (rate_inside - rate_outside)

    raise ValueError(
        "score_method must be one of: mean_diff, mean_ratio, cohen_d, activation_rate_diff, combined"
    )


def _transform_token_activations(
    activations: torch.Tensor,
    activation_transform: str,
) -> torch.Tensor:
    if activation_transform == "abs":
        return activations.abs()
    if activation_transform == "raw":
        return activations
    if activation_transform == "positive":
        return activations.clamp_min(0)
    raise ValueError("activation_transform must be one of: abs, raw, positive")


def _validate_token_level_latents(latent_activations: torch.Tensor) -> None:
    if latent_activations.ndim != 3:
        raise ValueError("token-level attribution requires shape [samples, sequence, latent]")


def _validate_token_samples(
    latent_activations: torch.Tensor,
    labels: tp.Sequence[str],
    tokens: tp.Sequence[tp.Sequence[str]] | None,
) -> None:
    if latent_activations.shape[0] != len(labels):
        raise ValueError(
            f"Number of activation samples ({latent_activations.shape[0]}) must match labels ({len(labels)})"
        )
    if tokens is not None and len(tokens) != len(labels):
        raise ValueError("tokens length must match concept_labels length")


def _get_token_text(
    tokens: tp.Sequence[tp.Sequence[str]] | None,
    sample_index: int,
    token_position: int,
) -> str | None:
    if tokens is None:
        return None
    if token_position >= len(tokens[sample_index]):
        return None
    return str(tokens[sample_index][token_position])


def _get_token_context(
    tokens: tp.Sequence[tp.Sequence[str]] | None,
    sample_index: int,
    token_position: int,
    context_window: int,
) -> tuple[str | None, str | None]:
    if tokens is None:
        return None, None

    sample_tokens = [str(token) for token in tokens[sample_index]]
    if token_position >= len(sample_tokens):
        return None, None

    left_start = max(0, token_position - context_window)
    right_end = min(len(sample_tokens), token_position + context_window + 1)
    left_context = " ".join(sample_tokens[left_start:token_position])
    right_context = " ".join(sample_tokens[token_position + 1 : right_end])
    return left_context, right_context


def _normalize_labels(concept_labels: tp.Sequence[str]) -> list[str]:
    if torch.is_tensor(concept_labels):
        concept_labels = concept_labels.detach().cpu().tolist()
    return [str(label) for label in concept_labels]


def _validate_samples(activations: torch.Tensor, labels: tp.Sequence[str]) -> None:
    if activations.ndim != 2:
        raise ValueError("aggregated activations must have shape [samples, latent]")
    if activations.shape[0] != len(labels):
        raise ValueError(
            f"Number of activation samples ({activations.shape[0]}) must match labels ({len(labels)})"
        )
    if activations.shape[0] < 2:
        raise ValueError("At least two samples are required for feature attribution")
