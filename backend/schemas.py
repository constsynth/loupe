from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LAYER_NAME = "model.layers.15"
DEFAULT_DASHBOARD_DIR = "data/sae_feature_dashboard"
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def default_sae_checkpoint_path() -> str:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(models_dir.glob("*.pt"))
    if not checkpoints:
        return f"There are no pretrained SAE for {DEFAULT_MODEL_NAME}"
    return str(checkpoints[0])


DEFAULT_SAE_CHECKPOINT_PATH = default_sae_checkpoint_path()


class LoupeBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class GenerationSettings(LoupeBaseModel):
    """Text generation settings forwarded to `transformers.generate`."""

    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    max_length: int = Field(default=512, ge=16, le=4096)
    do_sample: bool = False
    temperature: float | None = Field(default=None, gt=0.0, le=5.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT


class ConceptSpec(LoupeBaseModel):
    """User-facing concept definition mapped to SAE feature ids."""

    id: str
    name: str | None = None
    feature_ids: list[int] = Field(default_factory=list)
    strength: float = Field(default=1.0, ge=0.0, le=2.0)


class AttributionRequest(LoupeBaseModel):
    """Request for LLM generation plus token-level SAE attribution."""

    prompt: str = Field(min_length=1)
    model_name: str = DEFAULT_MODEL_NAME
    sae_checkpoint_path: str = Field(default_factory=default_sae_checkpoint_path)
    layer_name: str = DEFAULT_LAYER_NAME
    device: str = "auto"
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    top_k_features: int = Field(default=10, ge=1, le=100)
    top_k_token_features: int = Field(default=3, ge=1, le=20)
    concepts: list[ConceptSpec] = Field(default_factory=list)


class TokenFeatureScore(LoupeBaseModel):
    """Activation of one SAE feature at one token position."""

    feature_id: int
    activation: float
    raw_activation: float | None = None
    concept_id: str | None = None
    concept_label: str | None = None


class TokenAttribution(LoupeBaseModel):
    """Token-level SAE attribution for frontend highlighting."""

    text: str
    position: int
    activation: float
    raw_activation: float | None = None
    feature_id: int
    concept_id: str | None = None
    concept_label: str | None = None
    top_features: list[TokenFeatureScore]


class FeatureSummary(LoupeBaseModel):
    """Aggregated SAE feature activation summary over valid tokens."""

    feature_id: int
    activation: float
    concept_id: str | None = None
    concept_label: str | None = None
    score_method: str = "mean_abs"


class SAEMetrics(LoupeBaseModel):
    """Lightweight reconstruction and sparsity metrics for one request."""

    mse: float
    nmse: float
    cosine_similarity: float
    l0: float
    active_feature_share: float
    hoyer_sparsity: float
    normalized_entropy: float


class AttributionResponse(LoupeBaseModel):
    """Response for `/api/generate-attributions`."""

    text: str
    full_text: str
    model_name: str
    sae_checkpoint_path: str
    layer_name: str
    tokens: list[TokenAttribution]
    features: list[FeatureSummary]
    metrics: SAEMetrics


class InterventionRequest(LoupeBaseModel):
    """Request for baseline vs SAE-intervened generation."""

    prompt: str = Field(min_length=1)
    model_name: str = DEFAULT_MODEL_NAME
    sae_checkpoint_path: str = Field(default_factory=default_sae_checkpoint_path)
    layer_name: str = DEFAULT_LAYER_NAME
    device: str = "auto"
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    concepts: list[ConceptSpec] = Field(default_factory=list)
    feature_ids: list[int] = Field(default_factory=list)
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    token_positions: list[int] | None = None


class InterventionResponse(LoupeBaseModel):
    """Response for `/api/interventions`."""

    baseline_text: str
    intervened_text: str
    model_name: str
    sae_checkpoint_path: str
    layer_name: str
    feature_ids: list[int]
    intervention_value: float
    token_positions: list[int] | None


class DashboardTopConcept(LoupeBaseModel):
    """Concept score included in one feature dashboard row."""

    concept_label: str
    score: float
    mean_inside: float | None = None
    mean_outside: float | None = None


class DashboardFeature(LoupeBaseModel):
    """Persisted feature-level dashboard summary."""

    feature_index: int
    mean_activation: float
    max_activation: float
    activation_density: float
    top_concepts: list[DashboardTopConcept] = Field(default_factory=list)


class DashboardConceptScore(LoupeBaseModel):
    """Concept-level SAE feature candidate score."""

    concept_label: str
    feature_index: int
    score: float
    score_method: str | None = None
    mean_inside: float | None = None
    mean_outside: float | None = None
    activation_rate_inside: float | None = None
    activation_rate_outside: float | None = None
    n_inside: int | None = None
    n_outside: int | None = None


class DashboardTopToken(LoupeBaseModel):
    """Top activating token and context for one SAE feature."""

    feature_index: int
    sample_id: str | None = None
    sample_index: int | None = None
    token_position: int | None = None
    token_text: str | None = None
    activation: float
    raw_activation: float | None = None
    concept_label: str | None = None
    focus_concept: str | None = None
    left_context: str | None = None
    right_context: str | None = None
    text: str | None = None
    feature_score: float | None = None


class DashboardTopExample(LoupeBaseModel):
    """Full dataset example that strongly activates one SAE feature."""

    feature_index: int
    sample_id: str | None = None
    sample_index: int | None = None
    activation: float
    concept_label: str | None = None
    text: str | None = None


class FeatureDashboardResponse(LoupeBaseModel):
    """Read-only response backed by saved dashboard artifacts in `data/`."""

    dashboard_dir: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    features: list[DashboardFeature]
    concept_scores: list[DashboardConceptScore]
    top_tokens: list[DashboardTopToken]
    top_examples: list[DashboardTopExample]
    sample_token_attributions_count: int = 0
    method_note: str = (
        "Dashboard rows are activation-based SAE feature candidates, not causal proof."
    )


class DefaultConfigResponse(LoupeBaseModel):
    """Frontend defaults exposed by the backend."""

    model_name: str = DEFAULT_MODEL_NAME
    sae_checkpoint_path: str = Field(default_factory=default_sae_checkpoint_path)
    layer_name: str = DEFAULT_LAYER_NAME


class HealthResponse(LoupeBaseModel):
    status: str
