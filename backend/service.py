from __future__ import annotations

import json
import math
import threading
import typing as tp
from pathlib import Path

import pandas as pd
import torch

from interpretability.sae.sae import SAE
from utils.inference_utils.llm import LLM
from utils.stat_utils.sae_statistics import flatten_valid_tokens

from backend.sae_io import load_sae_checkpoint, resolve_project_path
from backend.schemas import (
    AttributionRequest,
    AttributionResponse,
    ConceptSpec,
    DEFAULT_DASHBOARD_DIR,
    DashboardConceptScore,
    DashboardFeature,
    DashboardTopConcept,
    DashboardTopExample,
    DashboardTopToken,
    FeatureSummary,
    FeatureDashboardResponse,
    GenerationSettings,
    InterventionRequest,
    InterventionResponse,
    SAEMetrics,
    TokenAttribution,
    TokenFeatureScore,
)


class LoupeBackendService:
    """Lazy runtime for LLM, SAE attribution, and SAE interventions."""

    def __init__(self) -> None:
        self._llm_cache: dict[tuple[str, str], LLM] = {}
        self._sae_cache: dict[tuple[str, str, str], SAE] = {}
        self._lock = threading.Lock()

    def generate_attributions(self, request: AttributionRequest) -> AttributionResponse:
        """Generate text and compute token-level SAE feature activations."""
        self._ensure_sae_checkpoint_exists(request.sae_checkpoint_path)
        llm = self._get_llm(request.model_name, request.device)
        sae = self._get_sae(request.sae_checkpoint_path, request.device, llm)

        generated_text = self._generate_completion(llm, request.prompt, request.generation)
        attribution_text = generated_text.strip() or request.prompt
        h, attention_mask, tokens = self._extract_layer_activations(
            llm=llm,
            texts=[attribution_text],
            layer_name=request.layer_name,
            max_length=request.generation.max_length,
        )

        device = next(sae.parameters()).device
        with torch.no_grad():
            sae_output = sae(h.to(device), return_output=True)

        z = sae_output.latent_activation.detach().cpu()
        h_hat = sae_output.reconstructed_hidden_state.detach().cpu()
        z = z.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        h_hat = h_hat.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        concept_lookup = self._build_dashboard_concept_lookup(DEFAULT_DASHBOARD_DIR)
        concept_lookup.update(self._build_concept_lookup(request.concepts))
        return AttributionResponse(
            text=generated_text,
            full_text=attribution_text,
            model_name=request.model_name,
            sae_checkpoint_path=request.sae_checkpoint_path,
            layer_name=request.layer_name,
            tokens=self._token_attributions(
                latent=z,
                attention_mask=attention_mask,
                tokens=tokens[0],
                top_k_token_features=request.top_k_token_features,
                concept_lookup=concept_lookup,
            ),
            features=self._feature_summary(
                latent=z,
                attention_mask=attention_mask,
                top_k=request.top_k_features,
                concept_lookup=concept_lookup,
            ),
            metrics=self._metrics(sae=sae, h=h, h_hat=h_hat, z=z, mask=attention_mask),
        )

    def run_intervention(self, request: InterventionRequest) -> InterventionResponse:
        """Compare baseline generation against one SAE latent intervention."""
        self._ensure_sae_checkpoint_exists(request.sae_checkpoint_path)
        llm = self._get_llm(request.model_name, request.device)
        sae = self._get_sae(request.sae_checkpoint_path, request.device, llm)
        feature_ids = self._resolve_intervention_features(request)
        if not feature_ids:
            raise ValueError("At least one intervention feature id is required")

        intervention_value = self._resolve_intervention_strength(request.strength)

        baseline_text = self._generate_completion(llm, request.prompt, request.generation)
        layer_handle = None
        try:
            layer_handle = llm.add_sae(
                sae=sae,
                layer_name=request.layer_name,
                feature_indices=feature_ids,
                intervention_value=intervention_value,
                token_positions=request.token_positions,
            )
            intervened_text = self._generate_completion(llm, request.prompt, request.generation)
        finally:
            if layer_handle is not None:
                llm.remove_sae(layer_handle)

        return InterventionResponse(
            baseline_text=baseline_text,
            intervened_text=intervened_text,
            model_name=request.model_name,
            sae_checkpoint_path=request.sae_checkpoint_path,
            layer_name=request.layer_name,
            feature_ids=feature_ids,
            intervention_value=intervention_value,
            token_positions=request.token_positions,
        )

    def get_feature_dashboard(
        self,
        dashboard_dir: str = DEFAULT_DASHBOARD_DIR,
        top_features: int = 20,
        top_concept_scores: int = 40,
        top_tokens: int = 80,
        top_examples: int = 40,
    ) -> FeatureDashboardResponse:
        """Read saved SAE dashboard artifacts from `data/sae_feature_dashboard`."""
        self._validate_dashboard_limits(
            top_features=top_features,
            top_concept_scores=top_concept_scores,
            top_tokens=top_tokens,
            top_examples=top_examples,
        )
        resolved_dir = resolve_project_path(dashboard_dir)
        if not resolved_dir.exists():
            raise FileNotFoundError(
                f"Feature dashboard not found: {resolved_dir}. "
                "Run examples/build_sae_feature_dashboard.ipynb first."
            )

        metadata = self._dashboard_metadata(resolved_dir / "dashboard_metadata.json")
        features_df = self._dashboard_csv(resolved_dir / "feature_dashboard.csv")
        concept_scores_df = self._dashboard_csv(resolved_dir / "feature_concept_scores.csv")
        top_tokens_df = self._dashboard_csv(resolved_dir / "feature_top_tokens.csv")
        top_examples_df = self._dashboard_csv(resolved_dir / "feature_top_examples.csv")
        sample_token_count = self._optional_dashboard_row_count(
            resolved_dir / "sample_token_attributions.csv"
        )

        return FeatureDashboardResponse(
            dashboard_dir=str(resolved_dir),
            metadata=metadata,
            features=self._dashboard_features(features_df, top_features),
            concept_scores=self._dashboard_concept_scores(
                concept_scores_df,
                top_concept_scores,
            ),
            top_tokens=self._dashboard_top_tokens(top_tokens_df, top_tokens),
            top_examples=self._dashboard_top_examples(top_examples_df, top_examples),
            sample_token_attributions_count=sample_token_count,
            method_note=metadata.get(
                "method_note",
                "Dashboard rows are activation-based SAE feature candidates, not causal proof.",
            ),
        )

    def _get_llm(self, model_name: str, device: str) -> LLM:
        resolved_device = self._resolve_device(device)
        key = (model_name, resolved_device)
        with self._lock:
            if key not in self._llm_cache:
                self._llm_cache[key] = LLM(model_name_or_path=model_name, device=resolved_device)
            return self._llm_cache[key]

    @staticmethod
    def _ensure_sae_checkpoint_exists(checkpoint_path: str) -> None:
        resolved_path = resolve_project_path(checkpoint_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {resolved_path}")

    def _get_sae(self, checkpoint_path: str, device: str, llm: LLM) -> SAE:
        model_device, model_dtype = llm._model_device_dtype()
        resolved_device = str(model_device if device == "auto" else self._resolve_device(device))
        key = (checkpoint_path, resolved_device, str(model_dtype))
        with self._lock:
            if key not in self._sae_cache:
                self._sae_cache[key] = load_sae_checkpoint(
                    checkpoint_path=checkpoint_path,
                    device=resolved_device,
                    dtype=model_dtype,
                )
            return self._sae_cache[key]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    @staticmethod
    def _generation_kwargs(settings: GenerationSettings) -> dict[str, tp.Any]:
        kwargs: dict[str, tp.Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
        }
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.top_p is not None:
            kwargs["top_p"] = settings.top_p
        return kwargs

    def _generate_completion(self, llm: LLM, prompt: str, settings: GenerationSettings) -> str:
        return llm.generate(
            prompt,
            input_max_length=settings.max_length,
            return_full_text=False,
            system_prompt=settings.system_prompt,
            **self._generation_kwargs(settings),
        )

    def _extract_layer_activations(
        self,
        llm: LLM,
        texts: list[str],
        layer_name: str,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
        named_modules = dict(llm.model.named_modules())
        if layer_name not in named_modules:
            raise ValueError(f"Layer not found: {layer_name}")

        inputs = llm.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ).to(llm.device)
        saved_hidden_states: list[torch.Tensor] = []

        def hook_fn(module, inputs, output):
            saved_hidden_states.append(
                llm._extract_hidden_state_from_module_output(output).detach().cpu()
            )

        handle = named_modules[layer_name].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = llm.model(**inputs)
        finally:
            handle.remove()

        if not saved_hidden_states:
            raise RuntimeError(f"Layer did not produce hidden states: {layer_name}")

        hidden_states = saved_hidden_states[-1]
        attention_mask = inputs["attention_mask"].detach().cpu().bool()
        tokens = [
            llm.tokenizer.convert_ids_to_tokens(row)
            for row in inputs["input_ids"].detach().cpu()
        ]
        if hidden_states.shape[:2] != attention_mask.shape:
            raise RuntimeError(
                f"Activation shape {tuple(hidden_states.shape)} is incompatible with "
                f"attention mask shape {tuple(attention_mask.shape)}"
            )
        return hidden_states, attention_mask, tokens

    @staticmethod
    def _build_concept_lookup(
        concepts: list[ConceptSpec],
    ) -> dict[int, tuple[str, str | None]]:
        lookup: dict[int, tuple[str, str | None]] = {}
        for concept in concepts:
            for feature_id in concept.feature_ids:
                lookup[int(feature_id)] = (concept.id, concept.name)
        return lookup

    @staticmethod
    def _build_dashboard_concept_lookup(
        dashboard_dir: str = DEFAULT_DASHBOARD_DIR,
    ) -> dict[int, tuple[str, str | None]]:
        resolved_dir = resolve_project_path(dashboard_dir)
        lookup: dict[int, tuple[str, str | None]] = {}

        concept_scores_path = resolved_dir / "feature_concept_scores.csv"
        if concept_scores_path.exists():
            try:
                concept_scores_df = pd.read_csv(concept_scores_path, dtype=str).fillna("")
            except pd.errors.EmptyDataError:
                concept_scores_df = pd.DataFrame()
            rows = LoupeBackendService._sort_by_float(concept_scores_df, "score")
            for row in rows.to_dict("records"):
                feature_id = LoupeBackendService._to_int(row.get("feature_index"), default=-1)
                concept_label = LoupeBackendService._empty_to_none(row.get("concept_label"))
                LoupeBackendService._set_dashboard_lookup_label(
                    lookup,
                    feature_id,
                    concept_label,
                )

        feature_dashboard_path = resolved_dir / "feature_dashboard.csv"
        if feature_dashboard_path.exists():
            try:
                feature_dashboard_df = pd.read_csv(feature_dashboard_path, dtype=str).fillna("")
            except pd.errors.EmptyDataError:
                feature_dashboard_df = pd.DataFrame()
            rows = LoupeBackendService._sort_by_float(feature_dashboard_df, "max_activation")
            for row in rows.to_dict("records"):
                feature_id = LoupeBackendService._to_int(row.get("feature_index"), default=-1)
                top_concepts = LoupeBackendService._parse_top_concepts(row.get("top_concepts"))
                concept_label = top_concepts[0].concept_label if top_concepts else None
                LoupeBackendService._set_dashboard_lookup_label(
                    lookup,
                    feature_id,
                    concept_label,
                )

        top_tokens_path = resolved_dir / "feature_top_tokens.csv"
        if top_tokens_path.exists():
            try:
                top_tokens_df = pd.read_csv(top_tokens_path, dtype=str).fillna("")
            except pd.errors.EmptyDataError:
                top_tokens_df = pd.DataFrame()
            rows = LoupeBackendService._sort_by_float(top_tokens_df, "activation")
            for row in rows.to_dict("records"):
                feature_id = LoupeBackendService._to_int(row.get("feature_index"), default=-1)
                concept_label = (
                    LoupeBackendService._empty_to_none(row.get("focus_concept"))
                    or LoupeBackendService._empty_to_none(row.get("concept_label"))
                )
                LoupeBackendService._set_dashboard_lookup_label(
                    lookup,
                    feature_id,
                    concept_label,
                )

        sample_token_path = resolved_dir / "sample_token_attributions.csv"
        if sample_token_path.exists():
            try:
                sample_token_df = pd.read_csv(sample_token_path, dtype=str).fillna("")
            except pd.errors.EmptyDataError:
                sample_token_df = pd.DataFrame()
            rows = LoupeBackendService._sort_by_float(sample_token_df, "activation")
            for row in rows.to_dict("records"):
                feature_id = LoupeBackendService._to_int(row.get("feature_index"), default=-1)
                concept_label = LoupeBackendService._empty_to_none(row.get("concept_label"))
                LoupeBackendService._set_dashboard_lookup_label(
                    lookup,
                    feature_id,
                    concept_label,
                )
        return lookup

    @staticmethod
    def _set_dashboard_lookup_label(
        lookup: dict[int, tuple[str, str | None]],
        feature_id: int,
        concept_label: str | None,
    ) -> None:
        if feature_id < 0 or concept_label is None or feature_id in lookup:
            return
        lookup[feature_id] = (concept_label, concept_label)

    @staticmethod
    def _token_attributions(
        latent: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: list[str],
        top_k_token_features: int,
        concept_lookup: dict[int, tuple[str, str | None]],
    ) -> list[TokenAttribution]:
        valid_positions = attention_mask[0].nonzero(as_tuple=False).flatten().tolist()
        raw_rows: list[dict[str, tp.Any]] = []
        primary_raw_activations: list[float] = []
        for position in valid_positions:
            token_latent = latent[0, position].float().abs()
            values, indices = torch.topk(
                token_latent,
                k=min(top_k_token_features, token_latent.numel()),
            )
            top_features: list[dict[str, tp.Any]] = []
            for feature_id_tensor, value_tensor in zip(indices, values):
                feature_id = int(feature_id_tensor.item())
                concept_id, concept_label = concept_lookup.get(feature_id, (None, None))
                top_features.append({
                    "feature_id": feature_id,
                    "raw_activation": float(value_tensor.item()),
                    "concept_id": concept_id,
                    "concept_label": concept_label,
                })

            primary = top_features[0]
            primary_raw_activations.append(primary["raw_activation"])
            raw_rows.append({
                "text": LoupeBackendService._clean_token_text(tokens[position]) or tokens[position],
                "position": int(position),
                "feature_id": primary["feature_id"],
                "raw_activation": primary["raw_activation"],
                "concept_id": primary["concept_id"],
                "concept_label": primary["concept_label"],
                "top_features": top_features,
            })

        normalized_primary = LoupeBackendService._normalize_activation_values(primary_raw_activations)
        output: list[TokenAttribution] = []
        for row, normalized_activation in zip(raw_rows, normalized_primary):
            primary_raw = row["raw_activation"]
            scale = normalized_activation / primary_raw if primary_raw > 0 else 0.0
            top_features = [
                TokenFeatureScore(
                    feature_id=feature["feature_id"],
                    activation=max(0.0, min(feature["raw_activation"] * scale, 1.0)),
                    raw_activation=feature["raw_activation"],
                    concept_id=feature["concept_id"],
                    concept_label=feature["concept_label"],
                )
                for feature in row["top_features"]
            ]
            output.append(
                TokenAttribution(
                    text=row["text"],
                    position=row["position"],
                    activation=normalized_activation,
                    raw_activation=primary_raw,
                    feature_id=row["feature_id"],
                    concept_id=row["concept_id"],
                    concept_label=row["concept_label"],
                    top_features=top_features,
                )
            )
        return output

    @staticmethod
    def _normalize_activation_values(values: list[float]) -> list[float]:
        """Normalize token activations to [0, 1] for color visualization."""
        if not values:
            return []

        tensor = torch.tensor(values, dtype=torch.float32)
        lower = torch.quantile(tensor, 0.05).item()
        upper = torch.quantile(tensor, 0.95).item()
        if upper <= lower:
            upper = float(tensor.max().item())
            lower = float(tensor.min().item())
        if upper <= lower:
            return [0.5 if value > 0 else 0.0 for value in values]

        normalized = ((tensor - lower) / (upper - lower)).clamp(0.0, 1.0)
        return [float(value.item()) for value in normalized]

    @staticmethod
    def _feature_summary(
        latent: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
        concept_lookup: dict[int, tuple[str, str | None]],
    ) -> list[FeatureSummary]:
        valid_latent = flatten_valid_tokens(latent, attention_mask)
        feature_indices, scores = SAE.top_activated_features(
            valid_latent,
            k=top_k,
            aggregate="mean_abs",
        )
        summaries: list[FeatureSummary] = []
        for feature_id_tensor, score_tensor in zip(feature_indices, scores):
            feature_id = int(feature_id_tensor.item())
            concept_id, concept_label = concept_lookup.get(feature_id, (None, None))
            summaries.append(
                FeatureSummary(
                    feature_id=feature_id,
                    activation=float(score_tensor.item()),
                    concept_id=concept_id,
                    concept_label=concept_label,
                )
            )
        return summaries

    @staticmethod
    def _metrics(
        sae: SAE,
        h: torch.Tensor,
        h_hat: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> SAEMetrics:
        valid_h = flatten_valid_tokens(h, mask)
        valid_h_hat = flatten_valid_tokens(h_hat, mask)
        valid_z = flatten_valid_tokens(z, mask).to(next(sae.parameters()).device)
        reconstruction = sae.reconstruction_metrics(valid_h, valid_h_hat)
        sparsity = sae.sparsity_metrics(valid_z)
        return SAEMetrics(
            mse=float(reconstruction["mse"].item()),
            nmse=float(reconstruction["nmse"].item()),
            cosine_similarity=float(reconstruction["cosine_similarity"].item()),
            l0=float(sparsity["l0"].item()),
            active_feature_share=float(sparsity["active_feature_share"].item()),
            hoyer_sparsity=float(sparsity["hoyer_sparsity"].item()),
            normalized_entropy=float(sparsity["normalized_entropy"].item()),
        )

    @staticmethod
    def _resolve_intervention_features(request: InterventionRequest) -> list[int]:
        feature_ids = [int(feature_id) for feature_id in request.feature_ids]
        for concept in request.concepts:
            feature_ids.extend(int(feature_id) for feature_id in concept.feature_ids)
        return sorted(set(feature_ids))

    @staticmethod
    def _resolve_intervention_strength(strength: float) -> float:
        return max(0.0, min(float(strength), 2.0))

    @staticmethod
    def _validate_dashboard_limits(
        top_features: int,
        top_concept_scores: int,
        top_tokens: int,
        top_examples: int,
    ) -> None:
        for name, value in {
            "top_features": top_features,
            "top_concept_scores": top_concept_scores,
            "top_tokens": top_tokens,
            "top_examples": top_examples,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @staticmethod
    def _dashboard_metadata(path: Path) -> dict[str, tp.Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError(f"Dashboard metadata must be a JSON object: {path}")
        return metadata

    @staticmethod
    def _dashboard_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Dashboard artifact not found: {path}. "
                "Run examples/build_sae_feature_dashboard.ipynb first."
            )
        try:
            return pd.read_csv(path, dtype=str).fillna("")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    @staticmethod
    def _optional_dashboard_row_count(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            return int(len(pd.read_csv(path, dtype=str, usecols=[0])))
        except pd.errors.EmptyDataError:
            return 0

    @staticmethod
    def _sort_by_float(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in df.columns:
            return df
        sorted_df = df.copy()
        sorted_df["_sort_value"] = sorted_df[column].map(
            lambda value: LoupeBackendService._to_float(value, float("-inf"))
        )
        return sorted_df.sort_values("_sort_value", ascending=False).drop(
            columns=["_sort_value"]
        )

    @staticmethod
    def _dashboard_features(df: pd.DataFrame, limit: int) -> list[DashboardFeature]:
        rows = LoupeBackendService._sort_by_float(df, "max_activation").head(limit)
        features: list[DashboardFeature] = []
        for row in rows.to_dict("records"):
            features.append(
                DashboardFeature(
                    feature_index=LoupeBackendService._to_int(row.get("feature_index")),
                    mean_activation=LoupeBackendService._to_float(
                        row.get("mean_activation")
                    ),
                    max_activation=LoupeBackendService._to_float(row.get("max_activation")),
                    activation_density=LoupeBackendService._to_float(
                        row.get("activation_density")
                    ),
                    top_concepts=LoupeBackendService._parse_top_concepts(
                        row.get("top_concepts")
                    ),
                )
            )
        return features

    @staticmethod
    def _dashboard_concept_scores(
        df: pd.DataFrame,
        limit: int,
    ) -> list[DashboardConceptScore]:
        rows = LoupeBackendService._sort_by_float(df, "score").head(limit)
        scores: list[DashboardConceptScore] = []
        for row in rows.to_dict("records"):
            scores.append(
                DashboardConceptScore(
                    concept_label=LoupeBackendService._empty_to_none(
                        row.get("concept_label")
                    )
                    or "unknown",
                    feature_index=LoupeBackendService._to_int(row.get("feature_index")),
                    score=LoupeBackendService._to_float(row.get("score")),
                    score_method=LoupeBackendService._empty_to_none(
                        row.get("score_method")
                    ),
                    mean_inside=LoupeBackendService._to_optional_float(
                        row.get("mean_inside")
                    ),
                    mean_outside=LoupeBackendService._to_optional_float(
                        row.get("mean_outside")
                    ),
                    activation_rate_inside=LoupeBackendService._to_optional_float(
                        row.get("activation_rate_inside")
                    ),
                    activation_rate_outside=LoupeBackendService._to_optional_float(
                        row.get("activation_rate_outside")
                    ),
                    n_inside=LoupeBackendService._to_optional_int(row.get("n_inside")),
                    n_outside=LoupeBackendService._to_optional_int(row.get("n_outside")),
                )
            )
        return scores

    @staticmethod
    def _dashboard_top_tokens(df: pd.DataFrame, limit: int) -> list[DashboardTopToken]:
        rows = LoupeBackendService._sort_by_float(df, "activation").head(limit)
        tokens: list[DashboardTopToken] = []
        for row in rows.to_dict("records"):
            tokens.append(
                DashboardTopToken(
                    feature_index=LoupeBackendService._to_int(row.get("feature_index")),
                    sample_id=LoupeBackendService._empty_to_none(row.get("sample_id")),
                    sample_index=LoupeBackendService._to_optional_int(
                        row.get("sample_index")
                    ),
                    token_position=LoupeBackendService._to_optional_int(
                        row.get("token_position")
                    ),
                    token_text=LoupeBackendService._clean_token_text(row.get("token_text")),
                    activation=LoupeBackendService._to_float(row.get("activation")),
                    raw_activation=LoupeBackendService._to_optional_float(
                        row.get("raw_activation")
                    ),
                    concept_label=LoupeBackendService._empty_to_none(
                        row.get("concept_label")
                    ),
                    focus_concept=LoupeBackendService._empty_to_none(
                        row.get("focus_concept")
                    ),
                    left_context=LoupeBackendService._clean_token_context(
                        row.get("left_context")
                    ),
                    right_context=LoupeBackendService._clean_token_context(
                        row.get("right_context")
                    ),
                    text=LoupeBackendService._empty_to_none(row.get("text")),
                    feature_score=LoupeBackendService._to_optional_float(
                        row.get("feature_score")
                    ),
                )
            )
        return tokens

    @staticmethod
    def _dashboard_top_examples(
        df: pd.DataFrame,
        limit: int,
    ) -> list[DashboardTopExample]:
        rows = LoupeBackendService._sort_by_float(df, "activation").head(limit)
        examples: list[DashboardTopExample] = []
        for row in rows.to_dict("records"):
            examples.append(
                DashboardTopExample(
                    feature_index=LoupeBackendService._to_int(row.get("feature_index")),
                    sample_id=LoupeBackendService._empty_to_none(row.get("sample_id")),
                    sample_index=LoupeBackendService._to_optional_int(
                        row.get("sample_index")
                    ),
                    activation=LoupeBackendService._to_float(row.get("activation")),
                    concept_label=LoupeBackendService._empty_to_none(
                        row.get("concept_label")
                    ),
                    text=LoupeBackendService._empty_to_none(row.get("text")),
                )
            )
        return examples

    @staticmethod
    def _parse_top_concepts(value: tp.Any) -> list[DashboardTopConcept]:
        raw_value = LoupeBackendService._empty_to_none(value)
        if raw_value is None:
            return []
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []

        concepts: list[DashboardTopConcept] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            concepts.append(
                DashboardTopConcept(
                    concept_label=LoupeBackendService._empty_to_none(
                        item.get("concept_label")
                    )
                    or "unknown",
                    score=LoupeBackendService._to_float(item.get("score")),
                    mean_inside=LoupeBackendService._to_optional_float(
                        item.get("mean_inside")
                    ),
                    mean_outside=LoupeBackendService._to_optional_float(
                        item.get("mean_outside")
                    ),
                )
            )
        return concepts

    @staticmethod
    def _empty_to_none(value: tp.Any) -> str | None:
        if value is None:
            return None
        value_str = str(value).strip()
        return value_str or None

    @staticmethod
    def _clean_token_text(value: tp.Any) -> str | None:
        token = LoupeBackendService._empty_to_none(value)
        if token is None:
            return None
        cleaned = (
            token.replace("Ġ", " ")
            .replace("▁", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
            .strip()
        )
        return cleaned or token

    @staticmethod
    def _clean_token_context(value: tp.Any) -> str | None:
        context = LoupeBackendService._empty_to_none(value)
        if context is None:
            return None
        cleaned = (
            context.replace("Ġ", " ")
            .replace("▁", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
        )
        cleaned = " ".join(cleaned.split())
        return cleaned or context

    @staticmethod
    def _to_float(value: tp.Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _to_optional_float(value: tp.Any) -> float | None:
        value_str = LoupeBackendService._empty_to_none(value)
        if value_str is None:
            return None
        number = LoupeBackendService._to_float(value_str, default=float("nan"))
        return number if math.isfinite(number) else None

    @staticmethod
    def _to_int(value: tp.Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_optional_int(value: tp.Any) -> int | None:
        value_str = LoupeBackendService._empty_to_none(value)
        if value_str is None:
            return None
        return LoupeBackendService._to_int(value_str)
