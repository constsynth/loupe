from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SAEForwardOutput:
    """
    Output container for SAE experiments.

    Shapes:
        hidden_state: [..., hidden]
        latent_activation: [..., latent]
        reconstructed_hidden_state: [..., hidden]
    """

    hidden_state: torch.Tensor
    latent_activation: torch.Tensor
    reconstructed_hidden_state: torch.Tensor


@dataclass
class SAELossOutput:
    """Loss container with reconstruction and sparsity terms."""

    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    output: SAEForwardOutput


class SAE(nn.Module):
    """
    Sparse autoencoder for transformer activations.

    The module is structured around the experiment notation from the thesis:
    `h -> z -> h_hat`, where `h` is the original activation, `z` is the sparse
    latent activation, and `h_hat` is the reconstructed activation.

    Supported input shapes:
        [batch, hidden]
        [batch, sequence, hidden]
        any shape ending with `hidden`
    """

    def __init__(
        self,
        in_hidden_state_size: int,
        sparse_hidden_state_size: int | None = None,
        sparsity_factor: int | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        use_batch_norm: bool = True,
        top_k: int | None = None,
    ) -> None:
        super().__init__()

        if in_hidden_state_size <= 0:
            raise ValueError("in_hidden_state_size must be positive")

        self.in_hidden_state_size = in_hidden_state_size
        self.sparse_hidden_state_size = self._resolve_sparse_hidden_state_size(
            in_hidden_state_size=in_hidden_state_size,
            sparse_hidden_state_size=sparse_hidden_state_size,
            sparsity_factor=sparsity_factor,
        )
        self.top_k = top_k

        device = self._resolve_device(device)
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)

        self.encoder, self.decoder = self.create_encoder_decoder(
            in_hidden_state_size=self.in_hidden_state_size,
            sparse_hidden_state_size=self.sparse_hidden_state_size,
            dtype=dtype,
            use_batch_norm=use_batch_norm,
        )
        self.to(device=device)

    @staticmethod
    def _resolve_sparse_hidden_state_size(
        in_hidden_state_size: int,
        sparse_hidden_state_size: int | None,
        sparsity_factor: int | None,
    ) -> int:
        if sparsity_factor is not None:
            if sparsity_factor <= 0:
                raise ValueError("sparsity_factor must be positive")
            return in_hidden_state_size * sparsity_factor

        if sparse_hidden_state_size is None or sparse_hidden_state_size <= 0:
            raise ValueError(
                "sparse_hidden_state_size must be positive when sparsity_factor is not provided"
            )
        return sparse_hidden_state_size

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    @staticmethod
    def cleanup_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def create_encoder_decoder(
        in_hidden_state_size: int,
        sparse_hidden_state_size: int,
        dtype: torch.dtype = torch.bfloat16,
        use_batch_norm: bool = True,
    ) -> tuple[nn.Sequential, nn.Linear]:
        """
        Create encoder `f_theta` and decoder `g_theta`.

        Encoder output is the latent activation `z`. Decoder columns are SAE
        feature directions in the original activation space.
        """
        encoder_layers: list[nn.Module] = [
            nn.Linear(
                in_hidden_state_size,
                sparse_hidden_state_size,
                dtype=dtype,
            )
        ]
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(sparse_hidden_state_size, dtype=dtype))
        encoder_layers.append(nn.ReLU())

        decoder = nn.Linear(
            sparse_hidden_state_size,
            in_hidden_state_size,
            dtype=dtype,
        )
        return nn.Sequential(*encoder_layers), decoder

    def _flatten_hidden_state(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        if hidden_state.shape[-1] != self.in_hidden_state_size:
            raise ValueError(
                f"Expected last dim {self.in_hidden_state_size}, got {hidden_state.shape[-1]}"
            )
        prefix_shape = hidden_state.shape[:-1]
        return hidden_state.reshape(-1, self.in_hidden_state_size), prefix_shape

    @staticmethod
    def _restore_prefix_shape(tensor: torch.Tensor, prefix_shape: torch.Size) -> torch.Tensor:
        return tensor.reshape(*prefix_shape, tensor.shape[-1])

    def _parameter_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype

    def _apply_top_k(self, latent_activation: torch.Tensor) -> torch.Tensor:
        if self.top_k is None:
            return latent_activation
        if self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        top_k = min(self.top_k, latent_activation.shape[-1])
        _, indices = torch.topk(latent_activation.abs(), k=top_k, dim=-1)
        mask = torch.zeros_like(latent_activation, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)
        return latent_activation * mask

    def encode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Encode transformer activations `h` into sparse latents `z`.

        Args:
            hidden_state: Tensor with shape [..., hidden].

        Returns:
            Tensor with shape [..., latent].
        """
        flat_hidden_state, prefix_shape = self._flatten_hidden_state(hidden_state)
        device, dtype = self._parameter_device_dtype()
        flat_hidden_state = flat_hidden_state.to(device=device, dtype=dtype)
        latent_activation = self.encoder(flat_hidden_state)
        latent_activation = self._apply_top_k(latent_activation)
        return self._restore_prefix_shape(latent_activation, prefix_shape)

    def decode(self, latent_activation: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse latents `z` into reconstructed activations `h_hat`.

        Args:
            latent_activation: Tensor with shape [..., latent].

        Returns:
            Tensor with shape [..., hidden].
        """
        if latent_activation.shape[-1] != self.sparse_hidden_state_size:
            raise ValueError(
                f"Expected last dim {self.sparse_hidden_state_size}, got {latent_activation.shape[-1]}"
            )
        prefix_shape = latent_activation.shape[:-1]
        device, dtype = self._parameter_device_dtype()
        flat_latent = latent_activation.reshape(-1, self.sparse_hidden_state_size)
        flat_latent = flat_latent.to(device=device, dtype=dtype)
        reconstructed_hidden_state = self.decoder(flat_latent)
        return self._restore_prefix_shape(reconstructed_hidden_state, prefix_shape)

    def forward(
        self,
        hidden_state: torch.Tensor,
        return_latent: bool = False,
        return_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | SAEForwardOutput:
        """
        Run `h -> z -> h_hat`.

        Args:
            hidden_state: Tensor with shape [..., hidden].
            return_latent: Return `(h_hat, z)` for lightweight experiment code.
            return_output: Return `SAEForwardOutput` with `h`, `z`, and `h_hat`.

        Returns:
            By default, only reconstructed hidden states.
        """
        latent_activation = self.encode(hidden_state)
        reconstructed_hidden_state = self.decode(latent_activation)
        reconstructed_hidden_state = reconstructed_hidden_state.to(
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )

        if return_output:
            return SAEForwardOutput(
                hidden_state=hidden_state,
                latent_activation=latent_activation,
                reconstructed_hidden_state=reconstructed_hidden_state,
            )
        if return_latent:
            return reconstructed_hidden_state, latent_activation
        return reconstructed_hidden_state

    def loss(
        self,
        hidden_state: torch.Tensor,
        sparsity_lambda: float = 0.0,
    ) -> SAELossOutput:
        """
        Compute SAE loss: `MSE(h_hat, h) + lambda * mean(abs(z))`.

        The reconstruction term supports information-preservation experiments;
        the sparsity term supports the density/interpretable-feature tradeoff.
        """
        output = self.forward(hidden_state, return_output=True)
        reconstruction_loss = F.mse_loss(output.reconstructed_hidden_state, hidden_state)
        sparsity_loss = output.latent_activation.abs().mean()
        total_loss = reconstruction_loss + sparsity_lambda * sparsity_loss
        return SAELossOutput(
            loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            output=output,
        )

    def reconstruction_metrics(
        self,
        hidden_state: torch.Tensor,
        reconstructed_hidden_state: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """
        Compute reconstruction metrics comparing `h` with `h_hat`.

        Returns MSE, NMSE, and mean cosine similarity. These metrics must not be
        applied directly between `h` and latent activations `z`.
        """
        if reconstructed_hidden_state is None:
            reconstructed_hidden_state = self.forward(hidden_state)

        return {
            "mse": self.reconstruction_mse(hidden_state, reconstructed_hidden_state),
            "nmse": self.normalized_mse(hidden_state, reconstructed_hidden_state, eps=eps),
            "cosine_similarity": self.mean_cosine_similarity(
                hidden_state,
                reconstructed_hidden_state,
                eps=eps,
            ),
        }

    def sparsity_metrics(
        self,
        latent_activation: torch.Tensor,
        threshold: float = 0.0,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """
        Compute sparsity metrics on latent activations `z`.

        Returns L0, active feature share, Hoyer sparsity, and normalized entropy.
        """
        return {
            "l0": self.l0_norm(latent_activation, threshold=threshold),
            "active_feature_share": self.active_feature_share(
                latent_activation,
                threshold=threshold,
            ),
            "hoyer_sparsity": self.hoyer_sparsity(latent_activation, eps=eps),
            "normalized_entropy": self.normalized_activation_entropy(
                latent_activation,
                eps=eps,
            ),
        }

    def experiment_metrics(
        self,
        hidden_state: torch.Tensor,
        threshold: float = 0.0,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """
        Compute first-level metrics for held-out SAE hypothesis checks.

        This covers reconstruction quality and latent sparsity. Distribution,
        concept-separability, and intervention tests should use the returned
        `SAEForwardOutput` or dedicated downstream code.
        """
        output = self.forward(hidden_state, return_output=True)
        return {
            **self.reconstruction_metrics(
                output.hidden_state,
                output.reconstructed_hidden_state,
                eps=eps,
            ),
            **self.sparsity_metrics(
                output.latent_activation,
                threshold=threshold,
                eps=eps,
            ),
        }

    def intervene_latent(
        self,
        latent_activation: torch.Tensor,
        feature_indices: int | list[int] | torch.Tensor,
        intervention_value: float = 1.0,
        token_positions: int | list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Add a scalar value to selected SAE features in `z`.

        Args:
            latent_activation: Tensor with shape [..., latent].
            feature_indices: SAE feature indices to change.
            intervention_value: Scalar intervention magnitude.
            token_positions: Optional token positions for tensors shaped
                [batch, sequence, latent]. If omitted, all positions are changed.

        Returns:
            Modified latent activation `z_prime`.
        """
        feature_indices = self._normalize_indices(
            feature_indices,
            upper_bound=latent_activation.shape[-1],
            device=latent_activation.device,
            name="feature_indices",
        )
        token_positions = (
            None
            if token_positions is None
            else self._normalize_indices(
                token_positions,
                upper_bound=latent_activation.shape[-2],
                device=latent_activation.device,
                name="token_positions",
            )
        )

        modified_latent = latent_activation.clone()
        mask = torch.zeros_like(modified_latent, dtype=torch.bool)

        for feature_idx in feature_indices.tolist():
            if token_positions is None:
                mask[..., feature_idx] = True
            else:
                if modified_latent.ndim < 3:
                    raise ValueError("token_positions require latent shape [batch, sequence, latent]")
                mask[:, token_positions, feature_idx] = True

        modified_latent[mask] = modified_latent[mask] + intervention_value

        return modified_latent

    def intervene(
        self,
        hidden_state: torch.Tensor,
        feature_indices: int | list[int] | torch.Tensor,
        intervention_value: float = 1.0,
        token_positions: int | list[int] | torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode `h`, intervene on selected SAE features, and decode `z_prime`.

        This supports the theory document's intervention path:
        `z -> z_prime -> h_prime`.
        """
        latent_activation = self.encode(hidden_state)
        modified_latent = self.intervene_latent(
            latent_activation=latent_activation,
            feature_indices=feature_indices,
            intervention_value=intervention_value,
            token_positions=token_positions,
        )
        modified_hidden_state = self.decode(modified_latent)
        modified_hidden_state = modified_hidden_state.to(
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        if return_latent:
            return modified_hidden_state, modified_latent
        return modified_hidden_state

    def decoder_directions(
        self,
        feature_indices: int | list[int] | torch.Tensor | None = None,
        normalize: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Return decoder feature directions.

        Decoder weight has shape [hidden, latent], so returned directions have
        shape [selected_features, hidden].
        """
        directions = self.decoder.weight.transpose(0, 1)
        if feature_indices is not None:
            feature_indices = self._normalize_indices(
                feature_indices,
                upper_bound=directions.shape[0],
                device=directions.device,
                name="feature_indices",
            )
            directions = directions.index_select(dim=0, index=feature_indices)
        if normalize:
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(eps)
        return directions

    def decoder_direction_for_feature(
        self,
        feature_indices: int | list[int] | torch.Tensor,
        intervention_value: float = 1.0,
        normalize: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Return the combined decoder direction for additive SAE steering.

        For a linear decoder, changing selected latent features as
        `z_prime_j = z_j + alpha` changes the decoded activation by
        `alpha * W_dec[:, j]`. For multiple features this method returns the
        sum of their decoder directions with the same scalar coefficient.

        Returns:
            Tensor with shape [hidden].
        """
        directions = self.decoder_directions(
            feature_indices=feature_indices,
            normalize=normalize,
            eps=eps,
        )
        return directions.sum(dim=0) * intervention_value

    @staticmethod
    def top_activated_features(
        latent_activation: torch.Tensor,
        k: int = 10,
        aggregate: str = "mean_abs",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return top SAE feature indices and scores.

        Args:
            latent_activation: Tensor with shape [..., latent].
            k: Number of features to return.
            aggregate: `mean_abs`, `mean`, or `max_abs`.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        flat_latent = latent_activation.reshape(-1, latent_activation.shape[-1])
        if aggregate == "mean_abs":
            scores = flat_latent.abs().mean(dim=0)
        elif aggregate == "mean":
            scores = flat_latent.mean(dim=0)
        elif aggregate == "max_abs":
            scores = flat_latent.abs().max(dim=0).values
        else:
            raise ValueError("aggregate must be one of: mean_abs, mean, max_abs")
        scores, indices = torch.topk(scores, k=min(k, scores.numel()))
        return indices, scores

    @staticmethod
    def reconstruction_mse(
        hidden_state: torch.Tensor,
        reconstructed_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean squared reconstruction error between `h` and `h_hat`."""
        return F.mse_loss(reconstructed_hidden_state, hidden_state)

    @staticmethod
    def normalized_mse(
        hidden_state: torch.Tensor,
        reconstructed_hidden_state: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Return NMSE = E||h - h_hat||^2 / E||h - mean(h)||^2.

        The mean activation is computed over all samples/tokens, preserving the
        hidden dimension.
        """
        hidden_flat = hidden_state.reshape(-1, hidden_state.shape[-1]).float()
        reconstructed_flat = reconstructed_hidden_state.reshape(
            -1,
            reconstructed_hidden_state.shape[-1],
        ).float()
        mean_hidden = hidden_flat.mean(dim=0, keepdim=True)

        numerator = (hidden_flat - reconstructed_flat).pow(2).sum(dim=-1).mean()
        denominator = (hidden_flat - mean_hidden).pow(2).sum(dim=-1).mean()
        return numerator / denominator.clamp_min(eps)

    @staticmethod
    def mean_cosine_similarity(
        hidden_state: torch.Tensor,
        reconstructed_hidden_state: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return mean cosine similarity between `h` and `h_hat`."""
        hidden_flat = hidden_state.reshape(-1, hidden_state.shape[-1]).float()
        reconstructed_flat = reconstructed_hidden_state.reshape(
            -1,
            reconstructed_hidden_state.shape[-1],
        ).float()
        return F.cosine_similarity(hidden_flat, reconstructed_flat, dim=-1, eps=eps).mean()

    @staticmethod
    def l0_norm(latent_activation: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Return mean number of active latent features per sample/token."""
        flat_latent = latent_activation.reshape(-1, latent_activation.shape[-1])
        return (flat_latent.abs() > threshold).sum(dim=-1).float().mean()

    @staticmethod
    def active_feature_share(
        latent_activation: torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Return share of active entries in the latent activation tensor."""
        return (latent_activation.abs() > threshold).float().mean()

    @staticmethod
    def hoyer_sparsity(latent_activation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Return mean Hoyer sparsity over samples/tokens.

        Values are near 0 for dense vectors and near 1 for highly sparse vectors.
        """
        flat_latent = latent_activation.reshape(-1, latent_activation.shape[-1]).float()
        n_features = flat_latent.shape[-1]
        if n_features <= 1:
            return torch.zeros((), device=latent_activation.device)

        l1 = flat_latent.abs().sum(dim=-1)
        l2 = flat_latent.norm(p=2, dim=-1).clamp_min(eps)
        sqrt_n = torch.sqrt(torch.tensor(float(n_features), device=flat_latent.device))
        sparsity = (sqrt_n - (l1 / l2)) / (sqrt_n - 1.0)
        return sparsity.clamp(0.0, 1.0).mean()

    @staticmethod
    def normalized_activation_entropy(
        latent_activation: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Return normalized entropy of absolute latent activation mass.

        Lower values indicate activation mass concentrated in fewer features.
        """
        flat_latent = latent_activation.reshape(-1, latent_activation.shape[-1]).float().abs()
        n_features = flat_latent.shape[-1]
        if n_features <= 1:
            return torch.zeros((), device=latent_activation.device)

        probabilities = flat_latent / flat_latent.sum(dim=-1, keepdim=True).clamp_min(eps)
        entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
        normalizer = torch.log(torch.tensor(float(n_features), device=flat_latent.device))
        return (entropy / normalizer.clamp_min(eps)).mean()

    @staticmethod
    def maximum_mean_discrepancy(
        original_activation: torch.Tensor,
        reconstructed_activation: torch.Tensor,
        bandwidth: float | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Estimate squared RBF-kernel MMD between `h` and `h_hat`.

        This supports distribution-preservation checks. For large activation
        sets, subsample before calling this method to control O(n^2) memory.
        """
        x = original_activation.reshape(-1, original_activation.shape[-1]).float()
        y = reconstructed_activation.reshape(-1, reconstructed_activation.shape[-1]).float()

        if bandwidth is None:
            xy = torch.cat([x, y], dim=0)
            sample_size = min(512, xy.shape[0])
            sample = xy[:sample_size]
            distances = torch.pdist(sample).pow(2)
            bandwidth = distances.median().clamp_min(eps).sqrt().item() if distances.numel() else 1.0

        gamma = 1.0 / (2.0 * max(bandwidth, eps) ** 2)
        k_xx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
        k_yy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
        k_xy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
        return k_xx + k_yy - 2.0 * k_xy

    @staticmethod
    def _normalize_indices(
        indices: int | list[int] | torch.Tensor,
        upper_bound: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if isinstance(indices, int):
            indices = [indices]
        indices = torch.as_tensor(indices, dtype=torch.long, device=device).flatten()
        if indices.numel() == 0:
            raise ValueError(f"{name} must contain at least one index")
        if (indices < 0).any() or (indices >= upper_bound).any():
            raise IndexError(f"{name} must be in [0, {upper_bound})")
        return indices
