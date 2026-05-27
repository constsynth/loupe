from __future__ import annotations
from pathlib import Path

import torch
from interpretability.sae.sae import SAE


def resolve_project_path(path: str | Path) -> Path:
    """Resolve user paths relative to the repository root."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def infer_sae_dimensions(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Infer `(hidden_size, latent_size)` from the current SAE state_dict format."""
    encoder_weight = state_dict.get("encoder.0.weight")
    decoder_weight = state_dict.get("decoder.weight")
    if encoder_weight is None or decoder_weight is None:
        raise ValueError(
            "SAE checkpoint must use the current explicit format with "
            "`encoder.0.weight` and `decoder.weight` keys"
        )
    if encoder_weight.ndim != 2 or decoder_weight.ndim != 2:
        raise ValueError("SAE encoder and decoder weights must be 2D tensors")

    latent_size, hidden_size = encoder_weight.shape
    decoder_hidden_size, decoder_latent_size = decoder_weight.shape
    if (hidden_size, latent_size) != (decoder_hidden_size, decoder_latent_size):
        raise ValueError(
            "SAE checkpoint encoder/decoder dimensions are inconsistent: "
            f"encoder={tuple(encoder_weight.shape)}, decoder={tuple(decoder_weight.shape)}"
        )
    return int(hidden_size), int(latent_size)


def load_sae_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
) -> SAE:
    """
    Load an SAE checkpoint for inference.

    The returned SAE preserves the dissertation notation `h -> z -> h_hat` and
    exposes token-level latent activations through `return_output=True`.
    Only the current explicit checkpoint format is supported:
    `encoder.*` and `decoder.*` keys.
    """
    resolved_path = resolve_project_path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("SAE checkpoint must be a state_dict or contain a state_dict key")

    hidden_size, latent_size = infer_sae_dimensions(checkpoint)
    use_batch_norm = "encoder.1.weight" in checkpoint
    sae = SAE(
        in_hidden_state_size=hidden_size,
        sparse_hidden_state_size=latent_size,
        device=device,
        dtype=dtype,
        use_batch_norm=use_batch_norm,
    )
    sae.load_state_dict(checkpoint, strict=True)
    sae.eval()
    return sae
