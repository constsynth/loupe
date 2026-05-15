import math
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score

from interpretability.sae.sae import SAE


def flatten_valid_tokens(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return tensor rows for non-padding token positions."""
    return tensor[mask].reshape(-1, tensor.shape[-1]).float()


def masked_mean_pool(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token activations over non-padding positions."""
    weights = mask.float().unsqueeze(-1)
    return (tensor.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_mean_abs_pool(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool absolute token activations over non-padding positions."""
    weights = mask.float().unsqueeze(-1)
    return (tensor.float().abs() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def cosine_per_valid_token(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Return cosine similarity between `h` and `h_hat` for valid tokens."""
    original_flat = flatten_valid_tokens(original, mask)
    reconstructed_flat = flatten_valid_tokens(reconstructed, mask)
    return F.cosine_similarity(original_flat, reconstructed_flat, dim=-1).cpu().numpy()


def reconstruction_statistics(
    sae: SAE,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
    nmse_tau: float,
    alpha: float = 0.05,
) -> tuple[dict[str, tp.Any], np.ndarray]:
    """
    Compute information-preservation statistics.

    Uses `SAE.normalized_mse` for global NMSE. Per-token squared errors are
    returned only for plotting the reconstruction error distribution.
    """
    valid_original = flatten_valid_tokens(original, mask)
    valid_reconstructed = flatten_valid_tokens(reconstructed, mask)
    nmse = float(sae.normalized_mse(valid_original, valid_reconstructed).item())
    cosine_values = cosine_per_valid_token(original, reconstructed, mask)
    per_token_squared_error = (
        (valid_original - valid_reconstructed).pow(2).mean(dim=-1).cpu().numpy()
    )

    # A one-sample test over token-level errors is only a diagnostic proxy. The
    # thesis-level decision should primarily use held-out global NMSE.
    normalized_token_errors = per_token_squared_error / (
        valid_original.var(dim=0).sum().clamp_min(1e-8).item()
    )
    nmse_test = (
        stats.ttest_1samp(normalized_token_errors, popmean=nmse_tau, alternative="less")
        if len(normalized_token_errors) > 1
        else None
    )

    return (
        {
            "nmse": nmse,
            "mean_cosine": float(np.mean(cosine_values)),
            "median_cosine": float(np.median(cosine_values)),
            "nmse_tau": nmse_tau,
            "token_error_ttest_pvalue": None if nmse_test is None else float(nmse_test.pvalue),
            "passed": bool(nmse < nmse_tau),
        },
        per_token_squared_error,
    )


def hoyer_per_row(x: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Return Hoyer sparsity per row."""
    x = x.float().abs()
    n_features = x.shape[-1]
    if n_features <= 1:
        return np.zeros(x.shape[0])
    l1 = x.sum(dim=-1)
    l2 = x.norm(p=2, dim=-1).clamp_min(eps)
    sqrt_n = math.sqrt(float(n_features))
    return (((sqrt_n - (l1 / l2)) / (sqrt_n - 1.0)).clamp(0.0, 1.0)).cpu().numpy()


def histogram_jsd(x: np.ndarray, y: np.ndarray, bins: int = 100) -> float:
    """Return Jensen-Shannon divergence between two 1D histograms."""
    hist_range = (
        min(float(np.min(x)), float(np.min(y))),
        max(float(np.max(x)), float(np.max(y))),
    )
    px, _ = np.histogram(x, bins=bins, range=hist_range, density=True)
    py, _ = np.histogram(y, bins=bins, range=hist_range, density=True)
    px = px + 1e-12
    py = py + 1e-12
    px = px / px.sum()
    py = py / py.sum()
    return float(jensenshannon(px, py, base=2.0) ** 2)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return boolean rejection mask using Benjamini-Hochberg FDR control."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = alpha * (np.arange(1, len(p_values) + 1) / len(p_values))
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros_like(p_values, dtype=bool)
    max_idx = np.where(passed)[0].max()
    rejected_sorted = np.zeros_like(p_values, dtype=bool)
    rejected_sorted[: max_idx + 1] = True
    rejected = np.zeros_like(p_values, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected


def distribution_preservation_statistics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
    mmd_epsilon: float,
    alpha: float = 0.05,
    max_tokens_for_mmd: int = 1024,
    max_dims_for_ks: int = 256,
) -> tuple[dict[str, tp.Any], dict[str, np.ndarray]]:
    """Compute distribution-preservation statistics for `h` and `h_hat`."""
    valid_original = flatten_valid_tokens(original, mask)
    valid_reconstructed = flatten_valid_tokens(reconstructed, mask)

    n_mmd = min(max_tokens_for_mmd, valid_original.shape[0])
    mmd_indices = torch.randperm(valid_original.shape[0])[:n_mmd]
    mmd_value = float(
        SAE.maximum_mean_discrepancy(
            valid_original[mmd_indices],
            valid_reconstructed[mmd_indices],
        ).item()
    )

    n_dims = min(max_dims_for_ks, valid_original.shape[1])
    dimension_indices = np.linspace(0, valid_original.shape[1] - 1, n_dims, dtype=int)
    ks_p_values = []
    ks_statistics = []
    wasserstein_values = []

    for dim in dimension_indices:
        x = valid_original[:, dim].cpu().numpy()
        y = valid_reconstructed[:, dim].cpu().numpy()
        ks = stats.ks_2samp(x, y)
        ks_statistics.append(float(ks.statistic))
        ks_p_values.append(float(ks.pvalue))
        wasserstein_values.append(float(stats.wasserstein_distance(x, y)))

    ks_p_values = np.array(ks_p_values)
    ks_statistics = np.array(ks_statistics)
    wasserstein_values = np.array(wasserstein_values)
    ks_rejected = benjamini_hochberg(ks_p_values, alpha=alpha)
    jsd_value = histogram_jsd(
        valid_original.flatten().cpu().numpy(),
        valid_reconstructed.flatten().cpu().numpy(),
    )

    return (
        {
            "mmd": mmd_value,
            "mean_ks_statistic": float(np.mean(ks_statistics)),
            "ks_rejected_share_bh": float(np.mean(ks_rejected)),
            "mean_wasserstein": float(np.mean(wasserstein_values)),
            "histogram_jsd": jsd_value,
            "passed": bool(mmd_value < mmd_epsilon and np.mean(ks_rejected) < 0.5),
        },
        {
            "ks_statistics": ks_statistics,
            "ks_p_values": ks_p_values,
            "ks_rejected": ks_rejected,
            "wasserstein_values": wasserstein_values,
        },
    )


def sparsity_statistics(
    sae: SAE,
    original: torch.Tensor,
    latent: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.0,
    alpha: float = 0.05,
) -> tuple[dict[str, tp.Any], dict[str, np.ndarray]]:
    """Compute sparsity statistics comparing dense `h` and SAE latents `z`."""
    valid_original = flatten_valid_tokens(original, mask)
    valid_latent = flatten_valid_tokens(latent, mask)

    z_hoyer = hoyer_per_row(valid_latent)
    h_hoyer = hoyer_per_row(valid_original)
    n = min(len(z_hoyer), len(h_hoyer))
    hoyer_test = stats.wilcoxon(z_hoyer[:n] - h_hoyer[:n], alternative="greater") if n > 1 else None
    latent_sparsity = sae.sparsity_metrics(valid_latent.to(next(sae.parameters()).device), threshold=threshold)

    return (
        {
            "z_l0": float(latent_sparsity["l0"].item()),
            "z_active_feature_share": float(latent_sparsity["active_feature_share"].item()),
            "z_hoyer": float(np.mean(z_hoyer)),
            "h_hoyer": float(np.mean(h_hoyer)),
            "z_normalized_entropy": float(latent_sparsity["normalized_entropy"].item()),
            "hoyer_wilcoxon_pvalue": None if hoyer_test is None else float(hoyer_test.pvalue),
            "passed": bool(np.mean(z_hoyer) > np.mean(h_hoyer) and (hoyer_test is None or hoyer_test.pvalue < alpha)),
        },
        {
            "z_hoyer": z_hoyer,
            "h_hoyer": h_hoyer,
        },
    )


def separability_statistics(
    original: torch.Tensor,
    latent: torch.Tensor,
    mask: torch.Tensor,
    concept_labels: tp.Sequence[str],
) -> tuple[dict[str, tp.Any], torch.Tensor, torch.Tensor]:
    """Compare concept separability in dense `h` and latent `z` sample spaces."""
    sample_h = masked_mean_pool(original, mask)
    sample_z = masked_mean_abs_pool(latent, mask)
    labels = [str(label) for label in concept_labels]
    unique_labels = sorted(set(labels))

    if len(unique_labels) > 1 and len(labels) > len(unique_labels):
        silhouette_h = float(silhouette_score(sample_h.numpy(), labels, metric="cosine"))
        silhouette_z = float(silhouette_score(sample_z.numpy(), labels, metric="cosine"))
    else:
        silhouette_h = np.nan
        silhouette_z = np.nan

    return (
        {
            "silhouette_h": silhouette_h,
            "silhouette_z": silhouette_z,
            "passed": bool(np.isfinite(silhouette_h) and np.isfinite(silhouette_z) and silhouette_z > silhouette_h),
        },
        sample_h,
        sample_z,
    )


def next_token_distribution(llm, prompt: str, max_length: int = 256) -> torch.Tensor:
    """Return next-token probability distribution for a prompt."""
    inputs = llm.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(llm.device)
    with torch.no_grad():
        logits = llm.model(**inputs).logits[:, -1, :]
    return torch.softmax(logits.float().cpu(), dim=-1).squeeze(0)


def target_token_id(llm, token_text: str) -> int:
    """Return first tokenizer id for target token text."""
    ids = llm.tokenizer(str(token_text), add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Could not tokenize target token: {token_text}")
    return int(ids[0])


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Return KL(p || q)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p.log() - q.log())).sum().item())
