import torch
import copy
from pathlib import Path
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from interpretability.sae.sae import SAE


def train_sae(
    sae: SAE,
    dataloader: DataLoader,
    checkpoint_name: str = "sae.pt",
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    sparsity_lambda: float = 0.0,
    return_plot: bool = False,
) -> dict[str, list[float]]:
    """
    Train SAE on transformer activation vectors.

    Args:
        sae: Sparse autoencoder to train.
        dataloader: Activation batches. Each batch must contain hidden states
            shaped [batch_size, hidden] or any shape supported by `SAE.loss`.
        checkpoint_name: File name for the saved SAE state_dict. The file is
            always saved under the repository root `models` directory.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        sparsity_lambda: L1 sparsity coefficient for latent activations `z`.
        return_plot: If True, show reconstruction, sparsity, and total loss
            curves after training.

    Returns:
        Dict with per-epoch `reconstruction_loss`, `sparsity_loss`, and
        `total_loss` histories.
    """
    device = next(sae.parameters()).device
    optimizer = optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, list[float]] = {
        "reconstruction_loss": [],
        "sparsity_loss": [],
        "total_loss": [],
    }
    best_epoch = 0
    best_losses: dict[str, float] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None

    sae.train()
    for epoch in range(1, epochs + 1):
        running: dict[str, float] = {
            "reconstruction_loss": 0.0,
            "sparsity_loss": 0.0,
            "total_loss": 0.0,
        }
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            # hidden states tensor shape [batch_size, in_hidden_state_size]
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            batch_size = batch.size(0)
            optimizer.zero_grad()
            loss_output = sae.loss(batch, sparsity_lambda=sparsity_lambda)
            loss_output.loss.backward()
            optimizer.step()
            running["reconstruction_loss"] += (
                loss_output.reconstruction_loss.item() * batch_size
            )
            running["sparsity_loss"] += loss_output.sparsity_loss.item() * batch_size
            running["total_loss"] += loss_output.loss.item() * batch_size

        dataset_size = _dataset_size(dataloader)
        epoch_losses = {
            key: value / dataset_size
            for key, value in running.items()
        }
        for key, value in epoch_losses.items():
            history[key].append(value)

        if best_losses is None or epoch_losses["total_loss"] < best_losses["total_loss"]:
            best_epoch = epoch
            best_losses = epoch_losses.copy()
            best_state_dict = copy.deepcopy(sae.state_dict())

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"reconstruction_loss={epoch_losses['reconstruction_loss']:.6f} | "
            f"sparsity_loss={epoch_losses['sparsity_loss']:.6f} | "
            f"total_loss={epoch_losses['total_loss']:.6f}"
        )
        cleanup_memory()

    if best_state_dict is None or best_losses is None:
        raise RuntimeError("SAE training did not produce a checkpoint state")

    sae.load_state_dict(best_state_dict)
    checkpoint_path = _checkpoint_path(checkpoint_name)
    torch.save(sae.state_dict(), checkpoint_path)
    print(
        f"Saved best SAE model to {checkpoint_path} | "
        f"best_epoch={best_epoch:03d}/{epochs:03d} | "
        f"reconstruction_loss={best_losses['reconstruction_loss']:.6f} | "
        f"sparsity_loss={best_losses['sparsity_loss']:.6f} | "
        f"total_loss={best_losses['total_loss']:.6f}"
    )

    if return_plot:
        _plot_training_history(history)

    return history


def _checkpoint_path(checkpoint_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    save_dir = repo_root / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / checkpoint_name


def _dataset_size(dataloader: DataLoader) -> int:
    try:
        return len(dataloader.dataset)
    except TypeError:
        return sum(
            batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
            for batch in dataloader
        )


def _plot_training_history(history: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["total_loss"]) + 1)
    loss_names = ("total_loss", "reconstruction_loss", "sparsity_loss")
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)

    for axis, loss_name in zip(axes, loss_names):
        axis.plot(epochs, history[loss_name], label=loss_name)
        axis.set_title(loss_name)
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("SAE training losses")
    plt.tight_layout()
    plt.show()


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
