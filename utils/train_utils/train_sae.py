import os
import torch
from torch import nn
from torch import optim
import typing as tp
from tqdm import tqdm
from torch.utils.data import DataLoader
from interpretability.sae.sae import SAE


def train_sae(
    sae: SAE,
    dataloader: DataLoader,
    path_to_save: str,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = 'cuda'
) -> str:
    """
    Обучает SAE на данных активаций (скрытые состояния).
    Сохраняет итоговую модель и возвращает путь к файлу.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    sae = sae.to(device)
    criterion = nn.MSELoss()  # reconstruction loss
    optimizer = optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)

    sae.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            # hidden states tensor shape [batch_size, in_hidden_state_size]
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = sae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
            cleanup_memory()
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")

    # Saving into a dir
    torch.save(sae.state_dict(), path_to_save)
    print(f"Saved SAE model to {path_to_save}")
    return path_to_save

def cleanup_memory():
    torch.cuda.empty_cache()