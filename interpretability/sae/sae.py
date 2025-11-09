import torch
import typing as tp
from torch import nn


class SAE(nn.Module):
    
    def __init__(
        self,
        in_hidden_state_size: int,
        sparse_hidden_state_size: int,
        sparsity_factor: int = None,
        device: str = 'cuda'
    ):
        super().__init__()

        self.sae = self.create_sae(
            in_hidden_state_size,
            sparse_hidden_state_size,
            sparsity_factor
        ).to(device)

    @staticmethod
    def create_sae(
        in_hidden_state_size: int,
        sparse_hidden_state_size: int,
        sparsity_factor: int = None,
    ) -> nn.Sequential:
        sparse_hidden_state_size = in_hidden_state_size * sparsity_factor if sparsity_factor else sparse_hidden_state_size
        sae = nn.Sequential(
            nn.Linear(
                in_hidden_state_size,
                sparse_hidden_state_size
            ),
            nn.BatchNorm1d(sparse_hidden_state_size),
            nn.ReLU(),
            nn.Linear(
                sparse_hidden_state_size,
                in_hidden_state_size
            )
        )
        return sae
    
    def forward(
        self,
        hidden_state: torch.Tensor
    ) -> torch.tensor:
        
        """
        Arguments:
            hidden_state: torch.Tensor with hidden states
        """
        return self.sae(hidden_state)
    