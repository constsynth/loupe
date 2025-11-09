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
    def cleanup_memory():
        torch.cuda.empty_cache()

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
                sparse_hidden_state_size,
                dtype=torch.bfloat16
            ),
            nn.BatchNorm1d(sparse_hidden_state_size, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(
                sparse_hidden_state_size,
                in_hidden_state_size,
                dtype=torch.bfloat16
            )
        )
        return sae
    
    def forward(
        self,
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        
        """
        Arguments:
            hidden_state: torch.Tensor with hidden states
        """
        reconstructed_hidden_states = self.sae(hidden_state)
        self.cleanup_memory()
        return reconstructed_hidden_states
    