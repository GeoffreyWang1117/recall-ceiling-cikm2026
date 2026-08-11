"""Base GNN encoder interface"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseGNN(nn.Module, ABC):
    """Base class for GNN encoders"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0
    ):
        """
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Learnable embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with Xavier uniform"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    @abstractmethod
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)

        Returns:
            user_embeddings: [num_users, embedding_dim]
            item_embeddings: [num_items, embedding_dim]
        """
        pass

    def get_embedding(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user and item embeddings (inference mode)"""
        self.eval()
        with torch.no_grad():
            return self.forward(edge_index, edge_weight)

    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict user-item scores

        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size] or [batch_size, num_items]
            user_emb: User embeddings [num_users, dim]
            item_emb: Item embeddings [num_items, dim]

        Returns:
            scores: [batch_size] or [batch_size, num_items]
        """
        user_embed = user_emb[user_ids]  # [batch_size, dim]

        if item_ids.dim() == 1:
            # Single item per user
            item_embed = item_emb[item_ids]  # [batch_size, dim]
            scores = (user_embed * item_embed).sum(dim=1)  # [batch_size]
        else:
            # Multiple items per user
            item_embed = item_emb[item_ids]  # [batch_size, num_items, dim]
            scores = torch.bmm(
                item_embed,
                user_embed.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size, num_items]

        return scores
