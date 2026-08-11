"""LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gnn import BaseGNN


class LightGCN(BaseGNN):
    """
    LightGCN model for collaborative filtering

    Reference:
    He et al. "LightGCN: Simplifying and Powering Graph Convolution Network
    for Recommendation" (SIGIR 2020)

    Key idea: Remove feature transformation and nonlinear activation,
    only keep neighbor aggregation
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        add_self_loops: bool = False
    ):
        """
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Embedding dimension
            num_layers: Number of LightGCN layers
            dropout: Dropout rate (applied to embeddings)
            add_self_loops: Whether to add self loops
        """
        super().__init__(num_users, num_items, embedding_dim, num_layers, dropout)
        self.add_self_loops = add_self_loops

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN forward pass

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]

        Returns:
            user_embeddings: [num_users, embedding_dim]
            item_embeddings: [num_items, embedding_dim]
        """
        # Initial embeddings
        user_emb_0 = self.user_embedding.weight
        item_emb_0 = self.item_embedding.weight

        # Combine user and item embeddings
        # Note: In bipartite graph, items are indexed as [num_users, num_users + num_items)
        all_emb_0 = torch.cat([user_emb_0, item_emb_0], dim=0)

        # Normalize adjacency matrix
        edge_index, edge_weight = self._normalize_adj(edge_index, edge_weight)

        # Layer-wise propagation
        all_embs = [all_emb_0]

        for layer in range(self.num_layers):
            # Graph convolution: aggregate neighbor embeddings
            all_emb = self._propagate(all_embs[-1], edge_index, edge_weight)

            # Dropout
            if self.training and self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout)

            all_embs.append(all_emb)

        # Final embedding: mean of all layers
        all_emb_final = torch.stack(all_embs, dim=0).mean(dim=0)

        # Split back to user and item
        user_emb_final = all_emb_final[:self.num_users]
        item_emb_final = all_emb_final[self.num_users:]

        return user_emb_final, item_emb_final

    def _normalize_adj(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize adjacency matrix: D^{-1/2} A D^{-1/2}

        Args:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]

        Returns:
            edge_index: [2, num_edges]
            edge_weight_norm: [num_edges]
        """
        num_nodes = self.num_users + self.num_items

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # Add self loops if needed
        if self.add_self_loops:
            loop_index = torch.arange(num_nodes, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            loop_weight = torch.ones(num_nodes, device=edge_index.device)

            edge_index = torch.cat([edge_index, loop_index], dim=1)
            edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        # Compute degree
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, row, edge_weight)
        deg.scatter_add_(0, col, edge_weight)

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalize edge weights
        edge_weight_norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight_norm

    def _propagate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Message passing: aggregate neighbor embeddings

        Args:
            x: Node embeddings [num_nodes, embedding_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]

        Returns:
            out: Aggregated embeddings [num_nodes, embedding_dim]
        """
        row, col = edge_index

        # Aggregate: sum of weighted neighbor embeddings
        out = torch.zeros_like(x)

        # Weighted sum
        weighted_x = x[col] * edge_weight.unsqueeze(-1)
        out.index_add_(0, row, weighted_x)

        return out

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        reg_weight: float = 1e-4
    ) -> torch.Tensor:
        """
        BPR (Bayesian Personalized Ranking) loss

        Args:
            user_ids: [batch_size]
            pos_item_ids: [batch_size]
            neg_item_ids: [batch_size]
            user_emb: [num_users, dim]
            item_emb: [num_items, dim]
            reg_weight: L2 regularization weight

        Returns:
            loss: Scalar loss
        """
        # Get embeddings
        user_embed = user_emb[user_ids]
        pos_item_embed = item_emb[pos_item_ids]
        neg_item_embed = item_emb[neg_item_ids]

        # Positive and negative scores
        pos_scores = (user_embed * pos_item_embed).sum(dim=1)
        neg_scores = (user_embed * neg_item_embed).sum(dim=1)

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization on initial embeddings
        reg_loss = (
            self.user_embedding(user_ids).norm(2).pow(2) +
            self.item_embedding(pos_item_ids).norm(2).pow(2) +
            self.item_embedding(neg_item_ids).norm(2).pow(2)
        ) / user_ids.size(0)

        total_loss = bpr_loss + reg_weight * reg_loss

        return total_loss
