"""GraphSAGE: Inductive Representation Learning on Large Graphs"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gnn import BaseGNN


class GraphSAGE(BaseGNN):
    """
    GraphSAGE model for recommendation

    Reference:
    Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)

    Key features:
    - Sampling-based aggregation
    - Support for multiple aggregator types
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        aggregator: str = "mean",
        normalize: bool = True
    ):
        """
        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Embedding dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            aggregator: Aggregator type ("mean", "gcn", "pool", "lstm")
            normalize: Whether to L2-normalize embeddings
        """
        super().__init__(num_users, num_items, embedding_dim, num_layers, dropout)
        self.aggregator_type = aggregator
        self.normalize = normalize

        # Build aggregator layers
        self.aggregators = nn.ModuleList()
        self.linears = nn.ModuleList()

        for _ in range(num_layers):
            if aggregator == "mean" or aggregator == "gcn":
                # Mean/GCN aggregator doesn't need parameters
                self.aggregators.append(None)
            elif aggregator == "pool":
                # Max pooling with MLP
                self.aggregators.append(
                    nn.Sequential(
                        nn.Linear(embedding_dim, embedding_dim),
                        nn.ReLU()
                    )
                )
            elif aggregator == "lstm":
                # LSTM aggregator
                self.aggregators.append(
                    nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
                )
            else:
                raise ValueError(f"Unknown aggregator: {aggregator}")

            # Linear transformation after aggregation
            if aggregator == "gcn":
                self.linears.append(nn.Linear(embedding_dim, embedding_dim))
            else:
                self.linears.append(nn.Linear(2 * embedding_dim, embedding_dim))

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GraphSAGE forward pass

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)

        Returns:
            user_embeddings: [num_users, embedding_dim]
            item_embeddings: [num_items, embedding_dim]
        """
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Combine
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        # Layer-wise propagation
        for layer in range(self.num_layers):
            all_emb = self._sage_conv(
                all_emb,
                edge_index,
                edge_weight,
                layer
            )

            # Activation
            all_emb = F.relu(all_emb)

            # Dropout
            if self.training and self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout)

            # L2 normalization
            if self.normalize:
                all_emb = F.normalize(all_emb, p=2, dim=1)

        # Split back
        user_emb_final = all_emb[:self.num_users]
        item_emb_final = all_emb[self.num_users:]

        return user_emb_final, item_emb_final

    def _sage_conv(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        layer: int
    ) -> torch.Tensor:
        """
        GraphSAGE convolution layer

        Args:
            x: Node embeddings [num_nodes, dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            layer: Layer index

        Returns:
            out: Updated embeddings [num_nodes, dim]
        """
        row, col = edge_index
        num_nodes = x.size(0)

        # Aggregate neighbor embeddings
        if self.aggregator_type == "mean":
            # Mean aggregator
            neighbor_emb = self._mean_aggregate(x, edge_index, edge_weight, num_nodes)

        elif self.aggregator_type == "gcn":
            # GCN aggregator (include self)
            neighbor_emb = self._gcn_aggregate(x, edge_index, edge_weight, num_nodes)

        elif self.aggregator_type == "pool":
            # Max pooling aggregator
            neighbor_emb = self._pool_aggregate(x, edge_index, layer, num_nodes)

        elif self.aggregator_type == "lstm":
            # LSTM aggregator
            neighbor_emb = self._lstm_aggregate(x, edge_index, layer, num_nodes)

        # Concatenate self and neighbor (except for GCN)
        if self.aggregator_type == "gcn":
            out = neighbor_emb
        else:
            out = torch.cat([x, neighbor_emb], dim=1)

        # Linear transformation
        out = self.linears[layer](out)

        return out

    def _mean_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int
    ) -> torch.Tensor:
        """Mean aggregator"""
        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        # Compute degree for normalization
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, edge_weight)
        deg = deg.clamp(min=1)

        # Aggregate
        out = torch.zeros_like(x)
        weighted_x = x[col] * edge_weight.unsqueeze(-1)
        out.index_add_(0, row, weighted_x)

        # Normalize by degree
        out = out / deg.unsqueeze(-1)

        return out

    def _gcn_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int
    ) -> torch.Tensor:
        """GCN-style aggregator (includes self)"""
        # Add self loops
        loop_index = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1) - num_nodes, device=x.device)

        loop_weight = torch.ones(num_nodes, device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        # Symmetric normalization
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, edge_weight)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_weight_norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Aggregate
        out = torch.zeros_like(x)
        weighted_x = x[col] * edge_weight_norm.unsqueeze(-1)
        out.index_add_(0, row, weighted_x)

        return out

    def _pool_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int,
        num_nodes: int
    ) -> torch.Tensor:
        """Max pooling aggregator"""
        row, col = edge_index

        # Transform neighbor features
        neighbor_x = self.aggregators[layer](x[col])

        # Max pooling
        out = torch.full((num_nodes, x.size(1)), float('-inf'), device=x.device)
        out.scatter_reduce_(0, row.unsqueeze(-1).expand_as(neighbor_x), neighbor_x, reduce='amax')

        # Handle nodes with no neighbors
        out[out == float('-inf')] = 0

        return out

    def _lstm_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int,
        num_nodes: int
    ) -> torch.Tensor:
        """LSTM aggregator (order matters)"""
        row, col = edge_index

        # Group neighbors by source node
        from collections import defaultdict
        node_neighbors = defaultdict(list)

        for src, dst in edge_index.t().tolist():
            node_neighbors[src].append(dst)

        # Process each node
        out = torch.zeros_like(x)

        for node_id in range(num_nodes):
            if node_id in node_neighbors:
                neighbor_ids = node_neighbors[node_id]
                neighbor_embs = x[neighbor_ids].unsqueeze(0)  # [1, num_neighbors, dim]

                # LSTM
                _, (h_n, _) = self.aggregators[layer](neighbor_embs)
                out[node_id] = h_n.squeeze(0)

        return out
