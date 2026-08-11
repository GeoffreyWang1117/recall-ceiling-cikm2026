"""Expert routing mechanisms for MoE"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module, ABC):
    """Base router class"""

    def __init__(self, num_experts: int, top_k: int = 1):
        """
        Args:
            num_experts: Number of experts
            top_k: Number of experts to route to
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route inputs to experts

        Args:
            x: Input features [batch_size, feature_dim]

        Returns:
            expert_indices: [batch_size, top_k] indices of selected experts
            expert_weights: [batch_size, top_k] weights for selected experts
        """
        pass


class LearnedRouter(Router):
    """
    Learned router with gating network

    Uses a linear layer + softmax to compute expert weights
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 1,
        noise_std: float = 0.0
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_experts: Number of experts
            top_k: Number of experts to select
            noise_std: Std of noise added to logits (for load balancing)
        """
        super().__init__(num_experts, top_k)
        self.gate = nn.Linear(input_dim, num_experts)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through learned router

        Args:
            x: [batch_size, input_dim]

        Returns:
            expert_indices: [batch_size, top_k]
            expert_weights: [batch_size, top_k]
        """
        # Compute logits
        logits = self.gate(x)  # [batch_size, num_experts]

        # Add noise during training for load balancing
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Select top-k experts
        expert_weights, expert_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_indices, expert_weights

    def load_balance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert usage

        Args:
            x: [batch_size, input_dim]

        Returns:
            loss: Scalar load balance loss
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)

        # Average probability per expert
        expert_probs = probs.mean(dim=0)  # [num_experts]

        # Ideal uniform distribution
        uniform_prob = 1.0 / self.num_experts

        # KL divergence from uniform
        kl_loss = F.kl_div(
            expert_probs.log(),
            torch.full_like(expert_probs, uniform_prob),
            reduction='sum'
        )

        return kl_loss


class RuleBasedRouter(Router):
    """
    Rule-based router using handcrafted features

    Routes based on:
    - Data sparsity
    - User/item popularity (head vs tail)
    - Cross-domain signals
    """

    def __init__(
        self,
        num_experts: int = 2,
        top_k: int = 1,
        sparsity_threshold: float = 0.5,
        tail_threshold: int = 10
    ):
        """
        Args:
            num_experts: Should be 2 (GNN expert, LLM expert)
            top_k: Number of experts to select
            sparsity_threshold: Sparsity threshold for routing
            tail_threshold: Interaction count threshold for tail users/items
        """
        super().__init__(num_experts, top_k)
        self.sparsity_threshold = sparsity_threshold
        self.tail_threshold = tail_threshold

        assert num_experts == 2, "RuleBasedRouter expects 2 experts: GNN (0), LLM (1)"

    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rule-based routing

        Args:
            features: Dict with keys:
                - user_degree: [batch_size] number of interactions
                - item_degree: [batch_size] number of interactions
                - is_cold_start: [batch_size] bool tensor

        Returns:
            expert_indices: [batch_size, top_k]
            expert_weights: [batch_size, top_k]
        """
        batch_size = features['user_degree'].size(0)

        # Initialize routing decisions
        expert_indices = torch.zeros(batch_size, self.top_k, dtype=torch.long)
        expert_weights = torch.zeros(batch_size, self.top_k)

        # Rules:
        # 1. Cold-start or tail users/items -> LLM expert (index 1)
        # 2. Dense, popular users/items -> GNN expert (index 0)

        is_cold_start = features.get('is_cold_start', torch.zeros(batch_size, dtype=torch.bool))
        user_degree = features.get('user_degree', torch.zeros(batch_size))
        item_degree = features.get('item_degree', torch.zeros(batch_size))

        # Check if tail (low degree)
        is_tail = (user_degree < self.tail_threshold) | (item_degree < self.tail_threshold)

        # Route to LLM if cold-start or tail
        route_to_llm = is_cold_start | is_tail

        # Set expert indices
        expert_indices[:, 0] = torch.where(route_to_llm, 1, 0)  # 1=LLM, 0=GNN

        # Set uniform weights for top-1
        expert_weights[:, 0] = 1.0

        return expert_indices, expert_weights

    def get_routing_statistics(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Get routing statistics for analysis

        Args:
            features: Input features

        Returns:
            Dict with routing statistics
        """
        expert_indices, _ = self.forward(features)

        # Count expert usage
        gnn_count = (expert_indices[:, 0] == 0).sum().item()
        llm_count = (expert_indices[:, 0] == 1).sum().item()
        total = expert_indices.size(0)

        return {
            'gnn_ratio': gnn_count / total,
            'llm_ratio': llm_count / total,
            'gnn_count': gnn_count,
            'llm_count': llm_count
        }
