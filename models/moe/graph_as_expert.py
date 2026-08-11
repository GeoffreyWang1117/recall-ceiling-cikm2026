"""Graph-as-Expert MoE architecture"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .router import Router, LearnedRouter, RuleBasedRouter
from ..gnn.base_gnn import BaseGNN
from ..llm.graph_llm import GraphLLM


class GraphAsExpertMoE(nn.Module):
    """
    Mixture-of-Experts with GNN and LLM as experts

    Key idea:
    - Expert 0: GNN (good for dense, structured data)
    - Expert 1: LLM (good for sparse, cold-start, cross-domain)
    - Router decides which expert(s) to use per sample
    """

    def __init__(
        self,
        gnn_model: BaseGNN,
        llm_model: GraphLLM,
        routing_strategy: str = "learned",
        routing_config: Optional[Dict] = None,
        top_k: int = 1,
        load_balance_weight: float = 0.01
    ):
        """
        Args:
            gnn_model: GNN expert
            llm_model: LLM expert
            routing_strategy: "learned" or "rule_based"
            routing_config: Router configuration
            top_k: Number of experts to activate per sample
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__()

        self.gnn_expert = gnn_model
        self.llm_expert = llm_model
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.routing_strategy = routing_strategy

        # Initialize router
        if routing_strategy == "learned":
            # Use combined GNN + LLM hidden size for routing
            routing_dim = gnn_model.embedding_dim + llm_model.llm.model.config.hidden_size

            self.router = LearnedRouter(
                input_dim=routing_dim,
                num_experts=2,
                top_k=top_k,
                noise_std=routing_config.get('noise_std', 0.1) if routing_config else 0.1
            )

        elif routing_strategy == "rule_based":
            self.router = RuleBasedRouter(
                num_experts=2,
                top_k=top_k,
                sparsity_threshold=routing_config.get('sparsity_threshold', 0.5) if routing_config else 0.5,
                tail_threshold=routing_config.get('tail_threshold', 10) if routing_config else 10
            )

        else:
            raise ValueError(f"Unknown routing strategy: {routing_strategy}")

        # Output fusion (when top_k > 1)
        if top_k > 1:
            self.output_fusion = nn.Linear(2, 1)  # Weighted combination

    def forward(
        self,
        # LLM inputs
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # GNN inputs
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        # Routing features (for rule-based)
        routing_features: Optional[Dict[str, torch.Tensor]] = None,
        # Labels
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE

        Args:
            input_ids: [batch_size, seq_len] for LLM
            attention_mask: [batch_size, seq_len]
            user_ids: [batch_size]
            item_ids: [batch_size]
            edge_index: Graph structure
            edge_weight: Edge weights
            routing_features: Features for routing (if rule-based)
            labels: [batch_size] (optional)

        Returns:
            Dict with logits, loss, expert_indices, expert_weights
        """
        batch_size = user_ids.size(0)

        # Get expert outputs
        # Expert 0: GNN
        with torch.set_grad_enabled(self.training):
            user_emb, item_emb = self.gnn_expert(edge_index, edge_weight)
            gnn_scores = self.gnn_expert.predict(user_ids, item_ids, user_emb, item_emb)

        # Expert 1: LLM
        llm_outputs = self.llm_expert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            item_ids=item_ids,
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        llm_scores = llm_outputs['logits']

        # Prepare routing input
        if self.routing_strategy == "learned":
            # Concatenate GNN and LLM representations
            gnn_repr = user_emb[user_ids] + item_emb[item_ids]
            llm_repr = llm_outputs['hidden_states']
            routing_input = torch.cat([gnn_repr, llm_repr], dim=-1)

            expert_indices, expert_weights = self.router(routing_input)

        elif self.routing_strategy == "rule_based":
            expert_indices, expert_weights = self.router(routing_features)

        # Combine expert outputs based on routing
        expert_outputs = torch.stack([gnn_scores, llm_scores], dim=1)  # [batch_size, 2]

        if self.top_k == 1:
            # Select single expert per sample
            batch_indices = torch.arange(batch_size, device=expert_outputs.device)
            final_scores = expert_outputs[batch_indices, expert_indices[:, 0]]

        else:
            # Weighted combination of top-k experts
            selected_outputs = torch.gather(
                expert_outputs,
                dim=1,
                index=expert_indices
            )  # [batch_size, top_k]

            final_scores = (selected_outputs * expert_weights).sum(dim=1)

        # Prepare results
        result = {
            'logits': final_scores,
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'gnn_scores': gnn_scores,
            'llm_scores': llm_scores
        }

        # Compute loss
        if labels is not None:
            # Task loss
            loss_fct = nn.BCEWithLogitsLoss()
            task_loss = loss_fct(final_scores, labels.float())

            # Load balance loss (only for learned router)
            if self.routing_strategy == "learned" and self.training:
                lb_loss = self.router.load_balance_loss(routing_input)
                total_loss = task_loss + self.load_balance_weight * lb_loss

                result['loss'] = total_loss
                result['task_loss'] = task_loss
                result['load_balance_loss'] = lb_loss
            else:
                result['loss'] = task_loss

        return result

    def get_expert_usage_stats(
        self,
        expert_indices: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute expert usage statistics

        Args:
            expert_indices: [batch_size, top_k]

        Returns:
            Dict with usage statistics
        """
        total = expert_indices.size(0)

        gnn_usage = (expert_indices == 0).sum().item()
        llm_usage = (expert_indices == 1).sum().item()

        return {
            'gnn_usage_ratio': gnn_usage / (total * self.top_k),
            'llm_usage_ratio': llm_usage / (total * self.top_k),
            'gnn_count': gnn_usage,
            'llm_count': llm_usage
        }

    def inference_with_expert(
        self,
        expert_id: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Force inference with specific expert (for ablation)

        Args:
            expert_id: 0 for GNN, 1 for LLM
            **kwargs: Expert-specific inputs

        Returns:
            scores: [batch_size]
        """
        if expert_id == 0:
            # GNN expert
            user_emb, item_emb = self.gnn_expert(
                kwargs['edge_index'],
                kwargs.get('edge_weight')
            )
            scores = self.gnn_expert.predict(
                kwargs['user_ids'],
                kwargs['item_ids'],
                user_emb,
                item_emb
            )

        elif expert_id == 1:
            # LLM expert
            outputs = self.llm_expert(
                input_ids=kwargs['input_ids'],
                attention_mask=kwargs['attention_mask'],
                user_ids=kwargs.get('user_ids'),
                item_ids=kwargs.get('item_ids'),
                edge_index=kwargs.get('edge_index'),
                edge_weight=kwargs.get('edge_weight')
            )
            scores = outputs['logits']

        else:
            raise ValueError(f"Invalid expert_id: {expert_id}")

        return scores
