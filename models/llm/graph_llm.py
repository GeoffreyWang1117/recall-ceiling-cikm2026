"""Graph-enhanced LLM recommender with multiple input strategies"""

from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn

from .llm_recommender import LLMRecommender
from ..gnn.base_gnn import BaseGNN


class GraphLLM(nn.Module):
    """
    Graph-enhanced LLM recommender

    Supports three strategies:
    1. Graph-as-Context: Serialize graph structure as text input
    2. Graph-as-Embedding: Inject GNN embeddings into LLM
    3. Hybrid: Combine both strategies
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        gnn_model: Optional[BaseGNN] = None,
        fusion_method: str = "cross_attention"
    ):
        """
        Args:
            llm_config: LLM model configuration
            gnn_model: Pre-trained GNN model (optional)
            fusion_method: How to fuse GNN and LLM ("concat", "cross_attention", "gate")
        """
        super().__init__()

        self.llm = LLMRecommender(llm_config)
        self.gnn = gnn_model
        self.fusion_method = fusion_method

        # Fusion modules
        if gnn_model is not None:
            hidden_size = self.llm.model.config.hidden_size
            gnn_dim = gnn_model.embedding_dim

            if fusion_method == "concat":
                # Simple concatenation + linear
                self.fusion_layer = nn.Linear(hidden_size + gnn_dim, hidden_size)

            elif fusion_method == "cross_attention":
                # Cross-attention between LLM and GNN
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=8,
                    batch_first=True
                )
                # Project GNN embeddings to LLM dimension
                self.gnn_proj = nn.Linear(gnn_dim, hidden_size)

            elif fusion_method == "gate":
                # Gating mechanism
                self.gnn_proj = nn.Linear(gnn_dim, hidden_size)
                self.gate = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.Sigmoid()
                )

            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional graph integration

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            user_ids: [batch_size] (for GNN)
            item_ids: [batch_size] (for GNN)
            edge_index: Graph edges (for GNN)
            edge_weight: Edge weights (for GNN)
            labels: [batch_size] (optional)

        Returns:
            Dict with logits, loss, hidden_states
        """
        # LLM forward
        llm_outputs = self.llm(input_ids, attention_mask)
        llm_hidden = llm_outputs['hidden_states']  # [batch_size, hidden_size]

        # If GNN is available and user/item IDs provided
        if self.gnn is not None and user_ids is not None and item_ids is not None:
            # Get GNN embeddings
            with torch.no_grad() if not self.training else torch.enable_grad():
                user_emb, item_emb = self.gnn(edge_index, edge_weight)

            # Get user and item embeddings for this batch
            batch_user_emb = user_emb[user_ids]  # [batch_size, gnn_dim]
            batch_item_emb = item_emb[item_ids]  # [batch_size, gnn_dim]

            # Combine user and item embeddings
            gnn_emb = batch_user_emb + batch_item_emb  # [batch_size, gnn_dim]

            # Fuse with LLM
            fused_hidden = self._fuse_representations(llm_hidden, gnn_emb)
        else:
            fused_hidden = llm_hidden

        # Recommendation head
        logits = self.llm.rec_head(fused_hidden).squeeze(-1)

        result = {
            'logits': logits,
            'hidden_states': fused_hidden
        }

        # Compute loss
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            result['loss'] = loss

        return result

    def _fuse_representations(
        self,
        llm_hidden: torch.Tensor,
        gnn_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse LLM and GNN representations

        Args:
            llm_hidden: [batch_size, llm_dim]
            gnn_emb: [batch_size, gnn_dim]

        Returns:
            fused: [batch_size, llm_dim]
        """
        if self.fusion_method == "concat":
            # Concatenate and project
            combined = torch.cat([llm_hidden, gnn_emb], dim=1)
            fused = self.fusion_layer(combined)

        elif self.fusion_method == "cross_attention":
            # Project GNN to LLM dimension
            gnn_proj = self.gnn_proj(gnn_emb).unsqueeze(1)  # [batch_size, 1, llm_dim]
            llm_query = llm_hidden.unsqueeze(1)  # [batch_size, 1, llm_dim]

            # Cross-attention: LLM attends to GNN
            attn_output, _ = self.cross_attn(
                query=llm_query,
                key=gnn_proj,
                value=gnn_proj
            )
            fused = attn_output.squeeze(1) + llm_hidden  # Residual

        elif self.fusion_method == "gate":
            # Project GNN
            gnn_proj = self.gnn_proj(gnn_emb)

            # Compute gate
            gate_input = torch.cat([llm_hidden, gnn_proj], dim=1)
            gate_values = self.gate(gate_input)

            # Gated fusion
            fused = gate_values * llm_hidden + (1 - gate_values) * gnn_proj

        return fused

    def predict_with_graph(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        prompts: list,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict with both text and graph signals

        Args:
            user_ids: [batch_size]
            item_ids: [batch_size]
            prompts: List of prompt strings
            edge_index: Graph structure
            edge_weight: Edge weights

        Returns:
            scores: [batch_size]
        """
        # Encode prompts
        input_ids, attention_mask = self.llm.encode_batch(prompts)
        input_ids = input_ids.to(self.llm.model.device)
        attention_mask = attention_mask.to(self.llm.model.device)
        user_ids = user_ids.to(self.llm.model.device)
        item_ids = item_ids.to(self.llm.model.device)

        # Forward
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_ids=user_ids,
                item_ids=item_ids,
                edge_index=edge_index,
                edge_weight=edge_weight
            )

        return outputs['logits']
