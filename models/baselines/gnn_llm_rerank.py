"""
GNN-LLM-Rerank Baseline

This is a critical baseline to prove Graph-as-Expert is NOT just reranking.

Pipeline:
1. LightGCN generates Top-K candidates (K=100)
2. For cold-start users (≤2 interactions), use LLM to rerank the Top-K
3. No MoE, no joint training, no router learning

This baseline represents the "naive combination" approach that reviewers
might claim is equivalent to Graph-as-Expert.
"""

from typing import Dict, List, Optional, Tuple
import time

import torch
import torch.nn as nn
import numpy as np

from ..gnn.lightgcn import LightGCN
from ..llm.llm_recommender import LLMRecommender


class GNNLLMRerank(nn.Module):
    """
    GNN-LLM Reranking Baseline

    Key differences from Graph-as-Expert:
    - No joint training (GNN and LLM trained separately)
    - No learned router (simple threshold)
    - LLM only reranks GNN candidates (doesn't generate from scratch)
    - No expert coordination
    """

    def __init__(
        self,
        gnn_model: LightGCN,
        llm_model: LLMRecommender,
        cold_start_threshold: int = 2,
        gnn_topk: int = 100,
        final_topk: int = 10,
        llm_batch_size: int = 1
    ):
        """
        Args:
            gnn_model: Pre-trained LightGCN (frozen)
            llm_model: LLM for reranking (can be fine-tuned)
            cold_start_threshold: Users with ≤ this many interactions use LLM reranking
            gnn_topk: Number of candidates from GNN
            final_topk: Final number of recommendations
            llm_batch_size: Batch size for LLM inference
        """
        super().__init__()

        self.gnn = gnn_model
        self.llm = llm_model
        self.cold_start_threshold = cold_start_threshold
        self.gnn_topk = gnn_topk
        self.final_topk = final_topk
        self.llm_batch_size = llm_batch_size

        # Freeze GNN (no joint training)
        for param in self.gnn.parameters():
            param.requires_grad = False

        # Statistics
        self.stats = {
            'total_users': 0,
            'gnn_only': 0,
            'llm_rerank': 0,
            'total_latency_ms': 0,
            'gnn_latency_ms': 0,
            'llm_latency_ms': 0
        }

    def get_gnn_candidates(
        self,
        user_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        exclude_items: Optional[List[List[int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Top-K candidates from GNN

        Args:
            user_ids: [batch_size]
            edge_index: Graph edges
            edge_weight: Edge weights
            exclude_items: Items to exclude for each user (already interacted)

        Returns:
            candidate_items: [batch_size, gnn_topk]
            candidate_scores: [batch_size, gnn_topk]
        """
        t0 = time.time()

        self.gnn.eval()
        with torch.no_grad():
            # Get embeddings
            user_emb, item_emb = self.gnn(edge_index, edge_weight)

            # Compute scores for all items
            batch_user_emb = user_emb[user_ids]  # [batch_size, emb_dim]
            scores = torch.matmul(batch_user_emb, item_emb.t())  # [batch_size, num_items]

            # Exclude already interacted items
            if exclude_items is not None:
                for i, exclude_list in enumerate(exclude_items):
                    if len(exclude_list) > 0:
                        scores[i, exclude_list] = -1e9

            # Get Top-K
            topk_scores, topk_items = torch.topk(scores, k=self.gnn_topk, dim=1)

        self.stats['gnn_latency_ms'] += (time.time() - t0) * 1000

        return topk_items, topk_scores

    def llm_rerank(
        self,
        user_ids: torch.Tensor,
        candidate_items: torch.Tensor,
        candidate_scores: torch.Tensor,
        user_history: List[List[int]],
        item_texts: Dict[int, str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use LLM to rerank GNN candidates

        Args:
            user_ids: [batch_size]
            candidate_items: [batch_size, gnn_topk]
            candidate_scores: [batch_size, gnn_topk] (GNN scores, for reference)
            user_history: List of historical item IDs for each user
            item_texts: Mapping from item ID to text description

        Returns:
            reranked_items: [batch_size, final_topk]
            reranked_scores: [batch_size, final_topk] (LLM scores)
        """
        t0 = time.time()

        batch_size = len(user_ids)
        reranked_items_list = []
        reranked_scores_list = []

        self.llm.eval()

        for i in range(batch_size):
            user_id = user_ids[i].item()
            history = user_history[i]
            candidates = candidate_items[i].cpu().numpy()

            # Build prompt
            prompt = self._build_rerank_prompt(
                user_id=user_id,
                history=history,
                candidates=candidates,
                item_texts=item_texts
            )

            # LLM inference
            with torch.no_grad():
                # Use LLM to score/rank candidates
                # For simplicity, we'll use the LLM to generate a ranked list
                reranked = self.llm.rerank(
                    prompt=prompt,
                    candidate_ids=candidates,
                    top_k=self.final_topk
                )

            reranked_items_list.append(reranked['item_ids'])
            reranked_scores_list.append(reranked['scores'])

        self.stats['llm_latency_ms'] += (time.time() - t0) * 1000

        # Convert to tensors
        reranked_items = torch.tensor(
            reranked_items_list,
            dtype=torch.long,
            device=user_ids.device
        )
        reranked_scores = torch.tensor(
            reranked_scores_list,
            dtype=torch.float,
            device=user_ids.device
        )

        return reranked_items, reranked_scores

    def forward(
        self,
        user_ids: torch.Tensor,
        edge_index: torch.Tensor,
        user_interaction_counts: torch.Tensor,
        user_history: List[List[int]],
        item_texts: Dict[int, str],
        edge_weight: Optional[torch.Tensor] = None,
        exclude_items: Optional[List[List[int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: GNN → LLM rerank (for cold-start users)

        Args:
            user_ids: [batch_size]
            edge_index: Graph structure
            user_interaction_counts: [batch_size] number of interactions per user
            user_history: Historical items for each user
            item_texts: Item text descriptions
            edge_weight: Edge weights
            exclude_items: Items to exclude

        Returns:
            Dict with:
                - recommendations: [batch_size, final_topk]
                - scores: [batch_size, final_topk]
                - routing_decisions: [batch_size] (0=GNN, 1=LLM rerank)
                - latency_ms: [batch_size]
        """
        t0 = time.time()
        batch_size = len(user_ids)

        # Step 1: GNN generates Top-K candidates for ALL users
        candidate_items, candidate_scores = self.get_gnn_candidates(
            user_ids=user_ids,
            edge_index=edge_index,
            edge_weight=edge_weight,
            exclude_items=exclude_items
        )

        # Step 2: Decide which users need LLM reranking
        cold_start_mask = user_interaction_counts <= self.cold_start_threshold

        # Step 3: For cold-start users, use LLM reranking
        final_items = torch.zeros(
            batch_size, self.final_topk,
            dtype=torch.long,
            device=user_ids.device
        )
        final_scores = torch.zeros(
            batch_size, self.final_topk,
            dtype=torch.float,
            device=user_ids.device
        )
        routing_decisions = torch.zeros(batch_size, dtype=torch.long)

        # GNN-only users
        gnn_mask = ~cold_start_mask
        if gnn_mask.any():
            gnn_indices = torch.where(gnn_mask)[0]
            final_items[gnn_indices] = candidate_items[gnn_indices, :self.final_topk]
            final_scores[gnn_indices] = candidate_scores[gnn_indices, :self.final_topk]
            routing_decisions[gnn_indices] = 0  # GNN

            self.stats['gnn_only'] += gnn_indices.size(0)

        # LLM rerank users
        if cold_start_mask.any():
            llm_indices = torch.where(cold_start_mask)[0]

            # Rerank with LLM
            reranked_items, reranked_scores = self.llm_rerank(
                user_ids=user_ids[llm_indices],
                candidate_items=candidate_items[llm_indices],
                candidate_scores=candidate_scores[llm_indices],
                user_history=[user_history[i] for i in llm_indices.cpu().numpy()],
                item_texts=item_texts
            )

            final_items[llm_indices] = reranked_items
            final_scores[llm_indices] = reranked_scores
            routing_decisions[llm_indices] = 1  # LLM rerank

            self.stats['llm_rerank'] += llm_indices.size(0)

        self.stats['total_users'] += batch_size

        total_time = (time.time() - t0) * 1000
        self.stats['total_latency_ms'] += total_time

        return {
            'recommendations': final_items,
            'scores': final_scores,
            'routing_decisions': routing_decisions,
            'latency_ms': torch.full((batch_size,), total_time / batch_size)
        }

    def _build_rerank_prompt(
        self,
        user_id: int,
        history: List[int],
        candidates: np.ndarray,
        item_texts: Dict[int, str]
    ) -> str:
        """Build prompt for LLM reranking"""

        # User history
        history_str = "User's interaction history:\n"
        for idx, item_id in enumerate(history[-10:], 1):  # Last 10 items
            item_text = item_texts.get(item_id, f"Item {item_id}")
            history_str += f"{idx}. {item_text}\n"

        # Candidates
        candidates_str = "\nCandidate items to rank:\n"
        for idx, item_id in enumerate(candidates[:20], 1):  # Top 20 candidates
            item_text = item_texts.get(item_id, f"Item {item_id}")
            candidates_str += f"{chr(64+idx)}. [ID:{item_id}] {item_text}\n"

        # Task
        task_str = (
            f"\nTask: Based on the user's history, rank these {len(candidates[:20])} "
            "candidates from most relevant to least relevant. "
            "Return the top 10 item IDs in order.\n"
            "Output format: item_id1, item_id2, item_id3, ..."
        )

        prompt = history_str + candidates_str + task_str

        return prompt

    def get_statistics(self) -> Dict[str, float]:
        """Get routing and latency statistics"""
        total = self.stats['total_users']
        if total == 0:
            return {}

        return {
            'gnn_routing_percentage': 100 * self.stats['gnn_only'] / total,
            'llm_routing_percentage': 100 * self.stats['llm_rerank'] / total,
            'avg_total_latency_ms': self.stats['total_latency_ms'] / total,
            'avg_gnn_latency_ms': self.stats['gnn_latency_ms'] / total,
            'avg_llm_latency_ms': self.stats['llm_latency_ms'] / total,
            'total_users': total
        }

    def reset_statistics(self):
        """Reset statistics counters"""
        for key in self.stats:
            self.stats[key] = 0
