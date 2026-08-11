"""Multi-hop emergence evaluator"""

from typing import Dict, Tuple, Set
from collections import defaultdict

import numpy as np
import torch
from loguru import logger


class MultiHopEvaluator:
    """
    Evaluate model's ability to perform multi-hop reasoning on graphs

    Tests if model can recommend items that are only reachable through
    k-hop paths (not directly connected)
    """

    def __init__(
        self,
        k_hop: int = 2,
        edge_masking_ratio: float = 0.5,
        min_samples: int = 100
    ):
        """
        Args:
            k_hop: Number of hops to test
            edge_masking_ratio: Ratio of direct edges to mask
            min_samples: Minimum test samples required
        """
        self.k_hop = k_hop
        self.edge_masking_ratio = edge_masking_ratio
        self.min_samples = min_samples

        self.test_samples = []
        self.adjacency = {}

    def prepare_test_set(
        self,
        edge_index: np.ndarray,
        user_item_pairs: np.ndarray,
        num_users: int,
        num_items: int
    ):
        """
        Prepare k-hop test set by masking direct edges

        Args:
            edge_index: Full graph [2, num_edges]
            user_item_pairs: Positive user-item pairs [num_pairs, 2]
            num_users: Number of users
            num_items: Number of items
        """
        logger.info(f"Preparing {self.k_hop}-hop test set...")

        # Build adjacency list (for path finding)
        self._build_adjacency(edge_index, num_users, num_items)

        # Find k-hop reachable pairs
        test_samples = []

        for user_id, item_id in user_item_pairs:
            # Check if item is k-hop reachable (but not directly connected)
            if not self._is_direct_neighbor(user_id, item_id):
                if self._is_k_hop_reachable(user_id, item_id, self.k_hop):
                    test_samples.append((user_id, item_id))

        self.test_samples = test_samples[:min(len(test_samples), self.min_samples * 10)]

        logger.info(f"Created {len(self.test_samples)} {self.k_hop}-hop test samples")

    def _build_adjacency(
        self,
        edge_index: np.ndarray,
        num_users: int,
        num_items: int
    ):
        """Build adjacency list from edge index"""
        self.adjacency = defaultdict(set)

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.adjacency[src].add(dst)
            self.adjacency[dst].add(src)  # Undirected

    def _is_direct_neighbor(self, user_id: int, item_id: int) -> bool:
        """Check if item is direct neighbor of user"""
        return item_id in self.adjacency.get(user_id, set())

    def _is_k_hop_reachable(
        self,
        user_id: int,
        item_id: int,
        k: int
    ) -> bool:
        """
        Check if item is reachable in exactly k hops via BFS

        Args:
            user_id: Source user
            item_id: Target item
            k: Number of hops

        Returns:
            True if reachable in k hops
        """
        if k == 0:
            return user_id == item_id

        # BFS to k hops
        visited = {user_id}
        current_level = {user_id}

        for hop in range(k):
            next_level = set()

            for node in current_level:
                for neighbor in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)

            current_level = next_level

            # Check if target reached at this hop
            if hop == k - 1:
                return item_id in current_level

        return False

    def evaluate(
        self,
        model,
        edge_index: torch.Tensor,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Evaluate model on k-hop test set

        Args:
            model: Recommendation model
            edge_index: Graph structure
            device: Device to run on

        Returns:
            Dict with metrics
        """
        if not self.test_samples:
            logger.warning("No test samples! Call prepare_test_set first.")
            return {}

        model.eval()

        user_ids = torch.tensor([u for u, _ in self.test_samples], device=device)
        item_ids = torch.tensor([i for _, i in self.test_samples], device=device)

        with torch.no_grad():
            # Get predictions
            if hasattr(model, 'gnn'):
                # GNN-based model
                user_emb, item_emb = model.gnn(edge_index.to(device))
                scores = model.gnn.predict(user_ids, item_ids, user_emb, item_emb)
            else:
                # Generic model
                scores = model.predict(user_ids, item_ids)

        # Compute metrics
        scores_np = scores.cpu().numpy()

        results = {
            f'{self.k_hop}_hop_mean_score': scores_np.mean(),
            f'{self.k_hop}_hop_median_score': np.median(scores_np),
            f'{self.k_hop}_hop_positive_ratio': (scores_np > 0).mean(),
            f'{self.k_hop}_hop_num_samples': len(self.test_samples)
        }

        logger.info(f"{self.k_hop}-hop evaluation: {results}")

        return results

    def track_emergence(
        self,
        current_metrics: Dict[str, float],
        history: list,
        threshold: float = 0.1
    ) -> bool:
        """
        Detect if multi-hop ability has emerged

        Args:
            current_metrics: Current evaluation metrics
            history: Historical metrics
            threshold: Emergence threshold

        Returns:
            True if emergence detected
        """
        key = f'{self.k_hop}_hop_positive_ratio'

        if key not in current_metrics:
            return False

        current_value = current_metrics[key]

        if len(history) < 5:
            return False

        # Check if sudden improvement from baseline
        baseline = np.mean([h.get(key, 0) for h in history[-10:-1]])
        improvement = current_value - baseline

        return improvement > threshold
