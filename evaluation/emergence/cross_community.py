"""Cross-community emergence evaluator"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from loguru import logger


class CrossCommunityEvaluator:
    """
    Evaluate model's ability to make cross-community recommendations

    Tests if model can overcome community structure bias
    and recommend items from different communities
    """

    def __init__(
        self,
        num_communities: int = 10,
        algorithm: str = "spectral",
        min_samples_per_pair: int = 10
    ):
        """
        Args:
            num_communities: Number of communities to detect
            algorithm: Community detection algorithm
            min_samples_per_pair: Min samples per community pair
        """
        self.num_communities = num_communities
        self.algorithm = algorithm
        self.min_samples_per_pair = min_samples_per_pair

        self.user_communities = None
        self.item_communities = None
        self.cross_community_samples = []

    def detect_communities(
        self,
        edge_index: np.ndarray,
        num_users: int,
        num_items: int
    ):
        """
        Detect communities in the bipartite graph

        Args:
            edge_index: Graph edges [2, num_edges]
            num_users: Number of users
            num_items: Number of items
        """
        logger.info(f"Detecting {self.num_communities} communities using {self.algorithm}...")

        # Build adjacency matrix (user-item bipartite)
        num_nodes = num_users + num_items
        adjacency = np.zeros((num_nodes, num_nodes))

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adjacency[src, dst] = 1
            adjacency[dst, src] = 1  # Symmetric

        # Spectral clustering
        if self.algorithm == "spectral":
            clustering = SpectralClustering(
                n_clusters=self.num_communities,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )

            labels = clustering.fit_predict(adjacency)

            self.user_communities = labels[:num_users]
            self.item_communities = labels[num_users:]

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        logger.info("Community detection complete")

        # Log community sizes
        user_comm_sizes = np.bincount(self.user_communities)
        item_comm_sizes = np.bincount(self.item_communities)

        logger.info(f"User community sizes: {user_comm_sizes}")
        logger.info(f"Item community sizes: {item_comm_sizes}")

    def prepare_cross_community_test(
        self,
        user_item_pairs: np.ndarray
    ):
        """
        Prepare test set with cross-community pairs

        Args:
            user_item_pairs: Positive user-item pairs [num_pairs, 2]
        """
        if self.user_communities is None:
            raise ValueError("Call detect_communities first!")

        logger.info("Preparing cross-community test set...")

        cross_comm_pairs = []

        for user_id, item_id in user_item_pairs:
            user_comm = self.user_communities[user_id]
            item_comm = self.item_communities[item_id]

            # Cross-community if different communities
            if user_comm != item_comm:
                cross_comm_pairs.append((user_id, item_id, user_comm, item_comm))

        self.cross_community_samples = cross_comm_pairs

        logger.info(f"Created {len(cross_comm_pairs)} cross-community test samples")

    def evaluate(
        self,
        model,
        edge_index: torch.Tensor,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Evaluate cross-community recommendation performance

        Args:
            model: Recommendation model
            edge_index: Graph structure
            device: Device

        Returns:
            Dict with metrics
        """
        if not self.cross_community_samples:
            logger.warning("No cross-community samples!")
            return {}

        model.eval()

        user_ids = torch.tensor([u for u, _, _, _ in self.cross_community_samples], device=device)
        item_ids = torch.tensor([i for _, i, _, _ in self.cross_community_samples], device=device)

        with torch.no_grad():
            if hasattr(model, 'gnn'):
                user_emb, item_emb = model.gnn(edge_index.to(device))
                scores = model.gnn.predict(user_ids, item_ids, user_emb, item_emb)
            else:
                scores = model.predict(user_ids, item_ids)

        scores_np = scores.cpu().numpy()

        # Compute metrics
        results = {
            'cross_community_mean_score': scores_np.mean(),
            'cross_community_median_score': np.median(scores_np),
            'cross_community_positive_ratio': (scores_np > 0).mean(),
            'cross_community_num_samples': len(self.cross_community_samples)
        }

        # Per-community-pair analysis
        pair_scores = {}
        for (user_id, item_id, user_comm, item_comm), score in zip(self.cross_community_samples, scores_np):
            pair_key = f"{user_comm}->{item_comm}"
            if pair_key not in pair_scores:
                pair_scores[pair_key] = []
            pair_scores[pair_key].append(score)

        # Community diversity score (entropy of community pairs)
        pair_counts = np.array([len(scores) for scores in pair_scores.values()])
        pair_probs = pair_counts / pair_counts.sum()
        diversity = -(pair_probs * np.log(pair_probs + 1e-10)).sum()

        results['cross_community_diversity'] = diversity

        logger.info(f"Cross-community evaluation: {results}")

        return results

    def track_emergence(
        self,
        current_metrics: Dict[str, float],
        history: list,
        threshold: float = 0.15
    ) -> bool:
        """
        Detect cross-community ability emergence

        Args:
            current_metrics: Current metrics
            history: Historical metrics
            threshold: Emergence threshold

        Returns:
            True if emergence detected
        """
        key = 'cross_community_positive_ratio'

        if key not in current_metrics:
            return False

        current_value = current_metrics[key]

        if len(history) < 5:
            return False

        baseline = np.mean([h.get(key, 0) for h in history[-10:-1]])
        improvement = current_value - baseline

        return improvement > threshold
