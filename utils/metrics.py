"""Evaluation metrics for recommendation systems"""

import torch
import numpy as np
from typing import List, Dict, Union


def calculate_metrics(
    predictions: Union[torch.Tensor, List],
    ground_truth: List,
    k_values: List[int] = [10, 20]
) -> Dict[str, float]:
    """
    Calculate recommendation metrics

    Args:
        predictions: Predicted item rankings [batch_size, k] or list of rankings
        ground_truth: List of ground truth items for each user
        k_values: List of k values for metrics

    Returns:
        Dictionary of metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, list) and len(predictions) > 0:
        if isinstance(predictions[0], torch.Tensor):
            predictions = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in predictions]
        # Don't convert to numpy array - keep as list to handle variable lengths
        # predictions = np.array(predictions)

    metrics = {}

    for k in k_values:
        ndcg_scores = []
        recall_scores = []
        hit_scores = []

        for pred, truth in zip(predictions, ground_truth):
            pred_k = pred[:k] if len(pred) >= k else pred
            truth_set = set(truth) if isinstance(truth, list) else {truth}

            # NDCG@k
            dcg = 0.0
            for i, item in enumerate(pred_k):
                if item in truth_set:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts from 0

            # Ideal DCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth_set), k)))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

            # Recall@k
            hits = len(set(pred_k) & truth_set)
            recall = hits / len(truth_set) if len(truth_set) > 0 else 0.0
            recall_scores.append(recall)

            # Hit@k
            hit_scores.append(1.0 if hits > 0 else 0.0)

        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
        metrics[f'recall@{k}'] = np.mean(recall_scores)
        metrics[f'hit@{k}'] = np.mean(hit_scores)

    # MRR (Mean Reciprocal Rank)
    mrr_scores = []
    for pred, truth in zip(predictions, ground_truth):
        truth_set = set(truth) if isinstance(truth, list) else {truth}
        for i, item in enumerate(pred):
            if item in truth_set:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0.0)

    metrics['mrr'] = np.mean(mrr_scores)

    return metrics
