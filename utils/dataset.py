"""PyTorch dataset classes"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """
    Dataset for recommendation task

    Supports:
    - User-item pairs with labels
    - Graph context injection
    - Negative sampling
    """

    def __init__(
        self,
        df: pd.DataFrame,
        edge_index: Optional[np.ndarray] = None,
        edge_weight: Optional[np.ndarray] = None,
        tokenizer=None,
        max_length: int = 512,
        num_negatives: int = 1,
        graph_context: bool = False
    ):
        """
        Args:
            df: DataFrame with user_id, item_id, label columns
            edge_index: Graph structure [2, num_edges]
            edge_weight: Edge weights [num_edges]
            tokenizer: Tokenizer for LLM input
            max_length: Max sequence length
            num_negatives: Number of negative samples per positive
            graph_context: Whether to include graph context
        """
        self.df = df
        self.edge_index = torch.from_numpy(edge_index).long() if edge_index is not None else None
        self.edge_weight = torch.from_numpy(edge_weight).float() if edge_weight is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.graph_context = graph_context

        # Extract data
        self.user_ids = df['user_id'].values
        self.item_ids = df['item_id'].values
        self.labels = df['label'].values if 'label' in df.columns else np.ones(len(df))

        # User degree (for routing features)
        if 'user_degree' in df.columns:
            self.user_degrees = df['user_degree'].values
        else:
            self.user_degrees = np.zeros(len(df))

        if 'item_degree' in df.columns:
            self.item_degrees = df['item_degree'].values
        else:
            self.item_degrees = np.zeros(len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        """Get item by index"""
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        label = self.labels[idx]

        # Build sample dict
        sample = {
            'user_ids': torch.tensor(user_id, dtype=torch.long),
            'item_ids': torch.tensor(item_id, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }

        # Add routing features
        sample['routing_features'] = {
            'user_degree': torch.tensor(self.user_degrees[idx], dtype=torch.float),
            'item_degree': torch.tensor(self.item_degrees[idx], dtype=torch.float),
            'is_cold_start': torch.tensor(
                self.user_degrees[idx] < 10 or self.item_degrees[idx] < 10,
                dtype=torch.bool
            )
        }

        # Add graph structure
        if self.edge_index is not None:
            sample['edge_index'] = self.edge_index
            if self.edge_weight is not None:
                sample['edge_weight'] = self.edge_weight

        # Generate prompt for LLM (if tokenizer provided)
        if self.tokenizer:
            prompt = self._generate_prompt(user_id, item_id)
            encoded = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            sample['input_ids'] = encoded['input_ids'].squeeze(0)
            sample['attention_mask'] = encoded['attention_mask'].squeeze(0)

        return sample

    def _generate_prompt(self, user_id: int, item_id: int) -> str:
        """Generate prompt for LLM"""
        prompt = f"### Recommendation Task\n"
        prompt += f"User: {user_id}\n"
        prompt += f"Candidate Item: {item_id}\n"

        # TODO: Add user history, graph context, etc.

        prompt += f"Will user interact with this item? (yes/no): "
        return prompt


class BatchCollator:
    """Custom collator for batching"""

    def __init__(self, graph_shared: bool = True):
        """
        Args:
            graph_shared: If True, graph is shared across batch (more efficient)
        """
        self.graph_shared = graph_shared

    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch"""
        collated = {}

        # Stack tensors
        for key in ['user_ids', 'item_ids', 'labels']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])

        # Stack routing features
        if 'routing_features' in batch[0]:
            routing_features = {}
            for key in batch[0]['routing_features'].keys():
                routing_features[key] = torch.stack([
                    item['routing_features'][key] for item in batch
                ])
            collated['routing_features'] = routing_features

        # Handle input_ids and attention_mask
        if 'input_ids' in batch[0]:
            collated['input_ids'] = torch.stack([item['input_ids'] for item in batch])
            collated['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])

        # Handle graph (shared across batch)
        if 'edge_index' in batch[0]:
            if self.graph_shared:
                collated['edge_index'] = batch[0]['edge_index']
                if 'edge_weight' in batch[0]:
                    collated['edge_weight'] = batch[0]['edge_weight']
            else:
                # Batch-specific subgraphs (not implemented yet)
                pass

        return collated
