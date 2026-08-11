#!/usr/bin/env python3
"""
Unified Data Interface for All Datasets
=========================================
Single entry point for loading any processed dataset. All experiment scripts
should use this instead of duplicating data loading logic.

Usage:
    from data_interface import load_dataset, list_datasets, DatasetInfo

    # List all available datasets
    for name, info in list_datasets().items():
        print(f"{name}: {info.n_users} users, {info.n_items} items")

    # Load a dataset
    ds = load_dataset('beauty')
    train_df, test_df = ds.train_df, ds.test_df
    test_samples = ds.get_test_samples(n_users=500, min_history=3, seed=42)

    # Access metadata
    text = ds.get_item_text(item_id=42)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json

project_root = Path(__file__).parent.parent


# ============================================================================
# DATASET REGISTRY
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str                # relative to data/processed/
    domain: str              # 'product', 'news', 'movie'
    description: str
    has_val: bool = False
    has_loo: bool = False    # separate LOO test set
    has_last5: bool = False  # separate last-5 test set


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    # Original Amazon datasets
    'beauty': DatasetConfig(
        name='Amazon Beauty', path='amazon_beauty_sampled',
        domain='product', description='Amazon All_Beauty reviews',
        has_val=True),
    'movies': DatasetConfig(
        name='Amazon Movies', path='amazon_movies_sampled',
        domain='product', description='Amazon Movies_and_TV reviews',
        has_val=True),
    'electronics': DatasetConfig(
        name='Amazon Electronics', path='amazon_electronics_sampled',
        domain='product', description='Amazon Electronics reviews',
        has_val=True),

    # New Amazon domains
    'toys': DatasetConfig(
        name='Amazon Toys', path='amazon_toys_sampled',
        domain='product', description='Amazon Toys_and_Games reviews'),
    'sports': DatasetConfig(
        name='Amazon Sports', path='amazon_sports_sampled',
        domain='product', description='Amazon Sports_and_Outdoors reviews'),
    'office': DatasetConfig(
        name='Amazon Office', path='amazon_office_sampled',
        domain='product', description='Amazon Office_Products reviews'),

    # Cross-domain
    'mind': DatasetConfig(
        name='MIND News', path='mind_news',
        domain='news', description='Microsoft MIND news recommendation'),
    'movielens': DatasetConfig(
        name='MovieLens-25M', path='movielens_25m',
        domain='movie', description='MovieLens-25M movie ratings',
        has_val=True, has_loo=True, has_last5=True),
}

# Short aliases
ALIASES = {
    'ml': 'movielens', 'ml25m': 'movielens',
    'mind_news': 'mind', 'news': 'mind',
    'amazon_beauty': 'beauty', 'amazon_movies': 'movies',
    'amazon_electronics': 'electronics',
    'amazon_toys': 'toys', 'amazon_sports': 'sports',
    'amazon_office': 'office',
}


# ============================================================================
# DATASET INFO (lightweight stats without loading data)
# ============================================================================

@dataclass
class DatasetInfo:
    """Lightweight dataset statistics."""
    name: str
    domain: str
    path: Path
    n_users: int
    n_items: int
    n_train: int
    n_test: int
    density: float
    exists: bool


def list_datasets() -> Dict[str, DatasetInfo]:
    """List all registered datasets with stats (without loading full data)."""
    result = {}
    for key, cfg in DATASET_REGISTRY.items():
        full_path = project_root / 'data' / 'processed' / cfg.path
        exists = (full_path / 'train.parquet').exists()

        if exists:
            # Quick stats from file size / stats file
            stats_file = full_path / 'dataset_stats.json'
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                info = DatasetInfo(
                    name=cfg.name, domain=cfg.domain, path=full_path,
                    n_users=stats.get('n_users', 0),
                    n_items=stats.get('n_items', 0),
                    n_train=stats.get('n_train', 0),
                    n_test=stats.get('n_test', 0),
                    density=stats.get('density', stats.get('density_pct', 0)),
                    exists=True)
            else:
                # Read parquet headers only
                train_df = pd.read_parquet(full_path / 'train.parquet')
                test_df = pd.read_parquet(full_path / 'test.parquet')
                n_users = train_df['user_id'].nunique()
                n_items = train_df['item_id'].nunique()
                density = len(train_df) / (n_users * n_items) * 100 if n_users * n_items > 0 else 0
                info = DatasetInfo(
                    name=cfg.name, domain=cfg.domain, path=full_path,
                    n_users=n_users, n_items=n_items,
                    n_train=len(train_df), n_test=len(test_df),
                    density=density, exists=True)
        else:
            info = DatasetInfo(
                name=cfg.name, domain=cfg.domain, path=full_path,
                n_users=0, n_items=0, n_train=0, n_test=0,
                density=0, exists=False)

        result[key] = info
    return result


# ============================================================================
# LOADED DATASET
# ============================================================================

class Dataset:
    """A loaded dataset ready for experiments."""

    def __init__(self, key: str, config: DatasetConfig):
        self.key = key
        self.config = config
        self.path = project_root / 'data' / 'processed' / config.path

        # Load dataframes
        self.train_df = pd.read_parquet(self.path / 'train.parquet')
        self.test_df = pd.read_parquet(self.path / 'test.parquet')

        # Ensure consistent dtypes
        for df in [self.train_df, self.test_df]:
            df['user_id'] = df['user_id'].astype(int)
            df['item_id'] = df['item_id'].astype(int)

        # Optional splits
        self.val_df = None
        if config.has_val and (self.path / 'val.parquet').exists():
            self.val_df = pd.read_parquet(self.path / 'val.parquet')
            self.val_df['user_id'] = self.val_df['user_id'].astype(int)
            self.val_df['item_id'] = self.val_df['item_id'].astype(int)

        self.test_loo_df = None
        if config.has_loo and (self.path / 'test_loo.parquet').exists():
            self.test_loo_df = pd.read_parquet(self.path / 'test_loo.parquet')

        # Computed properties
        self.n_users = self.train_df['user_id'].nunique()
        self.n_items = self.train_df['item_id'].nunique()
        self.n_interactions = len(self.train_df)
        self.density = self.n_interactions / (self.n_users * self.n_items) if self.n_users * self.n_items > 0 else 0

        # Caches
        self._user_history = None
        self._test_gt = None
        self._item_metadata = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def domain(self) -> str:
        return self.config.domain

    @property
    def catalog_size(self) -> int:
        return self.n_items

    @property
    def user_history(self) -> Dict[int, List[int]]:
        """User -> list of interacted item IDs (from train)."""
        if self._user_history is None:
            self._user_history = self.train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        return self._user_history

    @property
    def test_ground_truth(self) -> Dict[int, List[int]]:
        """User -> list of test item IDs."""
        if self._test_gt is None:
            self._test_gt = self.test_df.groupby('user_id')['item_id'].apply(list).to_dict()
        return self._test_gt

    @property
    def item_metadata(self) -> Dict[str, dict]:
        """Item ID -> metadata dict."""
        if self._item_metadata is None:
            meta_path = self.path / 'item_metadata.json'
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self._item_metadata = json.load(f)
            else:
                self._item_metadata = {}
        return self._item_metadata

    def get_item_text(self, item_id: int) -> str:
        """Get text description for an item."""
        meta = self.item_metadata.get(str(item_id), {})
        if meta is None:
            return f'Item {item_id}'
        return meta.get('text', meta.get('title', f'Item {item_id}'))

    def get_test_samples(self, n_users: int = 500, min_history: int = 3,
                         seed: int = 42) -> List[dict]:
        """Get test samples: [{user_id, history, ground_truth}, ...]."""
        history = self.user_history
        gt = self.test_ground_truth
        valid_users = [u for u in gt if u in history and len(history[u]) >= min_history]

        np.random.seed(seed)
        n = min(n_users, len(valid_users))
        sampled = np.random.choice(valid_users, size=n, replace=False)

        return [{'user_id': int(u),
                 'history': history[u],
                 'ground_truth': gt[u]}
                for u in sampled]

    def subsample_interactions(self, keep_ratio: float, seed: int = 42) -> 'Dataset':
        """Create a subsampled version (for density analysis).

        Returns a new Dataset with randomly dropped interactions.
        Does NOT modify the original.
        """
        np.random.seed(seed)
        mask = np.random.random(len(self.train_df)) < keep_ratio
        new_ds = Dataset.__new__(Dataset)
        new_ds.key = f"{self.key}_sub{int(keep_ratio*100)}"
        new_ds.config = self.config
        new_ds.path = self.path
        new_ds.train_df = self.train_df[mask].copy()
        new_ds.test_df = self.test_df.copy()
        new_ds.val_df = self.val_df
        new_ds.test_loo_df = self.test_loo_df
        new_ds.n_users = new_ds.train_df['user_id'].nunique()
        new_ds.n_items = new_ds.train_df['item_id'].nunique()
        new_ds.n_interactions = len(new_ds.train_df)
        new_ds.density = new_ds.n_interactions / (new_ds.n_users * new_ds.n_items) if new_ds.n_users * new_ds.n_items > 0 else 0
        new_ds._user_history = None
        new_ds._test_gt = None
        new_ds._item_metadata = self._item_metadata  # share metadata
        return new_ds

    def stats(self) -> dict:
        """Return a dict of dataset statistics."""
        return {
            'name': self.name,
            'key': self.key,
            'domain': self.domain,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'catalog_size': self.catalog_size,
            'n_interactions': self.n_interactions,
            'density_pct': round(self.density * 100, 4),
            'n_train': len(self.train_df),
            'n_test': len(self.test_df),
        }

    def __repr__(self):
        return (f"Dataset({self.key}: {self.n_users} users, {self.n_items} items, "
                f"density={self.density*100:.3f}%, domain={self.domain})")


# ============================================================================
# PUBLIC API
# ============================================================================

def resolve_name(name: str) -> str:
    """Resolve aliases and normalize dataset name."""
    name = name.lower().strip()
    return ALIASES.get(name, name)


def load_dataset(name: str) -> Dataset:
    """Load a dataset by name (supports aliases).

    Examples:
        ds = load_dataset('beauty')
        ds = load_dataset('mind')
        ds = load_dataset('ml25m')
    """
    key = resolve_name(name)
    if key not in DATASET_REGISTRY:
        available = ', '.join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    config = DATASET_REGISTRY[key]
    full_path = project_root / 'data' / 'processed' / config.path
    if not (full_path / 'train.parquet').exists():
        raise FileNotFoundError(
            f"Dataset '{key}' not processed. Run the processor first.\n"
            f"Expected path: {full_path}")

    return Dataset(key, config)


def load_all(domains: Optional[List[str]] = None,
             only_existing: bool = True) -> Dict[str, Dataset]:
    """Load all (or filtered) datasets.

    Args:
        domains: filter by domain ('product', 'news', 'movie')
        only_existing: skip datasets that haven't been processed
    """
    result = {}
    for key, cfg in DATASET_REGISTRY.items():
        if domains and cfg.domain not in domains:
            continue
        full_path = project_root / 'data' / 'processed' / cfg.path
        if only_existing and not (full_path / 'train.parquet').exists():
            continue
        try:
            result[key] = Dataset(key, cfg)
        except Exception as e:
            if not only_existing:
                raise
            print(f"  WARNING: Could not load {key}: {e}")
    return result


def get_dataset_groups() -> Dict[str, List[str]]:
    """Get datasets grouped by domain."""
    groups = {}
    for key, cfg in DATASET_REGISTRY.items():
        groups.setdefault(cfg.domain, []).append(key)
    return groups


# ============================================================================
# CLI: list datasets
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dataset interface')
    parser.add_argument('--list', action='store_true', help='List all datasets')
    parser.add_argument('--load', type=str, default=None, help='Load and show stats')
    parser.add_argument('--domain', type=str, default=None, help='Filter by domain')
    args = parser.parse_args()

    if args.list or (not args.load):
        print("=" * 70)
        print("AVAILABLE DATASETS")
        print("=" * 70)
        datasets = list_datasets()
        print(f"{'Key':15s} {'Name':20s} {'Domain':8s} {'Users':>8s} {'Items':>8s} "
              f"{'Density':>8s} {'Status':>8s}")
        print("-" * 80)
        for key, info in datasets.items():
            status = "Ready" if info.exists else "Missing"
            density = f"{info.density:.3f}%" if info.density < 1 else f"{info.density:.2f}%"
            print(f"{key:15s} {info.name:20s} {info.domain:8s} {info.n_users:>8d} "
                  f"{info.n_items:>8d} {density:>8s} {status:>8s}")

        groups = get_dataset_groups()
        print(f"\nDomains: {', '.join(f'{d}({len(v)})' for d, v in groups.items())}")

    if args.load:
        ds = load_dataset(args.load)
        print(f"\n{ds}")
        print(f"Stats: {json.dumps(ds.stats(), indent=2)}")
        samples = ds.get_test_samples(n_users=5)
        print(f"\nSample test users ({len(samples)}):")
        for s in samples[:3]:
            print(f"  user={s['user_id']}, history_len={len(s['history'])}, "
                  f"gt={s['ground_truth'][:3]}...")
