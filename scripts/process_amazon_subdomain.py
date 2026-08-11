#!/usr/bin/env python3
"""
Amazon Subdomain Processor
============================
Process additional Amazon review categories (Toys, Sports, Office, etc.)
into the same format as existing Beauty/Movies/Electronics datasets.

Input:  ~/DataSets/amazon/{Domain}_reviews.parquet + {Domain}_meta.parquet
Output: data/processed/amazon_{domain}_sampled/
        ├── train.parquet
        ├── test.parquet
        └── item_metadata.json

Usage:
    python scripts/process_amazon_subdomain.py --domain Toys_and_Games
    python scripts/process_amazon_subdomain.py --domain Sports_and_Outdoors
    python scripts/process_amazon_subdomain.py --domain Office_Products
    python scripts/process_amazon_subdomain.py --all
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm


DOMAIN_MAP = {
    'Toys_and_Games': {
        'reviews': 'Toys_and_Games_reviews.parquet',
        'meta': 'Toys_and_Games_meta.parquet',
        'output': 'amazon_toys_sampled',
        'short': 'toys'
    },
    'Sports_and_Outdoors': {
        'reviews': 'Sports_and_Outdoors_reviews.parquet',
        'meta': 'Sports_and_Outdoors_meta.parquet',
        'output': 'amazon_sports_sampled',
        'short': 'sports'
    },
    'Office_Products': {
        'reviews': 'Office_Products_reviews.parquet',
        'meta': 'Office_Products_meta.parquet',
        'output': 'amazon_office_sampled',
        'short': 'office'
    },
    'Home_and_Kitchen': {
        'reviews': 'Home_and_Kitchen_reviews.parquet',
        'meta': 'Home_and_Kitchen_meta.parquet',
        'output': 'amazon_home_sampled',
        'short': 'home'
    },
    'Arts_Crafts_and_Sewing': {
        'reviews': 'Arts_Crafts_and_Sewing_reviews.parquet',
        'meta': 'Arts_Crafts_and_Sewing_meta.parquet',
        'output': 'amazon_arts_sampled',
        'short': 'arts'
    },
    'Automotive': {
        'reviews': 'Automotive_reviews.parquet',
        'meta': 'Automotive_meta.parquet',
        'output': 'amazon_automotive_sampled',
        'short': 'automotive'
    }
}


def k_core_filter(df: pd.DataFrame, n_core: int = 5) -> pd.DataFrame:
    """Iteratively filter users and items with fewer than n_core interactions."""
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        valid_users = user_counts[user_counts >= n_core].index
        valid_items = item_counts[item_counts >= n_core].index
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    return df


def process_domain(domain: str, data_dir: Path, output_base: Path,
                   n_core: int = 5, max_users: int = 5000,
                   implicit_threshold: float = 4.0):
    """Process a single Amazon domain."""
    info = DOMAIN_MAP[domain]
    reviews_path = data_dir / info['reviews']
    meta_path = data_dir / info['meta']
    output_dir = output_base / info['output']
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PROCESSING: {domain}")
    print(f"{'='*70}")

    # Load reviews
    print(f"Loading reviews from {reviews_path.name}...")
    reviews_df = pd.read_parquet(reviews_path)
    print(f"  Raw: {len(reviews_df)} reviews")
    print(f"  Columns: {reviews_df.columns.tolist()}")

    # Determine column names (Amazon 2023 format)
    user_col = 'user_id'
    item_col = 'parent_asin' if 'parent_asin' in reviews_df.columns else 'asin'
    rating_col = 'rating'
    time_col = 'timestamp'

    # Convert to implicit feedback
    df = reviews_df[[user_col, item_col, rating_col, time_col]].copy()
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # Filter by rating threshold for implicit feedback
    df = df[df['rating'] >= implicit_threshold].copy()
    print(f"  After implicit filter (>={implicit_threshold}): {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    print(f"  After dedup: {len(df)}")

    # K-core filtering
    print(f"\n  Applying {n_core}-core filtering...")
    df = k_core_filter(df, n_core)
    print(f"  After {n_core}-core: {len(df)} interactions")
    print(f"  Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")

    if df['user_id'].nunique() < 100:
        print(f"  WARNING: Too few users after filtering. Skipping {domain}.")
        return None

    # Subsample users if too many
    if max_users and df['user_id'].nunique() > max_users:
        print(f"\n  Subsampling to {max_users} users (by interaction count)...")
        user_counts = df['user_id'].value_counts()
        # Mix: take top active users + random users for diversity
        n_top = max_users // 2
        n_random = max_users - n_top
        top_users = set(user_counts.head(n_top).index)
        remaining = set(user_counts.index) - top_users
        random_users = set(np.random.choice(list(remaining),
                                            size=min(n_random, len(remaining)),
                                            replace=False))
        selected_users = top_users | random_users
        df = df[df['user_id'].isin(selected_users)]
        df = k_core_filter(df, n_core)
        print(f"  After subsampling: {len(df)} interactions")
        print(f"  Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")

    # Create integer ID mappings
    users = sorted(df['user_id'].unique())
    items = sorted(df['item_id'].unique())
    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}

    orig_item_ids = {v: k for k, v in item2id.items()}

    df = df.copy()
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)

    # Temporal split: leave-one-out
    print("\n  Temporal split (leave-one-out)...")
    df = df.sort_values(['user_id', 'timestamp'])

    train_records = []
    test_records = []
    for uid, group in df.groupby('user_id'):
        records = group.to_dict('records')
        if len(records) < 3:
            continue
        test_records.append(records[-1])
        train_records.extend(records[:-1])

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Save train/test
    train_df[['user_id', 'item_id', 'timestamp']].to_parquet(
        output_dir / 'train.parquet', index=False)
    test_df[['user_id', 'item_id', 'timestamp']].to_parquet(
        output_dir / 'test.parquet', index=False)

    # Load and save item metadata
    print(f"\n  Loading metadata from {meta_path.name}...")
    try:
        meta_df = pd.read_parquet(meta_path)
        title_col = 'title' if 'title' in meta_df.columns else None
        desc_col = None
        for c in ['description', 'details', 'feature']:
            if c in meta_df.columns:
                desc_col = c
                break

        asin_col = 'parent_asin' if 'parent_asin' in meta_df.columns else 'asin'

        item_metadata = {}
        for new_id, orig_id in orig_item_ids.items():
            row = meta_df[meta_df[asin_col] == orig_id]
            if len(row) > 0:
                row = row.iloc[0]
                title = str(row[title_col]) if title_col and pd.notna(row.get(title_col)) else f"Item {orig_id}"
                desc = ''
                if desc_col and pd.notna(row.get(desc_col)):
                    val = row[desc_col]
                    if isinstance(val, list):
                        desc = '. '.join(str(v) for v in val[:3])
                    else:
                        desc = str(val)[:200]
                text = f"{title}. {desc}" if desc else title
                item_metadata[str(new_id)] = {
                    'title': title,
                    'text': text,
                    'asin': orig_id
                }
            else:
                item_metadata[str(new_id)] = {
                    'title': f'Item {orig_id}',
                    'text': f'Item {orig_id}',
                    'asin': orig_id
                }

        with open(output_dir / 'item_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(item_metadata, f, ensure_ascii=False, indent=1)
        print(f"  Metadata saved for {len(item_metadata)} items")

    except Exception as e:
        print(f"  WARNING: Could not load metadata: {e}")
        # Create minimal metadata
        item_metadata = {str(new_id): {'title': f'Item {orig_id}', 'text': f'Item {orig_id}'}
                         for new_id, orig_id in orig_item_ids.items()}
        with open(output_dir / 'item_metadata.json', 'w') as f:
            json.dump(item_metadata, f, indent=1)

    # Save ID mappings
    np.savez(output_dir / 'id_mappings.npz',
             user2id=user2id, item2id=item2id)

    # Summary stats
    catalog_size = df['item_id'].nunique()
    n_users_final = df['user_id'].nunique()
    density = len(df) / (n_users_final * catalog_size) * 100

    stats = {
        'dataset': f'Amazon {domain}',
        'domain': domain,
        'n_users': n_users_final,
        'n_items': catalog_size,
        'n_interactions': len(df),
        'density_pct': round(density, 4),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_core': n_core,
        'implicit_threshold': implicit_threshold,
        'processed_at': datetime.now().isoformat()
    }
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  DONE: {domain}")
    print(f"    Users: {n_users_final}, Items: {catalog_size}")
    print(f"    Density: {density:.4f}%")
    print(f"    Output: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Process Amazon subdomains')
    parser.add_argument('--domain', type=str, default=None,
                        choices=list(DOMAIN_MAP.keys()))
    parser.add_argument('--all', action='store_true',
                        help='Process all available domains')
    parser.add_argument('--data_dir', type=str,
                        default=str(Path.home() / 'DataSets' / 'amazon'))
    parser.add_argument('--n_core', type=int, default=5)
    parser.add_argument('--max_users', type=int, default=5000)
    parser.add_argument('--implicit_threshold', type=float, default=4.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_base = project_root / 'data' / 'processed'

    if args.all:
        domains = list(DOMAIN_MAP.keys())
    elif args.domain:
        domains = [args.domain]
    else:
        # Default: the three highest-value new domains
        domains = ['Toys_and_Games', 'Sports_and_Outdoors', 'Office_Products']

    np.random.seed(42)

    all_stats = {}
    for domain in domains:
        info = DOMAIN_MAP[domain]
        reviews_path = data_dir / info['reviews']
        if not reviews_path.exists():
            print(f"\n  SKIP: {domain} - {reviews_path} not found")
            continue
        stats = process_domain(domain, data_dir, output_base,
                               args.n_core, args.max_users, args.implicit_threshold)
        if stats:
            all_stats[domain] = stats

    # Summary
    print("\n" + "=" * 70)
    print("ALL DOMAINS PROCESSED")
    print("=" * 70)
    for domain, stats in all_stats.items():
        print(f"  {domain:30s}: {stats['n_users']:>6} users, {stats['n_items']:>6} items, "
              f"density={stats['density_pct']:.4f}%")

    # Save combined stats
    with open(output_base / 'new_domains_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)


if __name__ == '__main__':
    main()
