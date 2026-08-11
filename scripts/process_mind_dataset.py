#!/usr/bin/env python3
"""
MIND News Dataset Processor
============================
Converts Microsoft MIND news recommendation dataset into the same format
used by our Amazon experiments (train/test parquets + item_metadata.json).

MIND data: user click sequences on news articles.
We convert clicks to implicit feedback and apply temporal split.

Input:  ~/DataSets/MIND/MINDsmall_train.zip (or MINDlarge_*)
Output: data/processed/mind_news/
        ├── train.parquet
        ├── test.parquet
        └── item_metadata.json

Usage:
    python scripts/process_mind_dataset.py --size small --n_core 5
    python scripts/process_mind_dataset.py --size large --n_core 5 --max_users 10000
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import zipfile
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime


def load_news(news_path: str) -> dict:
    """Load news metadata from news.tsv."""
    news = {}
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                news_id = parts[0]
                category = parts[1]
                subcategory = parts[2]
                title = parts[3]
                abstract = parts[4] if len(parts) > 4 else ''
                news[news_id] = {
                    'title': title,
                    'category': category,
                    'subcategory': subcategory,
                    'abstract': abstract,
                    'text': f"{title}. {abstract}" if abstract else title
                }
    return news


def load_behaviors(behaviors_path: str) -> pd.DataFrame:
    """Load user behavior data from behaviors.tsv.

    Format: impression_id \t user_id \t time \t history \t impressions
    Impressions: newsID-0/1 (0=not clicked, 1=clicked)
    """
    records = []
    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            impression_id = parts[0]
            user_id = parts[1]
            timestamp_str = parts[2]  # e.g., "11/15/2019 7:28:32 AM"
            history = parts[3].split() if parts[3] else []
            impressions = parts[4].split() if parts[4] else []

            # Parse timestamp
            try:
                ts = datetime.strptime(timestamp_str, "%m/%d/%Y %I:%M:%S %p")
                timestamp = int(ts.timestamp())
            except (ValueError, TypeError):
                timestamp = 0

            # Extract clicked items from impressions
            for imp in impressions:
                if '-' in imp:
                    news_id, label = imp.rsplit('-', 1)
                    if label == '1':  # clicked
                        records.append({
                            'user_id': user_id,
                            'item_id': news_id,
                            'timestamp': timestamp,
                            'impression_id': impression_id
                        })

            # Also record history items (with earlier timestamps)
            for i, hist_item in enumerate(history):
                records.append({
                    'user_id': user_id,
                    'item_id': hist_item,
                    'timestamp': timestamp - len(history) + i,  # approximate ordering
                    'impression_id': 'history'
                })

    df = pd.DataFrame(records)
    # Remove duplicates (same user-item pair)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    return df


def k_core_filter(df: pd.DataFrame, n_core: int = 5) -> pd.DataFrame:
    """Iteratively filter users and items with fewer than n_core interactions."""
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()

        valid_users = user_counts[user_counts >= n_core].index
        valid_items = item_counts[item_counts >= n_core].index

        new_df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]

        if len(new_df) == len(df):
            break
        df = new_df

    return df


def temporal_split(df: pd.DataFrame) -> tuple:
    """Leave-one-out split: last interaction per user as test."""
    df = df.sort_values(['user_id', 'timestamp'])

    test_records = []
    train_records = []

    for user_id, group in df.groupby('user_id'):
        if len(group) < 3:
            continue
        items = group.to_dict('records')
        test_records.append(items[-1])  # last item as test
        train_records.extend(items[:-1])  # rest as train

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Process MIND dataset')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'large'],
                        help='MIND dataset size')
    parser.add_argument('--data_dir', type=str, default=str(Path.home() / 'DataSets' / 'MIND'))
    parser.add_argument('--output_dir', type=str,
                        default=str(project_root / 'data' / 'processed' / 'mind_news'))
    parser.add_argument('--n_core', type=int, default=5, help='K-core filtering threshold')
    parser.add_argument('--max_users', type=int, default=None, help='Max users to keep')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MIND Dataset Processing ({args.size})")
    print("=" * 70)

    # Extract zip files
    if args.size == 'small':
        train_zip = data_dir / 'MINDsmall_train.zip'
        dev_zip = data_dir / 'MINDsmall_dev.zip'
        train_folder = f'MINDsmall_train'
        dev_folder = f'MINDsmall_dev'
    else:
        train_zip = data_dir / 'MINDlarge_train.zip'
        dev_zip = data_dir / 'MINDlarge_dev.zip'
        train_folder = f'MINDlarge_train'
        dev_folder = f'MINDlarge_dev'

    extract_dir = data_dir / 'extracted'
    extract_dir.mkdir(exist_ok=True)

    for zf, folder in [(train_zip, train_folder), (dev_zip, dev_folder)]:
        if not (extract_dir / folder).exists():
            print(f"  Extracting {zf.name}...")
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(extract_dir)
        else:
            print(f"  {folder} already extracted.")

    # Load news metadata (from train set, usually superset)
    print("\nLoading news metadata...")
    news_meta = load_news(str(extract_dir / train_folder / 'news.tsv'))
    # Also load from dev set for completeness
    dev_news = load_news(str(extract_dir / dev_folder / 'news.tsv'))
    news_meta.update(dev_news)
    print(f"  Total news articles: {len(news_meta)}")

    # Load behaviors from both train and dev
    print("\nLoading user behaviors...")
    train_behaviors = load_behaviors(str(extract_dir / train_folder / 'behaviors.tsv'))
    dev_behaviors = load_behaviors(str(extract_dir / dev_folder / 'behaviors.tsv'))

    # Combine
    all_behaviors = pd.concat([train_behaviors, dev_behaviors], ignore_index=True)
    all_behaviors = all_behaviors.drop_duplicates(subset=['user_id', 'item_id'], keep='first')

    print(f"  Raw interactions: {len(all_behaviors)}")
    print(f"  Users: {all_behaviors['user_id'].nunique()}")
    print(f"  Items: {all_behaviors['item_id'].nunique()}")

    # Filter: only keep items with metadata
    all_behaviors = all_behaviors[all_behaviors['item_id'].isin(news_meta)]
    print(f"  After metadata filter: {len(all_behaviors)}")

    # K-core filtering
    print(f"\nApplying {args.n_core}-core filtering...")
    filtered = k_core_filter(all_behaviors, args.n_core)
    print(f"  After {args.n_core}-core: {len(filtered)} interactions")
    print(f"  Users: {filtered['user_id'].nunique()}, Items: {filtered['item_id'].nunique()}")

    # Optionally limit users
    if args.max_users and filtered['user_id'].nunique() > args.max_users:
        print(f"\nSubsampling to {args.max_users} users...")
        # Keep users with most interactions
        user_counts = filtered['user_id'].value_counts()
        top_users = user_counts.head(args.max_users).index
        filtered = filtered[filtered['user_id'].isin(top_users)]
        # Re-filter items
        filtered = k_core_filter(filtered, args.n_core)
        print(f"  After subsampling: {len(filtered)} interactions")
        print(f"  Users: {filtered['user_id'].nunique()}, Items: {filtered['item_id'].nunique()}")

    # Create ID mappings
    print("\nCreating ID mappings...")
    users = sorted(filtered['user_id'].unique())
    items = sorted(filtered['item_id'].unique())
    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}

    filtered = filtered.copy()
    filtered['user_id'] = filtered['user_id'].map(user2id)
    filtered['item_id'] = filtered['item_id'].map(item2id)

    # Temporal split
    print("\nSplitting (leave-one-out)...")
    train_df, test_df = temporal_split(filtered)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Save
    print(f"\nSaving to {output_dir}...")
    train_df[['user_id', 'item_id', 'timestamp']].to_parquet(output_dir / 'train.parquet', index=False)
    test_df[['user_id', 'item_id', 'timestamp']].to_parquet(output_dir / 'test.parquet', index=False)

    # Save item metadata (with mapped IDs)
    id2item_orig = {v: k for k, v in item2id.items()}
    item_metadata = {}
    for new_id, orig_id in id2item_orig.items():
        if orig_id in news_meta:
            item_metadata[str(new_id)] = news_meta[orig_id]

    with open(output_dir / 'item_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(item_metadata, f, ensure_ascii=False, indent=1)

    # Save ID mappings
    np.savez(output_dir / 'id_mappings.npz',
             user2id=user2id, item2id=item2id,
             id2user={v: k for k, v in user2id.items()},
             id2item=id2item_orig)

    # Summary
    print("\n" + "=" * 70)
    print("MIND DATASET PROCESSED SUCCESSFULLY")
    print("=" * 70)
    catalog_size = filtered['item_id'].nunique()
    n_users_final = filtered['user_id'].nunique()
    density = len(filtered) / (n_users_final * catalog_size) * 100
    print(f"  Users: {n_users_final}")
    print(f"  Items (catalog): {catalog_size}")
    print(f"  Interactions: {len(filtered)}")
    print(f"  Density: {density:.4f}%")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Output: {output_dir}")

    # Save dataset stats
    stats = {
        'dataset': 'MIND News',
        'size': args.size,
        'n_users': n_users_final,
        'n_items': catalog_size,
        'n_interactions': len(filtered),
        'density': density,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_core': args.n_core,
        'processed_at': datetime.now().isoformat()
    }
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
