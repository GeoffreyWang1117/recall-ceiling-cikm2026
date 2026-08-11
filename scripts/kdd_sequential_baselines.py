#!/usr/bin/env python3
"""
KDD 2026 Sequential Recommendation Baselines
=============================================
Implements SASRec and BERT4Rec as retrieval baselines to compare with CF methods.
Tests whether sequential models can break the recall bottleneck.

Usage:
    python scripts/kdd_sequential_baselines.py --n_users 200 --epochs 20
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# ============================================================================
# SASREC MODEL
# ============================================================================

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (SASRec).

    Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018
    """
    def __init__(self, n_items, hidden_size=64, max_seq_len=50, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Item embedding (0 is padding)
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer layers (causal attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, seq):
        """
        Args:
            seq: [batch_size, seq_len] item indices (0 = padding)
        Returns:
            [batch_size, seq_len, hidden_size] sequence representations
        """
        batch_size, seq_len = seq.shape

        # Get embeddings
        item_emb = self.item_embedding(seq)  # [B, L, H]
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # [1, L, H]

        x = self.dropout(item_emb + pos_emb)
        x = self.layer_norm(x)

        # Create causal mask (for unidirectional attention)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # Padding mask
        padding_mask = (seq == 0)  # [B, L]

        # Apply transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        return x

    def predict(self, seq):
        """Get predictions for all items based on sequence."""
        x = self.forward(seq)  # [B, L, H]

        # For left-padded sequences, last item is always at position -1
        last_hidden = x[:, -1, :]  # [B, H]

        # Score all items using dot product
        all_items = self.item_embedding.weight[1:]  # Exclude padding [N, H]
        scores = torch.matmul(last_hidden, all_items.t())  # [B, N]

        return scores


class BERT4Rec(nn.Module):
    """BERT-style Bidirectional Sequential Recommendation.

    Reference: Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations", CIKM 2019
    """
    def __init__(self, n_items, hidden_size=64, max_seq_len=50, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.mask_token = n_items + 1  # Special mask token

        # Item embedding (0=padding, n_items+1=mask)
        self.item_embedding = nn.Embedding(n_items + 2, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer layers (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_items)

    def forward(self, seq, mask_positions=None):
        """
        Args:
            seq: [batch_size, seq_len] item indices
            mask_positions: optional positions to predict
        Returns:
            [batch_size, seq_len, hidden_size] or predictions at mask positions
        """
        batch_size, seq_len = seq.shape

        # Get embeddings
        item_emb = self.item_embedding(seq)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        x = self.dropout(item_emb + pos_emb)
        x = self.layer_norm(x)

        # Padding mask only (bidirectional attention)
        padding_mask = (seq == 0)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        return x

    def predict(self, seq):
        """Get predictions for next item (append mask token and predict)."""
        # Append mask token at the end
        batch_size = seq.shape[0]
        mask_tokens = torch.full((batch_size, 1), self.mask_token, device=seq.device)
        seq_with_mask = torch.cat([seq[:, 1:], mask_tokens], dim=1)  # Shift and add mask

        x = self.forward(seq_with_mask)
        last_hidden = x[:, -1, :]  # [B, H]

        # Score all items
        all_items = self.item_embedding.weight[1:self.n_items+1]  # Exclude padding and mask
        scores = torch.matmul(last_hidden, all_items.t())

        return scores


# ============================================================================
# SEQUENTIAL RETRIEVAL WRAPPER
# ============================================================================

class SequentialRetrieval:
    """Wrapper for sequential models as retrieval methods."""

    def __init__(self, model_type='sasrec', hidden_size=64, max_seq_len=50, n_epochs=20,
                 batch_size=256, lr=0.001, device='cuda'):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'

    def _prepare_sequences(self, train_df):
        """Convert interactions to sequences."""
        # Sort by user and timestamp if available
        if 'timestamp' in train_df.columns:
            train_df = train_df.sort_values(['user_id', 'timestamp'])

        sequences = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        return sequences

    def _create_training_samples(self, sequences):
        """Create training samples from sequences."""
        samples = []
        for user_id, seq in sequences.items():
            if len(seq) < 2:
                continue
            # Use sliding window
            for i in range(1, len(seq)):
                input_seq = seq[max(0, i-self.max_seq_len):i]
                target = seq[i]
                samples.append((user_id, input_seq, target))
        return samples

    def _pad_sequence(self, seq):
        """Pad sequence to max_seq_len."""
        if len(seq) >= self.max_seq_len:
            return seq[-self.max_seq_len:]
        return [0] * (self.max_seq_len - len(seq)) + seq

    def fit(self, train_df):
        """Train the sequential model."""
        # Build item mapping
        items = train_df['item_id'].unique()
        self.item_to_idx = {item: idx + 1 for idx, item in enumerate(items)}  # 0 is padding
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        n_items = len(items)

        print(f"  Training {self.model_type.upper()} with {n_items} items...")

        # Create model
        if self.model_type == 'sasrec':
            self.model = SASRec(n_items, self.hidden_size, self.max_seq_len).to(self.device)
        else:
            self.model = BERT4Rec(n_items, self.hidden_size, self.max_seq_len).to(self.device)

        # Prepare training data
        sequences = self._prepare_sequences(train_df)
        self.user_sequences = sequences
        samples = self._create_training_samples(sequences)

        if len(samples) == 0:
            print("    No valid training samples!")
            return

        print(f"    {len(samples)} training samples")

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            np.random.shuffle(samples)

            n_batches = (len(samples) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(n_batches):
                batch = samples[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]

                # Prepare batch
                seqs = []
                targets = []
                for user_id, seq, target in batch:
                    # Convert to indices
                    seq_idx = [self.item_to_idx.get(i, 0) for i in seq]
                    seqs.append(self._pad_sequence(seq_idx))
                    targets.append(self.item_to_idx.get(target, 0) - 1)  # 0-indexed for loss

                seqs = torch.tensor(seqs, dtype=torch.long, device=self.device)
                targets = torch.tensor(targets, dtype=torch.long, device=self.device)

                # Filter out invalid targets
                valid_mask = targets >= 0
                if valid_mask.sum() == 0:
                    continue

                seqs = seqs[valid_mask]
                targets = targets[valid_mask]

                # Forward pass
                optimizer.zero_grad()
                scores = self.model.predict(seqs)
                loss = criterion(scores, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss/n_batches:.4f}")

        self.model.eval()

    def recall(self, user_id, K, exclude_ids):
        """Get top-K candidates for a user."""
        if user_id not in self.user_sequences:
            return [], []

        seq = self.user_sequences[user_id]
        seq_idx = [self.item_to_idx.get(i, 0) for i in seq]
        seq_padded = self._pad_sequence(seq_idx)

        with torch.no_grad():
            seq_tensor = torch.tensor([seq_padded], dtype=torch.long, device=self.device)
            scores = self.model.predict(seq_tensor)[0].cpu().numpy()

        # Exclude items
        for item in exclude_ids:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item] - 1] = -np.inf

        # Get top-K
        top_indices = np.argsort(scores)[::-1][:K]
        candidates = []
        item_scores = []

        for idx in top_indices:
            if scores[idx] > -np.inf:
                item_id = self.idx_to_item.get(idx + 1)
                if item_id is not None:
                    candidates.append(item_id)
                    item_scores.append(scores[idx])

        # Normalize scores
        if item_scores:
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]

        return candidates[:K], item_scores[:K]


# ============================================================================
# CF BASELINE FOR COMPARISON
# ============================================================================

class CFRecall:
    """SVD-based Collaborative Filtering (for comparison)."""
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD

        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))
        user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        n_components = min(self.n_factors, min(user_item.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = svd.fit_transform(user_item)
        self.item_factors = svd.components_.T
        self.popularity = train_df['item_id'].value_counts().to_dict()

    def recall(self, user_id, K, exclude_ids):
        if user_id not in self.user_to_idx:
            sorted_items = sorted(self.popularity.keys(), key=lambda x: self.popularity[x], reverse=True)
            candidates = [i for i in sorted_items if i not in set(exclude_ids)][:K]
            scores = [self.popularity.get(c, 0) / max(self.popularity.values()) for c in candidates]
            return candidates, scores

        u_idx = self.user_to_idx[user_id]
        scores = np.dot(self.item_factors, self.user_factors[u_idx])

        for item in exclude_ids:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices if scores[idx] > -np.inf]
        item_scores = [scores[self.item_to_idx[c]] for c in candidates]

        if item_scores:
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]
        return candidates[:K], item_scores[:K]


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval."""
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=200)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='beauty', choices=['beauty', 'movies', 'electronics'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("SEQUENTIAL RECOMMENDATION BASELINES")
    print("SASRec vs BERT4Rec vs CF-SVD")
    print("=" * 70)

    datasets = {
        'beauty': 'amazon_beauty_sampled',
        'movies': 'amazon_movies_sampled',
        'electronics': 'amazon_electronics_sampled'
    }

    ds_name = args.dataset
    ds_path = datasets[ds_name]

    print(f"\nDataset: {ds_name.upper()}")
    print("-" * 70)

    # Load data
    path = project_root / 'data' / 'processed' / ds_path
    train_df = pd.read_parquet(path / 'train.parquet')
    test_df = pd.read_parquet(path / 'test.parquet')

    print(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users, {train_df['item_id'].nunique()} items")
    print(f"Test: {len(test_df)} interactions")

    # Prepare test samples
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_gt.keys() if u in user_history and len(user_history[u]) >= 5]

    n_users = min(args.n_users, len(valid_users))
    sampled_users = np.random.choice(valid_users, size=n_users, replace=False)

    test_samples = [{'user_id': u, 'history': user_history[u], 'ground_truth': test_gt[u]} for u in sampled_users]
    print(f"Test users: {len(test_samples)} (with ≥5 history items)")

    # Build methods
    print("\n" + "=" * 70)
    print("TRAINING RETRIEVAL METHODS")
    print("=" * 70)

    methods = {}

    # CF-SVD baseline
    print("\nCF-SVD...")
    cf = CFRecall()
    cf.fit(train_df)
    methods['CF-SVD'] = cf

    # SASRec
    print("\nSASRec...")
    sasrec = SequentialRetrieval(
        model_type='sasrec',
        hidden_size=args.hidden_size,
        n_epochs=args.epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    sasrec.fit(train_df)
    methods['SASRec'] = sasrec

    # BERT4Rec
    print("\nBERT4Rec...")
    bert4rec = SequentialRetrieval(
        model_type='bert4rec',
        hidden_size=args.hidden_size,
        n_epochs=args.epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    bert4rec.fit(train_df)
    methods['BERT4Rec'] = bert4rec

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING RECALL@K")
    print("=" * 70)

    results = {}

    for method_name, method in methods.items():
        print(f"\n{method_name}:")
        recalls = []

        for sample in tqdm(test_samples, desc=f"  Evaluating"):
            user_id = sample['user_id']
            history = set(sample['history'])
            ground_truth = set(sample['ground_truth'])

            candidates, _ = method.recall(user_id, args.K, history)

            if len(ground_truth) > 0:
                hits = len(set(candidates) & ground_truth)
                recall = hits / len(ground_truth)
                recalls.append(recall)

        mean, ci_low, ci_high = bootstrap_ci(recalls)
        results[method_name] = {
            'recall@100': {
                'mean': float(mean),
                'ci_low': float(ci_low),
                'ci_high': float(ci_high)
            },
            'n_users': int(len(recalls))
        }

        print(f"  Recall@{args.K}: {mean*100:.2f}% [{ci_low*100:.2f}, {ci_high*100:.2f}]")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    from scipy.stats import wilcoxon

    # Re-compute per-user recalls for paired test
    method_recalls = {name: [] for name in methods.keys()}

    for sample in test_samples:
        user_id = sample['user_id']
        history = set(sample['history'])
        ground_truth = set(sample['ground_truth'])

        for method_name, method in methods.items():
            candidates, _ = method.recall(user_id, args.K, history)
            if len(ground_truth) > 0:
                hits = len(set(candidates) & ground_truth)
                recall = hits / len(ground_truth)
                method_recalls[method_name].append(recall)

    # Pairwise Wilcoxon tests
    comparisons = [
        ('SASRec', 'CF-SVD'),
        ('BERT4Rec', 'CF-SVD'),
        ('SASRec', 'BERT4Rec')
    ]

    stat_tests = []
    for m1, m2 in comparisons:
        r1, r2 = method_recalls[m1], method_recalls[m2]
        diff = np.array(r1) - np.array(r2)

        # Only test if there's variance
        if np.std(diff) > 0 and np.sum(diff != 0) >= 10:
            try:
                stat, p = wilcoxon(r1, r2, alternative='two-sided')
            except:
                p = 1.0
        else:
            p = 1.0

        mean_diff = np.mean(diff)
        stat_tests.append({
            'comparison': f'{m1} vs {m2}',
            'mean_diff': float(mean_diff),
            'p_value': float(p),
            'significant': bool(p < 0.05)
        })

        print(f"{m1} vs {m2}: Mean diff = {mean_diff*100:+.3f}%, p = {p:.4f} {'*' if p < 0.05 else ''}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dataset': ds_name,
            'n_users': len(test_samples),
            'K': args.K,
            'epochs': args.epochs,
            'hidden_size': args.hidden_size,
            'seed': args.seed
        },
        'retrieval_results': results,
        'statistical_tests': stat_tests,
        'key_finding': 'Sequential models (SASRec, BERT4Rec) compared to CF-SVD baseline'
    }

    output_path = project_root / 'experiments' / 'logs' / 'kdd_sequential_baselines.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_method = max(results.keys(), key=lambda x: results[x]['recall@100']['mean'])
    best_recall = results[best_method]['recall@100']['mean']

    print(f"\nBest method: {best_method} with Recall@{args.K} = {best_recall*100:.2f}%")

    # Check if sequential methods improve over CF
    cf_recall = results['CF-SVD']['recall@100']['mean']
    sasrec_recall = results['SASRec']['recall@100']['mean']
    bert4rec_recall = results['BERT4Rec']['recall@100']['mean']

    print(f"\nSequential vs CF comparison:")
    print(f"  SASRec vs CF-SVD: {(sasrec_recall - cf_recall)*100:+.2f}%")
    print(f"  BERT4Rec vs CF-SVD: {(bert4rec_recall - cf_recall)*100:+.2f}%")

    if best_recall < 0.10:
        print(f"\n⚠️  Even best sequential model achieves only {best_recall*100:.1f}% recall.")
        print("   The recall bottleneck persists - sequential methods cannot break it.")

    return output


if __name__ == '__main__':
    main()
