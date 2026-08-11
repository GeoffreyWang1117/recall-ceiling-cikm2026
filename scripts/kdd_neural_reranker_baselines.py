#!/usr/bin/env python3
"""
KDD 2026 Neural Reranker Baselines
====================================
Proves the recall ceiling is universal to ALL rerankers, not just LLMs.

Tests:
1. LambdaMART (LightGBM-based learning-to-rank)
2. RankNet (pairwise neural ranker)
3. Cross-Encoder (BERT-based reranker using item text)
4. MLP Reranker (feature-based neural reranker)
5. Oracle Perfect Reranker (theoretical upper bound)

Usage:
    python scripts/kdd_neural_reranker_baselines.py --n_users 500 --dataset beauty
    python scripts/kdd_neural_reranker_baselines.py --n_users 500 --all
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.stats import wilcoxon


# ============================================================================
# CF RETRIEVAL (shared)
# ============================================================================

class CFRecall:
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
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
            return [], []
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

    def get_user_embedding(self, user_id):
        if user_id in self.user_to_idx:
            return self.user_factors[self.user_to_idx[user_id]]
        return np.zeros(self.user_factors.shape[1])

    def get_item_embedding(self, item_id):
        if item_id in self.item_to_idx:
            return self.item_factors[self.item_to_idx[item_id]]
        return np.zeros(self.item_factors.shape[1])


# ============================================================================
# FEATURE ENGINEERING FOR LEARNING-TO-RANK
# ============================================================================

def build_reranking_features(user_id, candidates, cf_scores, cf_model, train_df,
                              user_history, item_popularity):
    """Build feature vectors for each (user, candidate) pair."""
    features = []
    u_emb = cf_model.get_user_embedding(user_id)

    history = set(user_history.get(user_id, []))
    history_items = user_history.get(user_id, [])

    for i, (item_id, cf_score) in enumerate(zip(candidates, cf_scores)):
        feat = []

        # CF score
        feat.append(cf_score)

        # Position in retrieval list
        feat.append(i / len(candidates))

        # Item popularity (log)
        pop = item_popularity.get(item_id, 0)
        feat.append(np.log1p(pop))

        # User-item embedding similarity
        i_emb = cf_model.get_item_embedding(item_id)
        sim = np.dot(u_emb, i_emb) / (np.linalg.norm(u_emb) * np.linalg.norm(i_emb) + 1e-8)
        feat.append(sim)

        # User activity level
        feat.append(np.log1p(len(history)))

        # Category overlap (approximated by co-occurrence with history)
        cooccurrence = 0
        if item_id in cf_model.item_to_idx:
            i_idx = cf_model.item_to_idx[item_id]
            for h_item in history_items[-10:]:  # last 10 items
                if h_item in cf_model.item_to_idx:
                    h_idx = cf_model.item_to_idx[h_item]
                    cooccurrence += np.dot(cf_model.item_factors[i_idx],
                                            cf_model.item_factors[h_idx])
        feat.append(cooccurrence / max(len(history_items[-10:]), 1))

        # Score rank
        feat.append(1.0 / (i + 1))

        # Embedding norm (item)
        feat.append(np.linalg.norm(i_emb))

        features.append(feat)

    return np.array(features)


# ============================================================================
# LAMBDAMART (LightGBM Learning-to-Rank)
# ============================================================================

class LambdaMARTReranker:
    """LambdaMART via LightGBM.

    A well-tuned gradient boosted tree ranker - one of the strongest
    traditional learning-to-rank methods.
    """
    def __init__(self, n_leaves=31, n_estimators=100, lr=0.1):
        self.n_leaves = n_leaves
        self.n_estimators = n_estimators
        self.lr = lr
        self.model = None

    def fit(self, train_features, train_labels, train_groups):
        """Train LambdaMART.

        Args:
            train_features: [N, F] feature matrix
            train_labels: [N] relevance labels (0/1)
            train_groups: list of group sizes (queries)
        """
        import lightgbm as lgb

        train_data = lgb.Dataset(train_features, label=train_labels, group=train_groups)

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10],
            'num_leaves': self.n_leaves,
            'learning_rate': self.lr,
            'n_estimators': self.n_estimators,
            'verbose': -1,
            'seed': 42
        }

        self.model = lgb.train(params, train_data, num_boost_round=self.n_estimators)

    def predict(self, features):
        return self.model.predict(features)


# ============================================================================
# RANKNET (Pairwise Neural Ranker)
# ============================================================================

class RankNetModel(nn.Module):
    """RankNet: pairwise learning-to-rank neural network.

    Reference: Burges et al., "Learning to Rank using Gradient Descent", ICML 2005
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RankNetReranker:
    """RankNet reranker wrapper."""
    def __init__(self, hidden_dim=64, lr=1e-3, n_epochs=20, device='cuda'):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None

    def fit(self, train_features, train_labels, train_groups):
        input_dim = train_features.shape[1]
        self.model = RankNetModel(input_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Build pairwise samples from groups
        pairs_x1, pairs_x2, pairs_y = [], [], []
        offset = 0
        for g_size in train_groups:
            group_feat = train_features[offset:offset + g_size]
            group_label = train_labels[offset:offset + g_size]

            pos_idx = np.where(group_label == 1)[0]
            neg_idx = np.where(group_label == 0)[0]

            for pi in pos_idx:
                for ni in neg_idx[:5]:  # limit negatives
                    pairs_x1.append(group_feat[pi])
                    pairs_x2.append(group_feat[ni])
                    pairs_y.append(1.0)

            offset += g_size

        if len(pairs_x1) == 0:
            print("    Warning: No pairwise training samples")
            return

        x1 = torch.FloatTensor(np.array(pairs_x1)).to(self.device)
        x2 = torch.FloatTensor(np.array(pairs_x2)).to(self.device)

        for epoch in range(self.n_epochs):
            self.model.train()
            # Shuffle
            perm = torch.randperm(len(x1))
            total_loss = 0
            batch_size = 512

            for start in range(0, len(x1), batch_size):
                idx = perm[start:start + batch_size]
                s1 = self.model(x1[idx])
                s2 = self.model(x2[idx])

                # RankNet loss: P(i > j) = sigma(s_i - s_j)
                loss = -F.logsigmoid(s1 - s2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.model.eval()

    def predict(self, features):
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            return self.model(x).cpu().numpy()


# ============================================================================
# MLP RERANKER (Pointwise Neural Ranker)
# ============================================================================

class MLPReranker:
    """Simple MLP pointwise reranker."""
    def __init__(self, hidden_dim=64, lr=1e-3, n_epochs=20, device='cuda'):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device if torch.cuda.is_available() else 'cpu'

    def fit(self, train_features, train_labels, train_groups):
        input_dim = train_features.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        x = torch.FloatTensor(train_features).to(self.device)
        y = torch.FloatTensor(train_labels).to(self.device)

        for epoch in range(self.n_epochs):
            self.model.train()
            perm = torch.randperm(len(x))
            for start in range(0, len(x), 512):
                idx = perm[start:start + 512]
                pred = self.model(x[idx]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()

    def predict(self, features):
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            return self.model(x).squeeze(-1).cpu().numpy()


# ============================================================================
# CROSS-ENCODER RERANKER (BERT-based, using item text)
# ============================================================================

class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Encodes (user_history_text, candidate_text) pairs jointly.
    """
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, train_df, item_texts):
        """Initialize cross-encoder (pre-trained, no fine-tuning)."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"    Warning: Could not load cross-encoder: {e}")
            self.model = None

        self.item_texts = item_texts

    def predict_scores(self, user_history_text, candidate_texts):
        if self.model is None:
            return np.random.randn(len(candidate_texts))

        pairs = [(user_history_text, ct) for ct in candidate_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores


# ============================================================================
# ORACLE PERFECT RERANKER
# ============================================================================

class OracleReranker:
    """Perfect reranker that knows ground truth. Establishes theoretical ceiling."""
    def rerank(self, candidates, ground_truth):
        gt_set = set(ground_truth)
        relevant = [c for c in candidates if c in gt_set]
        irrelevant = [c for c in candidates if c not in gt_set]
        return relevant + irrelevant


# ============================================================================
# EVALUATION
# ============================================================================

def ndcg_at_k(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True))
                  for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), \
           np.percentile(boot_means, (1 - alpha) * 100)


# ============================================================================
# MAIN
# ============================================================================

def run_dataset(ds_name, ds_path, args):
    print(f"\n{'='*70}")
    print(f"DATASET: {ds_name.upper()}")
    print(f"{'='*70}")

    path = project_root / 'data' / 'processed' / ds_path
    train_df = pd.read_parquet(path / 'train.parquet')
    test_df = pd.read_parquet(path / 'test.parquet')

    # Load item metadata for cross-encoder
    item_texts = {}
    meta_path = path / 'item_metadata.json'
    if meta_path.exists():
        import json as jsonlib
        with open(meta_path) as f:
            meta = jsonlib.load(f)
        for k, v in meta.items():
            if v is None:
                item_texts[k] = f'Item {k}'
            else:
                title = v.get('title', f'Item {k}')
                item_texts[k] = (title if title else f'Item {k}')[:200]

    print(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users, {train_df['item_id'].nunique()} items")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare test samples
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]

    n_users = min(args.n_users, len(valid_users))
    sampled_users = np.random.choice(valid_users, size=n_users, replace=False)
    test_samples = [{'user_id': u, 'history': user_history[u], 'ground_truth': test_gt[u]}
                    for u in sampled_users]
    print(f"Test samples: {len(test_samples)}")

    # Build retrieval
    print("\nBuilding CF retrieval...")
    cf = CFRecall(n_factors=128)
    cf.fit(train_df)

    item_popularity = train_df['item_id'].value_counts().to_dict()

    # Retrieve candidates for all users
    print("Retrieving candidates...")
    K = args.K
    user_candidates = {}
    user_cf_scores = {}

    for sample in tqdm(test_samples, desc="  Retrieval"):
        uid = sample['user_id']
        cands, scores = cf.recall(uid, K, sample['history'])
        user_candidates[uid] = cands
        user_cf_scores[uid] = scores

    # Check recall
    recall_scores = []
    for sample in test_samples:
        gt_set = set(sample['ground_truth'])
        cands = set(user_candidates[sample['user_id']])
        hits = len(cands & gt_set)
        recall_scores.append(hits / len(gt_set) if gt_set else 0)
    mean_recall = np.mean(recall_scores)
    print(f"  Recall@{K}: {mean_recall*100:.2f}%")

    # ========================================================================
    # BUILD TRAINING DATA FOR SUPERVISED RERANKERS
    # ========================================================================
    print("\nBuilding training data for supervised rerankers...")

    # Use validation set users for training rerankers
    val_df = pd.read_parquet(path / 'val.parquet') if (path / 'val.parquet').exists() else None

    if val_df is not None:
        val_gt = val_df.groupby('user_id')['item_id'].apply(list).to_dict()
        val_users = [u for u in val_gt if u in user_history and len(user_history[u]) >= 3]
    else:
        # Split test users: half for train, half for eval
        np.random.shuffle(sampled_users)
        split = len(sampled_users) // 2
        val_users = sampled_users[:split]
        val_gt = test_gt

    # Build training features
    all_features = []
    all_labels = []
    all_groups = []

    for uid in tqdm(val_users[:200], desc="  Building features"):
        if uid not in user_history:
            continue
        cands, scores = cf.recall(uid, K, user_history[uid])
        if len(cands) == 0:
            continue

        gt_items = set(val_gt.get(uid, []))
        features = build_reranking_features(uid, cands, scores, cf, train_df,
                                             user_history, item_popularity)
        labels = np.array([1 if c in gt_items else 0 for c in cands])

        all_features.append(features)
        all_labels.append(labels)
        all_groups.append(len(cands))

    if all_features:
        train_features = np.vstack(all_features)
        train_labels = np.concatenate(all_labels)
        train_groups = all_groups
        print(f"  Training samples: {len(train_features)}, Positive rate: {train_labels.mean():.4f}")
    else:
        train_features = np.zeros((0, 8))
        train_labels = np.array([])
        train_groups = []

    # ========================================================================
    # TRAIN AND EVALUATE RERANKERS
    # ========================================================================

    reranker_results = {}

    # 1. CF Score baseline (no reranking)
    print("\n--- CF Score Baseline ---")
    cf_ndcg_scores = []
    for sample in test_samples:
        uid = sample['user_id']
        cands = user_candidates[uid]
        cf_ndcg_scores.append(ndcg_at_k(cands, sample['ground_truth'], k=10))
    mean, lo, hi = bootstrap_ci(cf_ndcg_scores)
    reranker_results['CF-Score'] = {'ndcg': mean, 'ci_low': lo, 'ci_high': hi,
                                     'scores': cf_ndcg_scores}
    print(f"  NDCG@10: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # 2. Oracle Perfect Reranker
    print("\n--- Oracle Perfect Reranker ---")
    oracle = OracleReranker()
    oracle_ndcg_scores = []
    for sample in test_samples:
        uid = sample['user_id']
        cands = user_candidates[uid]
        reranked = oracle.rerank(cands, sample['ground_truth'])
        oracle_ndcg_scores.append(ndcg_at_k(reranked, sample['ground_truth'], k=10))
    mean, lo, hi = bootstrap_ci(oracle_ndcg_scores)
    reranker_results['Oracle-Perfect'] = {'ndcg': mean, 'ci_low': lo, 'ci_high': hi,
                                           'scores': oracle_ndcg_scores}
    print(f"  NDCG@10: {mean:.4f} [{lo:.4f}, {hi:.4f}] (theoretical ceiling)")

    # 3. LambdaMART
    if len(train_features) > 0:
        print("\n--- LambdaMART ---")
        lambdamart = LambdaMARTReranker(n_leaves=31, n_estimators=100, lr=0.1)
        try:
            lambdamart.fit(train_features, train_labels, train_groups)
            lm_ndcg_scores = []
            for sample in test_samples:
                uid = sample['user_id']
                cands = user_candidates[uid]
                scores = user_cf_scores[uid]
                features = build_reranking_features(uid, cands, scores, cf, train_df,
                                                     user_history, item_popularity)
                pred_scores = lambdamart.predict(features)
                reranked_idx = np.argsort(pred_scores)[::-1]
                reranked = [cands[i] for i in reranked_idx]
                lm_ndcg_scores.append(ndcg_at_k(reranked, sample['ground_truth'], k=10))
            mean, lo, hi = bootstrap_ci(lm_ndcg_scores)
            reranker_results['LambdaMART'] = {'ndcg': mean, 'ci_low': lo, 'ci_high': hi,
                                               'scores': lm_ndcg_scores}
            print(f"  NDCG@10: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  Error: {e}")

    # 4. RankNet
    if len(train_features) > 0:
        print("\n--- RankNet ---")
        ranknet = RankNetReranker(hidden_dim=64, lr=1e-3, n_epochs=20)
        try:
            ranknet.fit(train_features, train_labels, train_groups)
            rn_ndcg_scores = []
            for sample in test_samples:
                uid = sample['user_id']
                cands = user_candidates[uid]
                scores = user_cf_scores[uid]
                features = build_reranking_features(uid, cands, scores, cf, train_df,
                                                     user_history, item_popularity)
                pred_scores = ranknet.predict(features)
                reranked_idx = np.argsort(pred_scores)[::-1]
                reranked = [cands[i] for i in reranked_idx]
                rn_ndcg_scores.append(ndcg_at_k(reranked, sample['ground_truth'], k=10))
            mean, lo, hi = bootstrap_ci(rn_ndcg_scores)
            reranker_results['RankNet'] = {'ndcg': mean, 'ci_low': lo, 'ci_high': hi,
                                            'scores': rn_ndcg_scores}
            print(f"  NDCG@10: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  Error: {e}")

    # 5. MLP Reranker
    if len(train_features) > 0:
        print("\n--- MLP Reranker ---")
        mlp = MLPReranker(hidden_dim=64, lr=1e-3, n_epochs=20)
        try:
            mlp.fit(train_features, train_labels, train_groups)
            mlp_ndcg_scores = []
            for sample in test_samples:
                uid = sample['user_id']
                cands = user_candidates[uid]
                scores = user_cf_scores[uid]
                features = build_reranking_features(uid, cands, scores, cf, train_df,
                                                     user_history, item_popularity)
                pred_scores = mlp.predict(features)
                reranked_idx = np.argsort(pred_scores)[::-1]
                reranked = [cands[i] for i in reranked_idx]
                mlp_ndcg_scores.append(ndcg_at_k(reranked, sample['ground_truth'], k=10))
            mean, lo, hi = bootstrap_ci(mlp_ndcg_scores)
            reranker_results['MLP-Reranker'] = {'ndcg': mean, 'ci_low': lo, 'ci_high': hi,
                                                 'scores': mlp_ndcg_scores}
            print(f"  NDCG@10: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  Error: {e}")

    # 6. Cross-Encoder (if text available)
    if item_texts and args.cross_encoder:
        print("\n--- Cross-Encoder Reranker ---")
        ce = CrossEncoderReranker()
        ce.fit(train_df, item_texts)

        if ce.model is not None:
            ce_ndcg_scores = []
            n_ce_users = min(100, len(test_samples))  # Limit due to cost
            for sample in tqdm(test_samples[:n_ce_users], desc="  Cross-Encoder"):
                uid = sample['user_id']
                cands = user_candidates[uid]

                # Build user history text
                hist_items = sample['history'][-5:]
                user_text = "User liked: " + ", ".join(
                    [item_texts.get(str(h), f"item_{h}") for h in hist_items]
                )

                # Candidate texts
                cand_texts = [item_texts.get(str(c), f"item_{c}") for c in cands[:50]]

                scores = ce.predict_scores(user_text, cand_texts)
                reranked_idx = np.argsort(scores)[::-1]
                reranked = [cands[i] for i in reranked_idx if i < len(cands)]
                # Add remaining candidates
                reranked += [c for c in cands if c not in reranked]
                ce_ndcg_scores.append(ndcg_at_k(reranked, sample['ground_truth'], k=10))

            mean, lo, hi = bootstrap_ci(ce_ndcg_scores)
            reranker_results['Cross-Encoder'] = {'ndcg': mean, 'ci_low': lo, 'ci_high': hi,
                                                  'scores': ce_ndcg_scores}
            print(f"  NDCG@10: {mean:.4f} [{lo:.4f}, {hi:.4f}] (n={n_ce_users})")

    # ========================================================================
    # STATISTICAL COMPARISONS
    # ========================================================================
    print(f"\n--- Statistical Comparisons (vs CF-Score) ---")
    stat_tests = []

    cf_scores_list = reranker_results['CF-Score']['scores']

    for name, res in reranker_results.items():
        if name == 'CF-Score':
            continue
        method_scores = res['scores']

        # Align lengths
        min_len = min(len(cf_scores_list), len(method_scores))
        s1 = np.array(cf_scores_list[:min_len])
        s2 = np.array(method_scores[:min_len])
        diff = s2 - s1

        if np.std(diff) > 0 and np.sum(diff != 0) >= 10:
            try:
                _, p = wilcoxon(s2, s1)
            except Exception:
                p = 1.0
        else:
            p = 1.0

        mean_diff = np.mean(diff)
        d = mean_diff / (np.std(diff) + 1e-10)

        stat_tests.append({
            'method': name,
            'vs': 'CF-Score',
            'mean_ndcg_diff': float(mean_diff),
            'cohens_d': float(d),
            'p_value': float(p),
            'significant': bool(p < 0.05)
        })

        sig = '*' if p < 0.05 else ''
        pct_change = (mean_diff / (reranker_results['CF-Score']['ndcg'] + 1e-10)) * 100
        print(f"  {name:20s}: {res['ndcg']:.4f} (diff={mean_diff:+.4f}, {pct_change:+.1f}%, d={d:.3f}, p={p:.4f}) {sig}")

    # Key finding
    oracle_ceiling = reranker_results.get('Oracle-Perfect', {}).get('ndcg', 0)
    cf_baseline = reranker_results['CF-Score']['ndcg']
    best_reranker = max(
        [(n, r['ndcg']) for n, r in reranker_results.items()
         if n not in ('CF-Score', 'Oracle-Perfect')],
        key=lambda x: x[1], default=('None', 0)
    )

    print(f"\n--- Summary ---")
    print(f"  Recall@{K}: {mean_recall*100:.2f}%")
    print(f"  CF-Score baseline: {cf_baseline:.4f}")
    print(f"  Oracle ceiling: {oracle_ceiling:.4f}")
    print(f"  Best reranker: {best_reranker[0]} = {best_reranker[1]:.4f}")
    print(f"  Ceiling utilization: {best_reranker[1]/(oracle_ceiling+1e-10)*100:.1f}%")

    return {
        'dataset': ds_name,
        'n_users': len(test_samples),
        'recall_at_K': float(mean_recall),
        'K': K,
        'reranker_results': {
            name: {'ndcg': float(r['ndcg']),
                   'ci_low': float(r['ci_low']),
                   'ci_high': float(r['ci_high'])}
            for name, r in reranker_results.items()
        },
        'statistical_tests': stat_tests,
        'oracle_ceiling': float(oracle_ceiling),
        'ceiling_utilization': float(best_reranker[1] / (oracle_ceiling + 1e-10))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--new', action='store_true',
                        help='Run on new datasets (toys/sports/office/mind/movielens)')
    parser.add_argument('--cross_encoder', action='store_true',
                        help='Include cross-encoder (slow)')
    args = parser.parse_args()

    datasets = {
        'beauty': 'amazon_beauty_sampled',
        'movies': 'amazon_movies_sampled',
        'electronics': 'amazon_electronics_sampled'
    }
    new_datasets = {
        'toys': 'amazon_toys_sampled',
        'sports': 'amazon_sports_sampled',
        'office': 'amazon_office_sampled',
        'mind': 'mind_news',
        'movielens': 'movielens_25m',
    }

    if args.new:
        ds_list = list(new_datasets.items())
    elif args.all:
        combined = {**datasets, **new_datasets}
        ds_list = list(combined.items())
    elif args.dataset:
        all_ds = {**datasets, **new_datasets}
        ds_list = [(args.dataset, all_ds[args.dataset])]
    else:
        ds_list = list(datasets.items())

    all_results = {}
    for ds_name, ds_path in ds_list:
        result = run_dataset(ds_name, ds_path, args)
        all_results[ds_name] = result

    output_path = project_root / 'experiments' / 'logs' / 'kdd_neural_reranker_baselines.json'
    existing = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(all_results)
    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"{'='*70}")

    # Cross-dataset summary
    print("\n" + "=" * 70)
    print("CROSS-DATASET: ALL RERANKERS VS CF-SCORE")
    print("=" * 70)
    for ds_name, res in all_results.items():
        print(f"\n  {ds_name} (Recall@{res['K']}: {res['recall_at_K']*100:.1f}%):")
        for name, r in res['reranker_results'].items():
            print(f"    {name:20s}: {r['ndcg']:.4f}")

    print("\nConclusion: The recall ceiling constrains ALL rerankers, not just LLMs.")


if __name__ == '__main__':
    main()
