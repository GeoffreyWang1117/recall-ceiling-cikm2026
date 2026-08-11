"""
P9-4 v6: MoE without Normalization + Cold-start Segmented Analysis

Key improvements over v5:
1. NO Z-score normalization (preserves GNN signal strength)
2. Cold-start segmented metrics (separate analysis for cold/medium/active)
3. Simpler LLM prompts (direct ranking, no CoT)
4. Improved RRF with different k parameters
5. Score calibration: scale LLM scores to GNN range
6. K=500 (optimal from v4)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments' / 'realistic_recall'))

import time
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.llm.llm_recommender import LLMRecommender
from models.gnn.lightgcn import LightGCN
from utils.metrics import calculate_metrics
from cf_recall import CFRecall


# === Configuration ===
CONFIG = {
    'dataset': 'Movies',
    'n_test_users': 100,
    'candidate_size': 500,  # Back to v4's optimal K
    'top_k': 10,
    'seed': 42,
    'lightgcn_dim': 64,
    'lightgcn_layers': 3,
    'lightgcn_epochs': 100,
    'cf_factors': 128,
    # Key change: NO normalization
    'normalization': 'none',
    # User activity thresholds
    'cold_threshold': 20,
    'active_threshold': 100,
}


class MoviesDataset:
    """Simple loader for Movies dataset"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.train_df = pd.read_parquet(self.data_path / 'train.parquet')
        self.val_df = pd.read_parquet(self.data_path / 'val.parquet')
        self.test_df = pd.read_parquet(self.data_path / 'test.parquet')

        # Load item metadata
        self.item_texts = {}
        metadata_file = self.data_path / 'item_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            for item_id_str, meta in metadata.items():
                item_id = int(item_id_str)
                text = meta.get('text')
                if text:
                    self.item_texts[item_id] = text

        self.user_interaction_counts = self.train_df.groupby('user_id').size().to_dict()
        self.item_popularity = self.train_df['item_id'].value_counts().to_dict()
        self.all_items = set(self.train_df['item_id'].unique())

        print(f"Movies Dataset loaded:")
        print(f"  Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        print(f"  Users: {self.train_df['user_id'].nunique()}")
        print(f"  Items: {self.train_df['item_id'].nunique()}")
        print(f"  Items with text: {len(self.item_texts)}")

    def get_test_samples(self) -> List[Dict]:
        samples = []
        for user_id in self.test_df['user_id'].unique():
            history = self.train_df[self.train_df['user_id'] == user_id]['item_id'].tolist()
            ground_truth = self.test_df[self.test_df['user_id'] == user_id]['item_id'].tolist()
            if history and ground_truth:
                samples.append({
                    'user_id': user_id,
                    'history': history,
                    'ground_truth': ground_truth,
                    'interaction_count': len(history)
                })
        return samples


class LightGCNScorer:
    """LightGCN-based scoring with recall capability"""

    def __init__(self, embedding_dim=64, num_layers=3, lr=0.001, epochs=100, device='cuda'):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.model = None

    def fit(self, train_df: pd.DataFrame, verbose=True):
        if verbose:
            print("[LightGCN] Training...")

        self.user_ids = sorted(train_df['user_id'].unique())
        self.item_ids = sorted(train_df['item_id'].unique())
        self.user2idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        self.n_items = n_items

        user_indices = train_df['user_id'].map(self.user2idx).values
        item_indices = train_df['item_id'].map(self.item2idx).values + n_users

        self.edge_index = torch.tensor(
            np.array([
                np.concatenate([user_indices, item_indices]),
                np.concatenate([item_indices, user_indices])
            ]), dtype=torch.long, device=self.device
        )

        self.model = LightGCN(n_users, n_items, self.embedding_dim, self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        user_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.user_items = {self.user2idx[uid]: {self.item2idx[iid] for iid in items}
                          for uid, items in user_items.items()}

        train_pairs = []
        for user_id, pos_items in user_items.items():
            user_idx = self.user2idx[user_id]
            for pos_item in pos_items:
                pos_idx = self.item2idx[pos_item]
                neg_idx = np.random.randint(n_items)
                while self.item_ids[neg_idx] in pos_items:
                    neg_idx = np.random.randint(n_items)
                train_pairs.append([user_idx, pos_idx, neg_idx])
        train_pairs = np.array(train_pairs)

        self.model.train()
        for epoch in range(self.epochs):
            np.random.shuffle(train_pairs)
            total_loss = 0
            batch_size = 4096
            for i in range(0, len(train_pairs), batch_size):
                batch = train_pairs[i:i+batch_size]
                users = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
                pos = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)
                neg = torch.tensor(batch[:, 2], dtype=torch.long, device=self.device)

                user_emb, item_emb = self.model(self.edge_index)
                loss = self.model.bpr_loss(users, pos, neg, user_emb, item_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 25 == 0:
                avg_loss = total_loss / (len(train_pairs) // batch_size)
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.model.eval()
        with torch.no_grad():
            self.user_emb, self.item_emb = self.model(self.edge_index)
        if verbose:
            print("LightGCN trained")

    def score_raw(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Return raw scores"""
        scores = np.zeros(len(item_ids))
        if user_id not in self.user2idx:
            return scores

        user_idx = self.user2idx[user_id]
        with torch.no_grad():
            user_vec = self.user_emb[user_idx]
            for i, item_id in enumerate(item_ids):
                if item_id in self.item2idx:
                    item_idx = self.item2idx[item_id]
                    scores[i] = torch.dot(user_vec, self.item_emb[item_idx]).item()
        return scores

    def recall(self, user_id: int, K: int, exclude_ids: Set[int]) -> Tuple[List[int], np.ndarray]:
        """Recall top-K items for user"""
        if user_id not in self.user2idx:
            return [], np.array([])

        user_idx = self.user2idx[user_id]
        exclude_indices = {self.item2idx[iid] for iid in exclude_ids if iid in self.item2idx}

        with torch.no_grad():
            user_vec = self.user_emb[user_idx]
            scores = torch.matmul(user_vec, self.item_emb.T).cpu().numpy()

        for idx in exclude_indices:
            scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]
        top_items = [self.idx2item[idx] for idx in top_indices if scores[idx] > -np.inf]
        top_scores = scores[top_indices[:len(top_items)]]

        return top_items, top_scores


class HybridRecallV6:
    """Hybrid recall: CF + LightGCN + Popularity"""

    def __init__(self, cf_recall: CFRecall, gnn_scorer: LightGCNScorer,
                 item_popularity: Dict[int, int], all_items: Set[int]):
        self.cf_recall = cf_recall
        self.gnn_scorer = gnn_scorer
        self.item_popularity = item_popularity
        self.all_items = all_items
        self.popular_items = sorted(item_popularity.keys(),
                                    key=lambda x: item_popularity[x], reverse=True)[:1000]

    def recall(self, user_id: int, history: List[int], K: int = 500,
               exclude_ids: List[int] = None) -> Tuple[List[int], np.ndarray, Dict]:
        """Multi-source recall"""
        exclude_set = set(exclude_ids) if exclude_ids else set()
        debug = {}

        # Allocation
        cf_k = int(K * 0.6)
        gnn_k = int(K * 0.4)

        # CF recall
        cf_items, cf_scores = self.cf_recall.recall(user_id, cf_k, exclude_set)
        cf_scores = np.array(cf_scores) if not isinstance(cf_scores, np.ndarray) else cf_scores
        debug['cf_count'] = len(cf_items)

        # GNN recall
        gnn_items, gnn_scores = self.gnn_scorer.recall(user_id, gnn_k, exclude_set | set(cf_items))
        debug['gnn_count'] = len(gnn_items)

        # Merge
        all_items = list(cf_items) + list(gnn_items)
        all_scores = np.concatenate([cf_scores, gnn_scores]) if len(gnn_scores) > 0 else cf_scores

        # Popularity fallback
        if len(all_items) < K:
            existing = set(all_items)
            for item_id in self.popular_items:
                if item_id not in existing and item_id not in exclude_set:
                    all_items.append(item_id)
                    if len(all_items) >= K:
                        break
            debug['pop_added'] = len(all_items) - len(existing)

        debug['total'] = len(all_items)
        return all_items[:K], all_scores[:K] if len(all_scores) >= K else all_scores, debug


class LLMScorerSimple:
    """Simple LLM scorer without CoT (direct ranking)"""

    def __init__(self, llm_model: LLMRecommender, item_texts: Dict[int, str]):
        self.llm = llm_model
        self.item_texts = item_texts

    def _get_text(self, item_id: int) -> str:
        return self.item_texts.get(item_id, f"Movie {item_id}")

    def _clean_text(self, text: str, max_words: int = 8) -> str:
        """Simple text cleaning"""
        words = text.split()[:max_words]
        return ' '.join(words)

    def score(self, history: List[int], candidates: List[int], top_k: int = 10) -> Tuple[List[int], np.ndarray]:
        """Score candidates with simple direct prompt"""
        prompt = self._build_simple_prompt(history, candidates, top_k)

        try:
            result = self.llm.recommend(
                prompt=prompt,
                top_k=top_k,
                return_explanation=False,
                temperature=0.1,
                max_new_tokens=256
            )
            llm_ranking = result.get('item_ids', [])
            # Filter to only include valid candidates
            ranking = [item_id for item_id in llm_ranking if item_id in candidates]
            # Fill with candidates if not enough
            for c in candidates:
                if c not in ranking:
                    ranking.append(c)
                    if len(ranking) >= top_k:
                        break
            ranking = ranking[:top_k]
        except Exception as e:
            print(f"LLM error: {e}")
            ranking = candidates[:top_k]

        # Convert ranking to scores (position-based)
        scores = np.zeros(len(candidates))
        for rank, item_id in enumerate(ranking):
            if item_id in candidates:
                idx = candidates.index(item_id)
                scores[idx] = len(ranking) - rank  # Higher score for better rank

        return ranking, scores

    def _build_simple_prompt(self, history: List[int], candidates: List[int], top_k: int) -> str:
        """Simple direct prompt (no CoT)"""
        prompt = "Task: Recommend movies based on watch history.\n\n"

        prompt += "Watch history:\n"
        for idx, item_id in enumerate(history[-8:], 1):
            text = self._clean_text(self._get_text(item_id))
            prompt += f"  {idx}. {text}\n"

        prompt += f"\nCandidates ({min(len(candidates), 25)} shown):\n"
        for idx, item_id in enumerate(candidates[:25], 1):
            text = self._clean_text(self._get_text(item_id))
            prompt += f"  {idx}. Item {item_id}: {text}\n"

        prompt += f"\nOutput top {top_k} items by ID (one per line, format: Item <ID>):"
        return prompt

    def _parse_response(self, response: str, candidates: List[int], top_k: int) -> List[int]:
        """Parse LLM response"""
        import re
        ranking = []
        candidate_set = set(candidates)

        for line in response.split('\n'):
            matches = re.findall(r'Item\s*(\d+)', line, re.IGNORECASE)
            for m in matches:
                item_id = int(m)
                if item_id in candidate_set and item_id not in ranking:
                    ranking.append(item_id)
                    if len(ranking) >= top_k:
                        break
            if len(ranking) >= top_k:
                break

        # Fill with candidates if not enough
        for c in candidates:
            if c not in ranking:
                ranking.append(c)
                if len(ranking) >= top_k:
                    break

        return ranking[:top_k]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    """RRF: Reciprocal Rank Fusion with parameter k"""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            if item not in scores:
                scores[item] = 0
            scores[item] += 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_items


def scale_to_range(scores: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    """Scale scores to target range"""
    if len(scores) == 0 or scores.max() == scores.min():
        return scores
    normalized = (scores - scores.min()) / (scores.max() - scores.min())
    return normalized * (target_max - target_min) + target_min


def moe_rerank_v6(
    gnn_scorer: LightGCNScorer,
    llm_scorer: LLMScorerSimple,
    user_id: int,
    history: List[int],
    candidates: List[int],
    gnn_weight: float,
    llm_weight: float,
    fusion_method: str = 'weighted_avg',
    top_k: int = 10,
    rrf_k: int = 60
) -> Tuple[List[int], Dict]:
    """MoE reranking WITHOUT normalization"""
    debug = {}

    # Get raw GNN scores
    gnn_scores_raw = gnn_scorer.score_raw(user_id, candidates)
    debug['gnn_range'] = (float(gnn_scores_raw.min()), float(gnn_scores_raw.max()))

    # Get LLM ranking and scores
    llm_ranking, llm_scores_raw = llm_scorer.score(history, candidates, len(candidates))
    debug['llm_range'] = (float(llm_scores_raw.min()), float(llm_scores_raw.max()))

    if fusion_method == 'weighted_avg':
        # Scale LLM scores to match GNN range for fair fusion
        llm_scores_scaled = scale_to_range(llm_scores_raw, gnn_scores_raw.min(), gnn_scores_raw.max())
        combined = gnn_weight * gnn_scores_raw + llm_weight * llm_scores_scaled
        debug['method'] = 'weighted_avg_scaled'

    elif fusion_method == 'raw_weighted':
        # Raw weighted average (no scaling)
        combined = gnn_weight * gnn_scores_raw + llm_weight * llm_scores_raw
        debug['method'] = 'raw_weighted'

    elif fusion_method == 'rrf':
        # Reciprocal Rank Fusion
        gnn_ranking = [candidates[i] for i in np.argsort(gnn_scores_raw)[::-1]]
        combined_ranking = reciprocal_rank_fusion([gnn_ranking, llm_ranking], k=rrf_k)
        debug['method'] = f'rrf_k{rrf_k}'
        return combined_ranking[:top_k], debug

    elif fusion_method == 'rank_avg':
        # Average rank (lower is better)
        gnn_ranks = np.argsort(np.argsort(-gnn_scores_raw))
        llm_ranks = np.argsort(np.argsort(-llm_scores_raw))
        combined_ranks = gnn_weight * gnn_ranks + llm_weight * llm_ranks
        sorted_indices = np.argsort(combined_ranks)[:top_k]
        ranking = [candidates[i] for i in sorted_indices]
        debug['method'] = 'rank_avg'
        return ranking, debug

    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    sorted_indices = np.argsort(combined)[::-1][:top_k]
    ranking = [candidates[i] for i in sorted_indices]
    debug['combined_range'] = (float(combined.min()), float(combined.max()))

    return ranking, debug


def run_strategy_v6(
    name: str,
    gnn_scorer: LightGCNScorer,
    llm_scorer: LLMScorerSimple,
    hybrid_recall: HybridRecallV6,
    test_samples: List[Dict],
    gnn_weight: float,
    llm_weight: float,
    fusion_method: str,
    candidate_size: int,
    top_k: int,
    rrf_k: int = 60
) -> Dict:
    """Run single strategy with v6 enhancements (segmented metrics)"""

    # Segment users by activity
    cold_threshold = CONFIG['cold_threshold']
    active_threshold = CONFIG['active_threshold']

    results = {
        'all': {'ndcg': [], 'recall': [], 'recall_k': []},
        'cold': {'ndcg': [], 'recall': [], 'recall_k': []},
        'medium': {'ndcg': [], 'recall': [], 'recall_k': []},
        'active': {'ndcg': [], 'recall': [], 'recall_k': []},
        'debug_samples': []
    }

    for idx, sample in enumerate(tqdm(test_samples, desc=f"[{name}]")):
        user_id = sample['user_id']
        history = sample['history']
        ground_truth = sample['ground_truth']
        interaction_count = sample['interaction_count']

        # Determine user segment
        if interaction_count <= cold_threshold:
            segment = 'cold'
        elif interaction_count <= active_threshold:
            segment = 'medium'
        else:
            segment = 'active'

        # Hybrid recall
        candidates, _, recall_debug = hybrid_recall.recall(user_id, history, K=candidate_size, exclude_ids=history)
        gt_in_cand = set(candidates) & set(ground_truth)
        recall_quality = len(gt_in_cand) / len(ground_truth) if ground_truth else 0

        # Rerank
        if gnn_weight == 1.0:
            gnn_scores = gnn_scorer.score_raw(user_id, candidates)
            sorted_indices = np.argsort(gnn_scores)[::-1][:top_k]
            ranking = [candidates[i] for i in sorted_indices]
            debug = {'method': 'gnn_only'}
        elif llm_weight == 1.0:
            ranking, _ = llm_scorer.score(history, candidates, top_k)
            debug = {'method': 'llm_only'}
        else:
            ranking, debug = moe_rerank_v6(
                gnn_scorer, llm_scorer,
                user_id, history, candidates,
                gnn_weight, llm_weight,
                fusion_method, top_k, rrf_k
            )

        # Calculate metrics
        gt_for_metrics = ground_truth[:top_k]
        metrics = calculate_metrics([ranking], [gt_for_metrics], k_values=[top_k])
        ndcg = metrics['ndcg@10']
        recall = metrics['recall@10']

        # Store results
        results['all']['ndcg'].append(ndcg)
        results['all']['recall'].append(recall)
        results['all']['recall_k'].append(recall_quality)

        results[segment]['ndcg'].append(ndcg)
        results[segment]['recall'].append(recall)
        results[segment]['recall_k'].append(recall_quality)

        if idx < 3:
            results['debug_samples'].append({
                'user_id': user_id,
                'segment': segment,
                'hist_len': len(history),
                'gt_in_cand': len(gt_in_cand),
                'ndcg': ndcg,
                **debug
            })

    # Aggregate results (convert numpy types to native Python for JSON)
    output = {
        'strategy': name,
        'ndcg@10': float(np.mean(results['all']['ndcg'])),
        'recall@10': float(np.mean(results['all']['recall'])),
        f'recall@{candidate_size}': float(np.mean(results['all']['recall_k'])),
        'ndcg_std': float(np.std(results['all']['ndcg'])),
        'n_samples': int(len(results['all']['ndcg'])),
        # Segmented metrics
        'cold': {
            'n': int(len(results['cold']['ndcg'])),
            'ndcg@10': float(np.mean(results['cold']['ndcg'])) if results['cold']['ndcg'] else 0.0,
            'recall@10': float(np.mean(results['cold']['recall'])) if results['cold']['recall'] else 0.0,
        },
        'medium': {
            'n': int(len(results['medium']['ndcg'])),
            'ndcg@10': float(np.mean(results['medium']['ndcg'])) if results['medium']['ndcg'] else 0.0,
            'recall@10': float(np.mean(results['medium']['recall'])) if results['medium']['recall'] else 0.0,
        },
        'active': {
            'n': int(len(results['active']['ndcg'])),
            'ndcg@10': float(np.mean(results['active']['ndcg'])) if results['active']['ndcg'] else 0.0,
            'recall@10': float(np.mean(results['active']['recall'])) if results['active']['recall'] else 0.0,
        },
        'debug_samples': []  # Simplified to avoid serialization issues
    }

    return output


def main():
    print("=" * 80)
    print("P9-4 v6: MoE without Normalization + Cold-start Segmented Analysis")
    print("=" * 80)

    print(f"\nConfig: {json.dumps(CONFIG, indent=2)}")

    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])

    # Load data
    print("\n[1/6] Loading Movies dataset...")
    data_path = project_root / 'data' / 'processed' / 'amazon_movies_sampled'
    dataset = MoviesDataset(str(data_path))

    # Prepare test samples with stratification
    print("\n[2/6] Preparing test samples...")
    all_samples = dataset.get_test_samples()

    cold_users = [s for s in all_samples if s['interaction_count'] <= CONFIG['cold_threshold']]
    medium_users = [s for s in all_samples if CONFIG['cold_threshold'] < s['interaction_count'] <= CONFIG['active_threshold']]
    active_users = [s for s in all_samples if s['interaction_count'] > CONFIG['active_threshold']]

    print(f"  Cold-start (≤{CONFIG['cold_threshold']}): {len(cold_users)}")
    print(f"  Medium ({CONFIG['cold_threshold']+1}-{CONFIG['active_threshold']}): {len(medium_users)}")
    print(f"  Active (>{CONFIG['active_threshold']}): {len(active_users)}")

    # Stratified sampling
    n_per_segment = CONFIG['n_test_users'] // 3
    sampled = []
    if len(cold_users) >= n_per_segment:
        sampled.extend(np.random.choice(cold_users, n_per_segment, replace=False).tolist())
    else:
        sampled.extend(cold_users)
    if len(medium_users) >= n_per_segment:
        sampled.extend(np.random.choice(medium_users, n_per_segment, replace=False).tolist())
    else:
        sampled.extend(medium_users)
    if len(active_users) >= n_per_segment:
        sampled.extend(np.random.choice(active_users, n_per_segment, replace=False).tolist())
    else:
        sampled.extend(active_users)

    print(f"  Sampled: {len(sampled)} users (stratified)")

    # Build models
    print("\n[3/6] Building models...")
    cf_recall = CFRecall(n_factors=CONFIG['cf_factors'])
    cf_recall.fit(dataset.train_df)

    gnn_scorer = LightGCNScorer(
        embedding_dim=CONFIG['lightgcn_dim'],
        num_layers=CONFIG['lightgcn_layers'],
        epochs=CONFIG['lightgcn_epochs']
    )
    gnn_scorer.fit(dataset.train_df)

    # Build hybrid recall
    print("\n[4/6] Building hybrid recall...")
    hybrid_recall = HybridRecallV6(
        cf_recall, gnn_scorer,
        dataset.item_popularity, dataset.all_items
    )
    print("✓ Hybrid recall (CF + GNN + Pop) built")

    # Load LLM
    print("\n[5/6] Loading LLM...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm = LLMRecommender({
        'name': "Qwen/Qwen2.5-7B-Instruct",
        'quantization': '4bit',
        'max_length': 4096,
        'device': device
    })
    llm_scorer = LLMScorerSimple(llm, dataset.item_texts)
    print(f"LLM loaded on {device}")

    # Run experiments
    print(f"\n[6/6] Running v6 experiments (K={CONFIG['candidate_size']}, no normalization)...")
    start_time = time.time()

    results = {}

    # Strategy configurations
    strategies = [
        # Baseline: GNN-only
        ('GNN-only', 1.0, 0.0, 'weighted_avg', 60),
        # Baseline: LLM-only (simple prompt)
        ('LLM-only-Simple', 0.0, 1.0, 'weighted_avg', 60),
        # MoE with scaled fusion
        ('MoE-Scaled-0.7', 0.7, 0.3, 'weighted_avg', 60),
        ('MoE-Scaled-0.5', 0.5, 0.5, 'weighted_avg', 60),
        # MoE with rank average
        ('MoE-RankAvg-0.7', 0.7, 0.3, 'rank_avg', 60),
        ('MoE-RankAvg-0.5', 0.5, 0.5, 'rank_avg', 60),
        # MoE with RRF (different k values)
        ('MoE-RRF-k60', 0.5, 0.5, 'rrf', 60),
        ('MoE-RRF-k10', 0.5, 0.5, 'rrf', 10),
    ]

    for name, gnn_w, llm_w, method, rrf_k in strategies:
        print(f"\n--- {name} ---")
        results[name] = run_strategy_v6(
            name, gnn_scorer, llm_scorer, hybrid_recall, sampled,
            gnn_w, llm_w, method, CONFIG['candidate_size'], CONFIG['top_k'], rrf_k
        )

        # Print debug
        print(f"  Debug:")
        for d in results[name]['debug_samples']:
            print(f"    User {d['user_id']}: segment={d['segment']}, hist={d['hist_len']}, "
                  f"gt_in_cand={d['gt_in_cand']}, ndcg={d['ndcg']:.4f}")

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 80)
    print(f"Results Summary (v6: K={CONFIG['candidate_size']}, No Normalization, Segmented)")
    print("=" * 80)

    print("\n### Overall Results ###")
    cand_size = CONFIG['candidate_size']
    print(f"\n| Strategy         | NDCG@10 | Recall@10 | Recall@{cand_size} |")
    print("|------------------|---------|-----------|-------------|")
    for name, r in results.items():
        recall_k_key = f'recall@{cand_size}'
        print(f"| {name:16s} | {r['ndcg@10']:.4f}  | {r['recall@10']:.4f}    | {r[recall_k_key]:.4f}       |")

    print("\n### Segmented Results (NDCG@10) ###")
    print(f"\n| Strategy         | Cold (n) | Medium (n) | Active (n) |")
    print("|------------------|----------|------------|------------|")
    for name, r in results.items():
        cold_str = f"{r['cold']['ndcg@10']:.4f} ({r['cold']['n']})"
        medium_str = f"{r['medium']['ndcg@10']:.4f} ({r['medium']['n']})"
        active_str = f"{r['active']['ndcg@10']:.4f} ({r['active']['n']})"
        print(f"| {name:16s} | {cold_str:8s} | {medium_str:10s} | {active_str:10s} |")

    print(f"\nExecution time: {elapsed/60:.1f} minutes")

    # Save results
    output_path = project_root / 'experiments' / 'logs' / 'p9_4_moe_evaluation_v6.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'config': CONFIG,
            'results': results,
            'execution_time': elapsed
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 80)
    print("P9-4 v6 Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
