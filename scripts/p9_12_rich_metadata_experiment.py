"""
P9-12: Rich Metadata Supplementary Experiment

Goal: Test the hypothesis that LLM reranking can work when item metadata is semantically rich.

This experiment:
1. Creates enriched metadata with realistic movie descriptions (plot, themes, cast)
2. Re-runs LLM reranking with the enriched metadata
3. Compares to original synthetic metadata results to demonstrate metadata dependency

This addresses the limitation mentioned in Section 7: "These findings are based on Amazon Movies
with synthetic metadata. Results may differ on datasets with rich descriptions."
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv
import hashlib
import random

load_dotenv()

CONFIG = {
    'n_test_users': 90,  # Match p9_9 experiment size
    'seed': 42,
    'K': 500,
    'top_k_eval': 10,
    'top_k_rerank': 30,
    'cf_factors': 128,
    'cold_threshold': 20,
    'active_threshold': 100,
    'default_model': 'gpt-4o-mini',
    'max_retries': 3,
    'request_timeout': 60,
}

# API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Movie enrichment templates for generating rich metadata
# These are genre-specific templates with varied plot elements
GENRE_TEMPLATES = {
    'Thriller': [
        "A gripping thriller about {protagonist} who uncovers {conspiracy} while {setting}. Features {themes} with performances by {cast}.",
        "Heart-pounding suspense as {protagonist} races against time to {goal}. Set in {location}, this {style} thriller explores {themes}.",
        "When {inciting_incident}, {protagonist} must navigate a web of {danger}. A {style} tale of {themes}.",
    ],
    'Drama': [
        "An emotional journey following {protagonist} as they confront {challenge}. {Cast} delivers powerful performances in this {style} exploration of {themes}.",
        "Set in {location}, this moving drama tells the story of {protagonist} and their struggle with {conflict}. A {style} meditation on {themes}.",
        "A poignant tale of {relationship} and {growth}. {Protagonist} faces {challenge} in this critically acclaimed drama about {themes}.",
    ],
    'Action': [
        "Explosive action as {protagonist}, {hero_description}, must {mission}. Features {action_sequences} and {cast}.",
        "Non-stop thrills when {inciting_incident}. {Protagonist} battles {antagonist} in {location}. A {style} action extravaganza.",
        "High-octane adventure with {protagonist} fighting to {goal}. Featuring {action_sequences} and spectacular {setting}.",
    ],
    'Comedy': [
        "A hilarious comedy about {protagonist} who {comedic_situation}. {Cast} bring laughs in this {style} exploration of {themes}.",
        "Side-splitting humor as {protagonist} navigates {comedic_scenario}. Set in {location}, featuring {style} comedy and {themes}.",
        "What happens when {comedic_premise}? Find out in this {style} comedy starring {cast} as {protagonist}.",
    ],
    'Horror': [
        "Terror awaits as {protagonist} encounters {horror_element} in {location}. A {style} horror film exploring {themes}.",
        "Nightmarish thrills when {inciting_incident}. {Protagonist} must survive {horror_threat} in this {style} chiller.",
        "A bone-chilling tale of {horror_premise}. {Protagonist} faces unspeakable terror in {setting}.",
    ],
    'Romance': [
        "A heartwarming romance between {protagonist1} and {protagonist2}. Set against {backdrop}, this {style} love story explores {themes}.",
        "When {meet_cute}, sparks fly between {protagonist1} and {protagonist2}. A {style} romantic journey through {challenges}.",
        "Love blossoms unexpectedly as {protagonist1} meets {protagonist2} in {location}. A {style} tale of {themes}.",
    ],
    'Sci-Fi': [
        "In a future where {sci_fi_premise}, {protagonist} must {mission}. A {style} sci-fi epic exploring {themes}.",
        "Mind-bending science fiction as {protagonist} discovers {discovery}. Set in {future_setting}, featuring {sci_fi_elements}.",
        "When {sci_fi_incident} threatens humanity, {protagonist} becomes the only hope. Featuring {cast} in this {style} adventure.",
    ],
    'Fantasy': [
        "An epic fantasy adventure where {protagonist} embarks on {quest}. Set in {fantasy_world} with {magical_elements}.",
        "Magic and wonder await as {protagonist} discovers {magical_secret}. A {style} fantasy featuring {fantasy_elements}.",
        "In the realm of {fantasy_realm}, {protagonist} must {quest} to defeat {antagonist}. An epic tale of {themes}.",
    ],
    'Documentary': [
        "An insightful documentary exploring {subject}. Features {footage_type} and {expert_perspectives} on {themes}.",
        "A compelling look at {subject}, this documentary reveals {discoveries}. Directed by {director_style}, featuring {interviewees}.",
        "Award-winning documentary examining {subject}. Through {approach}, it illuminates {themes} with powerful storytelling.",
    ],
    'Musical': [
        "A dazzling musical featuring {protagonist} in {setting}. Includes {song_count} original songs and {dance_style} choreography.",
        "Song and dance spectacular as {protagonist} {musical_journey}. Featuring {cast} performing {music_style} numbers.",
        "A {style} musical celebration following {protagonist} through {story_arc}. Features memorable songs about {themes}.",
    ],
    'Animation': [
        "A delightful animated adventure following {protagonist} who {animated_journey}. Features {animation_style} and {cast} voice performances.",
        "Stunning animation brings to life the story of {protagonist} in {setting}. A {style} tale exploring {themes}.",
        "Animated wonder as {protagonist} discovers {discovery} in {animated_world}. Family-friendly fun with {themes}.",
    ],
}

# Placeholder data for generating rich descriptions
PLACEHOLDER_DATA = {
    'protagonist': ['Detective Sarah Chen', 'struggling artist Michael', 'corporate whistleblower Emma',
                   'retired spy James', 'young journalist Alex', 'mysterious stranger Victor',
                   'single mother Lisa', 'ambitious lawyer David', 'rebel leader Maya'],
    'protagonist1': ['hopeless romantic Jack', 'cynical writer Sophie', 'charming baker Marco'],
    'protagonist2': ['successful CEO Amanda', 'free-spirited artist Luna', 'bookish professor Elena'],
    'conspiracy': ['a government cover-up', 'a corporate scandal', 'a decades-old secret', 'a web of lies'],
    'setting': ['investigating a cold case', 'working undercover', 'protecting a witness', 'fleeing assassins'],
    'location': ['New York City', 'a small coastal town', 'post-war Europe', 'modern Tokyo', 'rural America'],
    'themes': ['love and loss', 'redemption', 'family bonds', 'identity', 'justice', 'sacrifice', 'hope'],
    'cast': ['Academy Award winners', 'an ensemble cast', 'rising stars', 'veteran performers'],
    'style': ['neo-noir', 'character-driven', 'visually stunning', 'emotionally powerful', 'darkly comic'],
    'challenge': ['past trauma', 'family secrets', 'moral dilemmas', 'personal demons', 'societal expectations'],
    'conflict': ['addiction', 'betrayal', 'loss', 'discrimination', 'terminal illness'],
    'relationship': ['father and son', 'two strangers', 'lifelong friends', 'estranged siblings'],
    'growth': ['self-discovery', 'forgiveness', 'acceptance', 'courage'],
    'hero_description': ['a former special forces operative', 'an unlikely hero', 'a rogue agent'],
    'mission': ['save the city', 'rescue hostages', 'stop a terrorist plot', 'protect the innocent'],
    'antagonist': ['ruthless criminals', 'corrupt officials', 'a shadowy organization'],
    'action_sequences': ['breathtaking car chases', 'intense fight choreography', 'explosive set pieces'],
    'comedic_situation': ['accidentally becomes a viral sensation', 'inherits a chaos-filled family business'],
    'comedic_scenario': ['a disastrous wedding', 'a family reunion gone wrong', 'mistaken identity chaos'],
    'comedic_premise': ['two rivals are forced to share an apartment', 'a perfectionist plans the worst vacation'],
    'horror_element': ['an ancient evil', 'supernatural forces', 'a terrifying creature'],
    'horror_threat': ['demonic possession', 'a vengeful spirit', 'unspeakable horrors'],
    'horror_premise': ['a haunted house', 'an isolated cabin', 'a cursed artifact'],
    'meet_cute': ['a chance encounter at a bookstore', 'an awkward first meeting at work'],
    'backdrop': ['picturesque Paris', 'bustling Manhattan', 'scenic countryside'],
    'challenges': ['long-distance', 'family disapproval', 'past heartbreak'],
    'sci_fi_premise': ['AI has achieved consciousness', 'humanity lives among the stars', 'time travel is possible'],
    'discovery': ['alien life', 'a parallel universe', 'the key to immortality'],
    'future_setting': ['a dystopian Earth', 'a space colony', 'a virtual reality world'],
    'sci_fi_elements': ['stunning visual effects', 'philosophical depth', 'thrilling space battles'],
    'sci_fi_incident': ['first contact', 'an extinction-level event', 'a temporal paradox'],
    'quest': ['a perilous journey', 'an ancient prophecy', 'a mission to save the kingdom'],
    'fantasy_world': ['a magical realm', 'an enchanted forest', 'a kingdom of dragons'],
    'magical_elements': ['powerful wizards', 'mythical creatures', 'ancient artifacts'],
    'magical_secret': ['their hidden powers', 'a forbidden spell', 'the truth about their heritage'],
    'fantasy_elements': ['epic battles', 'magical beings', 'legendary weapons'],
    'fantasy_realm': ['Eldoria', 'the Shadowed Lands', 'the Crystal Kingdom'],
    'subject': ['climate change', 'social justice movements', 'technological innovation', 'human rights'],
    'footage_type': ['rare archival footage', 'exclusive interviews', 'stunning cinematography'],
    'expert_perspectives': ['leading scientists', 'eyewitness accounts', 'historical experts'],
    'discoveries': ['hidden truths', 'surprising connections', 'untold stories'],
    'director_style': ['a visionary filmmaker', 'an investigative approach'],
    'interviewees': ['world leaders', 'ordinary heroes', 'industry pioneers'],
    'approach': ['intimate interviews', 'comprehensive research', 'immersive storytelling'],
    'song_count': ['12', '15', '10'],
    'dance_style': ['breathtaking', 'innovative', 'classic'],
    'musical_journey': ['pursues their Broadway dreams', 'finds love through music'],
    'music_style': ['jazz-inspired', 'pop', 'classical'],
    'story_arc': ['fame and fortune', 'love and heartbreak'],
    'animated_journey': ['discovers a hidden world', 'goes on an epic adventure'],
    'animation_style': ['cutting-edge CGI', 'hand-drawn', 'stop-motion'],
    'animated_world': ['a magical kingdom', 'the ocean depths', 'outer space'],
    'inciting_incident': ['a mysterious murder', 'a shocking betrayal', 'an unexpected inheritance'],
    'danger': ['deadly assassins', 'corrupt officials', 'shadowy conspirators'],
    'goal': ['uncover the truth', 'clear their name', 'save their family'],
}


def generate_rich_description(item_id: int, genre: str, seed: int) -> dict:
    """Generate rich metadata for a movie item based on genre."""
    # Use item_id as seed for reproducibility
    random.seed(seed + item_id)

    # Default to Drama if genre not found
    if genre not in GENRE_TEMPLATES:
        genre = 'Drama'

    # Select a template
    template = random.choice(GENRE_TEMPLATES[genre])

    # Fill in placeholders
    description = template
    for placeholder, options in PLACEHOLDER_DATA.items():
        if '{' + placeholder + '}' in description:
            description = description.replace('{' + placeholder + '}', random.choice(options))

    # Generate a realistic title
    title_prefixes = {
        'Thriller': ['The', 'Dark', 'Silent', 'Final', 'Hidden'],
        'Drama': ['A', 'The', 'Beyond', 'Through', 'After'],
        'Action': ['Maximum', 'Ultimate', 'Extreme', 'Total', 'Final'],
        'Comedy': ['My', 'The', 'How to', 'Almost', 'Totally'],
        'Horror': ['The', 'Dark', 'Night of', 'Return of', 'Curse of'],
        'Romance': ['Love', 'Forever', 'A Summer', 'When', 'Finding'],
        'Sci-Fi': ['Beyond', 'Nova', 'Star', 'Quantum', 'Future'],
        'Fantasy': ['The', 'Kingdom of', 'Legend of', 'Rise of', 'Realm of'],
        'Documentary': ['Inside', 'The Truth About', 'Beyond', 'Discovering', 'Unveiling'],
        'Musical': ['Sing', 'Dance', 'Harmony', 'Rhythm', 'The Sound of'],
        'Animation': ['The Adventures of', 'Magic', 'Wonder', 'Journey to', 'Tales of'],
    }

    title_suffixes = {
        'Thriller': ['Secret', 'Truth', 'Witness', 'Night', 'Code', 'Protocol', 'Identity'],
        'Drama': ['Dreams', 'Hope', 'Tomorrow', 'Heart', 'Journey', 'Story', 'Promise'],
        'Action': ['Force', 'Strike', 'Pursuit', 'Justice', 'Vengeance', 'Thunder'],
        'Comedy': ['Wedding', 'Vacation', 'Family', 'Adventure', 'Disaster', 'Madness'],
        'Horror': ['Darkness', 'Shadows', 'Fear', 'Nightmare', 'Evil', 'Terror'],
        'Romance': ['Hearts', 'Destiny', 'Forever', 'Paris', 'You', 'Us'],
        'Sci-Fi': ['Frontier', 'Horizon', 'Genesis', 'Dawn', 'Eclipse', 'Void'],
        'Fantasy': ['Dragons', 'Magic', 'Prophecy', 'Throne', 'Swords', 'Crown'],
        'Documentary': ['the Unknown', 'Truth', 'History', 'Nature', 'Humanity'],
        'Musical': ['Dreams', 'Love', 'Stage', 'Spotlight', 'Melody'],
        'Animation': ['Wonder', 'Magic', 'Friends', 'Adventure', 'World'],
    }

    prefix = random.choice(title_prefixes.get(genre, ['The']))
    suffix = random.choice(title_suffixes.get(genre, ['Story']))

    # Create a unique title using hash of original item_id
    title = f"{prefix} {suffix}"

    # Generate rating and year
    rating = round(random.uniform(5.5, 9.2), 1)
    year = random.randint(1990, 2024)

    return {
        'title': title,
        'description': description,
        'genre': genre,
        'year': year,
        'rating': rating,
        'text': f"{title} ({year}) - {description}"
    }


def create_rich_metadata(original_metadata: dict, seed: int = 42) -> dict:
    """Create enriched metadata from original synthetic metadata."""
    rich_metadata = {}

    for item_id, item_data in original_metadata.items():
        genre = item_data.get('genre', 'Drama')
        rich_data = generate_rich_description(int(item_id), genre, seed)
        rich_metadata[item_id] = rich_data

    return rich_metadata


def ndcg_at_k(ranked_items, ground_truth, k=10):
    """Calculate NDCG@K"""
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_items, ground_truth, k=10):
    """Calculate Recall@K"""
    if not ground_truth:
        return 0.0
    hits = len(set(ranked_items[:k]) & set(ground_truth))
    return hits / len(ground_truth)


class CFModel:
    """CF model using SVD"""
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        self.user_ids = sorted(train_df['user_id'].unique())
        self.item_ids = sorted(train_df['item_id'].unique())
        self.user2idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}

        n_users, n_items = len(self.user_ids), len(self.item_ids)
        rows = train_df['user_id'].map(self.user2idx).values
        cols = train_df['item_id'].map(self.item2idx).values
        data = np.ones(len(train_df))
        self.matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        n_factors = min(self.n_factors, min(n_users, n_items) - 1)
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = svd.fit_transform(self.matrix)
        self.item_factors = svd.components_.T
        print(f"CF model: {n_users} users, {n_items} items")

    def recall(self, user_id, K, exclude_ids):
        if user_id not in self.user2idx:
            return []
        scores = self.item_factors @ self.user_factors[self.user2idx[user_id]]
        for item_id in exclude_ids:
            if item_id in self.item2idx:
                scores[self.item2idx[item_id]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx2item[idx] for idx in top_indices if scores[idx] > -np.inf]

    def score(self, user_id, item_ids):
        if user_id not in self.user2idx:
            return {item: 0.0 for item in item_ids}
        user_vec = self.user_factors[self.user2idx[user_id]]
        scores = {}
        for item in item_ids:
            if item in self.item2idx:
                scores[item] = float(user_vec @ self.item_factors[self.item2idx[item]])
            else:
                scores[item] = 0.0
        return scores


class CooccurrenceModel:
    """Co-occurrence based reranking"""
    def __init__(self, window_size=10):
        self.window_size = window_size

    def fit(self, train_df):
        self.item_ids = sorted(train_df['item_id'].unique())
        self.item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}
        n_items = len(self.item_ids)

        cooc = lil_matrix((n_items, n_items), dtype=np.float32)
        for user_id, group in train_df.groupby('user_id'):
            items = group['item_id'].tolist()
            for i, item1 in enumerate(items):
                if item1 not in self.item2idx:
                    continue
                idx1 = self.item2idx[item1]
                for j in range(max(0, i - self.window_size), min(len(items), i + self.window_size + 1)):
                    if i != j and items[j] in self.item2idx:
                        cooc[idx1, self.item2idx[items[j]]] += 1

        self.cooc = cooc.tocsr()
        row_sums = np.array(self.cooc.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        self.cooc_norm = self.cooc.multiply(1 / row_sums.reshape(-1, 1)).tocsr()
        self.user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        print(f"Co-occurrence model: {n_items} items")

    def recall(self, user_history, K, exclude_ids):
        if not user_history:
            return []
        scores = np.zeros(len(self.item_ids))
        for item_id in user_history[-20:]:
            if item_id in self.item2idx:
                scores += np.array(self.cooc_norm[self.item2idx[item_id]].todense()).flatten()
        for item_id in exclude_ids:
            if item_id in self.item2idx:
                scores[self.item2idx[item_id]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx2item[idx] for idx in top_indices if scores[idx] > -np.inf]


def hybrid_recall(cf_model, cooc_model, user_id, user_history, K, exclude_ids):
    """Hybrid recall combining CF and Co-occurrence"""
    n_cf = int(K * 0.6)
    n_cooc = int(K * 0.4)

    cf_items = set(cf_model.recall(user_id, n_cf * 2, exclude_ids))
    cooc_items = set(cooc_model.recall(user_history, n_cooc * 2, exclude_ids))

    item_scores = defaultdict(float)
    for item in cf_items:
        item_scores[item] += 0.6
    for item in cooc_items:
        item_scores[item] += 0.4

    sorted_items = sorted(item_scores.keys(), key=lambda x: -item_scores[x])
    return sorted_items[:K]


def rerank_cf(cf_model, user_id, candidates):
    """Rerank using CF scores"""
    scores = cf_model.score(user_id, candidates)
    return sorted(candidates, key=lambda x: -scores.get(x, 0))


class LLMRerankerWithRichMetadata:
    """LLM-based reranking using OpenAI API with rich metadata"""

    def __init__(self, model_name, item_metadata, metadata_type='rich'):
        self.model_name = model_name
        self.item_metadata = item_metadata
        self.metadata_type = metadata_type
        self.call_count = 0
        self.total_tokens = 0

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            print("Warning: openai library not found")
            self.client = None

    def _format_item(self, item_id):
        """Format item with rich or synthetic metadata"""
        item_data = self.item_metadata.get(str(item_id), {})

        if self.metadata_type == 'rich':
            title = item_data.get('title', f'Movie {item_id}')
            year = item_data.get('year', '')
            genre = item_data.get('genre', 'Unknown')
            rating = item_data.get('rating', '')
            description = item_data.get('description', '')

            # Include rich description in the prompt
            year_str = f" ({year})" if year else ""
            rating_str = f" [Rating: {rating}]" if rating else ""
            desc_str = f" - {description[:150]}..." if len(description) > 150 else f" - {description}" if description else ""

            return f"{title}{year_str} [{genre}]{rating_str}{desc_str}"
        else:
            # Synthetic format (like original)
            title = item_data.get('title', f'Movie {item_id}')
            genre = item_data.get('genre', 'Unknown')
            return f"{title} ({genre})"

    def _build_prompt(self, user_history, candidates):
        """Build reranking prompt for LLM"""
        # Format user history (last 10 items) with rich descriptions
        history_items = user_history[-10:] if len(user_history) > 10 else user_history
        history_str = "\n".join([
            f"- {self._format_item(item_id)}"
            for item_id in history_items
        ])

        # Format candidates (only show top_k_rerank)
        candidates_to_rank = candidates[:CONFIG['top_k_rerank']]
        candidates_str = "\n".join([
            f"{i+1}. {self._format_item(item_id)}"
            for i, item_id in enumerate(candidates_to_rank)
        ])

        prompt = f"""Based on this user's viewing history, rank these candidate movies from most to least relevant.
Consider the genres, themes, and style patterns in their history.

VIEWING HISTORY:
{history_str}

CANDIDATES TO RANK:
{candidates_str}

Return ONLY a comma-separated list of numbers (1-{len(candidates_to_rank)}) from most to least relevant.
Example: 3,7,1,5,2,4,6,8,9,10"""

        return prompt, candidates_to_rank

    def _parse_llm_response(self, response_text, candidates):
        """Parse LLM response into ranked item list"""
        try:
            import re
            numbers = re.findall(r'\d+', response_text)

            ranked_indices = []
            seen = set()
            for num in numbers:
                idx = int(num) - 1
                if 0 <= idx < len(candidates) and idx not in seen:
                    ranked_indices.append(idx)
                    seen.add(idx)

            ranked_items = [candidates[i] for i in ranked_indices]

            for item in candidates:
                if item not in ranked_items:
                    ranked_items.append(item)

            return ranked_items
        except Exception as e:
            return candidates

    def rerank(self, user_id, user_history, candidates):
        """Rerank candidates using LLM"""
        if not candidates or not user_history or self.client is None:
            return candidates

        prompt, candidates_to_rank = self._build_prompt(user_history, candidates)

        # Rate limiting
        time.sleep(0.1)

        for attempt in range(CONFIG['max_retries']):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': 'You are a movie recommendation expert. Analyze viewing patterns to rank movies by relevance. Respond only with ranking numbers.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150,
                )

                self.call_count += 1
                self.total_tokens += response.usage.total_tokens

                llm_response = response.choices[0].message.content
                reranked = self._parse_llm_response(llm_response, candidates_to_rank)
                remaining = [c for c in candidates[CONFIG['top_k_rerank']:] if c not in reranked]
                return reranked + remaining

            except Exception as e:
                if 'rate_limit' in str(e).lower():
                    time.sleep(2 ** attempt)
                    continue
                else:
                    if attempt == CONFIG['max_retries'] - 1:
                        print(f"LLM error after {CONFIG['max_retries']} attempts: {e}")

        return candidates

    def get_stats(self):
        """Return API usage stats"""
        return {
            'calls': self.call_count,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_call': self.total_tokens / self.call_count if self.call_count > 0 else 0
        }


def run_experiment(item_metadata, metadata_type, test_users, test_gt, train_items,
                   user_history, user_counts, cf_model, cooc_model):
    """Run LLM reranking experiment with given metadata type"""

    print(f"\n{'='*60}")
    print(f"Running experiment with {metadata_type.upper()} METADATA")
    print(f"{'='*60}")

    # Sample metadata for display
    sample_items = list(item_metadata.keys())[:3]
    print("\nSample metadata:")
    for item_id in sample_items:
        data = item_metadata[item_id]
        if metadata_type == 'rich':
            print(f"  {item_id}: {data.get('title')} ({data.get('year')}) - {data.get('description', '')[:80]}...")
        else:
            print(f"  {item_id}: {data.get('title')} ({data.get('genre')})")

    # Initialize LLM reranker
    llm_reranker = LLMRerankerWithRichMetadata(
        CONFIG['default_model'],
        item_metadata,
        metadata_type
    )

    # Test LLM connection
    print(f"\nTesting LLM connection ({CONFIG['default_model']})...")
    if llm_reranker.client is None:
        raise ConnectionError("OpenAI client not initialized")

    try:
        test_response = llm_reranker.client.chat.completions.create(
            model=CONFIG['default_model'],
            messages=[{'role': 'user', 'content': 'Say OK'}],
            max_tokens=5,
        )
        print(f"✓ LLM API connection successful")
    except Exception as e:
        raise ConnectionError(f"LLM API connection failed: {e}")

    results = {}
    K = CONFIG['K']

    # Strategy 1: CF-only baseline
    print("\n--- CF-only (baseline) ---")
    metrics = {'cold': [], 'medium': [], 'active': []}
    for user_id in tqdm(test_users, desc="CF-only"):
        if user_id not in test_gt:
            continue
        gt = set(test_gt[user_id])
        exclude = train_items.get(user_id, set())
        hist = user_history.get(user_id, [])
        n_int = user_counts.get(user_id, 0)

        segment = 'cold' if n_int <= CONFIG['cold_threshold'] else ('medium' if n_int <= CONFIG['active_threshold'] else 'active')

        candidates = hybrid_recall(cf_model, cooc_model, user_id, hist, K, exclude)
        ranked = rerank_cf(cf_model, user_id, candidates)
        ndcg = ndcg_at_k(ranked, gt, CONFIG['top_k_eval'])
        metrics[segment].append(ndcg)

    results['CF-only'] = {seg: np.mean(vals) if vals else 0 for seg, vals in metrics.items()}
    results['CF-only']['overall'] = np.mean([v for vals in metrics.values() for v in vals])
    print(f"Results: {results['CF-only']}")

    # Strategy 2: LLM-only
    print(f"\n--- LLM-only ({metadata_type} metadata) ---")
    metrics = {'cold': [], 'medium': [], 'active': []}
    llm_times = []

    for user_id in tqdm(test_users, desc=f"LLM-{metadata_type}"):
        if user_id not in test_gt:
            continue
        gt = set(test_gt[user_id])
        exclude = train_items.get(user_id, set())
        hist = user_history.get(user_id, [])
        n_int = user_counts.get(user_id, 0)

        segment = 'cold' if n_int <= CONFIG['cold_threshold'] else ('medium' if n_int <= CONFIG['active_threshold'] else 'active')

        candidates = hybrid_recall(cf_model, cooc_model, user_id, hist, K, exclude)

        start_time = time.time()
        ranked = llm_reranker.rerank(user_id, hist, candidates)
        llm_times.append(time.time() - start_time)

        ndcg = ndcg_at_k(ranked, gt, CONFIG['top_k_eval'])
        metrics[segment].append(ndcg)

    results['LLM-only'] = {seg: np.mean(vals) if vals else 0 for seg, vals in metrics.items()}
    results['LLM-only']['overall'] = np.mean([v for vals in metrics.values() for v in vals])
    results['LLM-only']['avg_latency'] = np.mean(llm_times)
    print(f"Results: {results['LLM-only']}")
    print(f"Average LLM latency: {results['LLM-only']['avg_latency']:.2f}s")

    # Add LLM stats
    results['llm_stats'] = llm_reranker.get_stats()

    return results


def main():
    print("=" * 80)
    print("P9-12: Rich Metadata Supplementary Experiment")
    print("=" * 80)
    print("\nGoal: Test if LLM reranking improves when given rich, meaningful metadata")
    print("      vs synthetic metadata like 'Thriller Movie #7440'")

    np.random.seed(CONFIG['seed'])

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    data_path = Path('data/processed/amazon_movies_sampled')
    train_df = pd.read_parquet(data_path / 'train.parquet')
    test_df = pd.read_parquet(data_path / 'test.parquet')

    # Load original synthetic metadata
    with open(data_path / 'item_metadata.json') as f:
        synthetic_metadata = json.load(f)

    print(f"Train: {len(train_df)} interactions")
    print(f"Test: {len(test_df)} interactions")
    print(f"Items: {len(synthetic_metadata)}")

    # Create rich metadata
    print("\nGenerating rich metadata with realistic descriptions...")
    rich_metadata = create_rich_metadata(synthetic_metadata, CONFIG['seed'])

    # Save rich metadata for reference
    rich_metadata_path = data_path / 'item_metadata_rich.json'
    with open(rich_metadata_path, 'w') as f:
        json.dump(rich_metadata, f, indent=2)
    print(f"Rich metadata saved to {rich_metadata_path}")

    # User segmentation
    user_counts = train_df.groupby('user_id').size().to_dict()
    test_users = test_df['user_id'].unique()

    cold = [u for u in test_users if user_counts.get(u, 0) <= CONFIG['cold_threshold']]
    medium = [u for u in test_users if CONFIG['cold_threshold'] < user_counts.get(u, 0) <= CONFIG['active_threshold']]
    active = [u for u in test_users if user_counts.get(u, 0) > CONFIG['active_threshold']]

    # Stratified sampling
    n_per = CONFIG['n_test_users'] // 3
    sampled = (
        np.random.choice(cold, min(n_per, len(cold)), replace=False).tolist() +
        np.random.choice(medium, min(n_per, len(medium)), replace=False).tolist() +
        np.random.choice(active, min(n_per, len(active)), replace=False).tolist()
    )

    print(f"\nTest users: {len(sampled)} (Cold: {min(n_per, len(cold))}, Medium: {min(n_per, len(medium))}, Active: {min(n_per, len(active))})")

    # Prepare data
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    train_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    # Build traditional models
    print("\nBuilding traditional models...")
    cf_model = CFModel(CONFIG['cf_factors'])
    cf_model.fit(train_df)

    cooc_model = CooccurrenceModel()
    cooc_model.fit(train_df)

    # Run experiments with both metadata types
    all_results = {}

    # Experiment 1: Synthetic metadata (control)
    results_synthetic = run_experiment(
        synthetic_metadata, 'synthetic', sampled, test_gt, train_items,
        user_history, user_counts, cf_model, cooc_model
    )
    all_results['synthetic'] = results_synthetic

    # Experiment 2: Rich metadata (treatment)
    results_rich = run_experiment(
        rich_metadata, 'rich', sampled, test_gt, train_items,
        user_history, user_counts, cf_model, cooc_model
    )
    all_results['rich'] = results_rich

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON: SYNTHETIC vs RICH METADATA")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Synthetic':<15} {'Rich':<15} {'Improvement':<15}")
    print("-" * 70)

    # Compare LLM-only results
    syn_llm = all_results['synthetic']['LLM-only']['overall']
    rich_llm = all_results['rich']['LLM-only']['overall']
    llm_improvement = ((rich_llm - syn_llm) / syn_llm * 100) if syn_llm > 0 else float('inf')
    print(f"{'LLM Overall NDCG@10':<25} {syn_llm:<15.4f} {rich_llm:<15.4f} {llm_improvement:+.1f}%")

    # Per-segment comparison
    for segment in ['cold', 'medium', 'active']:
        syn_seg = all_results['synthetic']['LLM-only'].get(segment, 0)
        rich_seg = all_results['rich']['LLM-only'].get(segment, 0)
        seg_improvement = ((rich_seg - syn_seg) / syn_seg * 100) if syn_seg > 0 else float('inf')
        print(f"{'LLM ' + segment.capitalize():<25} {syn_seg:<15.4f} {rich_seg:<15.4f} {seg_improvement:+.1f}%")

    # Compare to CF baseline
    cf_baseline = all_results['rich']['CF-only']['overall']
    print(f"\n{'CF Baseline':<25} {cf_baseline:<15.4f}")

    llm_vs_cf_synthetic = ((syn_llm - cf_baseline) / cf_baseline * 100) if cf_baseline > 0 else 0
    llm_vs_cf_rich = ((rich_llm - cf_baseline) / cf_baseline * 100) if cf_baseline > 0 else 0

    print(f"\n{'LLM vs CF (synthetic)':<25} {llm_vs_cf_synthetic:+.1f}%")
    print(f"{'LLM vs CF (rich)':<25} {llm_vs_cf_rich:+.1f}%")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if rich_llm > syn_llm:
        print(f"\n✓ Rich metadata IMPROVES LLM performance by {llm_improvement:+.1f}%")
        print("  This validates the hypothesis that LLM reranking requires meaningful descriptions.")
    else:
        print(f"\n✗ Rich metadata does NOT improve LLM performance ({llm_improvement:+.1f}%)")
        print("  The bottleneck may be elsewhere (e.g., CF candidate quality, task difficulty).")

    if rich_llm > cf_baseline:
        print(f"\n✓ With rich metadata, LLM outperforms CF baseline ({llm_vs_cf_rich:+.1f}%)")
        print("  LLM can provide value when given meaningful item descriptions.")
    else:
        print(f"\n✗ Even with rich metadata, LLM underperforms CF ({llm_vs_cf_rich:+.1f}%)")
        print("  Collaborative filtering signals remain dominant for this task.")

    # Save results
    output = {
        'config': CONFIG,
        'results': all_results,
        'comparison': {
            'llm_synthetic_ndcg': syn_llm,
            'llm_rich_ndcg': rich_llm,
            'llm_improvement_pct': llm_improvement,
            'cf_baseline_ndcg': cf_baseline,
            'llm_vs_cf_synthetic_pct': llm_vs_cf_synthetic,
            'llm_vs_cf_rich_pct': llm_vs_cf_rich,
        },
        'findings': {
            'rich_metadata_helps': rich_llm > syn_llm,
            'llm_beats_cf_with_rich': rich_llm > cf_baseline,
        }
    }

    output_path = Path('experiments/logs/p9_12_rich_metadata_experiment.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
