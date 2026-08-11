"""OpenAI API helper for teacher & analyzer"""

import os
from typing import List, Dict, Optional

from openai import OpenAI
from loguru import logger


class OpenAIHelper:
    """
    Helper class for using OpenAI API as:
    1. Teacher: Generate explanations, preferences, rationales
    2. Analyzer: Analyze errors, cluster failure modes
    3. Baseline: Direct LLM recommendation for comparison
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """
        Args:
            api_key: OpenAI API key (if None, reads from env)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found! Set OPENAI_API_KEY env variable.")

        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model

    def generate_recommendation_rationale(
        self,
        user_history: List[str],
        candidate_item: str,
        context: str = ""
    ) -> str:
        """
        Generate explanation for why user might like item (teacher mode)

        Args:
            user_history: List of user's previous items
            candidate_item: Candidate item description
            context: Additional context (graph structure, etc.)

        Returns:
            Generated rationale
        """
        if not self.client:
            return ""

        prompt = f"""Given a user's purchase history and a candidate item, explain why the user might be interested in this item.

User History:
{chr(10).join(f"- {item}" for item in user_history)}

Candidate Item: {candidate_item}

Additional Context: {context}

Provide a concise explanation (2-3 sentences):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a recommendation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""

    def analyze_prediction_errors(
        self,
        error_cases: List[Dict],
        max_cases: int = 20
    ) -> str:
        """
        Analyze prediction errors and cluster failure modes (analyzer mode)

        Args:
            error_cases: List of error cases with user, item, predicted, actual
            max_cases: Maximum cases to analyze

        Returns:
            Analysis report
        """
        if not self.client or not error_cases:
            return ""

        # Sample error cases
        sampled_cases = error_cases[:max_cases]

        cases_text = "\n\n".join([
            f"Case {i+1}:\n"
            f"  User: {case['user']}\n"
            f"  Item: {case['item']}\n"
            f"  Predicted: {case['predicted']:.3f}\n"
            f"  Actual: {case['actual']}\n"
            f"  Error type: {case.get('error_type', 'unknown')}"
            for i, case in enumerate(sampled_cases)
        ])

        prompt = f"""Analyze the following recommendation prediction errors and identify common failure patterns.

Error Cases:
{cases_text}

Please provide:
1. Common error patterns (2-3 main categories)
2. Potential root causes
3. Suggestions for improvement

Keep analysis concise (max 300 words):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an ML error analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""

    def direct_recommendation(
        self,
        user_history: List[str],
        candidate_items: List[str],
        top_k: int = 5
    ) -> List[str]:
        """
        Use LLM directly for recommendation (baseline mode)

        Args:
            user_history: User's history
            candidate_items: List of candidates
            top_k: Number of recommendations

        Returns:
            Ranked list of item indices
        """
        if not self.client:
            return list(range(min(top_k, len(candidate_items))))

        prompt = f"""Given a user's purchase history, rank the following candidate items by relevance.

User History:
{chr(10).join(f"- {item}" for item in user_history)}

Candidate Items:
{chr(10).join(f"{i}. {item}" for i, item in enumerate(candidate_items))}

Return the indices of the top {top_k} most relevant items, in order (comma-separated):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a recommendation system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            # Parse response
            content = response.choices[0].message.content.strip()
            indices = [int(x.strip()) for x in content.split(',') if x.strip().isdigit()]

            return indices[:top_k]

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return list(range(min(top_k, len(candidate_items))))

    def generate_graph_context_summary(
        self,
        graph_paths: List[List[str]],
        user_id: str,
        item_id: str
    ) -> str:
        """
        Generate natural language summary of graph context

        Args:
            graph_paths: List of paths connecting user and item
            user_id: User identifier
            item_id: Item identifier

        Returns:
            Graph context summary
        """
        if not self.client or not graph_paths:
            return ""

        paths_text = "\n".join([
            f"Path {i+1}: {' -> '.join(path)}"
            for i, path in enumerate(graph_paths[:5])  # Max 5 paths
        ])

        prompt = f"""Summarize the graph connections between User {user_id} and Item {item_id} in 2-3 sentences.

Paths:
{paths_text}

Summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a graph analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""
