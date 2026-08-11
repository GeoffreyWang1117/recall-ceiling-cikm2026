"""Track and detect emergence across multiple protocols"""

from typing import Dict, List, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from .multi_hop import MultiHopEvaluator
from .cross_community import CrossCommunityEvaluator


class EmergenceTracker:
    """
    Unified tracker for all emergence phenomena

    Tracks:
    - Multi-hop reasoning emergence
    - Cross-community recommendation emergence
    - Cross-modal consistency emergence (if enabled)
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Emergence configuration from config.yaml
        """
        self.config = config
        self.evaluators = {}

        # Initialize evaluators
        if config.get('protocols', {}).get('multi_hop', {}).get('enable', False):
            self.evaluators['multi_hop'] = MultiHopEvaluator(
                k_hop=config['protocols']['multi_hop'].get('k_hop', 2),
                edge_masking_ratio=config['protocols']['multi_hop'].get('edge_masking_ratio', 0.5)
            )

        if config.get('protocols', {}).get('cross_community', {}).get('enable', False):
            self.evaluators['cross_community'] = CrossCommunityEvaluator(
                num_communities=config['protocols']['cross_community'].get('num_communities', 10),
                algorithm=config['protocols']['cross_community'].get('algorithm', 'spectral')
            )

        # History storage
        self.history = {name: [] for name in self.evaluators.keys()}
        self.emergence_events = {name: [] for name in self.evaluators.keys()}
        self.steps = []

    def prepare_evaluators(
        self,
        edge_index: np.ndarray,
        user_item_pairs: np.ndarray,
        num_users: int,
        num_items: int
    ):
        """
        Prepare all evaluators with data

        Args:
            edge_index: Graph structure
            user_item_pairs: Positive pairs
            num_users: Number of users
            num_items: Number of items
        """
        logger.info("Preparing emergence evaluators...")

        if 'multi_hop' in self.evaluators:
            self.evaluators['multi_hop'].prepare_test_set(
                edge_index, user_item_pairs, num_users, num_items
            )

        if 'cross_community' in self.evaluators:
            self.evaluators['cross_community'].detect_communities(
                edge_index, num_users, num_items
            )
            self.evaluators['cross_community'].prepare_cross_community_test(
                user_item_pairs
            )

        logger.info("All evaluators prepared")

    def evaluate_all(
        self,
        step: int,
        model,
        edge_index,
        device: str = 'cuda'
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all emergence evaluations

        Args:
            step: Training step
            model: Model to evaluate
            edge_index: Graph structure
            device: Device

        Returns:
            Dict of {protocol_name: metrics}
        """
        self.steps.append(step)
        all_results = {}

        for name, evaluator in self.evaluators.items():
            results = evaluator.evaluate(model, edge_index, device)
            all_results[name] = results

            # Store history
            self.history[name].append(results)

            # Check for emergence
            if evaluator.track_emergence(results, self.history[name]):
                self._register_emergence_event(name, step, results)

        return all_results

    def _register_emergence_event(
        self,
        protocol_name: str,
        step: int,
        metrics: Dict[str, float]
    ):
        """Register an emergence event"""
        event = {
            'step': step,
            'protocol': protocol_name,
            'metrics': metrics
        }

        self.emergence_events[protocol_name].append(event)

        logger.info(
            f"🌟 EMERGENCE DETECTED in {protocol_name} at step {step}! "
            f"Metrics: {metrics}"
        )

    def has_any_emerged(self) -> bool:
        """Check if any emergence has been detected"""
        return any(len(events) > 0 for events in self.emergence_events.values())

    def plot_emergence_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """Plot emergence curves over training"""
        num_protocols = len(self.evaluators)

        if num_protocols == 0:
            return

        fig, axes = plt.subplots(num_protocols, 1, figsize=(12, 4 * num_protocols))

        if num_protocols == 1:
            axes = [axes]

        for idx, (protocol_name, evaluator) in enumerate(self.evaluators.items()):
            ax = axes[idx]

            # Get primary metric for this protocol
            if protocol_name == 'multi_hop':
                key = f'{evaluator.k_hop}_hop_positive_ratio'
            elif protocol_name == 'cross_community':
                key = 'cross_community_positive_ratio'
            else:
                continue

            # Extract values
            values = [h.get(key, 0) for h in self.history[protocol_name]]

            if not values:
                continue

            # Plot curve
            ax.plot(self.steps[:len(values)], values, label=key, linewidth=2, marker='o')

            # Mark emergence events
            for event in self.emergence_events[protocol_name]:
                event_step = event['step']
                ax.axvline(event_step, color='red', linestyle='--', linewidth=2)
                ax.annotate(
                    'Emergence!',
                    xy=(event_step, event['metrics'].get(key, 0)),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red')
                )

            ax.set_xlabel('Training Step')
            ax.set_ylabel(key)
            ax.set_title(f'{protocol_name.replace("_", " ").title()} Emergence')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved emergence plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_summary_report(self) -> str:
        """Generate summary report"""
        report = "=" * 60 + "\n"
        report += "EMERGENCE ANALYSIS SUMMARY\n"
        report += "=" * 60 + "\n\n"

        if not any(self.emergence_events.values()):
            report += "No emergence events detected.\n"
        else:
            for protocol_name, events in self.emergence_events.items():
                if events:
                    report += f"\n{protocol_name.replace('_', ' ').title()}:\n"
                    report += "-" * 40 + "\n"

                    for i, event in enumerate(events, 1):
                        report += f"  Event {i}:\n"
                        report += f"    Step: {event['step']}\n"
                        report += f"    Metrics: {event['metrics']}\n\n"

        # Current status
        report += "\nCurrent Emergence Metrics:\n"
        report += "-" * 40 + "\n"

        for protocol_name, history in self.history.items():
            if history:
                latest = history[-1]
                report += f"\n{protocol_name}:\n"
                for key, value in latest.items():
                    report += f"  {key}: {value:.4f}\n"

        return report
