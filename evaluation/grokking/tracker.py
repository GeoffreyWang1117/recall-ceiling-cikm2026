"""Track multiple metrics for grokking analysis"""

from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from .detector import GrokkingDetector


class GrokkingTracker:
    """
    Track and visualize grokking across multiple metrics

    Tracks:
    - Overall performance (Recall, NDCG)
    - Head vs Tail item performance
    - Cold-start user performance
    - Cross-domain performance
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Grokking configuration from config.yaml
        """
        self.config = config
        self.detectors = {}

        # Create detector for each metric
        for metric in config.get('metrics', ['val_ndcg@10']):
            self.detectors[metric] = GrokkingDetector(
                window_size=config.get('window_size', 500),
                threshold_delta=config.get('threshold_delta', 0.05),
                plateau_patience=config.get('plateau_patience', 200)
            )

        # History storage
        self.history = defaultdict(list)
        self.steps = []

    def update(
        self,
        step: int,
        metrics: Dict[str, float],
        train_loss: float
    ):
        """
        Update all detectors

        Args:
            step: Training step
            metrics: Dict of metric_name -> value
            train_loss: Training loss
        """
        self.steps.append(step)
        self.history['train_loss'].append(train_loss)

        for metric_name, value in metrics.items():
            self.history[metric_name].append(value)

            # Update detector if exists
            if metric_name in self.detectors:
                self.detectors[metric_name].update(step, train_loss, value)

    def get_all_grokking_events(self) -> Dict[str, List[Dict]]:
        """Get grokking events for all metrics"""
        events = {}
        for metric_name, detector in self.detectors.items():
            events[metric_name] = detector.get_grokking_events()
        return events

    def has_any_grokked(self) -> bool:
        """Check if any metric has grokked"""
        return any(d.has_grokked() for d in self.detectors.values())

    def plot_grokking_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot training curves with grokking events highlighted

        Args:
            save_path: Path to save plot
            show: Whether to display plot
        """
        num_metrics = len(self.detectors)
        fig, axes = plt.subplots(num_metrics + 1, 1, figsize=(12, 4 * (num_metrics + 1)))

        if num_metrics == 0:
            return

        # Ensure axes is always iterable (plt.subplots returns single Axes if only 1 subplot)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Plot training loss
        ax = axes[0]
        ax.plot(self.steps, self.history['train_loss'], label='Train Loss', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot each metric
        for idx, (metric_name, detector) in enumerate(self.detectors.items()):
            ax = axes[idx + 1]

            # Plot metric curve
            metric_values = self.history[metric_name]
            if len(metric_values) == 0:
                ax.text(0.5, 0.5, f'No data for {metric_name}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_name} (No Data)')
                continue
            ax.plot(self.steps, metric_values, label=metric_name, alpha=0.7, linewidth=2)

            # Highlight grokking events
            events = detector.get_grokking_events()
            for event in events:
                # Shade plateau region
                plateau_start = event['plateau_start_step']
                jump_step = event['jump_step']

                ax.axvspan(plateau_start, jump_step, alpha=0.2, color='yellow', label='Plateau')
                ax.axvline(jump_step, color='red', linestyle='--', linewidth=2, label='Grokking Jump')

                # Annotate
                ax.annotate(
                    'Grokking!',
                    xy=(jump_step, event['val_metric_after']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', color='red')
                )

            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} - Grokking Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved grokking plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def export_to_csv(self, save_path: str):
        """Export history to CSV for analysis"""
        import pandas as pd

        data = {'step': self.steps}
        for key, values in self.history.items():
            data[key] = values

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        logger.info(f"Exported history to {save_path}")

    def get_summary_report(self) -> str:
        """Generate text summary of grokking analysis"""
        report = "=" * 60 + "\n"
        report += "GROKKING ANALYSIS SUMMARY\n"
        report += "=" * 60 + "\n\n"

        all_events = self.get_all_grokking_events()

        if not any(all_events.values()):
            report += "No grokking events detected.\n"
        else:
            for metric_name, events in all_events.items():
                if events:
                    report += f"\n{metric_name}:\n"
                    report += "-" * 40 + "\n"

                    for i, event in enumerate(events, 1):
                        report += f"  Event {i}:\n"
                        report += f"    Plateau: steps {event['plateau_start_step']} - {event['jump_step']}\n"
                        report += f"    Duration: {event['plateau_duration']} steps\n"
                        report += f"    Metric before: {event['val_metric_before']:.4f}\n"
                        report += f"    Metric after: {event['val_metric_after']:.4f}\n"
                        report += f"    Improvement: {event['val_metric_after'] - event['val_metric_before']:.4f}\n\n"

        # Current statistics
        report += "\nCurrent Statistics:\n"
        report += "-" * 40 + "\n"

        for metric_name, detector in self.detectors.items():
            stats = detector.get_statistics()
            if stats:
                report += f"\n{metric_name}:\n"
                report += f"  Current value: {stats.get('current_val_metric', 0):.4f}\n"
                report += f"  Std dev: {stats.get('val_metric_std', 0):.4f}\n"
                report += f"  Trend: {stats.get('val_metric_trend', 0):.6f}\n"
                report += f"  In plateau: {stats.get('in_plateau', False)}\n"

        return report
