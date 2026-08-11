"""Grokking phase transition detector"""

from typing import List, Dict, Optional, Tuple
from collections import deque

import numpy as np
from scipy import stats
from loguru import logger


class GrokkingDetector:
    """
    Detect grokking (sudden generalization) events during training

    Grokking signature:
    1. Training loss continues to decrease steadily
    2. Validation metric plateaus for extended period
    3. Sudden jump in validation metric
    4. Sustained improvement after jump
    """

    def __init__(
        self,
        window_size: int = 500,
        threshold_delta: float = 0.05,
        plateau_patience: int = 200,
        jump_significance: float = 2.0,
        sustain_steps: int = 100
    ):
        """
        Args:
            window_size: Window size for detecting changes
            threshold_delta: Minimum metric improvement to be significant
            plateau_patience: Min steps to consider plateau
            jump_significance: Z-score threshold for jump detection
            sustain_steps: Steps to confirm sustained improvement
        """
        self.window_size = window_size
        self.threshold_delta = threshold_delta
        self.plateau_patience = plateau_patience
        self.jump_significance = jump_significance
        self.sustain_steps = sustain_steps

        # State tracking
        self.train_losses = deque(maxlen=window_size * 2)
        self.val_metrics = deque(maxlen=window_size * 2)
        self.steps = []

        self.grokking_events = []
        self.in_plateau = False
        self.plateau_start_step = None

    def update(
        self,
        step: int,
        train_loss: float,
        val_metric: float
    ):
        """
        Update detector with new training step

        Args:
            step: Training step
            train_loss: Training loss value
            val_metric: Validation metric (higher is better, e.g., NDCG, Recall)
        """
        self.steps.append(step)
        self.train_losses.append(train_loss)
        self.val_metrics.append(val_metric)

        # Check for grokking if we have enough history
        if len(self.val_metrics) >= self.window_size:
            self._detect_grokking(step)

    def _detect_grokking(self, current_step: int):
        """Internal method to detect grokking events"""
        # Convert to numpy arrays
        val_history = np.array(list(self.val_metrics))
        train_history = np.array(list(self.train_losses))

        # Check if in plateau phase
        recent_val = val_history[-self.plateau_patience:]
        val_std = recent_val.std()
        val_trend = self._compute_trend(recent_val)

        # Plateau condition: low std and no significant trend
        is_plateau = (val_std < self.threshold_delta) and (abs(val_trend) < self.threshold_delta / self.plateau_patience)

        if is_plateau and not self.in_plateau:
            # Entering plateau
            self.in_plateau = True
            self.plateau_start_step = current_step
            logger.info(f"Step {current_step}: Entered plateau phase")

        elif not is_plateau and self.in_plateau:
            # Exiting plateau - check if it's a grokking jump
            plateau_duration = current_step - self.plateau_start_step

            if plateau_duration >= self.plateau_patience:
                # Check for sudden jump
                jump_detected = self._detect_jump(val_history)

                if jump_detected:
                    # Verify sustained improvement
                    if self._verify_sustained_improvement():
                        self._register_grokking_event(
                            plateau_start=self.plateau_start_step,
                            jump_step=current_step,
                            plateau_duration=plateau_duration
                        )

            self.in_plateau = False
            self.plateau_start_step = None

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend (slope)"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def _detect_jump(self, val_history: np.ndarray) -> bool:
        """
        Detect sudden jump using z-score

        Args:
            val_history: Full validation metric history

        Returns:
            True if jump detected
        """
        # Compare recent window to earlier baseline
        baseline = val_history[:-self.window_size // 2]
        recent = val_history[-self.window_size // 2:]

        if len(baseline) < 10 or len(recent) < 10:
            return False

        # Compute z-score of improvement
        baseline_mean = baseline.mean()
        baseline_std = baseline.std()

        if baseline_std < 1e-6:
            return False

        recent_mean = recent.mean()
        z_score = (recent_mean - baseline_mean) / baseline_std

        # Jump if recent performance significantly higher
        return z_score > self.jump_significance

    def _verify_sustained_improvement(self) -> bool:
        """Verify that improvement is sustained"""
        if len(self.val_metrics) < self.sustain_steps:
            return True  # Not enough data to verify, assume sustained

        recent = np.array(list(self.val_metrics)[-self.sustain_steps:])
        trend = self._compute_trend(recent)

        # Sustained if no significant negative trend
        return trend >= -self.threshold_delta / self.sustain_steps

    def _register_grokking_event(
        self,
        plateau_start: int,
        jump_step: int,
        plateau_duration: int
    ):
        """Register a detected grokking event"""
        event = {
            'plateau_start_step': plateau_start,
            'jump_step': jump_step,
            'plateau_duration': plateau_duration,
            'val_metric_before': list(self.val_metrics)[-self.window_size],
            'val_metric_after': list(self.val_metrics)[-1]
        }

        self.grokking_events.append(event)

        logger.info(
            f"🔥 GROKKING DETECTED! "
            f"Plateau: {plateau_start}-{jump_step} ({plateau_duration} steps), "
            f"Metric jump: {event['val_metric_before']:.4f} -> {event['val_metric_after']:.4f}"
        )

    def get_grokking_events(self) -> List[Dict]:
        """Get all detected grokking events"""
        return self.grokking_events

    def has_grokked(self) -> bool:
        """Check if grokking has been detected"""
        return len(self.grokking_events) > 0

    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        if len(self.val_metrics) == 0:
            return {}

        val_history = np.array(list(self.val_metrics))
        train_history = np.array(list(self.train_losses))

        return {
            'num_grokking_events': len(self.grokking_events),
            'in_plateau': self.in_plateau,
            'plateau_start_step': self.plateau_start_step,
            'current_val_metric': val_history[-1],
            'val_metric_std': val_history.std(),
            'val_metric_trend': self._compute_trend(val_history[-self.window_size:]),
            'train_loss_trend': self._compute_trend(train_history[-self.window_size:])
        }
