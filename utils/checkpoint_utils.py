"""
Checkpoint utilities for long-running experiments.

Allows experiments to:
1. Save progress periodically
2. Resume from interruptions
3. Avoid re-running completed work
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import time


class ExperimentCheckpoint:
    """Manages checkpoints for experiments"""

    def __init__(self, checkpoint_path: str, auto_save_interval: int = 5):
        """
        Args:
            checkpoint_path: Path to checkpoint file
            auto_save_interval: Save every N iterations
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.auto_save_interval = auto_save_interval
        self.data = {
            'completed_indices': [],
            'results': {},
            'metadata': {},
            'last_save_time': time.time()
        }

        # Try to load existing checkpoint
        if self.checkpoint_path.exists():
            self.load()

    def load(self):
        """Load checkpoint from disk"""
        try:
            with open(self.checkpoint_path, 'r') as f:
                self.data = json.load(f)
            print(f"✓ Loaded checkpoint: {len(self.data['completed_indices'])} items completed")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("  Starting from scratch...")

    def save(self, force: bool = False):
        """
        Save checkpoint to disk

        Args:
            force: Force save even if auto_save_interval not reached
        """
        should_save = force or len(self.data['completed_indices']) % self.auto_save_interval == 0

        if should_save:
            try:
                # Create directory if needed
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                # Save checkpoint
                self.data['last_save_time'] = time.time()
                with open(self.checkpoint_path, 'w') as f:
                    json.dump(self.data, f, indent=2)

            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")

    def mark_completed(self, index: int, result_data: Optional[Dict[str, Any]] = None):
        """
        Mark an item as completed

        Args:
            index: Index of completed item
            result_data: Optional result data to store
        """
        if index not in self.data['completed_indices']:
            self.data['completed_indices'].append(index)

            if result_data:
                # Store result for this index
                if 'per_item_results' not in self.data:
                    self.data['per_item_results'] = {}
                self.data['per_item_results'][str(index)] = result_data

            # Auto-save
            self.save()

    def is_completed(self, index: int) -> bool:
        """Check if index is already completed"""
        return index in self.data['completed_indices']

    def get_completed_indices(self):
        """Get list of completed indices"""
        return sorted(self.data['completed_indices'])

    def update_results(self, results: Dict[str, Any]):
        """Update aggregated results"""
        self.data['results'] = results
        self.save(force=True)

    def update_metadata(self, metadata: Dict[str, Any]):
        """Update experiment metadata"""
        self.data['metadata'].update(metadata)

    def get_results(self) -> Dict[str, Any]:
        """Get current aggregated results"""
        return self.data.get('results', {})

    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata"""
        return self.data.get('metadata', {})

    def get_per_item_results(self) -> Dict[str, Any]:
        """Get per-item results"""
        return self.data.get('per_item_results', {})

    def finalize(self, final_results: Dict[str, Any]):
        """Mark experiment as complete and save final results"""
        self.data['results'] = final_results
        self.data['metadata']['completed'] = True
        self.data['metadata']['completion_time'] = time.time()
        self.save(force=True)
        print(f"✓ Checkpoint finalized: {self.checkpoint_path}")

    def cleanup(self):
        """Remove checkpoint file (call after successful completion)"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"✓ Checkpoint cleaned up: {self.checkpoint_path}")

    def get_progress_str(self, total: int) -> str:
        """Get progress string for display"""
        completed = len(self.data['completed_indices'])
        pct = 100 * completed / total if total > 0 else 0
        return f"{completed}/{total} ({pct:.1f}%)"


def get_pending_items(all_items, checkpoint: ExperimentCheckpoint):
    """
    Get list of items that still need processing

    Args:
        all_items: List of all items to process
        checkpoint: Checkpoint object

    Returns:
        List of (index, item) tuples for pending items
    """
    pending = []
    for idx, item in enumerate(all_items):
        if not checkpoint.is_completed(idx):
            pending.append((idx, item))

    return pending
