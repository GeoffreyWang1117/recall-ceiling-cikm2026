"""Training utilities and Trainer class"""

from typing import Dict, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import wandb

from evaluation.grokking import GrokkingTracker
from evaluation.emergence import EmergenceTracker


class Trainer:
    """
    Main trainer class with support for:
    - Mixed precision training
    - Gradient accumulation
    - Distributed training
    - Grokking & Emergence tracking
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer
            device: Device to train on
            rank: Process rank (for DDP)
            world_size: Total number of processes
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Training config
        self.max_epochs = config['training']['max_epochs']
        self.gradient_accumulation_steps = config['hardware']['gradient_accumulation_steps']
        self.mixed_precision = config['hardware']['mixed_precision']

        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None

        # Checkpoint config
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = config['logging']['save_interval']

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric = 0.0

        # Grokking tracker
        if self.is_main_process and config.get('grokking', {}).get('enable', False):
            self.grokking_tracker = GrokkingTracker(config['grokking'])
        else:
            self.grokking_tracker = None

        # Emergence tracker
        if self.is_main_process and config.get('emergence', {}).get('enable', False):
            self.emergence_tracker = EmergenceTracker(config['emergence'])
        else:
            self.emergence_tracker = None

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, disable=not self.is_main_process)
        pbar.set_description(f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Mixed precision forward
            with autocast(enabled=(self.mixed_precision is not None)):
                outputs = self.model(**batch)
                loss = outputs['loss']

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': total_loss / num_batches})

            # Periodic evaluation & tracking
            if self.is_main_process and self.global_step % self.config['grokking'].get('log_interval', 100) == 0:
                self._periodic_evaluation()

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, disable=not self.is_main_process, desc="Validating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                outputs = self.model(**batch)

                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()

                # Collect predictions
                all_preds.append(outputs['logits'].cpu())
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())

        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0) if all_labels else None

        metrics = self._compute_metrics(all_preds, all_labels)
        metrics['val_loss'] = total_loss / len(self.val_loader)

        return metrics

    def _compute_metrics(
        self,
        preds: torch.Tensor,
        labels: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}

        if labels is not None:
            # AUC
            from sklearn.metrics import roc_auc_score
            try:
                metrics['auc'] = roc_auc_score(
                    labels.numpy(),
                    torch.sigmoid(preds).numpy()
                )
            except:
                pass

            # Accuracy
            pred_labels = (preds > 0).long()
            metrics['accuracy'] = (pred_labels == labels).float().mean().item()

        return metrics

    def _periodic_evaluation(self):
        """Periodic evaluation during training"""
        val_metrics = self.validate()

        # Log to wandb
        if self.config['logging'].get('use_wandb', False):
            wandb.log({
                'step': self.global_step,
                'epoch': self.current_epoch,
                **val_metrics
            })

        # Update grokking tracker
        if self.grokking_tracker:
            train_loss = val_metrics.get('train_loss', 0.0)
            self.grokking_tracker.update(
                step=self.global_step,
                metrics=val_metrics,
                train_loss=train_loss
            )

            # Check if grokking occurred
            if self.grokking_tracker.has_any_grokked():
                logger.info("🔥 Grokking detected!")
                # Save grokking plot
                self.grokking_tracker.plot_grokking_curves(
                    save_path=self.checkpoint_dir / "grokking_curves.png"
                )

        # Update emergence tracker
        if self.emergence_tracker and self.global_step % 1000 == 0:
            # Run emergence evaluation (expensive, less frequent)
            emergence_metrics = self.emergence_tracker.evaluate_all(
                step=self.global_step,
                model=self.model,
                edge_index=self.train_loader.dataset.edge_index,
                device=self.device
            )

            if self.emergence_tracker.has_any_emerged():
                logger.info("🌟 Emergence detected!")
                self.emergence_tracker.plot_emergence_curves(
                    save_path=self.checkpoint_dir / "emergence_curves.png"
                )

        # Save checkpoint
        if self.global_step % self.save_interval == 0:
            self.save_checkpoint(f"step_{self.global_step}")

    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.max_epochs} epochs")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch}: {train_metrics}")

            # Validate
            if self.is_main_process:
                val_metrics = self.validate()
                logger.info(f"Validation: {val_metrics}")

                # Log to wandb
                if self.config['logging'].get('use_wandb', False):
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics
                    })

                # Save best model
                val_metric = val_metrics.get('auc', 0.0)
                if val_metric > self.best_val_metric:
                    self.best_val_metric = val_metric
                    self.save_checkpoint('best')
                    logger.info(f"New best model! Val metric: {val_metric:.4f}")

        # Final evaluation
        if self.is_main_process:
            logger.info("Training complete! Running final evaluation...")
            test_metrics = self.evaluate_test()
            logger.info(f"Test metrics: {test_metrics}")

            # Generate final reports
            if self.grokking_tracker:
                report = self.grokking_tracker.get_summary_report()
                logger.info(f"\n{report}")

                # Save to file
                with open(self.checkpoint_dir / "grokking_report.txt", 'w') as f:
                    f.write(report)

            if self.emergence_tracker:
                report = self.emergence_tracker.get_summary_report()
                logger.info(f"\n{report}")

                with open(self.checkpoint_dir / "emergence_report.txt", 'w') as f:
                    f.write(report)

    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                outputs = self.model(**batch)
                all_preds.append(outputs['logits'].cpu())

                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0) if all_labels else None

        metrics = self._compute_metrics(all_preds, all_labels)
        return metrics

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
