# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 11:19
    @ Description:
"""

from typing import TYPE_CHECKING
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
from utils.load_split import loader
import json
import os
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import time
from utils.experiment_utils import create_experiment_name

if TYPE_CHECKING:
    from config.Experiment_Config import ExperimentConfig
else:
    from typing import Any as ExperimentConfig  # runtime type alias


def calculate_metrics(y_true, y_pred):
    """Calculate and return all metrics with zero_division handling"""
    # Calculate precision, recall, F1 (macro and micro) with zero_division=0
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='macro',
        zero_division=0  # Handle zero division case
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='micro',
        zero_division=0  # Handle zero division case
    )

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }

    return metrics


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config

        # Load data and set output dimension
        self.train_loader, self.val_loader, self.test_loader, self.label_encoder = loader(
            config)
        self._set_output_dimension()

        # Initialize model
        self.model = self.config.model_selection.get_model(config)
        self.model = self.model.to(config.training_settings.device)

        # Create full experiment name
        self.experiment_name = create_experiment_name(config, is_model=False)

        # Update model's experiment name to match trainer's
        self.model.experiment_name = self.experiment_name

        # Create directories
        self._setup_directories()

        # Initialize tracking variables
        self._initialize_tracking_variables()

        # Initialize training components
        self.criterion = self.config.model_settings.get_loss()

        # Convert model parameters to list before passing to optimizer
        model_params = list(self.model.parameters())
        self.optimizer = self.config.training_settings.get_optimizer(
            model_params)

        self.scheduler = self.config.training_settings.get_scheduler(
            self.optimizer)

        # Initialize best validation loss
        self.best_val_loss = float('inf')

        # Print detailed experiment information
        self._print_experiment_info()


    def _set_output_dimension(self):
        """Automatically set the output dimension based on number of classes"""
        num_classes = len(self.label_encoder.classes_)
        self.config.model_settings.output_dim = num_classes
        # Store class names for confusion matrix
        self.config.data_settings.class_names = self.label_encoder.classes_.tolist()

    def _initialize_tracking_variables(self):
        """Initialize variables for tracking training progress"""
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.continue_training = self.config.training_settings.continue_training

    def _setup_directories(self):
        """Setup all necessary directories for saving results"""
        # Get task type and data type
        task_type = self.config.training_settings.task_type
        data_type = self.config.data_settings.which  # 'question' or 'response'

        # Base directories with task type and data type
        self.model_dir = os.path.join(
            self.config.training_settings.save_model_dir,
            task_type,
            data_type,
            self.experiment_name
        )
        self.results_dir = os.path.join(
            self.config.training_settings.save_results_dir,
            task_type,
            data_type,
            self.experiment_name
        )

        # Setup fine-tuned models directory if needed
        if self.config.model_settings.fine_tune_embedding:
            self._setup_fine_tuned_directories()  # Replace the old code with this call

        # Create subdirectories for models
        for subdir in ['latest', 'best', 'checkpoints']:
            self._ensure_dir_exists(os.path.join(self.model_dir, subdir))

        # Create subdirectories for results
        self.metrics_dir = os.path.join(self.results_dir, 'metrics')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        for dir_path in [self.metrics_dir, self.figures_dir]:
            for subdir in ['latest', 'best', 'checkpoints']:
                self._ensure_dir_exists(os.path.join(dir_path, subdir))

    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint from the specified path or auto-detect latest checkpoint"""
        try:
            task_type = self.config.training_settings.task_type
            data_type = self.config.data_settings.which

            # 1. First determine which checkpoint to load
            if checkpoint_path is None:
                checkpoint_path = self._get_latest_checkpoint_path(
                    task_type, data_type)
                if checkpoint_path is None:
                    return None

            print(f"📂 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=True)

            # 2. Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 3. Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 4. Load scheduler state if it exists and scheduler is configured
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])

            # 5. Load training history directly
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.metrics_history = checkpoint.get('metrics_history', [])
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))

            # 6. Load corresponding embedding state if fine-tuning is enabled
            if self.config.model_settings.fine_tune_embedding:
                embedding_path = self._get_corresponding_embedding_path(
                    checkpoint_path)
                if embedding_path and os.path.exists(embedding_path):
                    embedding_checkpoint = torch.load(
                        embedding_path, weights_only=True)
                    self.model.embedding_model.load_state_dict(
                        embedding_checkpoint['state_dict'])
                    print(
                        f"✅ Successfully loaded corresponding fine-tuned embedding from: {embedding_path}")

            print(
                f"✅ Successfully loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
            return checkpoint

        except Exception as e:
            print(f"❌ Error loading checkpoint: {str(e)}")
            return None

    def _get_latest_checkpoint_path(self, task_type, data_type):
        """Helper to determine the most recent checkpoint path"""
        base_dir = os.path.join(
            self.config.training_settings.save_model_dir,
            task_type,
            data_type,
            self.experiment_name
        )

        # Check paths in order: best -> latest -> most recent checkpoint
        paths = [
            os.path.join(base_dir, 'best', 'model_best.pt'),
            os.path.join(base_dir, 'latest', 'model_latest.pt')
        ]

        for path in paths:
            if os.path.exists(path):
                return path

        return None

    def _get_corresponding_embedding_path(self, checkpoint_path):
        """Get the corresponding embedding path based on checkpoint path"""
        if 'best' in checkpoint_path:
            return os.path.join(self.fine_tuned_dir, 'best', 'embedding_model_best.pt')
        else:
            return os.path.join(self.fine_tuned_dir, 'latest', 'embedding_model_latest.pt')

    def _ensure_dir_exists(self, path):
        """Ensure directory exists, create if it doesn't"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True, mode=0o777)
        elif not os.access(path, os.W_OK):
            # Try to add write permission
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode | 0o777)

    def _create_epoch_directories(self, epoch):
        """Create epoch-specific directories for checkpoints, figures, and metrics"""
        epoch_str = f"epoch_{epoch}"

        # Create main epoch directories
        epoch_model_dir = os.path.join(
            self.model_dir, 'checkpoints', epoch_str)
        epoch_figures_dir = os.path.join(
            self.results_dir, 'figures', epoch_str)
        epoch_metrics_dir = os.path.join(
            self.results_dir, 'metrics', epoch_str)

        # Create fine-tuned model directory if needed
        if self.config.model_settings.fine_tune_embedding:
            epoch_fine_tuned_dir = os.path.join(
                self.fine_tuned_dir, 'checkpoints', epoch_str)
            self._ensure_dir_exists(epoch_fine_tuned_dir)

        for directory in [epoch_model_dir, epoch_figures_dir, epoch_metrics_dir]:
            self._ensure_dir_exists(directory)

        return epoch_model_dir, epoch_figures_dir, epoch_metrics_dir

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint with training history"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }

        # Save based on type (best/latest/checkpoint)
        if is_best:
            save_path = os.path.join(self.model_dir, 'best', 'model_best.pt')
            torch.save(checkpoint, save_path)
            print(f"🏆 Saved best model at epoch {epoch + 1}")
        else:
            save_path = os.path.join(
                self.model_dir, 'latest', 'model_latest.pt')
            torch.save(checkpoint, save_path)
            print(f"📝 Saved latest model at epoch {epoch + 1}")

        # Save periodic checkpoint if needed
        if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                self.model_dir,
                'checkpoints',
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch + 1}")

        # Save fine-tuned embedding if enabled
        if self.config.model_settings.fine_tune_embedding:
            self.model.save_and_reload_latest_model(epoch, is_best)

    def _save_fine_tuned_embedding(self, epoch, is_best=False):
        """Save fine-tuned embedding model with epoch information"""
        embedding_state = {
            'epoch': epoch + 1,
            'state_dict': self.model.embedding_model.state_dict(),
            'config': self.config.model_settings.to_dict(),
            'embedding_type': self.config.model_settings.embedding_type,
            'experiment_name': self.experiment_name
        }

        # Create base directories
        task_type = self.config.training_settings.task_type
        data_type = self.config.data_settings.which
        embedding_type = self.config.model_settings.embedding_type

        base_dir = os.path.join(
            self.config.training_settings.fine_tuned_models_dir,
            task_type,
            data_type,
            embedding_type,
            self.experiment_name
        )

        # Create all necessary directories
        for subdir in ['best', 'latest', 'checkpoints']:
            dir_path = os.path.join(base_dir, subdir)
            self._ensure_dir_exists(dir_path)

        # Save based on type (best/latest/checkpoint)
        if is_best:
            save_path = os.path.join(
                base_dir, 'best', 'embedding_model_best.pt')
            torch.save(embedding_state, save_path)
            print(f"🏆 Saved best fine-tuned embedding at epoch {epoch + 1}")
        else:
            save_path = os.path.join(
                base_dir, 'latest', 'embedding_model_latest.pt')
            torch.save(embedding_state, save_path)
            print(f"💾 Saved latest fine-tuned embedding at epoch {epoch + 1}")

        # Save periodic checkpoint
        if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                base_dir,
                'checkpoints',
                f'embedding_model_epoch_{epoch + 1}.pt'
            )
            torch.save(embedding_state, checkpoint_path)
            print(
                f"📁 Saved fine-tuned embedding checkpoint at epoch {epoch + 1}")

    def _save_figure_safely(self, save_path, fig=None, **save_kwargs):
        """Safely save a matplotlib figure with error handling and cleanup
        Args:
            save_path (str): Path where to save the figure
            fig (matplotlib.figure.Figure, optional): Figure to save. If None, uses current figure
            **save_kwargs: Additional arguments for plt.savefig
        """
        if fig is None:
            fig = plt.gcf()

        try:
            # Ensure parent directory exists
            dir_path = os.path.dirname(save_path)
            self._ensure_dir_exists(dir_path)

            # Close any existing figures with the same name
            plt.close(save_path)

            # If file exists, try to remove it
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except (PermissionError, OSError):
                    # If can't remove, create alternative filename with timestamp
                    base, ext = os.path.splitext(save_path)
                    save_path = f"{base}_{int(time.time())}{ext}"

            # Default save parameters
            save_params = {
                'bbox_inches': 'tight',
                'dpi': 300,
                'facecolor': 'white',
                'pad_inches': 0.1
            }
            save_params.update(save_kwargs)

            # Save the figure
            fig.savefig(save_path, **save_params)
            return True, save_path

        except Exception as e:
            print(f"❌ Error saving figure to {save_path}")
            print(f"Error details: {str(e)}")

            # Try saving to fallback location
            try:
                fallback_dir = os.path.join(os.getcwd(), 'fallback_results')
                os.makedirs(fallback_dir, exist_ok=True)
                fallback_path = os.path.join(
                    fallback_dir, os.path.basename(save_path))
                fig.savefig(fallback_path, **save_params)
                print(f"✓ Saved backup to: {fallback_path}")
                return True, fallback_path
            except Exception as backup_error:
                print(f"❌ Backup save also failed: {str(backup_error)}")
                return False, None
        finally:
            plt.close(fig)

    def plot_confusion_matrix(self, y_true, y_pred, save_dir, prefix=''):
        """Plot confusion matrix"""
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure with white background - increased figure size for better readability
        plt.figure(figsize=(20, 16), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        # Plot with better visibility
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=self.config.data_settings.class_names,
                    yticklabels=self.config.data_settings.class_names,
                    square=True,
                    cbar_kws={
                        'orientation': 'vertical',
                        'pad': 0.02,    # Reduce padding
                        'fraction': 0.046  # Make colorbar thinner
                    },
                    vmin=0,
                    annot_kws={'size': 7},  # Smaller font for numbers in cells
                    )

        # Get the colorbar and adjust its label size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)

        # Adjust label sizes and rotation
        plt.xticks(rotation=45, color='black', fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right')
        plt.yticks(rotation=0, color='black', fontsize=8)

        # Determine the correct title type based on the save directory
        if 'latest' in save_dir:
            title_type = 'Latest'
        elif 'best' in save_dir:
            title_type = 'Best'
        elif 'final' in save_dir:
            title_type = 'Final'
        elif 'checkpoints' in save_dir:
            title_type = 'Checkpoint'
        else:
            title_type = ''

        # Enhanced title with better spacing and formatting
        current_epoch = len(self.metrics_history)
        total_epochs = self.config.training_settings.num_epochs
        title_lines = [
            f'Confusion Matrix ({title_type})',
            f'{self.experiment_name}',
            f'Epoch {current_epoch}/{total_epochs}'
        ]

        plt.title('\n'.join(title_lines),
                  pad=20,
                  fontsize=14,
                  color='black',
                  linespacing=1.3,
                  ha='center',
                  x=0.5)

        # Adjust layout
        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9)

        # Add titles with appropriate sizing
        plt.xlabel('Predicted Label', labelpad=15, color='black', fontsize=12)
        plt.ylabel('True Label', labelpad=15, color='black', fontsize=12)

        # Add more padding to prevent label cutoff
        plt.tight_layout(pad=1.1)

        # Create full save path
        filename = f'{prefix}_conf.png'
        save_path = os.path.join(save_dir, filename)

        success, path = self._save_figure_safely(save_path)
        if not success:
            raise RuntimeError(
                f"Failed to save confusion matrix to {save_path}")

    def plot_class_performance(self, y_true, y_pred, save_dir, prefix=''):
        """Plot class performance with path checking"""
        # Calculate per-class metrics
        class_names = self.config.data_settings.class_names
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )

        # Create DataFrame for metrics
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': support
        })

        # Sort by F1 score
        metrics_df = metrics_df.sort_values('F1', ascending=True)

        # Plot with white background
        plt.figure(figsize=(15, 10), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        x = range(len(class_names))
        width = 0.25

        # Plot bars
        plt.barh([i - width for i in x], metrics_df['Precision'], width,
                 label='Precision', color='skyblue')
        plt.barh([i for i in x], metrics_df['Recall'], width,
                 label='Recall', color='lightgreen')
        plt.barh([i + width for i in x], metrics_df['F1'], width,
                 label='F1', color='salmon')

        # Style improvements with black text
        plt.yticks(x, metrics_df['Class'], color='black')
        plt.xlabel('Score', color='black')

        # Determine the correct title type based on the save directory
        if 'latest' in save_dir:
            title_type = 'Latest'
        elif 'best' in save_dir:
            title_type = 'Best'
        elif 'final' in save_dir:
            title_type = 'Final'
        elif 'checkpoints' in save_dir:
            title_type = 'Checkpoint'
        else:
            title_type = ''

        # Enhanced title with better spacing and formatting
        current_epoch = len(self.metrics_history)
        total_epochs = self.config.training_settings.num_epochs
        title_lines = [
            f'Per-class Performance ({title_type})',
            f'{self.experiment_name}',
            f'Epoch {current_epoch}/{total_epochs}'
        ]

        plt.title('\n'.join(title_lines),
                  pad=20,
                  fontsize=14,
                  color='black',
                  linespacing=1.3,
                  ha='center',      # Ensure horizontal center alignment
                  x=0.5)           # Set x position to center

        # Set legend with black text in upper left
        legend = plt.legend(
            facecolor='white',
            edgecolor='black',
            loc='upper left',         # Ensure legend is at upper left
            fontsize=10,
            bbox_to_anchor=(1.0, 1.0)  # Fine-tune position
        )
        plt.setp(legend.get_texts(), color='black')

        # Add support numbers
        for i, support in enumerate(metrics_df['Support']):
            plt.text(0.02, i, f'n={support}', va='center', color='black')

        # Set grid with light color
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')

        # Ensure all spines are black
        for spine in ax.spines.values():
            spine.set_color('black')

        # Set tick colors to black
        ax.tick_params(colors='black')

        # Adjust layout
        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.15, right=0.95)

        # Create full save path
        filename = f'{prefix}_perf.png'
        save_path = os.path.join(save_dir, filename)

        success, path = self._save_figure_safely(save_path)
        if not success:
            raise RuntimeError(
                f"Failed to save class performance plot to {save_path}")

    def _print_metrics(self, phase, metrics, epoch, batch_idx=None):
        """Print metrics in a clean, organized format with vertical alignment"""
        if batch_idx is not None:
            header = f"📊 {phase.capitalize()} Metrics (Epoch {epoch + 1}, Batch {batch_idx})"
        else:
            header = f"📊 {phase.capitalize()} Metrics (Epoch {epoch + 1})"

        print(f"\n{header}")
        print(f"{'═' * 65}")  # Top border

        # Format each metric with consistent width
        metrics_format = {
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 Macro': f"{metrics['f1_macro']:.4f}",
            'F1 Micro': f"{metrics['f1_micro']:.4f}",
            'Precision Macro': f"{metrics['precision_macro']:.4f}",
            'Precision Micro': f"{metrics['precision_micro']:.4f}",
            'Recall Macro': f"{metrics['recall_macro']:.4f}",
            'Recall Micro': f"{metrics['recall_micro']:.4f}"
        }

        # Print metrics in aligned columns with separators
        print(f"│ {'Accuracy':<13}│ {metrics_format['Accuracy']:<13}│")
        print(f"{'─' * 65}")  # Separator

        print(f"│ {'F1 Macro':<13}│ {metrics_format['F1 Macro']:<13}│ "
              f"{'F1 Micro':<13}│ {metrics_format['F1 Micro']:<13}│")
        print(f"{'─' * 65}")  # Separator

        print(f"│ {'Prec Macro':<13}│ {metrics_format['Precision Macro']:<13}│ "
              f"{'Prec Micro':<13}│ {metrics_format['Precision Micro']:<13}│")
        print(f"{'─' * 65}")  # Separator

        print(f"│ {'Recall Macro':<13}│ {metrics_format['Recall Macro']:<13}│ "
              f"{'Recall Micro':<13}│ {metrics_format['Recall Micro']:<13}│")
        print(f"{'═' * 65}")  # Bottom border

    def train(self):
        """Main training loop"""
        # 1. Handle training continuation
        if self.continue_training:
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                checkpoint = self.load_checkpoint(latest_checkpoint)
                if checkpoint is not None:
                    self.start_epoch = checkpoint['epoch'] + 1
                    # Update model's current epoch
                    self.model.update_epoch(self.start_epoch)
                    print(
                        f"✅ Resuming training from epoch {self.start_epoch + 1}")
                else:
                    print("⚠️ Failed to load checkpoint, starting from scratch")
                    self.start_epoch = 0
            else:
                print("⚠️ No checkpoint found, starting training from scratch")
                self.start_epoch = 0
        else:
            self.start_epoch = 0

        # 2. Main training loop
        for epoch in range(self.start_epoch, self.config.training_settings.num_epochs):
            # Update current epoch in model before training
            self.model.update_epoch(epoch)

            print(f"\n{'─'*50}")
            print(
                f"⏳ Epoch {epoch + 1}/{self.config.training_settings.num_epochs}")
            print(f"{'─'*50}")

            # Train one epoch
            train_metrics, train_preds, train_labels = self.train_epoch()

            # Evaluate on validation set
            val_metrics, val_preds, val_labels = self.evaluate(
                self.val_loader, phase='val')

            # Update metrics history
            self._update_tracking(train_metrics, val_metrics)

            # Save latest figures and metrics
            latest_dir = os.path.join(self.figures_dir, 'latest')
            self._ensure_dir_exists(latest_dir)
            self._save_epoch_figures(
                train_labels, train_preds, val_labels, val_preds, latest_dir)
            self._save_metrics_data(
                epoch, train_metrics, val_metrics, save_type='latest')
            self._save_latest_checkpoint(epoch, val_metrics)

            # Save periodic checkpoints
            if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
                # Save checkpoint figures
                checkpoint_figures_dir = os.path.join(
                    self.figures_dir, 'checkpoints', f'epoch_{epoch + 1}')
                self._ensure_dir_exists(checkpoint_figures_dir)
                self._save_epoch_figures(
                    train_labels, train_preds, val_labels, val_preds, checkpoint_figures_dir)

                # Save checkpoint metrics
                self._save_metrics_data(
                    epoch, train_metrics, val_metrics, save_type='checkpoint')

                # Save model checkpoint
                checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
                self._ensure_dir_exists(checkpoint_dir)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                self.save_checkpoint(epoch, val_metrics['loss'], is_best=False)

                # Save fine-tuned embedding checkpoint if enabled
                if self.config.model_settings.fine_tune_embedding:
                    checkpoint_embedding_dir = os.path.join(
                        self.config.training_settings.fine_tuned_models_dir,
                        self.config.training_settings.task_type.lower(),
                        self.config.data_settings.which,
                        self.config.model_settings.embedding_type,
                        self.experiment_name,
                        'checkpoints'
                    )
                    self._ensure_dir_exists(checkpoint_embedding_dir)
                    checkpoint_embedding_path = os.path.join(
                        checkpoint_embedding_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                    self._save_fine_tuned_embedding(epoch, is_best=False)

            # Check if this is the best model
            current_val_loss = val_metrics['loss']
            is_best = current_val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0

                # Save best figures
                best_dir = os.path.join(self.figures_dir, 'best')
                self._ensure_dir_exists(best_dir)
                self._save_epoch_figures(
                    train_labels, train_preds, val_labels, val_preds, best_dir)

                # Save best metrics
                self._save_metrics_data(
                    epoch, train_metrics, val_metrics, save_type='best')

                # Save best model
                self.save_checkpoint(epoch, val_metrics['loss'], is_best=True)

                # Save best fine-tuned embedding if enabled
                if self.config.model_settings.fine_tune_embedding:
                    self._save_fine_tuned_embedding(epoch, is_best=True)

            # Early stopping check
            if self.patience_counter >= self.config.training_settings.early_stopping_patience:
                print(f"⚠️ Early stopping triggered after {epoch + 1} epochs")
                # Save final state before breaking
                self._save_final_state(
                    train_labels, train_preds, val_labels, val_preds)
                break

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(current_val_loss)

            # Save whole period metrics
            self.save_whole_period_metrics()

        # Save final state at end of training if not stopped early
        if self.patience_counter < self.config.training_settings.early_stopping_patience:
            self._save_final_state(
                train_labels, train_preds, val_labels, val_preds)

        return self.metrics_history

    def _update_tracking(self, train_metrics, val_metrics):
        """Update training history with new metrics"""
        # Add losses to tracking lists
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])

        # Create metrics dictionary with the exact structure needed for plotting
        metrics = {
            'epoch': len(self.metrics_history) + 1,
            'train': {
                'accuracy': train_metrics['accuracy'],
                'precision_macro': train_metrics['precision_macro'],
                'precision_micro': train_metrics['precision_micro'],
                'recall_macro': train_metrics['recall_macro'],
                'recall_micro': train_metrics['recall_micro'],
                'f1_macro': train_metrics['f1_macro'],
                'f1_micro': train_metrics['f1_micro']
            },
            'val': {
                'accuracy': val_metrics['accuracy'],
                'precision_macro': val_metrics['precision_macro'],
                'precision_micro': val_metrics['precision_micro'],
                'recall_macro': val_metrics['recall_macro'],
                'recall_micro': val_metrics['recall_micro'],
                'f1_macro': val_metrics['f1_macro'],
                'f1_micro': val_metrics['f1_micro']
            }
        }

        # Add the metrics to history
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = []
        self.metrics_history.append(metrics)

        # Update early stopping counter
        if val_metrics['loss'] >= self.best_val_loss:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        train_labels, train_preds = [], []
        total_loss = 0
        num_samples = 0
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            self.optimizer.zero_grad()

            # Move batch to device
            inputs = batch[0].to(self.config.training_settings.device)
            labels = batch[1].to(self.config.training_settings.device)

            outputs, activations = self.model(inputs)

            if self.config.model_settings.final_activation == 'softmax':
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
            elif self.config.model_settings.final_activation == 'sigmoid':
                loss = self.criterion(outputs, labels.float())
                preds = (outputs > 0.5).long()
            else:  # 'linear' or no activation
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                preds = torch.argmax(outputs, dim=1)

            loss.backward()

            # Add monitoring call here
            self._monitor_training_dynamics(
                len(self.metrics_history), batch_idx, loss.item(), outputs, activations)

            # Gradient clipping if configured
            if self.config.training_settings.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training_settings.gradient_clip
                )

            self.optimizer.step()

            # Update metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            # Store predictions and labels for metric calculation
            labels_np = labels.detach().cpu().numpy().astype(np.int64)
            preds_np = preds.detach().cpu().numpy().astype(np.int64)
            train_labels.extend(labels_np)
            train_preds.extend(preds_np)

        # Calculate average loss and metrics
        avg_loss = total_loss / num_samples

        metrics = calculate_metrics(train_labels, train_preds)
        metrics['loss'] = avg_loss

        # Calculate and print metrics only at epoch end
        self._print_metrics('train', metrics, len(self.metrics_history))

        # Save latest state after each epoch
        latest_figures_dir = os.path.join(self.figures_dir, 'latest')
        self._ensure_dir_exists(latest_figures_dir)
        self._save_epoch_figures(
            train_labels, train_preds, None, None, latest_figures_dir)

        # Save checkpoint state if at checkpoint frequency
        if (len(self.metrics_history) + 1) % self.config.training_settings.checkpoint_freq == 0:
            checkpoint_figures_dir = os.path.join(
                self.figures_dir, 'checkpoints', f'epoch_{len(self.metrics_history) + 1}')
            self._ensure_dir_exists(checkpoint_figures_dir)
            self._save_epoch_figures(
                train_labels, train_preds, None, None, checkpoint_figures_dir)

        if self.model.config.model_settings.fine_tune_embedding:
            self.model.save_and_reload_latest_model(
                len(self.metrics_history), is_best=False)

        return metrics, train_preds, train_labels

    def evaluate(self, loader, phase='val'):
        """Validate or test the model"""
        self.model.eval()

        # Handle fine-tuned model loading based on phase
        if self.config.model_settings.fine_tune_embedding:
            if phase == 'val':
                # For validation, use latest fine-tuned model
                embedding_path = os.path.join(
                    self.config.training_settings.fine_tuned_models_dir,
                    self.config.training_settings.task_type,
                    self.config.data_settings.which,
                    self.config.model_settings.embedding_type,
                    self.experiment_name,
                    'latest',
                    'embedding_model_latest.pt'
                )
                model_version = 'latest'
            else:  # phase == 'test'
                # For testing, use best fine-tuned model
                embedding_path = os.path.join(
                    self.config.training_settings.fine_tuned_models_dir,
                    self.config.training_settings.task_type,
                    self.config.data_settings.which,
                    self.config.model_settings.embedding_type,
                    self.experiment_name,
                    'best',
                    'embedding_model_best.pt'
                )
                model_version = 'best'

            if os.path.exists(embedding_path):
                self._load_fine_tuned_embedding(embedding_path)
                print(
                    f"✅ Loaded {model_version} fine-tuned embedding for {phase} evaluation")
            else:
                print(
                    f"⚠️ No {model_version} fine-tuned model found for {phase} evaluation")

        total_loss = 0
        num_samples = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Evaluating ({phase})'):
                # Skip empty batches
                if batch[0].size(0) == 0:
                    continue

                # Use consistent unpacking
                inputs = batch[0].to(self.config.training_settings.device)
                labels = batch[1].to(self.config.training_settings.device)

                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / num_samples

        # Calculate and print metrics
        self._print_metrics(phase, metrics, len(self.metrics_history))

        return metrics, all_labels, all_preds

    def _setup_fine_tuned_directories(self):
        """Set up directories for fine-tuned embedding models"""
        if not self.config.model_settings.fine_tune_embedding:
            return

        # Match BaseLSTM's directory structure
        task_type = self.config.training_settings.task_type
        data_type = self.config.data_settings.which

        self.fine_tuned_dir = os.path.join(
            self.config.training_settings.fine_tuned_models_dir,
            task_type,
            data_type,
            self.config.model_settings.embedding_type,
            self.experiment_name
        )

        # Create subdirectories
        for subdir in ['latest', 'best', 'checkpoints']:
            dir_path = os.path.join(self.fine_tuned_dir, subdir)
            self._ensure_dir_exists(dir_path)

    def _print_experiment_info(self):
        """Print detailed experiment configuration and setup"""
        print("\n" + "="*50)
        print("EXPERIMENT CONFIGURATION")
        print("="*50)

        # Basic experiment info
        print(f"\n📊 Experiment Name: {self.experiment_name}")
        print(f"📁 Model Directory: {self.model_dir}")
        print(f"📈 Results Directory: {self.results_dir}")

        # Model configuration
        print("\n🔧 Model Configuration:")
        print(f"- Model Type: {self.config.model_selection.model_type}")
        print(f"- Embedding Type: {self.config.model_settings.embedding_type}")

        # Print hidden dimensions based on configuration
        if self.config.model_settings.custom_hidden_dims is not None:
            print(
                f"- Hidden Dimensions: {self.config.model_settings.custom_hidden_dims}")
        else:
            print(
                f"- Initial Hidden Dim: {self.config.model_settings.init_hidden_dim}")

        print(f"- Bidirectional: {self.config.model_settings.bidirectional}")
        print(f"- Dropout Rate: {self.config.model_settings.dropout_rate}")
        print(f"- Attention: {self.config.model_settings.use_attention}")
        if self.config.model_settings.use_attention:
            print(f"  • Type: {self.config.model_settings.attention_type}")
            print(
                f"  • Heads: {self.config.model_settings.num_attention_heads}")
            print(
                f"  • Dimension: {self.config.model_settings.attention_dim}")
            print(
                f"  • Dropout: {self.config.model_settings.attention_dropout}")
        print(
            f"- Fine-tune Embedding: {self.config.model_settings.fine_tune_embedding}")
        print(f"- Weight Init: {self.config.model_settings.weight_init}")
        print(f"- Activation: {self.config.model_settings.activation}")

        # Training configuration
        print("\n⚙️ Training Configuration:")
        print(f"- Batch Size: {self.config.training_settings.batch_size}")
        print(f"- Max Length: {self.config.tokenizer_settings.max_length}")
        print(
            f"- Learning Rate: {self.config.training_settings.learning_rate}")
        print(f"- Weight Decay: {self.config.training_settings.weight_decay}")
        print(f"- Num Epochs: {self.config.training_settings.num_epochs}")
        print(f"- Loss Function: {self.config.model_settings.loss}")
        print(f"- Optimizer: {self.config.training_settings.optimizer_type}")
        if self.config.training_settings.gradient_clip:
            print(
                f"- Gradient Clipping: {self.config.training_settings.gradient_clip}")
        print(
            f"- Early Stopping Patience: {self.config.training_settings.early_stopping_patience}")
        print(
            f"- Scheduler Patience: {self.config.training_settings.scheduler_patience}")
        print(
            f"- Scheduler Factor: {self.config.training_settings.scheduler_factor}")
        print(
            f"- Min Learning Rate: {self.config.training_settings.min_learning_rate}")
        print(
            f"- Checkpoint Frequency: {self.config.training_settings.checkpoint_freq}")

        # Data configuration
        print("\n📊 Data Configuration:")
        print(
            f"- Imbalanced Strategy: {self.config.data_settings.imbalanced_strategy}")
        if self.config.data_settings.imbalanced_strategy == 'weighted_sampler':
            print(
                f"  • Alpha: {self.config.data_settings.weighted_sampler_alpha}")
        print(
            f"- Number of Classes: {len(self.config.data_settings.class_names)}")

        # Directory configuration
        print("\n📂 Directory Configuration:")
        print(
            f"- Model Save Dir: {self.config.training_settings.save_model_dir}")
        print(
            f"- Results Save Dir: {self.config.training_settings.save_results_dir}")
        if self.config.model_settings.fine_tune_embedding:
            print(
                f"- Fine-tuned Models Dir: {self.config.training_settings.fine_tuned_models_dir}")

        # Hardware configuration
        print("\n💻 Hardware Configuration:")
        print(f"- Device: {self.config.training_settings.device}")
        print(f"- MPS available: {torch.backends.mps.is_available()}")
        print(f"- CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"- CUDA devices: {torch.cuda.device_count()}")

        # Dataset information
        print("\n Dataset Information:")
        print(f"- Training samples: {len(self.train_loader.dataset)}")
        print(f"- Validation samples: {len(self.val_loader.dataset)}")
        print(f"- Test samples: {len(self.test_loader.dataset)}")
        print(f"- Number of batches (train): {len(self.train_loader)}")

        # Add fine-tuning information
        if self.config.model_settings.fine_tune_embedding:
            print("\n🔄 Fine-tuning Configuration:")
            print(f"- Mode: {self.config.model_settings.fine_tune_mode}")
            print(f"- Fine-tuned Models Dir: {self.fine_tuned_dir}")
            if self.config.model_settings.use_discriminative_lr:
                print(
                    f"- Discriminative LR Decay: {self.config.model_settings.lr_decay_factor}")
            if hasattr(self.config.model_settings, 'fine_tune_lr'):
                print(
                    f"- Fine-tuning Learning Rate: {self.config.model_settings.fine_tune_lr}")

        print("\n" + "="*50 + "\n")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint by comparing both checkpoints directory and latest directory"""
        latest_epoch = -1
        latest_checkpoint_path = None

        # Check if directories exist
        latest_dir = os.path.join(self.model_dir, 'latest')
        checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')

        latest_exists = os.path.exists(latest_dir)
        checkpoints_exists = os.path.exists(checkpoint_dir)

        if not latest_exists and not checkpoints_exists:
            print(
                "⚠️ Neither latest nor checkpoints directory exists, starting fresh training")
            return None

        # Try to get epoch from latest model
        latest_model_path = os.path.join(latest_dir, 'model_latest.pt')
        latest_epoch = -1

        if latest_exists and os.path.exists(latest_model_path):
            try:
                checkpoint = torch.load(latest_model_path, weights_only=True)
                latest_epoch = checkpoint.get('epoch', -1)
                latest_checkpoint_path = latest_model_path
                print(
                    f"📝 Found model in latest folder from epoch {latest_epoch + 1}")
            except Exception as e:
                print(f"⚠️ Error reading latest model: {str(e)}")
                latest_epoch = -1

        # Try to get max epoch from checkpoints
        max_checkpoint_epoch = -1
        max_checkpoint_path = None

        if checkpoints_exists:
            checkpoints = [f for f in os.listdir(checkpoint_dir)
                           if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(
                    re.search(r'epoch_(\d+)', x).group(1)))
                max_checkpoint_epoch = int(
                    re.search(r'epoch_(\d+)', latest_checkpoint).group(1)) - 1
                max_checkpoint_path = os.path.join(
                    checkpoint_dir, latest_checkpoint)
                print(
                    f"📁 Found checkpoint from epoch {max_checkpoint_epoch + 1}")

        # Compare epochs and return the most recent checkpoint
        if latest_epoch >= max_checkpoint_epoch and latest_epoch >= 0:
            print(f"✅ Using latest model from epoch {latest_epoch + 1}")
            return latest_checkpoint_path
        elif max_checkpoint_epoch >= 0:
            print(f"✅ Using checkpoint from epoch {max_checkpoint_epoch + 1}")
            return max_checkpoint_path

        print("⚠️ No valid checkpoints found, starting fresh training")
        return None

    def _save_epoch_figures(self, train_labels, train_preds, val_labels, val_preds, save_dir):
        """Save figures for current epoch in the appropriate directory"""
        current_epoch = len(
            self.metrics_history)  # This will match the current epoch number

        # 1. Save training phase figures if available
        if train_labels is not None and train_preds is not None:
            # Save confusion matrix for training
            self.plot_confusion_matrix(
                train_labels, train_preds,
                save_dir=save_dir,
                prefix='train'
            )

            # Save per-class performance for training
            self.plot_class_performance(
                train_labels, train_preds,
                save_dir=save_dir,
                prefix='train'
            )
            print(f"📊 Saved training plots for epoch {self.start_epoch + 1}")

        # 2. Save validation phase figures if available
        if val_labels is not None and val_preds is not None:
            # Save confusion matrix for validation
            self.plot_confusion_matrix(
                val_labels, val_preds,
                save_dir=save_dir,
                prefix='val'
            )

            # Save per-class performance for validation
            self.plot_class_performance(
                val_labels, val_preds,
                save_dir=save_dir,
                prefix='val'
            )
            print(f"📊 Saved validation plots for epoch {self.start_epoch + 1}")

    def save_whole_period_metrics(self):
        """Create and save a metrics plot for the entire training period so far"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 12
        })

        # Create figure
        fig = plt.figure(figsize=(16, 12), dpi=300, facecolor='white')

        # Title formatting
        current_epoch = len(self.metrics_history)
        total_epochs = self.config.training_settings.num_epochs
        title_lines = [
            'Training Metrics (Whole Period)',
            f'{self.experiment_name}',
            f'Epoch {current_epoch}/{total_epochs}'
        ]

        # Main title
        fig.suptitle('\n'.join(title_lines),
                     y=0.95,
                     fontsize=14,
                     fontweight='bold',
                     color='black',
                     linespacing=1.5)

        # Create subplots for different metrics
        metric_pairs = [
            ('accuracy', 'Accuracy'),
            (('precision_macro', 'precision_micro'), 'Precision (Macro vs Micro)'),
            (('recall_macro', 'recall_micro'), 'Recall (Macro vs Micro)'),
            (('f1_macro', 'f1_micro'), 'F1 (Macro vs Micro)')
        ]
        if not hasattr(self, 'metrics_history') or not self.metrics_history:
            print("⚠️ No metrics history available to plot")
            return

        for i, (metric, title) in enumerate(metric_pairs, 1):
            ax = fig.add_subplot(2, 2, i)
            ax.set_facecolor('white')

            # Get epochs for x-axis
            epochs = list(range(1, len(self.metrics_history) + 1))

            if isinstance(metric, tuple):
                # Plot macro metrics
                train_macro = [m.get('train', {}).get(metric[0], 0)
                               for m in self.metrics_history]
                val_macro = [m.get('val', {}).get(metric[0], 0)
                             for m in self.metrics_history]
                ax.plot(epochs, train_macro, '-', color='#1f77b4',
                        label='Train (Macro)', marker='o', markersize=2)
                ax.plot(epochs, val_macro, '-', color='#7cc7ff',
                        label='Val (Macro)', marker='o', markersize=2)

                # Plot micro metrics
                train_micro = [m.get('train', {}).get(metric[1], 0)
                               for m in self.metrics_history]
                val_micro = [m.get('val', {}).get(metric[1], 0)
                             for m in self.metrics_history]
                ax.plot(epochs, train_micro, '-', color='#d62728',
                        label='Train (Micro)', marker='s', markersize=2)
                ax.plot(epochs, val_micro, '-', color='#ff9999',
                        label='Val (Micro)', marker='s', markersize=2)
            else:
                train_metric = [m.get('train', {}).get(metric, 0)
                                for m in self.metrics_history]
                val_metric = [m.get('val', {}).get(metric, 0)
                              for m in self.metrics_history]
                ax.plot(epochs, train_metric, '-', color='#1f77b4',
                        label='Train', marker='o', markersize=2)
                ax.plot(epochs, val_metric, '-', color='#7cc7ff',
                        label='Val', marker='o', markersize=2)

            # Set integer ticks on x-axis
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Customize axis and labels
            ax.set_xlabel('Epoch', fontsize=11, color='black', labelpad=8)
            ax.set_ylabel('Score', fontsize=11, color='black', labelpad=8)
            ax.set_title(title, pad=10, fontsize=12,
                         fontweight='bold', color='black')
            ax.set_ylim(0.0, 1.0)

            # Adjust legend position and parameters
            legend = ax.legend(
                facecolor='white',
                edgecolor='none',
                loc='upper left',
                bbox_to_anchor=(0.02, 0.98),
                framealpha=0.9,
                ncol=1
            )
            plt.setp(legend.get_texts(), color='black')

        # Adjust subplot spacing to ensure legends are visible
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.85,      # Keep space for main title
            bottom=0.08,   # Space at bottom
            left=0.08,     # Space at left
            right=0.98,    # Space at right
            hspace=0.25,   # Horizontal space between subplots
            wspace=0.20    # Vertical space between subplots
        )

        # Create shorter filename
        filename = 'whole_period_metrics.png'
        save_path = os.path.join(self.figures_dir, filename)

        success, path = self._save_figure_safely(save_path)
        if not success:
            raise RuntimeError(
                f"Failed to save whole period metrics to {save_path}")

    def _check_early_stopping(self, val_loss):
        """Check if training should be stopped based on validation loss"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.training_settings.early_stopping_patience:
                return True
            return False

    def _load_fine_tuned_embedding(self, path: str):
        """Helper method to load fine-tuned embedding model"""
        try:
            checkpoint = torch.load(path)
            self.model.embedding_model.load_state_dict(
                checkpoint['state_dict'])
            print(f"✅ Successfully loaded fine-tuned embedding from: {path}")
        except Exception as e:
            print(f"⚠️ Error loading fine-tuned embedding: {str(e)}")

    def _save_fine_tuned_embedding(self, epoch: int, is_best: bool = False):
        """Helper method to save fine-tuned embedding model"""
        if not self.config.model_settings.fine_tune_embedding:
            return

        save_dir = os.path.join(
            self.config.training_settings.fine_tuned_models_dir,
            self.config.training_settings.task_type,
            self.config.data_settings.which,
            self.config.model_settings.embedding_type,
            self.experiment_name,
            'best' if is_best else 'latest'
        )

        os.makedirs(save_dir, exist_ok=True)
        file_name = 'embedding_model_best.pt' if is_best else 'embedding_model_latest.pt'
        save_path = os.path.join(save_dir, file_name)

        state_dict = {
            'epoch': epoch,
            'state_dict': self.model.embedding_model.state_dict(),
            'fine_tune_config': {
                'mode': self.config.model_settings.fine_tune_mode,
                'unfrozen_layers': [name for name, param in self.model.embedding_model.named_parameters()
                                    if param.requires_grad],
                'lr_scales': {name: getattr(param, 'lr_scale', 1.0)
                              for name, param in self.model.embedding_model.named_parameters()}
            }
        }

        torch.save(state_dict, save_path)
        print(
            f"💾 Saved {'best' if is_best else 'latest'} fine-tuned embedding at epoch {self.start_epoch + 1}")

    def _monitor_training_dynamics(self, epoch: int, batch_idx: int, loss: float, outputs: torch.Tensor, activations: dict):
        """Monitor training dynamics including gradients, activations, and loss patterns"""
        # Only monitor every n batches to reduce overhead
        # Adjust as needed
        monitor_freq = len(
            self.train_loader)
        if batch_idx % monitor_freq != 0:
            return

        print(
            f"\n📊 Training Dynamics Monitor (Epoch {epoch + 1}):")

        # 1. Monitor Loss
        if loss > 1e3:  # Adjust threshold as needed
            print(f"⚠️ High loss detected: {loss:.4f}")

        # 2. Monitor Gradients
        grad_stats = {name: [] for name in ['lstm', 'fc', 'attention']}
        grad_norms = {name: [] for name in ['lstm', 'fc', 'attention']}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                # Categorize parameter
                if 'lstm' in name:
                    grad_stats['lstm'].append(param.grad.abs().mean().item())
                    grad_norms['lstm'].append(grad_norm)
                elif 'fc' in name:
                    grad_stats['fc'].append(param.grad.abs().mean().item())
                    grad_norms['fc'].append(grad_norm)
                elif 'attention' in name:
                    grad_stats['attention'].append(
                        param.grad.abs().mean().item())
                    grad_norms['attention'].append(grad_norm)

                # Check for gradient explosion
                if grad_norm > 10:  # Adjust threshold as needed
                    print(f"⚠️ Large gradient in {name}: {grad_norm:.4f}")
                # Check for gradient vanishing
                elif grad_norm < 1e-7:  # Adjust threshold as needed
                    print(f"⚠️ Vanishing gradient in {name}: {grad_norm:.4f}")

        # Print gradient statistics
        for layer_type in grad_stats:
            if grad_stats[layer_type]:
                mean_grad = sum(grad_stats[layer_type]) / \
                    len(grad_stats[layer_type])
                max_norm = max(grad_norms[layer_type]
                               ) if grad_norms[layer_type] else 0
                print(
                    f"{layer_type.upper()} - Mean Grad: {mean_grad:.2e}, Max Norm: {max_norm:.2e}")

        # 3. Monitor Activations
        if activations:
            for layer_name, activation in activations.items():
                if activation is not None:
                    act_mean = activation.abs().mean().item()
                    act_std = activation.std().item()
                    print(f"Activations {layer_name}:")
                    print(f"  Mean: {act_mean:.4f}, Std: {act_std:.4f}")

                    # Check for activation saturation
                    if act_mean > 0.9:  # Adjust threshold as needed
                        print(
                            f"⚠️ Possible activation saturation in {layer_name}")
                    elif act_mean < 0.1:  # Adjust threshold as needed
                        print(f"⚠️ Low activation values in {layer_name}")

        # 4. Monitor Output Distribution
        if outputs is not None:
            output_mean = outputs.abs().mean().item()
            output_std = outputs.std().item()
            print(f"Outputs - Mean: {output_mean:.4f}, Std: {output_std:.4f}")

            # Check for potential issues in output distribution
            if output_std < 1e-4:
                print("⚠️ Very low output variance detected")
            if torch.isnan(outputs).any():
                print("⚠️ NaN values detected in outputs")

        print("=" * 50)

    def plot_activation_stats(self, save_dir):
        """Plot activation statistics over time"""
        plt.figure(figsize=(10, 12), facecolor='white')

        # Plot means
        plt.subplot(2, 1, 1)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.gca().set_facecolor('white')

        for layer in ['first_layer', 'middle_layer', 'last_layer']:
            means = [stats[layer]['mean']
                     for stats in self.activation_stats if stats[layer]]
            if means:  # Only plot if we have data
                plt.plot(means, label=f'{layer} mean',
                         marker='o', markersize=2)

        plt.title('Activation Means During Training', pad=10,
                  fontsize=12, fontweight='bold', color='black')
        plt.xlabel('Steps (x10)', fontsize=11, color='black')
        plt.ylabel('Mean Activation', fontsize=11, color='black')
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            facecolor='white',
            edgecolor='none',
            framealpha=0.9
        )

        # Plot standard deviations
        plt.subplot(2, 1, 2)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.gca().set_facecolor('white')

        for layer in ['first_layer', 'middle_layer', 'last_layer']:
            stds = [stats[layer]['std']
                    for stats in self.activation_stats if stats[layer]]
            if stds:  # Only plot if we have data
                plt.plot(stds, label=f'{layer} std', marker='o', markersize=2)

        plt.title('Activation Standard Deviations During Training',
                  pad=10, fontsize=12, fontweight='bold', color='black')
        plt.xlabel('Steps (x10)', fontsize=11, color='black')
        plt.ylabel('Activation std', fontsize=11, color='black')
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            facecolor='white',
            edgecolor='none',
            framealpha=0.9
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, 'activation_stats.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1
        )
        plt.close()

    def plot_gradient_norms(self, save_dir):
        """Plot gradient norms over time"""
        plt.figure(figsize=(10, 6), facecolor='white')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.gca().set_facecolor('white')

        for layer_type in ['lstm', 'fc', 'attention']:
            norms = [stats[layer_type]
                     for stats in self.gradient_norms if stats[layer_type]]
            if norms:  # Only plot if we have data
                plt.plot(
                    norms, label=f'{layer_type.upper()} layers', marker='o', markersize=2)

        plt.title('Gradient Norms During Training', pad=10,
                  fontsize=12, fontweight='bold', color='black')
        plt.xlabel('Steps (x10)', fontsize=11, color='black')
        plt.ylabel('Gradient Norm', fontsize=11, color='black')
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            facecolor='white',
            edgecolor='none',
            framealpha=0.9
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, 'gradient_norms.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1
        )
        plt.close()

    def _save_metrics_data(self, epoch, train_metrics, val_metrics, save_type='latest'):
        """
        Save essential metrics data in appropriate directory
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary containing training metrics
            val_metrics: Dictionary containing validation metrics
            save_type: 'latest', 'best', 'checkpoint', or 'final'
        """
        # Prepare simplified metrics data
        metrics_data = {
            'epoch': epoch + 1,
            'train': {
                'loss': self.train_losses[-1],
                'accuracy': train_metrics['accuracy'],
                'precision': {
                    'macro': train_metrics['precision_macro'],
                    'micro': train_metrics['precision_micro']
                },
                'recall': {
                    'macro': train_metrics['recall_macro'],
                    'micro': train_metrics['recall_micro']
                },
                'f1': {
                    'macro': train_metrics['f1_macro'],
                    'micro': train_metrics['f1_micro']
                }
            },
            'val': {
                'loss': self.val_losses[-1],
                'accuracy': val_metrics['accuracy'],
                'precision': {
                    'macro': val_metrics['precision_macro'],
                    'micro': val_metrics['precision_micro']
                },
                'recall': {
                    'macro': val_metrics['recall_macro'],
                    'micro': val_metrics['recall_micro']
                },
                'f1': {
                    'macro': val_metrics['f1_macro'],
                    'micro': val_metrics['f1_micro']
                }
            }
        }

        # Determine save directory based on save_type
        if save_type == 'latest':
            metrics_dir = os.path.join(self.metrics_dir, 'latest')
        elif save_type == 'best':
            metrics_dir = os.path.join(self.metrics_dir, 'best')
        elif save_type == 'checkpoint':
            metrics_dir = os.path.join(
                self.metrics_dir, 'checkpoints', f'epoch_{epoch + 1}')
        elif save_type == 'final':
            metrics_dir = os.path.join(self.metrics_dir, 'final')

        # Ensure directory exists using class method
        self._ensure_dir_exists(metrics_dir)

        # Create full path for metrics file
        metrics_path = os.path.join(metrics_dir, 'metrics.json')

        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)

        print(f"💾 Saved {save_type} metrics at epoch {epoch + 1}")

    def _save_latest_checkpoint(self, epoch, val_metrics):
        """Save the latest model checkpoint"""
        # Ensure latest directory exists
        latest_dir = os.path.join(self.model_dir, 'latest')
        self._ensure_dir_exists(latest_dir)

        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_metrics['loss'],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }

        latest_model_path = os.path.join(latest_dir, 'model_latest.pt')
        torch.save(checkpoint, latest_model_path)

        # Save fine-tuned embedding if enabled
        if self.config.model_settings.fine_tune_embedding:
            latest_embedding_dir = os.path.join(
                self.config.training_settings.fine_tuned_models_dir,
                self.config.training_settings.task_type.lower(),
                self.config.data_settings.which,
                self.config.model_settings.embedding_type,
                self.experiment_name,
                'latest'
            )
            self._ensure_dir_exists(latest_embedding_dir)

            embedding_state = {
                'epoch': epoch + 1,
                'state_dict': self.model.embedding_model.state_dict(),
                'config': self.config.model_settings.to_dict()
            }

            latest_embedding_path = os.path.join(
                latest_embedding_dir, 'embedding_model_latest.pt')
            torch.save(embedding_state, latest_embedding_path)
