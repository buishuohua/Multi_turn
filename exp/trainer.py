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
from datetime import timedelta

if TYPE_CHECKING:
    from config.Experiment_Config import ExperimentConfig
else:
    from typing import Any as ExperimentConfig  # runtime type alias


def calculate_metrics(y_true, y_pred):
    """Calculate and return all metrics"""
    # Calculate precision, recall, F1 (macro and micro)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
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

        # Load data automatically
        self.train_loader, self.val_loader, self.test_loader, self.label_encoder = loader(
            config)

        # Automatically set output dimension based on number of classes
        self._set_output_dimension()

        # Initialize model with correct output dimension
        self.model = self.config.model_selection.get_model(config)
        self.model = self.model.to(self.config.training_settings.device)

        # Create experiment name with timestamp and key parameters
        self.experiment_name = self._create_experiment_name()

        # Create directories
        self._setup_directories()

        # Initialize tracking variables
        self._initialize_tracking_variables()

        # Create fine-tuned models directory if needed
        if config.model_settings.fine_tune_embedding:
            self.fine_tuned_dir = os.path.join(
                config.training_settings.fine_tuned_models_dir,
                config.model_settings.embedding_type
            )
            self._setup_fine_tuned_directories()

        # Initialize training components
        self.criterion = self.config.model_settings.get_loss()
        self.optimizer = self.config.training_settings.get_optimizer(
            self.model.parameters())
        self.scheduler = self.config.training_settings.get_scheduler(
            self.optimizer)

        # Print detailed experiment information
        self._print_experiment_info()

        self.epoch_times = []  # Track time for each epoch

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
        """Set up necessary directories for saving results"""
        def normalize_path(*paths):
            """Normalize path for cross-platform compatibility"""
            return os.path.normpath(os.path.join(*paths))

        # Store normalize function for use in other methods
        self._normalize_path = normalize_path

        # Set up main directories
        self.model_dir = normalize_path(
            self.config.training_settings.save_model_dir,
            self.experiment_name
        )
        self.results_dir = normalize_path(
            self.config.training_settings.save_results_dir,
            self.experiment_name
        )

        # Define all required directories
        self.figures_dir = normalize_path(self.results_dir, 'figures')
        self.metrics_dir = normalize_path(self.results_dir, 'metrics')

        # Create base directories
        for directory in [self.model_dir, self.results_dir, self.figures_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)

        # Create subdirectories
        special_dirs = ['latest', 'best', 'final', 'checkpoints']
        for special_dir in special_dirs:
            os.makedirs(normalize_path(
                self.model_dir, special_dir), exist_ok=True)
            os.makedirs(normalize_path(
                self.figures_dir, special_dir), exist_ok=True)
            os.makedirs(normalize_path(
                self.metrics_dir, special_dir), exist_ok=True)

        print(f"✅ Created all necessary directories")

    def _create_experiment_name(self):
        """Create concise but informative experiment name"""
        components = []

        # 1. Model Architecture (including bidirectional)
        model_prefix = 'Bi' if self.config.model_settings.bidirectional else ''
        components.append(
            f"{model_prefix}{self.config.model_selection.model_type}")

        # 2. Embedding Info
        emb_name = self.config.model_settings.embedding_type.split('_')[
            0]  # BERT/RoBERTa/glove
        components.append(emb_name)

        # 3. Core Model Parameters
        # Number of layers
        components.append(f"L{self.config.model_settings.num_layers}")
        # Hidden dimension
        components.append(f"H{self.config.model_settings.init_hidden_dim}")
        components.append(
            f"M{self.config.tokenizer_settings.max_length}")  # Max length

        # 4. Training Strategy
        components.append(self.config.model_settings.loss.replace(
            '_', ''))  # Loss function

        # 5. Initialization Function
        init_map = {
            'xavier_uniform': 'xavier',
            'xavier_normal': 'xavier',
            'kaiming_uniform': 'kaiming',
            'kaiming_normal': 'kaiming',
            'orthogonal': 'ortho',
            'zeros': 'zero',
            'ones': 'one'
        }
        init_name = init_map.get(self.config.model_settings.weight_init,
                                 self.config.model_settings.weight_init[:3])
        components.append(init_name)

        # 6. Data Strategy (if any)
        if self.config.data_settings.imbalanced_strategy != 'none':
            imbal_map = {
                'weighted_sampler': 'ws',
                'class_weights': 'cw',
                'random_oversample': 'ros',
                'random_undersample': 'rus',
                'smote': 'sm',
                'adasyn': 'ada',
                'tomek': 'tmk'
            }
            imbal_code = imbal_map.get(self.config.data_settings.imbalanced_strategy,
                                       self.config.data_settings.imbalanced_strategy[:3])
            components.append(imbal_code)

        # 7. Special Features (as flags)
        flags = []
        if self.config.model_settings.use_attention:
            flags.append('A')  # Attention
        if self.config.model_settings.fine_tune_embedding:
            flags.append('FT')  # Fine-tuning
        if flags:
            components.append(''.join(flags))

        return '_'.join(components)

    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint with enhanced fallback strategy"""
        latest_path = os.path.join(
            self.model_dir, 'latest', f'{self.experiment_name}_latest.pt')

        if not os.path.exists(latest_path):
            print("No checkpoint found. Starting fresh training...")
            return False

        try:
            checkpoint = torch.load(latest_path)

            # Load model and training state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['val_loss']

            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.val_losses = checkpoint['val_losses']
                self.metrics_history = checkpoint['metrics_history']

            return True

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Starting fresh training...")
            return False

    def _ensure_dir_exists(self, path):
        """Ensure directory exists, create if it doesn't"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

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
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }

        # Save latest checkpoint
        latest_dir = os.path.join(self.model_dir, 'latest')
        self._ensure_dir_exists(latest_dir)
        latest_path = os.path.join(
            latest_dir, f'{self.experiment_name}_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint in epoch-specific directory
        if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
            epoch_model_dir, _, _ = self._create_epoch_directories(epoch + 1)
            checkpoint_path = os.path.join(
                epoch_model_dir,
                f'{self.experiment_name}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")

        # Save best model if needed
        if is_best:
            best_dir = os.path.join(self.model_dir, 'best')
            self._ensure_dir_exists(best_dir)
            best_path = os.path.join(
                best_dir, f'{self.experiment_name}_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch + 1}")

    def _save_fine_tuned_embedding(self, epoch):
        """Save fine-tuned embedding model with epoch information"""
        embedding_state = {
            'epoch': epoch + 1,
            'state_dict': self.model.embedding_model.state_dict()
        }

        # Save latest version
        latest_dir = os.path.join(self.fine_tuned_dir, 'latest')
        self._ensure_dir_exists(latest_dir)
        latest_path = os.path.join(latest_dir, 'embedding_model_latest.pt')
        torch.save(embedding_state, latest_path)

        # Save periodic checkpoint in epoch-specific directory
        if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
            _, _, _ = self._create_epoch_directories(epoch + 1)
            checkpoint_path = os.path.join(
                self.fine_tuned_dir,
                'checkpoints',
                f'epoch_{epoch + 1}',
                'embedding_model.pt'
            )
            torch.save(embedding_state, checkpoint_path)
            print(f"Saved fine-tuned embedding model at epoch {epoch + 1}")

    def plot_confusion_matrix(self, y_true, y_pred, epoch, prefix='val'):
        """Plot and save confusion matrix"""
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
                        'label': 'Count',
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
        cbar.set_label('Count', size=10)

        # Adjust label sizes and rotation
        plt.xticks(rotation=45, color='black', fontsize=8)
        # Set horizontal alignment separately
        ax.set_xticklabels(ax.get_xticklabels(), ha='right')
        plt.yticks(rotation=0, color='black', fontsize=8)

        # Enhanced title with better spacing and formatting
        if epoch == 'latest':
            current_epoch = len(self.metrics_history)
            title_lines = [
                'Confusion Matrix (Latest)',
                f'{self.experiment_name}',
                f'Epoch {current_epoch}/{self.config.training_settings.num_epochs}'
            ]
        else:
            title_lines = [
                'Confusion Matrix (Checkpoint)',
                f'{self.experiment_name}',
                f'Epoch {epoch}/{self.config.training_settings.num_epochs}'
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

        # Determine save directory and create full path
        if isinstance(epoch, int):
            save_dir = self._normalize_path(self.figures_dir, f'epoch_{epoch}')
        else:
            save_dir = self._normalize_path(self.figures_dir, str(epoch))

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Create full save path
        filename = f'{prefix}_confusion_matrix_{self.experiment_name}.png'
        save_path = self._normalize_path(save_dir, filename)

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save figure
            plt.savefig(save_path, bbox_inches='tight',
                        dpi=300, facecolor='white')
            print(f"✅ Saved confusion matrix to: {save_path}")
        except Exception as e:
            print(f"❌ Error saving confusion matrix")
            print(f"Attempted path: {save_path}")
            print(f"Error details: {str(e)}")
            raise
        finally:
            plt.close()

    def plot_metrics(self, epoch=None):
        # Use a default style and customize it
        plt.style.use('default')  # Reset to default style
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.linewidth': 1.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'grid.color': 'gray',
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.5,
            'ytick.minor.width': 0.5,
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
        })

        metric_pairs = [
            ('accuracy', 'Accuracy'),
            (('precision_macro', 'precision_micro'), 'Precision (Macro vs Micro)'),
            (('recall_macro', 'recall_micro'), 'Recall (Macro vs Micro)'),
            (('f1_macro', 'f1_micro'), 'F1 (Macro vs Micro)')
        ]

        # Create figure with higher DPI
        fig = plt.figure(figsize=(16, 12), dpi=300, facecolor='white')

        # Title formatting
        current_epoch = len(self.metrics_history)
        title_lines = [
            'Training Metrics (Latest)' if epoch is None or epoch == 'latest' else 'Training Metrics (Checkpoint)',
            f'{self.experiment_name}',
            f'Epoch {current_epoch}/{self.config.training_settings.num_epochs}'
        ]

        # Main title with better spacing
        fig.suptitle('\n'.join(title_lines),
                     y=0.95,
                     fontsize=14,
                     fontweight='bold',
                     color='black',
                     linespacing=1.5)

        for i, (metric, title) in enumerate(metric_pairs, 1):
            ax = fig.add_subplot(2, 2, i)
            ax.set_facecolor('white')

            if isinstance(metric, tuple):
                for m, (linestyle, marker) in zip(metric, [('-', 'o'), ('--', 's')]):
                    train_metric = [h[f'train_{m}']
                                    for h in self.metrics_history]
                    val_metric = [h[f'val_{m}'] for h in self.metrics_history]

                    if 'macro' in m:
                        ax.plot(train_metric, linestyle=linestyle, marker=marker,
                                label='Train (Macro)', color='#1f77b4',
                                markersize=5, linewidth=1.5, markeredgewidth=1.5)
                        ax.plot(val_metric, linestyle=linestyle, marker=marker,
                                label='Val (Macro)', color='#7cc7ff',
                                markersize=5, linewidth=1.5, markeredgewidth=1.5)
                    else:
                        ax.plot(train_metric, linestyle=linestyle, marker=marker,
                                label='Train (Micro)', color='#d62728',
                                markersize=5, linewidth=1.5, markeredgewidth=1.5)
                        ax.plot(val_metric, linestyle=linestyle, marker=marker,
                                label='Val (Micro)', color='#ff9999',
                                markersize=5, linewidth=1.5, markeredgewidth=1.5)
            else:
                train_metric = [h[f'train_{metric}']
                                for h in self.metrics_history]
                val_metric = [h[f'val_{metric}'] for h in self.metrics_history]
                ax.plot(train_metric, '-o', label='Train', color='#1f77b4',
                        markersize=5, linewidth=1.5, markeredgewidth=1.5)
                ax.plot(val_metric, '-o', label='Val', color='#7cc7ff',
                        markersize=5, linewidth=1.5, markeredgewidth=1.5)

            # Enhanced grid and axis styling
            ax.grid(True, linestyle='--', alpha=0.3,
                    color='gray', linewidth=0.5)
            ax.set_axisbelow(True)

            # Set axis limits and ticks
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(left=0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Title and labels with consistent black color
            ax.set_title(title, pad=10, fontsize=12,
                         fontweight='bold', color='black')
            ax.set_xlabel('Epoch', fontsize=11, color='black', labelpad=8)
            ax.set_ylabel('Score', fontsize=11, color='black', labelpad=8)

            # Spine and tick styling
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.0)
            ax.tick_params(which='both', colors='black', width=1.0)
            ax.tick_params(which='major', length=6)
            ax.tick_params(which='minor', length=3)

            # Legend styling
            legend = ax.legend(
                facecolor='white',
                edgecolor='none',
                loc='upper right',
                bbox_to_anchor=(0.98, 0.98),
                framealpha=0.9,
                ncol=1,
                handlelength=1.5,
                handletextpad=0.5,
                columnspacing=1.0,
                borderaxespad=0.2,
                markerscale=1.0
            )
            plt.setp(legend.get_texts(), color='black')
            legend.get_frame().set_linewidth(0.5)

        # Adjust subplot spacing
        plt.subplots_adjust(
            top=0.88,
            bottom=0.08,
            left=0.08,
            right=0.98,
            hspace=0.25,
            wspace=0.20
        )

        # Before saving, ensure all directories exist
        base_path = os.path.join(self.results_dir, 'figures')
        latest_figures_dir = os.path.join(base_path, 'latest')
        self._ensure_dir_exists(base_path)
        self._ensure_dir_exists(latest_figures_dir)

        for path in [base_path, latest_figures_dir]:
            plt.savefig(
                os.path.join(path, f'metrics_plot_{self.experiment_name}.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1
            )
        plt.close()

    def plot_class_performance(self, y_true, y_pred, epoch, prefix='val'):
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

        # Enhanced title with better spacing and formatting
        if epoch == 'latest':
            current_epoch = len(self.metrics_history)
            title_lines = [
                'Per-class Performance (Latest)',
                f'{self.experiment_name}',
                f'Epoch {current_epoch}/{self.config.training_settings.num_epochs}'
            ]
        else:
            title_lines = [
                'Per-class Performance (Checkpoint)',
                f'{self.experiment_name}',
                f'Epoch {epoch}/{self.config.training_settings.num_epochs}'
            ]

        plt.title('\n'.join(title_lines),
                  pad=20,
                  fontsize=14,
                  color='black',
                  linespacing=1.3,
                  ha='center',      # Ensure horizontal center alignment
                  x=0.5)           # Set x position to center

        # Set legend with black text in upper right
        legend = plt.legend(
            facecolor='white',
            edgecolor='black',
            loc='upper right',         # Ensure legend is at upper right
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

        # Ensure directories exist
        base_path = os.path.join(self.results_dir, 'figures')
        latest_figures_dir = os.path.join(base_path, 'latest')
        self._ensure_dir_exists(base_path)
        self._ensure_dir_exists(latest_figures_dir)

        if epoch == 'latest':
            save_path = os.path.join(
                latest_figures_dir, f'{prefix}_class_performance_{self.experiment_name}.png')
        else:
            epoch_figures_dir = os.path.join(base_path, f'epoch_{epoch}')
            self._ensure_dir_exists(epoch_figures_dir)
            save_path = os.path.join(
                epoch_figures_dir, f'{prefix}_class_performance_{self.experiment_name}.png')

        # Ensure the immediate parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()

    def _display_time_statistics(self, epoch_time=None, avg_time=None, remaining_time=None):
        """Display formatted time statistics during training"""
        print("\n┌───────────────────────────────────────────┐")
        print("│ ⏱️  Time Statistics                        │")
        print("├───────────────────────────────────────────┤")
        print(f"│  • Current Epoch: {self._format_time(epoch_time):<15} │")
        print(f"│  • Average Time:  {self._format_time(avg_time):<15} │")
        print(f"│  • Remaining:     {self._format_time(remaining_time):<15} │")
        print("└───────────────────────────────────────────┘")

    def _format_time(self, seconds):
        """Format time in HH:MM:SS format"""
        if seconds is None:
            return "--:--:--"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def train(self):
        """Train the model"""
        print("\n🚀 Starting training...\n")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.training_settings.num_epochs):
            epoch_start_time = time.time()

            print(f"{'='*50}")
            print(
                f"⏳ Epoch {epoch + 1}/{self.config.training_settings.num_epochs}")
            print(f"{'='*50}")

            # Calculate and store epoch time
            epoch_duration = time.time() - epoch_start_time
            self.epoch_times.append(epoch_duration)

            # Calculate average epoch time and estimate remaining time
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            estimated_remaining = self._estimate_remaining_time(
                epoch + 1, avg_epoch_time)

            # Print time information in a box
            self._display_time_statistics(
                epoch_duration, avg_epoch_time, estimated_remaining)

            # Training phase
            train_metrics, train_preds, train_labels = self.train_epoch()

            # Validation phase
            val_metrics, val_preds, val_labels = self.evaluate(
                self.val_loader, 'val')

            # Store losses and metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])

            epoch_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
            epoch_metrics.update(
                {f'val_{k}': v for k, v in val_metrics.items()})
            self.metrics_history.append(epoch_metrics)

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])

            # Print metrics
            print(f"\nEpoch {epoch + 1} Results:")
            print("\nTraining Metrics:")
            for k, v in train_metrics.items():
                print(f"{k}: {v:.4f}")
            print("\nValidation Metrics:")
            for k, v in val_metrics.items():
                print(f"{k}: {v:.4f}")

            # Always save latest model and plots after each epoch
            self.save_checkpoint(epoch + 1, val_metrics['loss'], is_best=False)

            # Save latest plots for both training and validation
            self.plot_confusion_matrix(
                train_labels, train_preds, 'latest', prefix='train')
            self.plot_confusion_matrix(
                val_labels, val_preds, 'latest', prefix='val')
            self.plot_class_performance(
                train_labels, train_preds, 'latest', prefix='train')
            self.plot_class_performance(
                val_labels, val_preds, 'latest', prefix='val')

            self.save_metrics(train_metrics, 'latest', 'train')
            self.save_metrics(val_metrics, 'latest', 'val')
            self.plot_metrics(epoch=None)  # This will go to latest directory

            # Periodic checkpoint saving
            if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
                # Save numbered checkpoint
                checkpoint_path = os.path.join(
                    self.model_dir,
                    'checkpoints',
                    f'checkpoint_epoch_{epoch + 1}.pt'
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': self.config.to_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'metrics_history': self.metrics_history
                }, checkpoint_path)

                # Save periodic plots for both training and validation
                self.plot_confusion_matrix(
                    train_labels, train_preds, epoch + 1, prefix='train')
                self.plot_confusion_matrix(
                    val_labels, val_preds, epoch + 1, prefix='val')
                self.plot_class_performance(
                    train_labels, train_preds, epoch + 1, prefix='train')
                self.plot_class_performance(
                    val_labels, val_preds, epoch + 1, prefix='val')

                self.save_metrics(train_metrics, epoch + 1, 'train')
                self.save_metrics(val_metrics, epoch + 1, 'val')
                self.plot_metrics(epoch=epoch+1)

            # Best model saving logic
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model and plots
                best_path = os.path.join(self.model_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': self.config.to_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'metrics_history': self.metrics_history
                }, best_path)

                # Save best plots for both training and validation
                self.plot_confusion_matrix(
                    train_labels, train_preds, 'best', prefix='train')
                self.plot_confusion_matrix(
                    val_labels, val_preds, 'best', prefix='val')
                self.plot_class_performance(
                    train_labels, train_preds, 'best', prefix='train')
                self.plot_class_performance(
                    val_labels, val_preds, 'best', prefix='val')

                self.save_metrics(train_metrics, 'best', 'train')
                self.save_metrics(val_metrics, 'best', 'val')
            else:
                self.patience_counter += 1

            # Early stopping check
            if self.patience_counter >= self.config.training_settings.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Final evaluation on test set
        test_metrics, test_preds, test_labels = self.evaluate(
            self.test_loader, 'test')
        self.save_metrics(test_metrics, 'final', 'test')

        print("\nFinal Test Results:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        # Plot final confusion matrix
        self.plot_confusion_matrix(test_labels, test_preds, 'final')

        # Print total training time at the end
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("🎉 Training completed!")
        print(f"⏰ Total training time: {self._format_time(total_time)}")
        print("="*50 + "\n")

        return self.metrics_history

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc='Training'):
            self.optimizer.zero_grad()

            # First element is input
            inputs = batch[0].to(self.config.training_settings.device)
            # Second element is label
            labels = batch[1].to(self.config.training_settings.device)

            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()

            if self.config.training_settings.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training_settings.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.train_loader)

        return metrics, all_preds, all_labels

    def evaluate(self, loader, phase='val'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Evaluating ({phase})'):
                inputs, labels = batch  # Changed this line
                inputs = inputs.to(self.config.training_settings.device)
                labels = labels.to(self.config.training_settings.device)

                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(loader)

        return metrics, all_preds, all_labels

    def save_metrics(self, metrics, epoch, phase):
        """Save metrics to JSON file"""
        if isinstance(epoch, int):
            # For periodic saves
            if (epoch) % self.config.training_settings.checkpoint_freq == 0:
                _, _, epoch_metrics_dir = self._create_epoch_directories(epoch)
                metrics_file = os.path.join(
                    epoch_metrics_dir,
                    f'{phase}_metrics_{self.experiment_name}.json'
                )
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
        else:
            # For 'latest', 'best', or 'final' saves
            metrics_dir = os.path.join(self.results_dir, 'metrics', epoch)
            self._ensure_dir_exists(metrics_dir)
            metrics_file = os.path.join(
                metrics_dir,
                f'{phase}_metrics_{self.experiment_name}.json'
            )
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

    def _setup_fine_tuned_directories(self):
        """Set up directories for fine-tuned embedding models"""
        # Create main directory
        self._ensure_dir_exists(self.fine_tuned_dir)

        # Create subdirectories for checkpoints and latest models
        self._ensure_dir_exists(os.path.join(
            self.fine_tuned_dir, 'checkpoints'))
        self._ensure_dir_exists(os.path.join(self.fine_tuned_dir, 'latest'))

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
        print(f"- Hidden Dimensions: {self.config.model_settings.hidden_dims}")
        print(f"- Bidirectional: {self.config.model_settings.bidirectional}")
        print(f"- Dropout Rate: {self.config.model_settings.dropout_rate}")
        print(f"- Attention: {self.config.model_settings.use_attention}")
        if self.config.model_settings.use_attention:
            print(f"  • Type: {self.config.model_settings.attention_type}")
            print(
                f"  • Heads: {self.config.model_settings.num_attention_heads}")
            print(f"  • Dimension: {self.config.model_settings.attention_dim}")
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
        print("\n📚 Dataset Information:")
        print(f"- Training samples: {len(self.train_loader.dataset)}")
        print(f"- Validation samples: {len(self.val_loader.dataset)}")
        print(f"- Test samples: {len(self.test_loader.dataset)}")
        print(f"- Number of batches (train): {len(self.train_loader)}")

        print("\n" + "="*50 + "\n")
