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

        # Load data automatically
        self.train_loader, self.val_loader, self.test_loader, self.label_encoder = loader(
            config)

        # Automatically set output dimension based on number of classes
        self._set_output_dimension()

        # Initialize model with correct output dimension
        self.model = self.config.model_selection.get_model(config)
        self.model = self.model.to(self.config.training_settings.device)

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

        # 1. Task Type Prefix
        task_prefix = 'ID' if self.config.training_settings.task_type == 'identification' else 'CLS'
        components.append(task_prefix)

        # 2. Model Architecture (including bidirectional and resnet)
        model_prefix = []
        if self.config.model_settings.use_res_net:
            model_prefix.append('Res')
        model_prefix.append(self.config.model_selection.model_type)
        components.append(''.join(model_prefix))

        # 3. Embedding Info
        emb_name = self.config.model_settings.embedding_type.split('_')[0]
        components.append(emb_name)

        # 4. Core Model Parameters
        # Number of layers
        components.append(f"L{self.config.model_settings.num_layers}")
        # Hidden dimension
        components.append(f"H{self.config.model_settings.init_hidden_dim}")
        components.append(
            f"M{self.config.tokenizer_settings.max_length}")  # Max length

        # Add learning rate (with scientific notation for small values)
        lr = self.config.training_settings.learning_rate
        if lr < 0.01:
            # Simplify scientific notation
            lr_str = f"{lr:.0e}".replace('e-0', 'e-')
        else:
            lr_str = f"{lr:.3f}".rstrip('0').rstrip('.')
        components.append(f"LR{lr_str}")

        # Add activation function (first 4 chars)
        components.append(f"Act{self.config.model_settings.activation[:4]}")

        # 5. Training Strategy
        components.append(self.config.model_settings.loss.replace(
            '_', ''))  # Loss function

        # 6. Initialization Function
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

        # 7. Data Strategy (if any)
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

        # 8. Special Features (as flags)
        flags = []
        if self.config.model_settings.use_attention:
            flags.append('A')  # Attention
        if self.config.model_settings.fine_tune_embedding:
            flags.append('FT')  # Fine-tuning
        if self.config.model_settings.use_layer_norm:
            flags.append('LN')  # Layer Normalization
        if flags:
            components.append(''.join(flags))

        return '_'.join(components)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and return checkpoint data"""
        if not checkpoint_path:
            print("⚠️ No checkpoint path provided")
            return None

        try:
            print(f"📂 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.metrics_history = checkpoint.get('metrics_history', [])
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))

            print(
                f"✅ Successfully loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
            return checkpoint

        except Exception as e:
            print(f"❌ Error loading checkpoint: {str(e)}")
            return None

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
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }

        # Save model checkpoint
        if is_best:
            best_dir = os.path.join(self.model_dir, 'best')
            self._ensure_dir_exists(best_dir)
            best_path = os.path.join(best_dir, 'model_best.pt')
            torch.save(checkpoint, best_path)
            print(f"🏆 Saved best model at epoch {epoch + 1}")
        else:
            latest_dir = os.path.join(self.model_dir, 'latest')
            self._ensure_dir_exists(latest_dir)
            latest_path = os.path.join(latest_dir, 'model_latest.pt')
            torch.save(checkpoint, latest_path)
            print(f"📝 Saved latest model at epoch {epoch + 1}")

        # Save periodic checkpoint if configured
        if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
            checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
            self._ensure_dir_exists(checkpoint_dir)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch + 1}")

        # Save fine-tuned embedding model if enabled
        if self.config.model_settings.fine_tune_embedding:
            self._save_fine_tuned_embedding(epoch, is_best)

    def _save_fine_tuned_embedding(self, epoch, is_best=False):
        """Save fine-tuned embedding model with epoch information"""
        # Prepare embedding state
        embedding_state = {
            'epoch': epoch + 1,
            'state_dict': self.model.embedding_model.state_dict(),
            'config': self.config.model_settings.to_dict(),
            'embedding_type': self.config.model_settings.embedding_type
        }

        # 1. Save latest version
        latest_dir = os.path.join(self.fine_tuned_dir, 'latest')
        self._ensure_dir_exists(latest_dir)
        latest_path = os.path.join(latest_dir, 'embedding_model_latest.pt')
        torch.save(embedding_state, latest_path)
        print(
            f"💾 Saved latest fine-tuned embedding model at epoch {epoch + 1}")

        # 2. Save best version if applicable
        if is_best:
            best_dir = os.path.join(self.fine_tuned_dir, 'best')
            self._ensure_dir_exists(best_dir)
            best_path = os.path.join(best_dir, 'embedding_model_best.pt')
            torch.save(embedding_state, best_path)
            print(
                f"🏆 Saved best fine-tuned embedding model at epoch {epoch + 1}")

        # 3. Save periodic checkpoint based on frequency
        if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
            checkpoint_dir = os.path.join(self.fine_tuned_dir, 'checkpoints')
            self._ensure_dir_exists(checkpoint_dir)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'embedding_model_epoch_{epoch + 1}.pt'
            )
            torch.save(embedding_state, checkpoint_path)
            print(
                f"📁 Saved fine-tuned embedding checkpoint at epoch {epoch + 1}")

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
        filename = f'{prefix}_confusion_matrix_{self.experiment_name}.png'
        save_path = os.path.join(save_dir, filename)

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save figure
            plt.savefig(save_path, bbox_inches='tight',
                        dpi=300, facecolor='white')
        except Exception as e:
            print(f"❌ Error saving confusion matrix")
            print(f"Attempted path: {save_path}")
            print(f"Error details: {str(e)}")
            raise
        finally:
            plt.close()

    def plot_metrics(self, save_dir):
        """Plot training and validation metrics"""
        # Use a default style and customize it
        plt.style.use('default')
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

        # Create figure with higher DPI
        fig = plt.figure(figsize=(16, 12), dpi=300, facecolor='white')

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

        # Title formatting
        current_epoch = len(self.metrics_history)
        total_epochs = self.config.training_settings.num_epochs
        title_lines = [
            f'Training Metrics ({title_type})',
            f'{self.experiment_name}',
            f'Epoch {current_epoch}/{total_epochs}'
        ]

        # Main title with better spacing
        fig.suptitle('\n'.join(title_lines),
                     y=0.95,
                     fontsize=14,
                     fontweight='bold',
                     color='black',
                     linespacing=1.5)

        metric_pairs = [
            ('accuracy', 'Accuracy'),
            (('precision_macro', 'precision_micro'), 'Precision (Macro vs Micro)'),
            (('recall_macro', 'recall_micro'), 'Recall (Macro vs Micro)'),
            (('f1_macro', 'f1_micro'), 'F1 (Macro vs Micro)')
        ]

        for i, (metric, title) in enumerate(metric_pairs, 1):
            ax = fig.add_subplot(2, 2, i)
            ax.set_facecolor('white')

            # Plot metrics
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

        # Create full save path
        filename = f'{prefix}_class_performance_{self.experiment_name}.png'
        save_path = os.path.join(save_dir, filename)

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save figure
            plt.savefig(save_path, bbox_inches='tight',
                        dpi=300, facecolor='white')
        except Exception as e:
            print(f"❌ Error saving class performance plot")
            print(f"Attempted path: {save_path}")
            print(f"Error details: {str(e)}")
            raise
        finally:
            plt.close()

    def _print_metrics(self, metrics, phase='train'):
        """Print metrics in a formatted way"""
        print(f"\n📊 {phase.capitalize()} Metrics:")
        print(
            f"Loss: {metrics['loss']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        print(
            f"Macro → P: {metrics['precision_macro']:.4f} | R: {metrics['recall_macro']:.4f} | F1: {metrics['f1_macro']:.4f}")
        print(
            f"Micro → P: {metrics['precision_micro']:.4f} | R: {metrics['recall_micro']:.4f} | F1: {metrics['f1_micro']:.4f}")
        print(f"\n")

    def train(self):
        """Main training loop"""
        # 1. Handle training continuation
        if self.continue_training:
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                checkpoint = self.load_checkpoint(latest_checkpoint)
                if checkpoint is not None:
                    self.start_epoch = checkpoint['epoch'] + 1
                    print(f"✅ Resuming training from epoch {self.start_epoch + 1}")
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
            print(f"\n{'─'*50}")
            print(f"⏳ Epoch {epoch + 1}/{self.config.training_settings.num_epochs}")
            print(f"{'─'*50}")

            # Training and validation phases
            train_metrics, train_preds, train_labels = self.train_epoch()
            val_metrics, val_preds, val_labels = self.evaluate(self.val_loader, 'val')

            # Update tracking variables
            self._update_tracking(train_metrics, val_metrics)
            
            # 3. Save latest state
            self._save_latest_state(epoch, train_labels, train_preds, val_labels, val_preds)

            # 4. Save periodic checkpoints
            if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
                self._save_checkpoint_state(epoch, train_labels, train_preds, val_labels, val_preds)

            # 5. Handle best model saving
            if val_metrics['loss'] < self.best_val_loss:
                self._save_best_state(epoch, train_labels, train_preds, val_labels, val_preds, val_metrics['loss'])

            # 6. Early stopping check
            if self._check_early_stopping(val_metrics['loss']):
                print(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
                break

        # 7. Save final state
        self._save_final_state(train_labels, train_preds, val_labels, val_preds)

        return self.metrics_history

    def _update_tracking(self, train_metrics, val_metrics):
        """Update tracking variables with new metrics"""
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        epoch_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
        epoch_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        self.metrics_history.append(epoch_metrics)

    def _save_latest_state(self, epoch, train_labels, train_preds, val_labels, val_preds):
        """Save latest model state and figures"""
        latest_dir = os.path.join(self.figures_dir, 'latest')
        self._ensure_dir_exists(latest_dir)
        self._save_epoch_figures(train_labels, train_preds, val_labels, val_preds, latest_dir)
        self.save_checkpoint(epoch, self.val_losses[-1], is_best=False)

    def _save_checkpoint_state(self, epoch, train_labels, train_preds, val_labels, val_preds):
        """Save periodic checkpoint state and figures"""
        checkpoint_figures_dir = os.path.join(self.figures_dir, 'checkpoints', f'epoch_{epoch + 1}')
        self._ensure_dir_exists(checkpoint_figures_dir)
        self._save_epoch_figures(train_labels, train_preds, val_labels, val_preds, checkpoint_figures_dir)

    def _save_best_state(self, epoch, train_labels, train_preds, val_labels, val_preds, val_loss):
        """Save best model state and figures"""
        self.best_val_loss = val_loss
        best_dir = os.path.join(self.figures_dir, 'best')
        self._ensure_dir_exists(best_dir)
        self._save_epoch_figures(train_labels, train_preds, val_labels, val_preds, best_dir)
        self.save_checkpoint(epoch, val_loss, is_best=True)
        print(f"🏆 New best model achieved at epoch {epoch + 1}!")

    def _save_final_state(self, train_labels, train_preds, val_labels, val_preds):
        """Save final model state and figures"""
        final_dir = os.path.join(self.figures_dir, 'final')
        self._ensure_dir_exists(final_dir)
        self._save_epoch_figures(train_labels, train_preds, val_labels, val_preds, final_dir)

    def train_epoch(self):
        """Training loop for one epoch"""
        self.model.train()
        total_loss = 0
        num_samples = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc='Training'):
            self.optimizer.zero_grad()

            # Move batch to device
            inputs = batch[0].to(self.config.training_settings.device)
            labels = batch[1].to(self.config.training_settings.device)

            # Create attention mask for padding tokens if model supports it
            attention_mask = (inputs != 0).to(
                self.config.training_settings.device)

            # Forward pass - check if model accepts attention_mask
            try:
                outputs, _ = self.model(inputs, attention_mask=attention_mask)
            except TypeError:
                # If model doesn't accept attention_mask, call without it
                outputs, _ = self.model(inputs)

            # Handle loss calculation based on task type
            if self.config.training_settings.task_type == 'classification':
                # For classification, CrossEntropyLoss expects class indices, not one-hot
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
            else:
                # For other tasks, might need different handling
                loss = self.criterion(outputs, labels)
                preds = outputs

            # Backward pass
            loss.backward()

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
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

        # Calculate average loss and metrics
        avg_loss = total_loss / num_samples
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        metrics['loss'] = avg_loss

        return metrics, all_preds, all_labels

    def evaluate(self, loader, phase='val'):
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0
        num_samples = 0
        all_preds = []
        all_labels = []

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
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / num_samples

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
                checkpoint = torch.load(latest_model_path)
                latest_epoch = checkpoint.get('epoch', -1)
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

        # Decide which checkpoint to use
        if latest_epoch > max_checkpoint_epoch:
            if latest_epoch >= 0:
                print(f"✅ Using latest model from epoch {latest_epoch + 1}")
                return latest_model_path
        elif max_checkpoint_epoch >= 0:
            print(f"✅ Using checkpoint from epoch {max_checkpoint_epoch + 1}")
            return max_checkpoint_path

        print("⚠️ No valid checkpoints found, starting fresh training")
        return None

    def _save_epoch_figures(self, train_labels, train_preds, val_labels, val_preds, save_dir):
        """Save all figures for an epoch in the specified directory"""
        # Get directory type from path for message
        dir_type = 'latest' if 'latest' in save_dir else 'best' if 'best' in save_dir else 'checkpoint'
        epoch = len(self.train_losses)  # Current epoch

        # Always save metrics plot
        self.plot_metrics(save_dir=save_dir)
        print(
            f"📈 Saved {'latest' if dir_type == 'latest' else dir_type} metrics plot at epoch {epoch}")

        # Save confusion matrices and class performance plots only for latest
        if dir_type == 'latest':
            # Save confusion matrices
            self.plot_confusion_matrix(
                train_labels, train_preds, save_dir=save_dir, prefix='train')
            self.plot_confusion_matrix(
                val_labels, val_preds, save_dir=save_dir, prefix='val')
            print(f"🔄 Saved latest confusion matrices at epoch {epoch}")

            # Save class performance plots
            self.plot_class_performance(
                train_labels, train_preds, save_dir=save_dir, prefix='train')
            self.plot_class_performance(
                val_labels, val_preds, save_dir=save_dir, prefix='val')
            print(f"📊 Saved latest performance plots at epoch {epoch}")

        # Create and save whole period metrics if we're at a checkpoint epoch
        if dir_type == 'checkpoint' or dir_type == 'best' or dir_type == 'latest':
            self.save_whole_period_metrics()

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

        for idx, (metric, title) in enumerate(metric_pairs, 1):
            ax = plt.subplot(2, 2, idx)
            ax.set_facecolor('white')
            ax.grid(True, linestyle='--', alpha=0.7)

            epochs = range(1, len(self.metrics_history) + 1)

            if isinstance(metric, tuple):
                # Plot macro metrics
                train_macro = [m[f'train_{metric[0]}']
                               for m in self.metrics_history]
                val_macro = [m[f'val_{metric[0]}']
                             for m in self.metrics_history]
                ax.plot(epochs, train_macro, 'b-',
                        label=f'Train (Macro)', marker='o', markersize=2)
                ax.plot(epochs, val_macro, 'c-',
                        label=f'Val (Macro)', marker='o', markersize=2)

                # Plot micro metrics
                train_micro = [m[f'train_{metric[1]}']
                               for m in self.metrics_history]
                val_micro = [m[f'val_{metric[1]}']
                             for m in self.metrics_history]
                ax.plot(epochs, train_micro, 'r-',
                        label=f'Train (Micro)', marker='s', markersize=2)
                ax.plot(epochs, val_micro, 'salmon',
                        label=f'Val (Micro)', marker='s', markersize=2)
            else:
                train_metric = [m[f'train_{metric}']
                                for m in self.metrics_history]
                val_metric = [m[f'val_{metric}']
                              for m in self.metrics_history]
                ax.plot(epochs, train_metric, 'b-',
                        label='Train', marker='o', markersize=2)
                ax.plot(epochs, val_metric, 'c-',
                        label='Val', marker='o', markersize=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.legend(loc='upper left')
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Save in the main figures directory
        save_path = os.path.join(
            self.figures_dir, f'metrics_whole_period_{self.experiment_name}.png')
        plt.savefig(save_path,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.1)
        plt.close()

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

    def _load_fine_tuned_embedding(self, checkpoint_path):
        """Load fine-tuned embedding model"""
        try:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.embedding_model.load_state_dict(
                    checkpoint['state_dict'])
                print(
                    f"✅ Loaded fine-tuned embedding model from epoch {checkpoint['epoch']}")
            else:
                self.model.embedding_model.load_state_dict(checkpoint)
                print("✅ Loaded fine-tuned embedding model")
        except Exception as e:
            print(f"❌ Error loading fine-tuned embedding model: {str(e)}")
            raise
