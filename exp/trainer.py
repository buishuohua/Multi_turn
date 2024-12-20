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

        # Initialize training components
        self.criterion = self.config.model_settings.get_loss()
        self.optimizer = self.config.training_settings.get_optimizer(
            self.model.parameters())
        self.scheduler = self.config.training_settings.get_scheduler(
            self.optimizer)

        # Create directories
        self._setup_directories()

        # Initialize tracking variables
        self._initialize_tracking_variables()

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
        self.model_dir = os.path.join(
            self.config.training_settings.save_model_dir,
            self.experiment_name
        )
        self.results_dir = os.path.join(
            self.config.training_settings.save_results_dir,
            self.experiment_name
        )

        # Create model directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'latest'), exist_ok=True)

        # Create results directories
        os.makedirs(os.path.join(self.results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir,
                    'figures', 'latest'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir,
                    'metrics', 'latest'), exist_ok=True)

    def _create_experiment_name(self):
        """Create unique experiment name with key parameters"""
        model_type = self.config.model_selection.model_type
        tokenizer_name = self.config.tokenizer_settings.name
        layers = self.config.model_settings.num_layers
        return f"{model_type}_{tokenizer_name}_l{layers}"

    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint for continuing training"""
        if checkpoint_path is None:
            # Try to load latest checkpoint first
            latest_path = os.path.join(self.model_dir, 'latest_model.pt')
            if os.path.exists(latest_path):
                checkpoint_path = latest_path
            else:
                # Fall back to finding the most recent checkpoint in checkpoints directory
                checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
                if not os.path.exists(checkpoint_dir):
                    return False

                checkpoints = [f for f in os.listdir(
                    checkpoint_dir) if f.endswith('.pt')]
                if not checkpoints:
                    return False

                # Extract epoch numbers and find the latest
                epoch_numbers = [int(re.search(r'epoch_(\d+)', cp).group(1))
                                 for cp in checkpoints]
                latest_checkpoint = checkpoints[epoch_numbers.index(
                    max(epoch_numbers))]
                checkpoint_path = os.path.join(
                    checkpoint_dir, latest_checkpoint)

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Load model and training state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']

        # Load training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.metrics_history = checkpoint['metrics_history']

        return True

    def _ensure_dir_exists(self, path):
        """Ensure directory exists, create if it doesn't"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

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

        # Ensure directories exist
        latest_dir = os.path.join(self.model_dir, 'latest')
        checkpoints_dir = os.path.join(self.model_dir, 'checkpoints')
        self._ensure_dir_exists(latest_dir)
        self._ensure_dir_exists(checkpoints_dir)

        # Save latest model
        latest_path = os.path.join(latest_dir, f'{self.experiment_name}.pt')
        torch.save(checkpoint, latest_path)

        # Save periodic checkpoint
        if isinstance(epoch, int):
            checkpoint_path = os.path.join(
                checkpoints_dir, f'{self.experiment_name}_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch}")

        # Save best model
        if is_best:
            best_path = os.path.join(
                self.model_dir, f'{self.experiment_name}_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")

    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot enhanced confusion matrix with better readability for many classes"""
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
                    cbar_kws={'label': 'Count'},  # Removed labelsize from here
                    vmin=0,
                    annot_kws={'size': 7},  # Smaller font for numbers in cells
                    )

        # Get the colorbar and adjust its label size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)  # Set colorbar tick label size
        cbar.set_label('Count', size=10)   # Set colorbar label size

        # Adjust label sizes and rotation
        plt.xticks(rotation=45, ha='right', color='black', fontsize=8)
        plt.yticks(rotation=0, color='black', fontsize=8)

        # Move x-axis labels up slightly to prevent cutoff
        ax.xaxis.set_tick_params(
            labelsize=8, rotation=45, ha='right', rotation_mode='anchor')

        # Add titles with appropriate sizing
        plt.title(f'Confusion Matrix - {self.experiment_name}\nEpoch {epoch}',
                  pad=20, fontsize=14, color='black')
        plt.xlabel('Predicted Label', labelpad=15, color='black', fontsize=12)
        plt.ylabel('True Label', labelpad=15, color='black', fontsize=12)

        # Add more padding to prevent label cutoff
        plt.tight_layout(pad=1.1)

        # Ensure directories exist and save
        base_path = os.path.join(self.results_dir, 'figures')
        latest_figures_dir = os.path.join(base_path, 'latest')
        self._ensure_dir_exists(base_path)
        self._ensure_dir_exists(latest_figures_dir)

        if epoch == 'latest':
            save_path = os.path.join(
                latest_figures_dir, f'confusion_matrix_{self.experiment_name}.png')
        else:
            save_path = os.path.join(
                base_path, f'confusion_matrix_{self.experiment_name}_epoch_{epoch}.png')

        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white',
                    pad_inches=0.5)
        plt.close()

    def plot_metrics(self):
        """Plot metrics with path checking"""
        metric_pairs = [
            ('accuracy', 'Accuracy'),
            (('precision_macro', 'precision_micro'), 'Precision (Macro vs Micro)'),
            (('recall_macro', 'recall_micro'), 'Recall (Macro vs Micro)'),
            (('f1_macro', 'f1_micro'), 'F1 (Macro vs Micro)')
        ]

        plt.figure(figsize=(20, 15), facecolor='white')

        for i, (metric, title) in enumerate(metric_pairs, 1):
            plt.subplot(2, 2, i)
            ax = plt.gca()
            ax.set_facecolor('white')

            if isinstance(metric, tuple):
                # Plot macro and micro metrics together
                for m, style in zip(metric, ['-o', '--s']):
                    train_metric = [h[f'train_{m}']
                                    for h in self.metrics_history]
                    val_metric = [h[f'val_{m}'] for h in self.metrics_history]
                    label_suffix = '(Macro)' if 'macro' in m else '(Micro)'
                    plt.plot(train_metric, style, label=f'Train {label_suffix}',
                             markersize=4, color='blue' if 'macro' in m else 'red')
                    plt.plot(val_metric, style, label=f'Val {label_suffix}',
                             markersize=4, color='lightblue' if 'macro' in m else 'lightcoral')
            else:
                # Plot single metric
                train_metric = [h[f'train_{metric}']
                                for h in self.metrics_history]
                val_metric = [h[f'val_{metric}'] for h in self.metrics_history]
                plt.plot(train_metric, '-o', label='Train',
                         markersize=4, color='blue')
                plt.plot(val_metric, '--s', label='Val',
                         markersize=4, color='lightblue')

            plt.title(title, color='black', pad=10)
            plt.xlabel('Epoch', color='black')
            plt.ylabel('Score', color='black')
            plt.grid(True, linestyle='--', alpha=0.3, color='gray')

            # Set legend with black text
            legend = plt.legend(facecolor='white', edgecolor='black')
            plt.setp(legend.get_texts(), color='black')

            # Set spine colors to black
            for spine in ax.spines.values():
                spine.set_color('black')

            # Set tick colors to black
            ax.tick_params(colors='black')

        plt.suptitle(f'Training Metrics - {self.experiment_name}',
                     color='black', fontsize=16, y=1.02)
        plt.tight_layout()

        # Ensure directories exist
        base_path = os.path.join(self.results_dir, 'figures')
        latest_figures_dir = os.path.join(base_path, 'latest')
        self._ensure_dir_exists(base_path)
        self._ensure_dir_exists(latest_figures_dir)

        # Save both versions
        plt.savefig(os.path.join(base_path, f'metrics_plot_{self.experiment_name}.png'),
                    bbox_inches='tight', dpi=300, facecolor='white')
        plt.savefig(os.path.join(latest_figures_dir, f'metrics_plot_{self.experiment_name}.png'),
                    bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()

    def plot_class_performance(self, y_true, y_pred, epoch):
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
        plt.title(f'Per-class Performance - {self.experiment_name}\nEpoch {epoch}',
                  color='black')

        # Set legend with black text
        legend = plt.legend(facecolor='white', edgecolor='black')
        # Set legend text color to black
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

        plt.tight_layout()

        # Ensure directories exist
        base_path = os.path.join(self.results_dir, 'figures')
        latest_figures_dir = os.path.join(base_path, 'latest')
        self._ensure_dir_exists(base_path)
        self._ensure_dir_exists(latest_figures_dir)

        if epoch == 'latest':
            save_path = os.path.join(
                latest_figures_dir, f'class_performance_{self.experiment_name}.png')
        else:
            save_path = os.path.join(
                base_path, f'class_performance_{self.experiment_name}_epoch_{epoch}.png')

        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()

    def train(self):
        """Train model with option to continue from checkpoint"""
        if self.continue_training:
            loaded = self.load_checkpoint()
            if loaded:
                print(f"Continuing training from epoch {self.start_epoch + 1}")
            else:
                print("No checkpoint found, starting fresh training")
                self.start_epoch = 0

        print(f"Experiment name: {self.experiment_name}")
        self.model = self.model.to(self.config.training_settings.device)

        for epoch in range(self.start_epoch, self.config.training_settings.num_epochs):
            print(
                f"\nEpoch {epoch + 1}/{self.config.training_settings.num_epochs} - "
                f"{self.experiment_name}")

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics, val_preds, val_labels = self.evaluate(self.val_loader)

            # Store losses and metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            epoch_metrics = {
                f'train_{k}': v for k, v in train_metrics.items()
            }
            epoch_metrics.update({
                f'val_{k}': v for k, v in val_metrics.items()
            })
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
            # Save latest model
            self.save_checkpoint(epoch + 1, val_metrics['loss'], is_best=False)
            self.plot_confusion_matrix(val_labels, val_preds, 'latest')
            self.plot_class_performance(val_labels, val_preds, 'latest')
            self.save_metrics(train_metrics, 'latest', 'train')
            self.save_metrics(val_metrics, 'latest', 'val')
            self.plot_metrics()

            # Periodic checkpoint saving remains the same
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

                # Save periodic plots and metrics
                self.plot_confusion_matrix(val_labels, val_preds, epoch + 1)
                self.plot_class_performance(val_labels, val_preds, epoch + 1)
                self.save_metrics(train_metrics, epoch + 1, 'train')
                self.save_metrics(val_metrics, epoch + 1, 'val')

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
                self.plot_confusion_matrix(val_labels, val_preds, 'best')
                self.plot_class_performance(val_labels, val_preds, 'best')
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
            # outputs shape should be [batch_size, num_classes]
            # labels shape should be [batch_size]
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

        return metrics

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
        metrics_dir = os.path.join(self.results_dir, 'metrics')
        latest_metrics_dir = os.path.join(metrics_dir, 'latest')

        # Ensure directories exist
        self._ensure_dir_exists(metrics_dir)
        self._ensure_dir_exists(latest_metrics_dir)

        if epoch == 'latest':
            metrics_file = os.path.join(
                latest_metrics_dir, f'{phase}_metrics_{self.experiment_name}.json')
        else:
            metrics_file = os.path.join(
                metrics_dir, f'{phase}_metrics_{self.experiment_name}_epoch_{epoch}.json')

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
