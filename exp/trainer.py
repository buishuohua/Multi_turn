# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/15 11:19
    @ Description:
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from config.Experiment_Config import ExperimentConfig
import re
from datetime import datetime
import numpy as np


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
    def __init__(
            self,
            model,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            config: ExperimentConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Create experiment name with timestamp and key parameters
        self.experiment_name = self._create_experiment_name()
        
        # Initialize training components
        self.criterion = self.config.model_settings.get_loss()
        self.optimizer = self.config.training_settings.get_optimizer(self.model.parameters())
        self.scheduler = self.config.training_settings.get_scheduler(self.optimizer)
        
        # Create directories with experiment name
        self.model_dir = os.path.join(config.training_settings.save_model_dir, self.experiment_name)
        self.results_dir = os.path.join(config.training_settings.save_results_dir, self.experiment_name)
        self._create_directories()
        
        # Initialize tracking variables
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _create_experiment_name(self):
        """Create unique experiment name with key parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.model_selection.model_type
        layers = self.config.model_settings.num_layers
        hidden = self.config.model_settings.init_hidden_dim
        lr = self.config.training_settings.learning_rate
        return f"{model_type}_l{layers}_h{hidden}_lr{lr}_{timestamp}"

    def _create_directories(self):
        """Create necessary directories for saving results"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'metrics'), exist_ok=True)

    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint for continuing training"""
        if checkpoint_path is None:
            # Find the latest checkpoint
            checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                return False
                
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if not checkpoints:
                return False
                
            # Extract epoch numbers and find the latest
            epoch_numbers = [int(re.search(r'epoch_(\d+)', cp).group(1)) for cp in checkpoints]
            latest_checkpoint = checkpoints[epoch_numbers.index(max(epoch_numbers))]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
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

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.model_dir,
            'checkpoints',
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.model_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot enhanced confusion matrix"""
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation text with both count and percentage
        annotations = np.array([f'{count}\n({percent:.1f}%)'
                              for count, percent in zip(cm.flatten(), cm_percent.flatten())])
        annotations = annotations.reshape(cm.shape)
        
        # Plot with improved aesthetics
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=self.config.data_settings.class_names,
                   yticklabels=self.config.data_settings.class_names)
        
        plt.title(f'Confusion Matrix - {self.experiment_name}\nEpoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save with experiment name
        plt.savefig(os.path.join(
            self.results_dir,
            'figures',
            f'confusion_matrix_epoch_{epoch}.png'
        ), bbox_inches='tight', dpi=300)
        plt.close()

    def plot_metrics(self):
        """Plot enhanced metrics history"""
        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 2, i)
            train_metric = [m[f'train_{metric}'] for m in self.metrics_history]
            val_metric = [m[f'val_{metric}'] for m in self.metrics_history]
            
            plt.plot(train_metric, label=f'Train {metric}', marker='o', markersize=4)
            plt.plot(val_metric, label=f'Val {metric}', marker='o', markersize=4)
            plt.title(f'{metric.replace("_", " ").title()}\n{self.experiment_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir,
            'figures',
            f'metrics_plot_{self.experiment_name}.png'
        ), bbox_inches='tight', dpi=300)
        plt.close()

    def train(self, continue_training=False):
        """Train model with option to continue from checkpoint"""
        if continue_training:
            loaded = self.load_checkpoint()
            if loaded:
                print(f"Continuing training from epoch {self.start_epoch + 1}")
            else:
                print("No checkpoint found, starting fresh training")
                self.start_epoch = 0

        print(f"Training on device: {self.config.training_settings.device}")
        print(f"Experiment name: {self.experiment_name}")
        
        self.model = self.model.to(self.config.training_settings.device)

        for epoch in range(self.config.training_settings.num_epochs - 1):
            print(
                f"\nEpoch {epoch + 1}/{self.config.training_settings.num_epochs} - "
                f"{self.config.training_settings.experiment_name}")

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics, val_preds, val_labels = self.evaluate(self.val_loader)

            # Store losses
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])

            # Store metrics
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

            # Save checkpoints and plots every N epochs
            if (epoch + 1) % self.config.training_settings.checkpoint_freq == 0:
                # Save metrics
                self.save_metrics(train_metrics, epoch + 1, 'train')
                self.save_metrics(val_metrics, epoch + 1, 'val')

                # Save checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                self.save_checkpoint(epoch + 1, val_metrics['loss'], is_best)

                # Plot confusion matrix
                self.plot_confusion_matrix(val_labels, val_preds, epoch + 1)

                # Plot metrics
                self.plot_metrics()

            # Early stopping
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
        metrics_file = os.path.join(self.config.training_settings.save_results_dir, 'metrics',
                                    f'{phase}_metrics_epoch_{epoch}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
