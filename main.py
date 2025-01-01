from config.Experiment_Config import ExperimentConfig
from exp.trainer import Trainer
import argparse
import os
import json


def config_set():
    config = ExperimentConfig.get_default_config()
    config.model_settings.embedding_type = 'XLM_roberta_large'
    config.training_settings.num_epochs = 1000
    config.training_settings.batch_size = 32
    config.tokenizer_settings.max_length = 256
    config.model_settings.weight_init = 'kaiming_normal'
    config.data_settings.imbalanced_strategy = 'weighted_sampler'
    # config.data_settings.weighted_sampler_alpha = 0.5
    config.model_settings.activation = 'gelu'
    config.training_settings.early_stopping_patience = 100
    config.training_settings.task_type = 'Multi'
    config.model_settings.num_layers = 14
    config.model_settings.custom_hidden_dims = [
        1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 64, 64, 64, 64]
    config.training_settings.learning_rate = 5e-5
    config.model_settings.fine_tune_lr = 2e-5
    config.model_settings.use_res_net = True
    config.training_settings.gradient_clip = 1.0
    config.training_settings.continue_training = True
    config.model_settings.use_layer_norm = True
    config.model_settings.attention_temperature = 1.0
    config.model_settings.attention_positions = ['embedding', 'inter_lstm']
    config.model_settings.use_attention = True
    config.model_settings.fine_tune_embedding = True
    # config.model_settings.fine_tune_loading_strategies = [
    #     'plateau', 'ensemble']
    # config.model_settings.fine_tune_mode = 'gradual'
    config.model_settings.num_frozen_layers = 8
    config.model_settings.dropout_rate = 0.2
    config.model_settings.gradual_unfreeze_epochs = 100
    config.model_settings.use_discriminative_lr = True
    config.model_settings.bidirectional = True
    return config


def save_config(config, experiment_name):
    """Save configuration to a JSON file"""
    config_dir = os.path.join('saved_models', config.training_settings.task_type,
                              config.data_settings.which, experiment_name, 'config')
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, 'model_config.json')
    config_dict = {
        'model_settings': config.model_settings.to_dict(),
        'training_settings': vars(config.training_settings),
        'data_settings': vars(config.data_settings),
        'tokenizer_settings': vars(config.tokenizer_settings)
    }

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Saved configuration to {config_path}")


def load_config(experiment_name):
    """Load configuration from saved JSON file"""
    config = ExperimentConfig.get_default_config()
    config_dir = os.path.join('saved_models', config.training_settings.task_type,
                              config.data_settings.which, experiment_name, 'config')
    config_path = os.path.join(config_dir, 'model_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No saved configuration found at {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Update config with saved settings
    for key, value in config_dict['model_settings'].items():
        setattr(config.model_settings, key, value)
    for key, value in config_dict['training_settings'].items():
        setattr(config.training_settings, key, value)
    for key, value in config_dict['data_settings'].items():
        setattr(config.data_settings, key, value)
    for key, value in config_dict['tokenizer_settings'].items():
        setattr(config.tokenizer_settings, key, value)

    return config


def train():
    """Training function"""
    config = config_set()
    trainer = Trainer(config)
    save_config(config, trainer.experiment_name)
    trainer.train()
    return trainer


def test(trainer=None):
    """
    Testing function that can work independently or with a provided trainer
    Args:
        trainer: Optional[Trainer], existing trainer instance
    """
    if trainer is None:
        # Load the same configuration used for training
        try:
            config = load_config(trainer.experiment_name)
            print("Loaded saved configuration for testing")
        except FileNotFoundError:
            print("No saved configuration found, using default config")
            config = config_set()

        trainer = Trainer(config)
        print("\n" + "="*50)
        print("Loading Best Model for Testing...")
        print("="*50)

    test_metrics, test_labels, test_preds = trainer.evaluate(
        trainer.test_loader, phase='test')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Train or test the model')
    # parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'],
    #                     default='both', help='Run mode: train, test, or both')
    # args = parser.parse_args()

    # if args.mode == 'train':
    #     train()
    # elif args.mode == 'test':
    #     test()
    # else:  # both
    #     trainer = train()
    #     test(trainer)

    trainer = train()
    test(trainer)
