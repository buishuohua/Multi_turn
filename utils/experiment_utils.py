def create_experiment_name(config, is_model=True):
    """Create experiment name based on configuration

    Args:
        config: The experiment configuration
        is_model: If True, creates a simpler name for model-specific paths
    """
    components = []

    # 1. Model Type (e.g., "BL" for BiLSTM)
    model_map = {'BiLSTM': 'BL', 'LSTM': 'L'}
    model = model_map[config.model_selection.model_type]
    components.append(model)

    # 2. Embedding Info (e.g., "BB" for BERT_base)
    emb_map = {
        'BERT_base_uncased': 'BB',
        'BERT_large_uncased': 'BL',
        'BERT_base_multilingual_cased': 'BM',
        'XLM_roberta_base': 'XB',
        'XLM_roberta_large': 'XL',
        'T5_small': 'TS',
        'T5_base': 'TB',
        'T5_large': 'TL',
    }
    components.append(emb_map[config.model_settings.embedding_type])

    # For model-specific paths, use simpler name
    if is_model:
        return '_'.join(components)

    # 3. Architecture and Sequence Length (e.g., "L4H384M256")
    arch = f"L{config.model_settings.num_layers}"
    arch += f"H{config.model_settings.init_hidden_dim}"
    arch += f"M{config.tokenizer_settings.max_length}"
    components.append(arch)

    # 4. Training Config, Loss and Activation (e.g., "W5e4FT2e5CE-ReLU")
    opt_map = {'adam': 'A', 'adamw': 'W', 'sgd': 'S'}
    loss_map = {
        'cross_entropy': 'CE',
        'bce_with_logits': 'BCE',
        'weighted_cross_entropy': 'WCE',
        'focal': 'FL',
        'label_smoothing_ce': 'LSE',
        'kl_div': 'KL',
        'mse': 'MSE',
        'mae': 'MAE'
    }
    act_map = {
        'relu': 'ReLU',
        'gelu': 'GeLU',
        'tanh': 'Tanh',
        'sigmoid': 'Sig'
    }

    opt = opt_map[config.training_settings.optimizer_type]
    lr = f"{config.training_settings.learning_rate:.0e}".replace('e-0', 'e-')

    # Add fine-tuning learning rate if enabled
    ft_lr = ""
    if config.model_settings.fine_tune_embedding:
        ft_lr = f"FT{config.model_settings.fine_tune_lr:.0e}".replace(
            'e-0', 'e-')

    loss = loss_map[config.model_settings.loss]
    act = act_map[config.model_settings.activation]
    components.append(f"{opt}{lr}{ft_lr}{loss}-{act}")

    # 5. Imbalanced Strategy (e.g., "SMT" for SMOTE)
    imb_map = {
        'none': '',
        'weighted_sampler': 'WS',
        'random_oversample': 'ROS',
        'smote': 'SMT',
        'borderline1': 'BL1',
        'borderline2': 'BL2',
        'svm_smote': 'SVMS',
        'adasyn': 'ADA',
        'random_undersample': 'RUS',
        'tomek': 'TMK',
        'edited_nearest_neighbors': 'ENN',
        'cluster_centroids': 'CC',
        'near_miss': 'NM',
        'instance_hardness_threshold': 'IHT'
    }
    if config.data_settings.imbalanced_strategy != 'none':
        components.append(imb_map[config.data_settings.imbalanced_strategy])

    # NEW: 6. Attention Position (e.g., "ATN-EIO")
    if config.model_settings.use_attention:
        pos_map = {
            'embedding': 'E',
            'inter_lstm': 'I',
            'output': 'O'
        }
        pos_str = ''.join(pos_map[pos] for pos in sorted(
            config.model_settings.attention_positions))
        components.append(f"ATN-{pos_str}")

    # NEW: 7. Loading Strategy (e.g., "LD-PL20")
    if config.model_settings.fine_tune_embedding:
        strategy_map = {
            'periodic': 'P',
            'plateau': 'L',
        }
        strategy_str = ''.join(strategy_map[s] for s in sorted(
            config.model_settings.fine_tune_loading_strategies))
        reload_freq = f"{config.model_settings.fine_tune_reload_freq}"
        components.append(f"LD-{strategy_str}{reload_freq}")

    # 8. Key Features as Flags (e.g., "ARTG")
    features = []
    if config.model_settings.use_attention:
        features.append('A')  # Attention
    if config.model_settings.use_res_net:
        features.append('R')  # Residual
    if config.model_settings.use_layer_norm:
        features.append('N')  # Normalization
    if config.model_settings.fine_tune_embedding:
        ft_map = {'full': 'F', 'last_n': 'L', 'gradual': 'G', 'selective': 'S'}
        features.append(
            f"T{ft_map.get(config.model_settings.fine_tune_mode, 'X')}")

    if features:
        components.append(''.join(features))

    return '_'.join(components)
