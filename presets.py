experiments_20 = {
    'data':
        {'n_experiments': 20,
         'max_set_size': 500,
         'network_filename': 'H_sapiens.net', #'S_cerevisiae.net'
         'directed_interactions_filename': 'KPI_dataset',
         'sources_filename': 'drug_targets.txt',
         'terminals_filename': 'drug_expressions.txt',
         'load_prop_scores': False,
         'save_prop_scores': False,
         'balance_dataset': True,
         'prop_scores_filename': 'balanced_kpi_prop_scores',
         'random_seed': 0,
         'normalization_method': 'power',   # Standard, Power
         'split_type': 'normal'}, # 'regular'/harsh
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [64, 32],
         'classifier_layers': [64, 32],
         'pulling_func': 'mean',
         'exp_emb_size': 4,
         'feature_extractor_dropout': 0,
         'classifier_dropout': 0,
         'pair_degree_feature': 0,
         'share_source_terminal_weights': False,
         },
    'train':
        {'intermediate_loss_weight': 0,
         'intermediate_loss_type': 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.66, 0.14, 0.2],  # sum([train, val, test])=1
         'train_batch_size': 32,
         'test_batch_size': 32,
         'n_epochs': 100,
         'eval_interval': 3,
         'learning_rate': 1e-3,
         'max_evals_no_imp': 3,
         'optimizer' : 'ADAM' # ADAM/WADAM
         }}


experiments_50 = {
    'data':
        {'n_experiments': 50,
         'max_set_size': 500,
         'network_filename': 'H_sapiens.net',
         'directed_interactions_filename': 'KPI_dataset',
         'sources_filename': 'drug_targets.txt',
         'terminals_filename': 'drug_expressions.txt',
         'load_prop_scores': True,
         'save_prop_scores': False,
         'prop_scores_filename': 'balanced_kpi_prop_scores',
         'random_seed': 0,
         'normalization_method': 'standard'
         },
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [128, 64],
         'classifier_layers': [128, 64],
         'pulling_func': 'mean',
         'exp_emb_size': 12,
         'feature_extractor_dropout': 0,
         'classifier_dropout': 0,
         'pair_degree_feature': 0,
         'share_source_terminal_weights': False,
         },
    'train':
        {'intermediate_loss_weight': 0.5,
         'intermediate_loss_type': 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 32,
         'test_batch_size': 32,
         'n_epochs': 4,
         'eval_interval': 2,
         'learning_rate': 5e-4,
         'max_evals_no_imp': 3,
         'optimizer' : 'ADAM' # ADAM/WADAM
         }}


experiments_0 = {
    'data':
        {'n_experiments': 0,
         'max_set_size': 500,
         'network_filename': 'H_sapiens.net',
         'directed_interactions_filename': ['KPI'],
         'sources_filename': 'targets_drug',
         'terminals_filename': 'expressions_drug',
         'load_prop_scores': True,
         'save_prop_scores': False,
         'balance_dataset': True,
         'prop_scores_filename': 'drug_KPI_0',
         'random_seed': 0,
         'normalization_method': 'power',   # Standard, Power
         'split_type': 'normal'}, # 'regular'/harsh
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [128, 64],
         'classifier_layers': [64],
         'pulling_func': 'mean',
         'exp_emb_size': 16,
         'feature_extractor_dropout': 0,
         'classifier_dropout': 0,
         'pair_degree_feature': 0,
         'share_source_terminal_weights': False,
         },
    'train':
        {'intermediate_loss_weight': 0.5,
         'intermediate_loss_type': 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 4,
         'test_batch_size': 32,
         'n_epochs': 4,
         'eval_interval': 2,
         'learning_rate': 1e-3,
         'max_evals_no_imp': 3,
         'optimizer': 'ADAMW'  # ADAM/WADAM
         }}