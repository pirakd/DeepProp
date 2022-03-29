experiments_20 = {
    'data':
        {'n_experiments': 20,
         'max_set_size': 400,
         'network_filename': 'H_sapiens.net',
         'directed_interactions_filename': 'KPI_dataset',
         'sources_filename': 'drug_targets.txt',
         'terminals_filename': 'drug_expressions.txt',
         'load_prop_scores': False,
         'save_prop_scores': False,
         'prop_scores_filename': 'balanced_kpi_prop_scores',
         'random_seed': 0,
         'normalization_method': 'standard'},  # Standard, Power
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [64, 32, 16],
         'classifier_layers': [32, 16],
         'pulling_func': 'mean',
         'exp_emb_size': 4,
         'feature_extractor_dropout': 0,
         'classifier_dropout': 0,
         'pair_degree_feature': 1
         },
    'train':
        {'intermediate_loss_weight': 0.5,
         'intermediate_loss_type': 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 4,
         'test_batch_size': 32,
         'n_epochs':50 ,
         'eval_interval': 10,
         'learning_rate': 1e-3,
         'max_evals_no_imp': 25}}

experiments_50 = {
    'data':
        {'n_experiments': 50,
         'max_set_size': 400,
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
        {'feature_extractor_layers': [64, 32],
         'classifier_layers': [128, 64],
         'pulling_func': 'mean',
         'exp_emb_size': 12,
         'feature_extractor_dropout': 0,
         'classifier_dropout': 0,
         'pair_degree_feature': 0
         },
    'train':
        {'intermediate_loss_weight': 0.5,
         'intermediate_loss_type': 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 16,
         'test_batch_size': 32,
         'n_epochs': 1000 ,
         'eval_interval': 2,
         'learning_rate': 5e-4,
         'max_evals_no_imp': 25,

         }}


experiments_all = {
    'data':
        {'n_experiments': 'all',
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
        {'feature_extractor_layers': [128, 64, 32],
         'classifier_layers': [128, 64],
         'pulling_func': 'mean',
         'exp_emb_size': 20,
         'feature_extractor_dropout': 0,
         'classifier_dropout': 0,
         'pair_degree_feature': 0
         },
    'train':
        {'intermediate_loss_weight': 0.75,
         'intermediate_loss_type': 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 8,
         'test_batch_size': 8,
         'n_epochs': 2000,
         'eval_interval': 3,
         'learning_rate': 5e-3,
         'max_evals_no_imp': 25,
         }}
