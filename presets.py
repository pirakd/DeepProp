experiments_20 = {
    'data':
        {'n_experiments': 20,
         'train_test_split': 0.8,
         'dataset_type': 'balanced_kpi'},
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [64, 32, 16],
         'classifier_layers': [128, 64],
         'pulling_func': 'mean',
         'exp_emb_size': 8},
    'train':
        {'intermediate_loss_weight': None,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 12,
         'test_batch_size': 32,
         'n_epochs':1000 ,
         'eval_interval': 2,
         'learning_rate': 1e-3,
         'n_evals_no_improvement': 25}}

experiments_50 = {
    'data':
        {'n_experiments': 50,
         'train_test_split': 0.8,
         'dataset_type': 'balanced_kpi'},
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [64, 32],
         'classifier_layers': [128, 64],
         'pulling_func': 'mean',
         'exp_emb_size': 12},
    'train':
        {'intermediate_loss_weight': 0.5,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 16,
         'test_batch_size': 32,
         'n_epochs':2000 ,
         'eval_interval': 2,
         'learning_rate': 5e-4,
         'n_evals_no_improvement': 25}}