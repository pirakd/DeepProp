from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import makedirs
from deep_learning.data_loaders import LightDataset
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import read_data, generate_raw_propagation_scores, log_results, get_root_path, save_propagation_score,\
    load_pickle, train_test_split, get_normalization_constants, get_loss_function
import torch
import numpy as np
import random

device = torch.device("cuda".format() if torch.cuda.is_available() else "cpu")
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder)
makedirs(output_file_path, exist_ok=True)

root_path = get_root_path()
input_file = path.join(root_path, 'input')
NETWORK_FILENAME = path.join(input_file, 'networks', "H_sapiens.net")
DIRECTED_INTERACTIONS_FILENAME = path.join(input_file, 'directed_interactions', "KPI_dataset")
SOURCES_FILENAME = path.join(input_file, 'priors', "drug_targets.txt")
TERMINALS_FILENAME = path.join(input_file, 'priors', "drug_expressions.txt")
# torch.set_default_dtype(torch.float64)

args = {
    'data':
        {'n_experiments': 2,
         'dataset_type': 'balanced_kpi',
         'load_scores': False,
         'scores_file_name': 'balanced_kpi_prop_scores',
         'random_seed': 0,
         'save_scores': False},
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
        {'intermediate_loss_weight': 0,
         'intermediate_loss_type' : 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.8, 0.2, 0],
         'test_batch_size': 32,
         'train_batch_size': 32,
         'n_epochs': 100,
         'eval_interval': 10,
         'learning_rate': 1e-3,
         'max_evals_no_imp': 10 },}

device = torch.device("cuda".format() if torch.cuda.is_available() else "cpu")
cmd_args = [int(arg) for arg in sys.argv[1:]]
if len(cmd_args) == 2:
    args['data']['n_experiments'] = cmd_args[1]
    device = torch.device("cuda:{}".format(cmd_args[0]) if torch.cuda.is_available() else "cpu")
    print('n_expriments: {}, device: {}'.format(args['data']['n_experiments'], device))

# data read
rng = np.random.RandomState(args['data']['random_seed'])
network, directed_interactions, sources, terminals =\
    read_data(NETWORK_FILENAME, DIRECTED_INTERACTIONS_FILENAME, SOURCES_FILENAME, TERMINALS_FILENAME,
              rng)

# filter experiments
experiments = set(sorted(list(sources.keys() & terminals.keys()))[:args['data']['n_experiments']])
sources = {name: genes for name, genes in sources.items() if name in experiments}
terminals = {name: genes for name, genes in terminals.items() if name in experiments}

directed_interactions_pairs_list = np.array(directed_interactions.index)
genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

if args['data']['load_scores']:
    scores_file_path = path.join(root_path, 'input', 'propagation_scores', args['data']['scores_file_name'])
    scores_dict = load_pickle(scores_file_path)
    propagation_scores = scores_dict['propagation_scores']
    row_id_to_idx, col_id_to_idx = scores_dict['row_id_to_idx'], scores_dict['col_id_to_idx']
    assert scores_dict['data_args']['random_seed'] == args['data']['random_seed'], 'random seed of loaded data does not much current one'
else:
    propagation_scores, row_id_to_idx, col_id_to_idx = generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep,
                                                                  args['propagation']['alpha'], args['propagation']['n_iterations'],
                                                                  args['propagation']['eps'])
    if args['data']['save_scores']:
        save_propagation_score(propagation_scores, row_id_to_idx, col_id_to_idx, args['propagation'], args['data'],
                               'balanced_kpi_prop_scores')

sources_indexes = [[row_id_to_idx[id] for id in set] for set in sources.values()]
terminals_indexes = [[row_id_to_idx[id] for id in set] for set in terminals.values()]
pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
normalization_constants = get_normalization_constants(pairs_indexes, sources_indexes, terminals_indexes, propagation_scores)

train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                            random_state=rng)

train_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[train_indexes], sources, terminals, normalization_constants)
val_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[val_indexes], sources, terminals, normalization_constants)

feature_extractor_layers = [[128,64], [64, 32, 16], [64,32], [16, 8]]
classifier_layers = [[128, 64, 32], [128, 64], [32, 16, 8],  [32, 16], [64, 32, 16]]
learning_rate = [1e-4, 5e-4, 1e-3, 5e-3, 5e-5]
intermediate_loss_weight = [0.25, 0.5, 0.75, 0.9]
train_batch_size = [16, 32, 64]
exp_embedding_size = [4, 8, 12, 16]
intermediate_loss_type_list = ['BCE', 'FOCAL']
focal_gamma = [0.5, 1, 2, 3]

while True:
    args['model']['feature_extractor_layers'] = random.choice(feature_extractor_layers)
    args['model']['classifier_layers'] = random.choice(classifier_layers)
    args['train']['learning_rate'] = random.choice(learning_rate)
    args['train']['intermediate_loss_weight'] = random.choice(intermediate_loss_weight)
    args['model']['exp_emb_size'] = random.choice(exp_embedding_size)
    args['train']['train_batch_size'] = random.choice(train_batch_size)
    args['train']['focal_gamma'] = random.choice(focal_gamma)
    args['train']['intermediate_loss_type'] = np.random.choice(intermediate_loss_type_list, p=[0.2, 0.8])
    intermediate_loss_type =  get_loss_function(args['train']['intermediate_loss_type'],
                                                                focal_gamma=args['train']['focal_gamma'])

    train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=True)

    deep_prop_model = DeepProp(args['model']['feature_extractor_layers'], args['model']['pulling_func'],
                               args['model']['classifier_layers'], args['data']['n_experiments'],
                               args['model']['exp_emb_size'])
    model = DeepPropClassifier(deep_prop_model, args['data']['n_experiments'])
    model.to(device=device)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])

    trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(),
                                intermediate_criteria=intermediate_loss_type,
                                intermediate_loss_weight=args['train']['intermediate_loss_weight'], device=device,
                                optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'])

    train_stats, best_model =\
        trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model, max_evals_no_improvement=args['train']['max_evals_no_imp'])

    results_dict = {'args': args, 'train_stats': train_stats}

    log_results(results_dict, output_file_path)
