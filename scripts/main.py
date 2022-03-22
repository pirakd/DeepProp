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
from utils import read_data, generate_raw_propagation_scores,\
    get_root_path, save_propagation_score, load_pickle, train_test_split, get_normalization_constants, get_loss_function
import torch
import numpy as np

device = torch.device("cuda".format() if torch.cuda.is_available() else "cpu")
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder)
makedirs(output_file_path, exist_ok=True)

root_path = get_root_path()


args = {
    'data':
        {'n_experiments': 10,
         'max_set_size': 500,
         'network_filename': 'H_sapiens.net',
         'directed_interactions_filename': 'STKE_dataset',
         'sources_filename': 'targets_drug',
         'terminals_filename': 'expressions_drug',
         'load_prop_scores': False,
         'save_prop_scores': False,
         'prop_scores_filename': 'AML',
         'random_seed': 0},
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [64, 32],
         'classifier_layers': [32, 16],
         'pulling_func': 'mean',
         'exp_emb_size': 4},
    'train':
        {'intermediate_loss_weight': 0.9,
         'intermediate_loss_type': 'BCE',
         'focal_gamma': 1,
         'train_val_test_split': [0.8, 0.1, 0.1], # sum([train, val, test])=1
         'train_batch_size': 16,
         'test_batch_size': 32,
         'n_epochs':2000 ,
         'eval_interval': 20,
         'learning_rate': 5e-4,
         'max_evals_no_imp': 25}}

device = torch.device("cuda".format() if torch.cuda.is_available() else "cpu")
cmd_args = [int(arg) for arg in sys.argv[1:]]
if len(cmd_args) == 2:
    args['data']['n_experiments'] = cmd_args[1]
    device = torch.device("cuda:{}".format(cmd_args[0]) if torch.cuda.is_available() else "cpu")
    print('n_expriments: {}, device: {}'.format(args['data']['n_experiments'], device))
rng = np.random.RandomState(args['data']['random_seed'])

# data read
network, directed_interactions, sources, terminals =\
    read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
              args['data']['sources_filename'], args['data']['terminals_filename'],
              args['data']['n_experiments'], args['data']['max_set_size'], rng)


# filter experiments
n_experiments = len(sources.keys())
directed_interactions_pairs_list = np.array(directed_interactions.index)
genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

if args['data']['load_prop_scores']:
    scores_file_path = path.join(root_path, 'input', 'propagation_scores', args['data']['prop_scores_filename'])
    scores_dict = load_pickle(scores_file_path)
    propagation_scores = scores_dict['propagation_scores']
    row_id_to_idx, col_id_to_idx = scores_dict['row_id_to_idx'], scores_dict['col_id_to_idx']
    normalization_constants_dict = scores_dict['normalization_constants']
    sources_indexes = [[row_id_to_idx[id] for id in set] for set in sources.values()]
    terminals_indexes = [[row_id_to_idx[id] for id in set] for set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
    assert scores_dict['data_args']['random_seed'] == args['data']['random_seed'], 'random seed of loaded data does not much current one'
else:
    propagation_scores, row_id_to_idx, col_id_to_idx = generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep,
                                                                  args['propagation']['alpha'], args['propagation']['n_iterations'],
                                                                  args['propagation']['eps'])
    sources_indexes = [[row_id_to_idx[id] for id in set] for set in sources.values()]
    terminals_indexes = [[row_id_to_idx[id] for id in set] for set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
    normalization_constants_dict = get_normalization_constants(pairs_indexes, sources_indexes, terminals_indexes,
                                                               propagation_scores)
    if args['data']['save_prop_scores']:
        save_propagation_score(propagation_scores, normalization_constants_dict, row_id_to_idx, col_id_to_idx,
                               args['propagation'], args['data'], 'balanced_kpi_prop_scores')


train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                            random_state=rng)
train_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[train_indexes], sources, terminals, normalization_constants_dict)
train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True, pin_memory=True)
val_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[val_indexes], sources, terminals, normalization_constants_dict)
val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=True)
test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[test_indexes], sources, terminals, normalization_constants_dict)
test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=True, )

deep_prop_model = DeepProp(args['model']['feature_extractor_layers'], args['model']['pulling_func'],
                           args['model']['classifier_layers'], n_experiments,
                           args['model']['exp_emb_size'])

model = DeepPropClassifier(deep_prop_model)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])
intermediate_loss_type = get_loss_function(args['train']['intermediate_loss_type'],
                                           focal_gamma=args['train']['focal_gamma'])
trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(), intermediate_criteria=intermediate_loss_type,
                            intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                            optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'], device='cpu')
train_stats, best_model = trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model,
                                        max_evals_no_improvement=args['train']['max_evals_no_imp'])
avg_eval_loss, avg_eval_intermediate_loss, avg_eval_classifier_loss, eval_acc, mean_auc, precision, recall = \
    trainer.eval(best_model, test_loader)
print('Test PR-AUC: {:.2f}'.format(mean_auc))
