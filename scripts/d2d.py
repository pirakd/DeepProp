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
    get_root_path, save_propagation_score, load_pickle, train_test_split, get_normalization_constants
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from D2D import generate_D2D_features_from_propagation_scores, eval_D2D, eval_D2D_2
device = torch.device("cuda".format() if torch.cuda.is_available() else "cpu")
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder)
makedirs(output_file_path, exist_ok=True)

root_path = get_root_path()
input_file = path.join(root_path, 'input')
NETWORK_FILENAME = path.join(input_file, 'networks', "H_sapiens.net")
DIRECTED_INTERACTIONS_FILENAME = path.join(input_file, 'directed_interactions', "KPI_dataset")
SOURCES_FILENAME = path.join(input_file, 'priors', "mutations_AML")
TERMINALS_FILENAME = path.join(input_file, 'priors', "gene_expression_AML")
# torch.set_default_dtype(torch.float64)
n_experiments = 551
args = {
    'data':
        {'n_experiments': 'all',
         'max_set_size': 500,
         'train_test_split': 0.8,
         'dataset_type': 'balanced_kpi',
         'scores_filename': 'AML',
         'random_seed': 0,
         'load_prop_scores': True,
         'save_prop_scores': False},
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
         'train_val_test_split': [0.8, 0, 0.2],
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
rng = np.random.RandomState(args['data']['random_seed'])

# data read
network, directed_interactions, sources, terminals =\
    read_data(NETWORK_FILENAME, DIRECTED_INTERACTIONS_FILENAME, SOURCES_FILENAME, TERMINALS_FILENAME,
              args['data']['n_experiments'], args['data']['max_set_size'], rng)

directed_interactions_pairs_list = np.array(directed_interactions.index)
genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

if args['data']['load_prop_scores']:
    scores_file_path = path.join(root_path, 'input', 'propagation_scores', args['data']['scores_filename'])
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
                               args['propagation'], args['data'], args['data']['scores_filename'])


a=1
train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                            random_state=rng)

features, deconstructed_features= generate_D2D_features_from_propagation_scores(propagation_scores, pairs_indexes, sources_indexes, terminals_indexes)
d2d_probs, d2d_labels = eval_D2D(features[train_indexes], features[test_indexes])
d2d_precision, d2d_recall, d2d_thresholds = precision_recall_curve(d2d_labels, d2d_probs[:, 1])
d2d_auc = auc(d2d_recall, d2d_precision)

d2d_probs_2, d2d_labels_2 = eval_D2D_2(deconstructed_features[train_indexes], deconstructed_features[test_indexes])
d2d_precision_2, d2d_recall_2, d2d_thresholds_2 = precision_recall_curve(d2d_labels_2, d2d_probs_2[:, 1])
d2d_auc_2 = auc(d2d_recall_2, d2d_precision_2)

print('D2D: {:.4f}, D2D_2:{:.4f}'.format(d2d_auc, d2d_auc_2))
