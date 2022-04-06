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
    get_root_path, save_propagation_score, load_pickle, train_test_split, gen_propagation_scores, get_time
import torch
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import argparse
from D2D import generate_D2D_features_from_propagation_scores, eval_D2D, eval_D2D_2
from presets import experiments_20

def run(sys_args):
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    n_experiments = sys_args.n_experiments
    args = experiments_20

    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    args['data']['sources_filename'] = sources_filenmae_dict[sys_args.experiments_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[sys_args.experiments_type]
    args['data']['n_experiments']  = n_experiments
    rng = np.random.RandomState(args['data']['random_seed'])

    # data read
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng, translate_genes=False)

    directed_interactions_pairs_list = np.array(directed_interactions.index)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list)

    train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                                random_state=rng)
    train_indexes = np.sort(np.concatenate([train_indexes, val_indexes]))
    sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
    terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]

    features, deconstructed_features = generate_D2D_features_from_propagation_scores(propagation_scores, pairs_indexes, sources_indexes, terminals_indexes)
    directed_interactions_source_type = np.array(directed_interactions.source)

    d2d_results_dict = eval_D2D(features[train_indexes], features[test_indexes])
    d2d_precision, d2d_recall, d2d_thresholds = precision_recall_curve(d2d_results_dict['overall']['labels'], d2d_results_dict['overall']['probs'][:, 1])
    d2d_auc = auc(d2d_recall, d2d_precision)

    d2d_2_results_dict = eval_D2D_2(deconstructed_features[train_indexes], deconstructed_features[test_indexes])
    d2d_precision_2, d2d_recall_2, d2d_thresholds_2 = precision_recall_curve(d2d_2_results_dict['overall']['labels'], d2d_2_results_dict['overall']['probs'][:, 1])
    d2d_auc_2 = auc(d2d_recall_2, d2d_precision_2)

    d2d_probs, d2d_probs_2 = np.array(d2d_2_results_dict['overall']['probs']), np.array(d2d_2_results_dict['overall']['probs'])
    d2d_labels, d2d_labels_2 = np.array(d2d_results_dict['overall']['labels']), np.array(d2d_2_results_dict['overall']['labels'])

    results_by_source_type = {}
    unique_source_type = np.unique(directed_interactions_source_type)
    for source_type in unique_source_type:
        doubled_test_indexes = np.concatenate([test_indexes, test_indexes])
        type_d2d_probs = d2d_probs[directed_interactions_source_type[doubled_test_indexes] == source_type][:, 1]
        type_labels = d2d_labels[directed_interactions_source_type[doubled_test_indexes] == source_type]
        type_d2d_precision, type_d2d_recall, type_d2d_thresholds = precision_recall_curve(type_labels, type_d2d_probs)
        type_d2d_auc = auc(type_d2d_recall, type_d2d_precision)

        type_d2d_probs_2 = d2d_probs_2[directed_interactions_source_type[doubled_test_indexes] == source_type][:, 1]
        type_labels_2 = d2d_labels_2[directed_interactions_source_type[doubled_test_indexes] == source_type]
        type_d2d_precision_2, type_d2d_recall_2, type_d2d_thresholds_2 = precision_recall_curve(type_labels_2, type_d2d_probs_2)
        type_d2d_auc_2 = auc(type_d2d_recall_2, type_d2d_precision_2)
        results_by_source_type[source_type] = {'d2d_auc': type_d2d_auc, 'd2d_reconstructed': type_d2d_auc_2}
    print('D2D: {:.4f}, D2D_2:{:.4f}'.format(d2d_auc, d2d_auc_2))

if __name__ == '__main__':
    input_type = 'yeast'
    load_prop = False
    save_prop = False
    n_exp = 20
    split = [0.66, 0.14, 0.2]
    interaction_type = ['yeast_KPI']
    # interaction_type = ['STKE']
    device = 'cpu'
    prop_scores_filename = 'yeast_KPI'

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str, help='name of experiment type(drug, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int, help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores',  action='store_true', default=False, help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores',  action='store_true', default=False, help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split',  nargs=3, help='[train, val, test] sums to 1', default=split, type=float)
    parser.add_argument('-d', '--device', type=str, help='cpu or gpu number',  default=device)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', nargs='*', type=str,
                        help='KPI/STKE', default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)

    args = parser.parse_args()
    # args.load_prop_scores = True
    # args.save_prop_scores = True
    run(args)
