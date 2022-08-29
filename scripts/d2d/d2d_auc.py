from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))
from os import path, makedirs
import torch
from utils import read_data, get_root_path, train_test_split, get_time, \
    gen_propagation_scores, redirect_output
from D2D import eval_D2D, eval_D2D_2, generate_D2D_features_from_propagation_scores
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from presets import experiments_20, experiments_50, experiments_0
import json
import argparse
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
from collections import defaultdict

def run(sys_args):
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    n_experiments = sys_args.n_experiments
    args = experiments_0

    if sys_args.device:
        device = torch.device("cuda:{}".format(sys_args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    args['data']['sources_filename'] = sources_filenmae_dict[sys_args.experiments_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[sys_args.experiments_type]
    args['data']['n_experiments'] = n_experiments
    print(json.dumps(args, indent=4))

    # data read and filtering
    rng = np.random.RandomState(args['data']['random_seed'])

    network, directed_interactions, sources, terminals, id_to_degree = \
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)
    n_experiments = len(sources)

    # merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list)

    d2d_2_probs_list, d2d_2_labels_list, d2d_probs_list, d2d_labels_list = [
        defaultdict(list) for x in range(4)]
    d2d_2_folds_stats, d2d_folds_stats = [], []

    for fold in range(n_folds):
        print('\n Evaluating Fold {} \n'.format(fold))
        train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'],
                                                                    len(directed_interactions_pairs_list),
                                                                    args['train']['train_val_test_split'],
                                                                    random_state=rng)  # feature generation
        d2d_train_indexes = np.concatenate([train_indexes, val_indexes])

        # d2d evaluation
        sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
        terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
        pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
        features, deconstructed_features = generate_D2D_features_from_propagation_scores(propagation_scores,
                                                                                         pairs_indexes,
                                                                                         sources_indexes,
                                                                                         terminals_indexes)
        d2d_results_dict, model = eval_D2D(features[d2d_train_indexes], features[test_indexes],
                                           directed_interactions_source_type[test_indexes])
        d2d_2_results_dict, model = eval_D2D_2(deconstructed_features[d2d_train_indexes],
                                               deconstructed_features[test_indexes],
                                               directed_interactions_source_type[test_indexes])
        d2d_folds_stats.append({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                                d2d_results_dict.items()})
        d2d_2_folds_stats.append({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                                  d2d_2_results_dict.items()})

        for key in d2d_results_dict.keys():
            d2d_probs_list[key].append(d2d_results_dict[key]['probs'][:, 1])
            d2d_labels_list[key].append(d2d_results_dict[key]['labels'])
            d2d_2_probs_list[key].append(d2d_2_results_dict[key]['probs'][:, 1])
            d2d_2_labels_list[key].append(d2d_2_results_dict[key]['labels'])

    final_stats =  {'d2d': {'folds_stats': d2d_folds_stats, 'final': {}, },
                   'd2d_2': {'folds_stats': d2d_2_folds_stats, 'final': {} }}

    d2d_2_probs_by_type, d2d_2_labels_by_type, d2d_probs_by_type, d2d_labels_by_type = [
        defaultdict(list) for x in range(4)]
    for source_type in d2d_results_dict.keys():
        d2d_probs_by_type[source_type] = np.hstack(d2d_probs_list[source_type])
        d2d_labels_by_type[source_type] = np.hstack(d2d_labels_list[source_type])
        d2d_2_probs_by_type[source_type] = np.hstack(d2d_2_probs_list[source_type])
        d2d_2_labels_by_type[source_type] = np.hstack(d2d_2_labels_list[source_type])
        d2d_acc = np.mean((np.array(d2d_probs_by_type[source_type]) > 0.5).astype(int) == d2d_labels_by_type[source_type])
        d2d_2_acc = np.mean((np.array(d2d_2_probs_by_type[source_type]) > 0.5).astype(int) == d2d_2_labels_by_type[source_type])

        d2d_precision, d2d_recall, d2d_thresholds = precision_recall_curve(d2d_labels_by_type[source_type],
                                                                           d2d_probs_by_type[source_type])

        d2d_auc = auc(d2d_recall, d2d_precision)
        if len(d2d_precision) == 2:
            d2d_auc = 0.5
            d2d_precision = [0.5, 0.5]

        d2d_precision_2, d2d_recall_2, d2d_thresholds_2 = precision_recall_curve(d2d_2_labels_by_type[source_type],
                                                                                 d2d_2_probs_by_type[source_type])
        d2d_2_auc = auc(d2d_recall_2, d2d_precision_2)
        if len(d2d_precision_2) == 2:
            d2d_2_auc = 0.5
            d2d_precision_2 = [0.5, 0.5]

        final_stats['d2d']['final'][source_type] = {'auc': d2d_auc,
                                                    'acc': d2d_acc,
                                                    'precision': list(d2d_precision),
                                                    'recall': list(d2d_recall)}
        final_stats['d2d_2']['final'][source_type] = {'auc':d2d_2_auc,
                                                      'acc': d2d_2_acc,
                                                      'precision': list(d2d_precision_2),
                                                      'recall': list(d2d_recall_2)
                                                      }

    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'results'), 'w') as f:
        json.dump(final_stats, f, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    n_folds = 5
    input_type = 'drug'
    load_prop = False
    save_prop = False
    n_exp = 5
    split = [0.66, 0.14, 0.2]
    interaction_type = sorted(['KPI', 'STKE', 'EGFR', 'E3', 'PDI'])
    prop_scores_filename = 'drug_KPI'

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str,
                        help='name of experiment type(drug, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int,
                        help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores', action='store_true', default=False,
                        help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores', action='store_true', default=False,
                        help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split', nargs=3, help='[train, val, test] sums to 1',
                        default=split, type=float)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', nargs='*', type=str,
                        help='KPI/STKE', default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)
    parser.add_argument('-f', '--n_folds', dest='n_folds', type=str,
                        help='Name of prop score file(save/load)', default=n_folds)
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)
    parser.add_argument('-d', dest='device', type=int, help='gpu number', default=None)
    args = parser.parse_args()
    args.directed_interactions_filename = sorted(args.directed_interactions_filename)
    args.prop_scores_filename = args.experiments_type + '_' + '_'.join(args.directed_interactions_filename) + '_{}'.format(args.n_experiments)

    # args.load_prop_scores =  True
    args.save_prop_scores = True
    run(args)
