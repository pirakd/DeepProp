from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))
from os import path, makedirs
import torch
from utils import read_data, get_root_path, train_test_split, get_time, \
    gen_propagation_scores, redirect_output
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from presets import experiments_20, experiments_50, experiments_0
import json
import argparse
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
from collections import defaultdict
from gene_name_translator.gene_translator import GeneTranslator
from Vinayagam import generate_vinyagam_feature, count_sp_edges, infer_vinayagam


def run(sys_args):
    gene_translator = GeneTranslator(verbosity=True)
    gene_translator.load_dictionary()

    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    n_experiments = sys_args.n_experiments
    args = experiments_0


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

    transcription_factors_df = pd.read_csv(path.join(get_root_path(), 'input', 'other', 'transcription_factors.tsv'), sep='\t')
    receptors_df = pd.read_csv(path.join(get_root_path(), 'input', 'other', 'receptors.tsv'), sep='\t')
    transcription_factors_df['DBD'] = transcription_factors_df['DBD'].str.split(';').str[0]
    receptors_df['Classification'] = receptors_df['Classification']
    all_transcription_factors = transcription_factors_df['Symbol'].to_list()
    all_receptors = receptors_df['Entrez_ID'].to_list()
    symbol_to_entrez = gene_translator.translate(all_transcription_factors, 'symbol', 'entrez_id')
    entrez_to_entrez = gene_translator.translate(all_receptors, 'entrez_id', 'entrez_id')

    transcription_factors_df['Symbol'] = transcription_factors_df['Symbol'].apply(lambda x: symbol_to_entrez.get(x, None))
    receptors_df['Entrez_ID'] = receptors_df['Entrez_ID'].apply(lambda x: entrez_to_entrez.get(x, None))
    transcription_factors_df.rename(columns={'Symbol':'Entrez_ID'}, inplace=True)
    transcription_factors_df.dropna(inplace=True)
    tf_groups = transcription_factors_df.set_index('Entrez_ID').to_dict()['DBD']
    receptor_groups = receptors_df.set_index('Entrez_ID').to_dict()['Classification']
    transcription_factors_df.to_dict()
    network['edge_score'] = 1-network['edge_score']
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')

    counts, grouped_counts = count_sp_edges(graph, receptor_groups, tf_groups)
    probs_list, labels_list, probs_list, labels_list = [
        defaultdict(list) for x in range(4)]
    folds_stats, folds_stats = [], []

    for fold in range(n_folds):
        print('\n Evaluating Fold {} \n'.format(fold))

        #evaluation
        train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'],
                                                                    len(directed_interactions_pairs_list),
                                                                    args['train']['train_val_test_split'],
                                                                    random_state=rng)  # feature generation
        train_indexes = np.concatenate([train_indexes, val_indexes])

        train_features, train_labels = generate_vinyagam_feature(graph, counts, grouped_counts,
                                                                 directed_interactions_pairs_list[train_indexes])
        test_features, test_labels = generate_vinyagam_feature(graph, counts, grouped_counts,
                                                               directed_interactions_pairs_list[test_indexes])
        results_dict, model = infer_vinayagam(train_features, train_labels, test_features, test_labels,
                                              directed_interactions_source_type[test_indexes])
        # merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])

        vinayagam_stats = ({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                            results_dict.items()})

        folds_stats.append({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                                results_dict.items()})


        for key in results_dict.keys():
            probs_list[key].append(results_dict[key]['probs'][:, 1])
            labels_list[key].append(results_dict[key]['labels'])
            probs_list[key].append(results_dict[key]['probs'][:, 1])
            labels_list[key].append(results_dict[key]['labels'])

    final_stats =  {'vinayagam': {'folds_stats': folds_stats, 'final': {}, }}

    probs_by_type, labels_by_type = [
        defaultdict(list) for x in range(2)]
    for source_type in results_dict.keys():
        probs_by_type[source_type] = np.hstack(probs_list[source_type])
        labels_by_type[source_type] = np.hstack(labels_list[source_type])

        acc = np.mean((np.array(probs_by_type[source_type]) > 0.5).astype(int) == labels_by_type[source_type])
        precision, recall, thresholds = precision_recall_curve(labels_by_type[source_type],
                                                                           probs_by_type[source_type])

        _auc = auc(recall, precision)
        if len(precision) == 2:
            _auc = 0.5
            precision = [0.5, 0.5]



        final_stats['vinayagam']['final'][source_type] = {'auc': _auc,
                                                    'acc': acc,
                                                    'precision': list(precision),
                                                    'recall': list(recall)}

    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'results'), 'w') as f:
        json.dump(final_stats, f, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    n_folds = 5
    input_type = 'drug'
    load_prop = False
    save_prop = False
    n_exp = 2
    split = [0.66, 0.14, 0.2]
    interaction_type = ['KPI', 'STKE']
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

    args.load_prop_scores =  True
    # args.save_prop_scores = True
    run(args)
