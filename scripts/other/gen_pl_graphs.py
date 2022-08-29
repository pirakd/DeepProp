from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))
from os import makedirs
import json
from utils import read_data, log_results, get_time, get_root_path, train_test_split, get_loss_function,\
    gen_propagation_scores, redirect_output, get_optimizer, save_model
import numpy as np
from presets import experiments_20, experiments_50, experiments_0, experiments_all_datasets
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import argparse
import pandas as pd
import copy
import networkx as nx

def run(sys_args):
    directions_predictions_files = {'breast': '08_05_2022__14_11_30',
                                    'ovary': '08_05_2022__13_28_47',
                                    'AML': '08_05_2022__13_27_08',
                                    'colon': '08_05_2022__14_08_28',
                                    'd2d_breast': '08_05_2022__14_27_16',
                                    'd2d_AML': '08_05_2022__15_58_37',
                                    'd2d_colon': '08_05_2022__14_26_35',
                                     'd2d_ovary': '08_05_2022__14_25_49'}
    directions_predictions_files = {'breast': 'breast',
                                    'ovary': 'ovary',
                                    'AML': 'AML',
                                    'colon': 'colon',
                                    'd2d_breast': 'd2d_breast',
                                    'd2d_AML': 'd2d_AML',
                                    'd2d_colon': 'd2d_colon',
                                    'd2d_ovary': 'd2d_ovary'}
    root_path = get_root_path()
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
    # interaction_type = sorted(['KEGG'])
    predction_folder = path.join(root_path, 'input', 'predicted_interactions')
    consensus_threshold = 3 / 4
    ratio_threshold = 1.1
    prediction_types = ['ovary', 'AML', 'colon', 'breast']
    # prediction_types = ['d2d_ovary', 'd2d_colon', 'd2d_breast', 'd2d_AML']

    args_dict = {'interaction_types': interaction_type,
                 'ratio_threshold': ratio_threshold,
                 'consensus_threshold': consensus_threshold,
                 'output_file_path': output_file_path,
                 'prediction_types': prediction_types}
    results_dict = {}
    results_dict['args'] = args_dict
    results_dict['results'] = {}
    args = experiments_0
    args['data']['directed_interactions_filename'] = interaction_type
    rng = np.random.RandomState(args['data']['random_seed'])

    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))
    n_experiments = sys_args.n_experiments
    args = experiments_0

    args['data']['directed_interactions_filename'] = interaction_type
    args['data']['sources_filename'] = sources_filenmae_dict[sys_args.experiments_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[sys_args.experiments_type]
    args['data']['n_experiments']  = n_experiments
    rng = np.random.RandomState(args['data']['random_seed'])
    print(json.dumps(args, indent=4))

    # data read
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)

    network = network.reset_index().drop(columns='source')
    network_2 = copy.copy(network)
    network_2[0] = network[1]
    network_2[1] = network[0]
    merged_network = pd.concat([network, network_2])
    merged_network = merged_network.set_index([0,1])
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1)


    n_interactions = len(list(directed_interactions.index))
    # generating datasets
    directed_interactions_set = set(directed_interactions.index)

    predicted_edges = {}
    for name in prediction_types:
        prediction_file_path = path.join(predction_folder, directions_predictions_files[name], 'directed_network')
        prediction = pd.read_csv(prediction_file_path, sep='\t', index_col=[0, 1])
        predictions_dict = prediction[['direction_prob']].to_dict()['direction_prob']
        predicted_edges[name] = [x for x in predictions_dict.keys() if
                                 (predictions_dict[x] / (predictions_dict[(x[1], x[0])] + 1e-12) > ratio_threshold)]

    all_edges = list(set.union(*[set(edges) for edges in predicted_edges.values()]))
    edge_to_idx = {edge: idx for idx, edge in enumerate(all_edges)}
    idx_to_edge = {xx: x for x, xx in edge_to_idx.items()}
    consensus_array = np.zeros((len(all_edges), len(predicted_edges.keys())))
    for n, name in enumerate(predicted_edges.keys()):
        for edge in predicted_edges[name]:
            consensus_array[edge_to_idx[edge], n] = 1

    consensus_idxs = np.nonzero(np.mean(consensus_array, axis=1) >= consensus_threshold)[0]
    consensus_predictions = [idx_to_edge[idx] for idx in consensus_idxs]
    consensus_predictions = set(consensus_predictions).union(directed_interactions_set)
    flipped_predictions = [(x[1], x[0]) for x in consensus_predictions]
    directed_network = merged_network.drop(flipped_predictions)

    merged_network.to_csv(path.join(output_file_path, 'undirected_h_sapiens.net'), sep='\t')
    directed_network.to_csv(path.join(output_file_path,'directed_h_sapiens.net'), sep='\t')
    a=1
if __name__ == '__main__':
    n_models = 1
    input_type = 'drug'
    n_exp = 2
    split = [0.8, 0.2, 0]
    interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
    interaction_type = sorted(['KEGG'])
    device = None
    prop_scores_filename = input_type + '_' + '_'.join(interaction_type) + '_{}'.format(n_exp)

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
    parser.add_argument('--n_models', dest='n_models', type=int,
                        help='number_of_models_to_train', default=n_models)
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)
    args = parser.parse_args()
    args.prop_scores_filename = args.experiments_type + '_' + '_'.join(args.directed_interactions_filename) + '_{}'.format(args.n_experiments)
    # args.load_prop_scores = True
    # args.save_prop_scores = True
    run(args)

