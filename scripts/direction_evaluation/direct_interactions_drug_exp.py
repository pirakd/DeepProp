import pandas as pd
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from utils import read_data, get_root_path
import numpy as np
from utils import read_network, propagate_directed_network
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict

# read the network
# understand directionality
# tweak edges according to predictions
from presets import experiments_all
experiments_type = 'drug'
interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
root_path = get_root_path()
predction_folder = path.join(root_path, 'input', 'predicted_interactions')
direction_prob_threshold = 0.5
consensus_threshold = 3/4
ratio_threshold = 1.2
prediction_names = ['AML', 'ovary', 'colon', 'breast']
# prediction_names = ['d2d_AML', 'd2d_ovary', 'd2d_colon', 'd2d_breast']

args = experiments_all
args['data']['sources_filename'] = sources_filenmae_dict[experiments_type]
args['data']['terminals_filename'] = terminals_filenmae_dict[experiments_type]
args['data']['directed_interactions_filename'] = interaction_type
rng = np.random.RandomState(args['data']['random_seed'])

network, directed_interactions, sources, terminals, id_to_degree = \
    read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
              args['data']['sources_filename'], args['data']['terminals_filename'],
              args['data']['n_experiments'], args['data']['max_set_size'], rng)

predicted_edges = {}
for name in prediction_names:
    prediction_file_path = path.join(predction_folder, name, 'directed_network')
    prediction = pd.read_csv(prediction_file_path, sep='\t', index_col=[0,1])
    predictions_dict = prediction[['direction_prob']].to_dict()['direction_prob']
    predicted_edges[name] = [x for x in predictions_dict.keys() if (predictions_dict[x]/(predictions_dict[(x[1],x[0])] + 1e-12) > ratio_threshold)]
    # set(prediction[prediction['direction_prob'] > direction_prob_threshold].index.to_list())
    # predicted_edges[name] = set(prediction[prediction['direction_prob'] > direction_prob_threshold].index.to_list())

all_edges = list(set.union(*[set(edges) for edges in predicted_edges.values()]))
edge_to_idx = {edge:idx for idx, edge in enumerate(all_edges)}
idx_to_edge = {xx:x for x, xx in edge_to_idx.items()}
consensus_array = np.zeros((len(all_edges), len(predicted_edges.keys())))
for n, name in enumerate(predicted_edges.keys()):
    for edge in predicted_edges[name]:
        consensus_array[edge_to_idx[edge], n] = 1

consensus_predictions = [idx_to_edge[idx] for idx in np.nonzero(np.mean(consensus_array, axis=1)>= consensus_threshold)[0]]
consensus_predictions_flipped = [(pair[1], pair[0]) for pair in consensus_predictions]
n_keep = 5
first_n_keep_experiments = list(sources.keys())[:n_keep]
keep_source= {key: sources[key] for key in first_n_keep_experiments}
keep_terminals = {key: terminals[key] for key in first_n_keep_experiments}

directed_propagation_scores, _, _ = propagate_directed_network(undirected_network=network,
                                                                              directed_edges=consensus_predictions_flipped,
                                                                              sources=keep_terminals,
                                                                              terminals=keep_source, args=args)
propagation_scores, row_id_to_idx, col_id_to_idx = propagate_directed_network(undirected_network=network,
                                                                              directed_edges=[],
                                                                              sources=keep_terminals,
                                                                              terminals=keep_source, args=args)


undirected_source_ranks, directed_source_ranks = [], []
undirected_source_scores, directed_source_scores = [], []

source_ranks = []
for source_name, source_values in keep_source.items():
    terminal_indexes = [row_id_to_idx[x] for x in terminals[source_name]]
    sources_indexes = [col_id_to_idx[x] for x in sources[source_name]]
    source_undirected_prop_scores = np.sum(propagation_scores[terminal_indexes, :], 0)
    source_directed_prop_scores = np.sum(directed_propagation_scores[terminal_indexes, :], 0)
    rank_undirected = np.argsort(np.argsort(source_undirected_prop_scores))
    rank_directed = np.argsort(np.argsort(source_directed_prop_scores))
    undirected_source_ranks.append(rank_undirected[sources_indexes])
    directed_source_ranks.append(rank_directed[sources_indexes])

    undirected_source_scores.append(source_undirected_prop_scores[sources_indexes])
    directed_source_scores.append(source_directed_prop_scores[sources_indexes])

directed_ranks = [x for xx in directed_source_ranks for x in xx]
undirected_ranks = [x for xx in undirected_source_ranks for x in xx]
mean_undirected_ranks = np.mean(undirected_ranks)
mean_directed_ranks = np.mean(directed_ranks)

directed_scores = [x for xx in directed_source_scores for x in xx]
undirected_scores = [x for xx in undirected_source_scores for x in xx]
mean_undirected_scores = np.mean(undirected_scores)
mean_directed_scores = np.mean(directed_scores)
print('mean directed rank: {}, mean undirected rank: {}'.format(mean_directed_ranks, mean_undirected_ranks))