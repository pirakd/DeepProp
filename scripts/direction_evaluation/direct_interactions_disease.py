import pandas as pd
from os import path, makedirs
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from utils import read_data, get_root_path, train_test_split
import numpy as np
from utils import read_network, propagate_directed_network, redirect_output, get_time, train_test_split
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
from gene_name_translator.gene_translator import GeneTranslator
import networkx as nx
translator = GeneTranslator()
from presets import experiments_0
import json
translator.load_dictionary()
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, auc
directions_predictions_files = {'breast': '08_05_2022__14_11_30',
                                'ovary':'08_05_2022__13_28_47',
                                'AML':'08_05_2022__13_27_08',
                                'colon':'08_05_2022__14_08_28',
                                'd2d_breast': '08_05_2022__14_27_16',
                                'd2d_AML': '08_05_2022__15_58_37',
                                'd2d_colon': '08_05_2022__14_26_35',
                                'd2d_ovary': '08_05_2022__14_25_49'}
# directions_predictions_files = {'breast': 'breast',
#                                 'ovary':'ovary',
#                                 'AML':'AML',
#                                 'colon':'colon',
#                                 'drug': 'drug',
#                                 'd2d_breast': 'd2d_breast',
#                                 'd2d_AML': 'd2d_AML',
#                                 'd2d_colon': 'd2d_colon',
#                                 'd2d_ovary': 'd2d_ovary',
#                                 'd2d_drug': 'd2d_drug'}


maximum_set_size = 10000
minimum_set_size = 10
results_dict = {}
root_path = get_root_path()
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)
redirect_output(path.join(output_file_path, 'log'))

interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
# interaction_type = sorted(['KEGG'])
predction_folder = path.join(root_path, 'input', 'predicted_interactions')
consensus_threshold = 2
threshold_type = 'ratio'
ratio_threshold = 1.01
prob_threshold = 0.7
# prediction_types = ['ovary', 'AML', 'colon', 'breast', 'drug']
# prediction_types = ['drug']
# prediction_types = ['d2d_ovary', 'd2d_colon', 'd2d_breast', 'd2d_AML', ]
# prediction_types = ['d2d_ovary', 'd2d_colon', 'd2d_breast', 'd2d_AML']
# prediction_types = ['d2d_drug']
prediction_types = ['d2d_ovary']

args_dict = {'interaction_types':interaction_type,
             'ratio_threshold':ratio_threshold,
             'consensus_threshold': consensus_threshold,
             'output_file_path':output_file_path,
             'prediction_types':prediction_types}
results_dict['args'] = args_dict
results_dict['results'] = {}

args = experiments_0
args['data']['directed_interactions_filename'] = interaction_type
rng = np.random.RandomState(args['data']['random_seed'])

network, directed_interactions, _, _, id_to_degree = \
    read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
              args['data']['sources_filename'], args['data']['terminals_filename'],
              args['data']['n_experiments'], args['data']['max_set_size'], rng)
network_genes = np.unique(network.ge)
graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1)

target_diesease_file_path = path.join(root_path, 'input', 'other', 'disease_target_up')
source_diesease_file_path = path.join(root_path, 'input', 'other', 'disease_source')

target_disease_genes = dict()
with open(target_diesease_file_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        genes = []
        for x in line[1:]:
            translated_gene = translator.translate(x.upper(), 'symbol', 'entrez_id')
            if translated_gene is not None and translated_gene in graph:
                genes.append(translated_gene)
        if len(genes) <= maximum_set_size and len(genes) >= minimum_set_size:
            target_disease_genes[line[0]] = genes


source_disease_genes = dict()
with open(source_diesease_file_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        genes = []
        for x in line[1:]:
            translated_gene = translator.translate(int(x), 'entrez_id', 'entrez_id')
            if translated_gene is not None and translated_gene in graph:
                genes.append(translated_gene)
        if len(genes) <= maximum_set_size and len(genes) >= minimum_set_size:
            source_disease_genes[line[0]] = genes

keep_experiments = source_disease_genes.keys() & set(target_disease_genes)
source_disease_genes = {x: source_disease_genes[x] for x in keep_experiments}
target_disease_genes = {x: target_disease_genes[x] for x in keep_experiments}


n_interactions = len(list(directed_interactions.index))
# generating datasets
directed_interactions_set = set(directed_interactions.index)

predicted_edges = {}
for name in prediction_types:
    prediction_file_path = path.join(predction_folder, directions_predictions_files[name], 'directed_network')
    prediction = pd.read_csv(prediction_file_path, sep='\t', index_col=[0,1])
    predictions_dict = prediction[['direction_prob']].to_dict()['direction_prob']
    predicted_edges[name] = [x for x in predictions_dict.keys() if (predictions_dict[x]/(predictions_dict[(x[1],x[0])] + 1e-12) > ratio_threshold)]

all_edges = list(set.union(*[set(edges) for edges in predicted_edges.values()]))
edge_to_idx = {edge:idx for idx, edge in enumerate(all_edges)}
idx_to_edge = {xx:x for x, xx in edge_to_idx.items()}
consensus_array = np.zeros((len(all_edges), len(predicted_edges.keys())))
for n, name in enumerate(predicted_edges.keys()):
    for edge in predicted_edges[name]:
        consensus_array[edge_to_idx[edge], n] = 1

consensus_idxs = np.nonzero(np.mean(consensus_array, axis=1) >= consensus_threshold)[0]
consensus_predictions = [idx_to_edge[idx] for idx in consensus_idxs]
consensus_predictions = set(consensus_predictions).union(directed_interactions_set)
consensus_predictions_flipped = [(pair[1], pair[0]) for pair in consensus_predictions]

overlaps = set(consensus_predictions_flipped).intersection(set(consensus_predictions))

directed_propagation_scores, col_id_to_idx = propagate_directed_network(undirected_network=network,
                                                            directed_edges=consensus_predictions_flipped,
                                                            sources=target_disease_genes, args=args)
undirected_propagation_scores, col_id_to_idx = propagate_directed_network(undirected_network=network,
                                                            directed_edges=[],
                                                            sources=target_disease_genes, args=args)

num_genes_in_source_precentile = [len(directed_propagation_scores[0]) * x * 0.01 for x in [0.05, 0.1, 0.25, 0.5, 1, 5, 10]]
directed_test_ranks = []
auc_list = []
undirected_auc_list = []
undirected_ranks_list = []
directed_ranks_list = []
labels_list = []
for e, exp in enumerate(source_disease_genes.keys()):
    directed_source_ranks = [], []
    directed_source_scores = [], []
    source_ranks = []

    test_gene_indexes = [col_id_to_idx[x] for x in source_disease_genes[exp] if x in col_id_to_idx]
    directed_ranks = np.argsort(np.argsort(directed_propagation_scores[e]))
    undirected_ranks = np.argsort(np.argsort(undirected_propagation_scores[e]))
    labels = np.zeros_like(directed_ranks)
    labels[test_gene_indexes] = 1

    undirected_ranks_list.append(undirected_ranks)
    directed_ranks_list.append(directed_ranks)
    labels_list.append(labels)


directed_ranks_list = np.array([x for xx in directed_ranks_list for x in xx])
undirected_ranks_list = np.array([x for xx in undirected_ranks_list for x in xx])
labels_list = np.array([x for xx in labels_list for x in xx])
#TODO continue from here
undirected_percent_of_cancer_genes_of_percentile = [
    np.sum(np.array(undirected_ranks_list[labels_list == 1]) > (len(directed_propagation_scores[0]) - x)) / (x * len(target_disease_genes.keys())) for x in
    num_genes_in_source_precentile]
directed_percent_of_cancer_genes_of_percentile = [
    np.sum(np.array(directed_ranks_list[labels_list == 1 ]) > (len(directed_propagation_scores[0]) - x)) / (x * len(target_disease_genes.keys())) for x in num_genes_in_source_precentile]

# directed_test_ranks.append(ranks[test_gene_indexes])

precision, recall, _ = precision_recall_curve(labels_list, directed_ranks_list)
directed_auc = auc(recall, precision)

precision, recall, _ = precision_recall_curve(labels_list, undirected_ranks_list)
undirected_auc = auc(recall, precision)

results_dict['results'] = \
    {'directed_network': {'auc': directed_auc,
                          'percentiles':directed_percent_of_cancer_genes_of_percentile},
     'undirected_network': {'auc':undirected_auc,
                           'percentiles': undirected_percent_of_cancer_genes_of_percentile}}

with open(path.join(output_file_path, 'results'), 'w') as f:
    json.dump(results_dict, f, indent=4, separators=(',', ': '))