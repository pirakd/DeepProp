import pandas as pd
from os import path, makedirs
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from utils import read_data, get_root_path, train_test_split
import numpy as np
from utils import read_network, propagate_directed_network, redirect_output, get_time, train_test_split
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
from gene_name_translator.gene_translator import GeneTranslator
translator = GeneTranslator()
translator.load_dictionary()
from presets import experiments_0
import json
from scipy.stats import hypergeom

results_dict = {}
root_path = get_root_path()
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)

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
#                                 'd2d_breast': '08_05_2022__14_27_16',
#                                 'd2d_AML': '08_05_2022__15_58_37',
#                                 'd2d_colon': '08_05_2022__14_26_35',
#                                 'd2d_ovary': '08_05_2022__14_25_49'}
cancer_genes_datasets = ['cosmic', 'uniprot']

# interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
interaction_type = sorted(['KEGG'])
predction_folder = path.join(root_path, 'input', 'predicted_interactions')
consensus_threshold = 2 / 3
ratio_threshold = 1.2
experiments_types = ['ovary']
prediction_types = [ 'ovary', 'AML', 'colon', 'breast']
# prediction_types = ['d2d_ovary', 'd2d_colon', 'd2d_breast', 'd2d_AML']
redirect_output(path.join(output_file_path, 'log'))

args_dict = {'interaction_types':interaction_type,
             'ratio_threshold':ratio_threshold,
             'consensus_threshold': consensus_threshold,
             'experiments_types': experiments_types,
             'output_file_path':output_file_path,
             'cancer_genes_datasets': cancer_genes_datasets,
             'prediction_types':prediction_types}
results_dict['args'] = args_dict
results_dict['results'] = {}
for experiment_type in experiments_types:

    undropped_predictions = [x for x in prediction_types if experiment_type not in x]
    cancer_genes_dict = {}
    for cancer_genes_dataset in cancer_genes_datasets:
        driver_genes_path = path.join(root_path, 'input', 'other', '{}_cancer_genes.tsv'.format(cancer_genes_dataset))
        cancer_genes = list(pd.read_csv(driver_genes_path, sep='\t')['Gene Symbol'].str.split(pat=' ').str[0])
        cancer_genes_dict[cancer_genes_dataset] = list(translator.translate(cancer_genes, 'symbol', 'entrez_id').values())
    cancer_genes_dict['overall'] = set(x for xx in cancer_genes_dict.values() for x in xx)

    args = experiments_0
    args['data']['sources_filename'] = sources_filenmae_dict[experiment_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[experiment_type]
    args['data']['directed_interactions_filename'] = interaction_type
    rng = np.random.RandomState(args['data']['random_seed'])

    network, directed_interactions, sources, terminals, id_to_degree = \
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)

    n_interactions = len(list(directed_interactions.index))
    # generating datasets
    directed_interactions_set = set(directed_interactions.index)
    predicted_edges = {}
    for name in undropped_predictions:
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
    consensus_predictions_flipped = list(set(consensus_predictions_flipped).difference(overlaps))
    directed_interactions_flipped = [(pair[1], pair[0]) for pair in directed_interactions_set]
    directed_propagation_scores, _ = propagate_directed_network(undirected_network=network,
                                                                                  directed_edges=consensus_predictions_flipped,
                                                                                  sources=terminals, args=args)

    propagation_scores, col_id_to_idx = propagate_directed_network(undirected_network=network,
                                                                                  directed_edges=[],
                                                                                  sources=terminals, args=args)

    single_input_results_dict = {}

    for cancer_genes_dataset_name, cancer_genes_dataset_ids in cancer_genes_dict.items():
        undirected_source_ranks, directed_source_ranks = [], []
        undirected_source_scores, directed_source_scores = [], []
        source_ranks = []
        single_input_results_dict[cancer_genes_dataset_name] = {}
        for exp_idx in range(len(sources)):
            cancer_gene_indexes = [col_id_to_idx[x] for x in cancer_genes_dataset_ids if x in col_id_to_idx]
            rank_undirected = np.argsort(np.argsort(propagation_scores[exp_idx]))
            rank_directed = np.argsort(np.argsort(directed_propagation_scores[exp_idx]))
            undirected_source_ranks.append(rank_undirected[cancer_gene_indexes])
            directed_source_ranks.append(rank_directed[cancer_gene_indexes])

        directed_ranks = [x for xx in directed_source_ranks for x in xx]
        undirected_ranks = [x for xx in undirected_source_ranks for x in xx]
        num_genes_in_source_precentile = [len(propagation_scores[0]) * x * 0.01 for x in [0.25, 0.5, 1, 5, 10]]
        undirected_percent_of_cancer_genes_of_percentile = [
            np.sum(np.array(undirected_ranks) > len(propagation_scores[0]) - x) / (x * len(sources)) for x in
            num_genes_in_source_precentile]
        directed_percent_of_cancer_genes_of_percentile = [
            np.sum(np.array(directed_ranks) > len(propagation_scores[0]) - x) / (x * len(sources)) for x in num_genes_in_source_precentile]

        single_input_results_dict[cancer_genes_dataset_name]['undirected_network'] = {'percentiles':undirected_percent_of_cancer_genes_of_percentile}
        single_input_results_dict[cancer_genes_dataset_name]['directed_network'] = {'percentiles': directed_percent_of_cancer_genes_of_percentile}

    results_dict['results'][experiment_type] = single_input_results_dict

with open(path.join(output_file_path, 'results'), 'w') as f:
    json.dump(results_dict, f, indent=4, separators=(',', ': '))