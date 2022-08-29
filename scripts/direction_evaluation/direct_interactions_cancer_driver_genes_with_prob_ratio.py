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
dict_to_log = {}

root_path = get_root_path()
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)
percentiles = [0.25, 0.5, 1, 5, 10]
max_set_size = 10000
min_set_size = 0
quantile_threshold = 0.55

# directions_predictions_files = {'breast': '08_05_2022__14_11_30',
#                                 'ovary':'08_05_2022__13_28_47',
#                                 'AML':'08_05_2022__13_27_08',
#                                 'colon':'08_05_2022__14_08_28',
#                                 'd2d_breast': '08_05_2022__14_27_16',
#                                 'd2d_AML': '08_05_2022__15_58_37',
#                                 'd2d_colon': '08_05_2022__14_26_35',
#                                 'd2d_ovary': '08_05_2022__14_25_49'}
directions_predictions_files = {'breast': 'breast',
                                'ovary':'ovary',
                                'AML':'AML',
                                'colon':'colon',
                                'drug':'drug',
                                'd2d_breast': 'd2d_breast',
                                'd2d_AML': 'd2d_AML',
                                'd2d_colon': 'd2d_colon',
                                'd2d_ovary': 'd2d_ovary',
                                'd2d_drug':'d2d_drug'}

# directions_predictions_files = {'breast': '23_06_2022__20_31_55',
#                                 'ovary':'23_06_2022__19_50_57',
#                                 'AML':'23_06_2022__19_50_30',
#                                 'colon':'25_06_2022__12_28_46',
#                                 'drug':'23_06_2022__19_50_08',
#                                 'd2d_breast': '25_06_2022__12_30_55',
#                                 'd2d_AML': '23_06_2022__18_56_52',
#                                 'd2d_colon': '25_06_2022__12_31_12',
#                                 'd2d_ovary': '23_06_2022__18_56_40',
#                                 'd2d_drug':'23_06_2022__18_07_16'}
cancer_genes_datasets = ['cosmic', 'uniprot']

interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
# interaction_type = sorted(['KEGG'])
predction_folder = path.join(root_path, 'input', 'predicted_interactions')
experiments_types = ['ovary', 'colon', 'breast', 'AML']
prediction_types = [ 'ovary', 'AML', 'colon', 'breast']
prediction_types = [ 'ovary', 'AML', 'colon', 'breast']
# prediction_types = ['d2d_ovary', 'd2d_colon', 'd2d_breast', 'd2d_AML']
# prediction_types = ['d2d_ovary', 'd2d_AML', 'd2d_colon', 'd2d_breast']
redirect_output(path.join(output_file_path, 'log'))


for experiment_type in experiments_types:

    undropped_predictions = [x for x in prediction_types if experiment_type not in x]
    undropped_predictions = [x for x in prediction_types]
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
                  args['data']['n_experiments'], 10000, rng)

    sources = {key:value for key,value in sources.items() if len(value)>min_set_size and len(value)<max_set_size}
    terminals = {key:value for key,value in terminals.items() if len(value)>min_set_size and len(value)<max_set_size}
    filtered_experiments = sorted(sources.keys() & terminals.keys())
    sources = {exp_name: sources[exp_name] for exp_name in filtered_experiments}
    terminals = {exp_name: terminals[exp_name] for exp_name in filtered_experiments}

    n_interactions = len(list(directed_interactions.index))
    # generating datasets
    directed_interactions_set = set(directed_interactions.index)
    predictions = []
    for name in undropped_predictions:
        prediction_file_path = path.join(predction_folder, directions_predictions_files[name], 'directed_network')
        prediction = pd.read_csv(prediction_file_path, sep='\t', index_col=[0,1])
        predictions_dict = prediction[['direction_prob']].to_dict()['direction_prob']
        predicted_edges = {x: predictions_dict[x]/(predictions_dict[(x[1],x[0])] + 1e-12) for x in predictions_dict.keys()}
        prediction['direction_prob'].loc[predicted_edges.keys()] = list(predicted_edges.values())
        predictions.append(prediction)

    predictions = pd.concat(predictions)
    predictions['direction_prob'] = np.log(predictions['direction_prob'] + 1e-9)
    predictions['direction_prob'] = predictions.groupby(level=[0, 1])['direction_prob'].mean()
    predictions = predictions.reset_index().drop_duplicates(keep='first').set_index(keys=['0', '1'])

    consensus_predictions = set(predictions[predictions['direction_prob'] > predictions['direction_prob'].quantile(quantile_threshold)].index.to_list())
    consensus_predictions = set(consensus_predictions).union(directed_interactions_set)
    consensus_predictions_flipped = [(pair[1], pair[0]) for pair in consensus_predictions]


    overlaps = set(consensus_predictions_flipped).intersection(set(consensus_predictions))
    consensus_predictions_flipped = list(set(consensus_predictions_flipped).difference(overlaps))
    directed_propagation_scores, _ = propagate_directed_network(undirected_network=network,
                                                                                  directed_edges=consensus_predictions_flipped,
                                                                                  sources=terminals, args=args)

    propagation_scores, col_id_to_idx = propagate_directed_network(undirected_network=network,
                                                                                  directed_edges=[],
                                                                                  sources=terminals, args=args)

    single_input_results_dict = {}

    for cancer_genes_dataset_name, cancer_genes_dataset_ids in cancer_genes_dict.items():
        dataset_fraction_of_total_genes = len(cancer_genes_dataset_ids) / len(propagation_scores[0])
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
        num_genes_in_source_precentile = [len(propagation_scores[0]) * x * 0.01 for x in percentiles]
        undirected_percent_of_cancer_genes_of_percentile = [
            np.sum(np.array(undirected_ranks) > len(propagation_scores[0]) - x) / (x * len(sources)) for x in
            num_genes_in_source_precentile]
        directed_percent_of_cancer_genes_of_percentile = [
            np.sum(np.array(directed_ranks) > len(propagation_scores[0]) - x) / (x * len(sources)) for x in num_genes_in_source_precentile]
        directed_fold_enrichment = [x/ dataset_fraction_of_total_genes for x in directed_percent_of_cancer_genes_of_percentile]
        undirected_fold_enrichment = [x/ dataset_fraction_of_total_genes for x in undirected_percent_of_cancer_genes_of_percentile]
        single_input_results_dict[cancer_genes_dataset_name]['undirected_network'] = {'percentiles':undirected_percent_of_cancer_genes_of_percentile,
                                                                                      'fold_enrichment': undirected_fold_enrichment}
        single_input_results_dict[cancer_genes_dataset_name]['directed_network'] = {'percentiles': directed_percent_of_cancer_genes_of_percentile,
                                                                                    'fold_enrichment': directed_fold_enrichment}


    results_dict[experiment_type] = single_input_results_dict


args_dict = {'prediction_files':directions_predictions_files,
             'interaction_types':interaction_type,
             'quantile_threshold':quantile_threshold,
             'max_set_size':max_set_size,
             'min_set_size':min_set_size,
             'experiments_types': experiments_types,
             'output_file_path':output_file_path,
             'cancer_genes_datasets': cancer_genes_datasets,
             'prediction_types':prediction_types,
             'percentiles':percentiles}
dict_to_log['args'] = args_dict
dict_to_log['results'] = results_dict
with open(path.join(output_file_path, 'results'), 'w') as f:
    json.dump(dict_to_log, f, indent=4, separators=(',', ': '))