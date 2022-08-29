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
from itertools import combinations
import networkx as nx
import pandas as pd
root_path = get_root_path()
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)
redirect_output(path.join(output_file_path, 'log'))

constrain_to_train_set_neighbors = False
quantile_thresholds = list(np.linspace(0.8, 1, 5, endpoint=False))
args = experiments_0
interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
args['data']['directed_interactions_filename'] = interaction_type
rng = np.random.RandomState(args['data']['random_seed'])
network, directed_interactions, _, _, id_to_degree = \
    read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
              args['data']['sources_filename'], args['data']['terminals_filename'],
              args['data']['n_experiments'], args['data']['max_set_size'], rng)

predction_folder = path.join(root_path, 'input', 'predicted_interactions')
prediction_types = ['d2d_ovary', 'd2d_AML', 'd2d_colon', 'd2d_breast']

# prediction_types = ['ovary', 'AML', 'colon', 'breast']

complex_file = path.join(root_path, 'input', 'other', 'humanComplexes.txt')

directions_predictions_files = {'breast': 'breast',
                                'ovary':'ovary',
                                'AML':'AML',
                                'colon':'colon',
                                'drug': 'drug',
                                'd2d_breast': 'd2d_breast',
                                'd2d_AML': 'd2d_AML',
                                'd2d_colon': 'd2d_colon',
                                'd2d_ovary': 'd2d_ovary',
                                'd2d_drug': 'd2d_drug'}
# directions_predictions_files = {'breast': '23_06_2022__20_31_55',
#                             'ovary':'23_06_2022__19_50_57',
#                             'AML':'23_06_2022__19_50_30',
#                             'colon':'25_06_2022__12_28_46',
#                             'drug':'23_06_2022__19_50_08',
#                             'd2d_breast': 'd2d_breast',
#                             'd2d_AML': 'd2d_AML',
#                             'd2d_colon': 'd2d_colon',
#                             'd2d_ovary': 'd2d_ovary',
#                             'd2d_drug': 'd2d_drug'}

directed_interactions_pairs_list = np.array(directed_interactions.index)
train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'],
                                                            len(directed_interactions_pairs_list),
                                                            args['train']['train_val_test_split'],
                                                            random_state=rng)  # feature generation


predictions = []
for name in prediction_types:
    prediction_file_path = path.join(predction_folder, directions_predictions_files[name], 'directed_network')
    prediction = pd.read_csv(prediction_file_path, sep='\t', index_col=[0, 1])
    predictions_dict = prediction[['direction_prob']].to_dict()['direction_prob']
    predicted_edges = {tuple(sorted(x)): np.max(
        [predictions_dict[x] / (predictions_dict[(x[1], x[0])] + 1e-12),
        predictions_dict[(x[1], x[0])] / (predictions_dict[x] + 1e-12)]) for x in
                       predictions_dict.keys()}
    predictions_score_df = pd.DataFrame(
        data={'direction_prob': list(predicted_edges.values())},
        index=pd.MultiIndex.from_tuples(tuples=predicted_edges.keys(), names=[0, 1]))    # prediction['direction_prob'].loc[predicted_edges.keys()] = list(predicted_edges.values())
    predictions.append(predictions_score_df)

predictions = pd.concat(predictions)
# predictions['direction_prob'] = np.log(predictions['direction_prob'] + 1e-9)
# predictions['direction_prob'] = predictions.groupby(level=[0, 1])['direction_prob'].mean()
predictions['direction_prob'] = predictions.groupby(level=[0, 1])['direction_prob'].prod()
predictions = predictions.reset_index().drop_duplicates(keep='first').set_index(keys=[0, 1])

if constrain_to_train_set_neighbors:
    graph =  nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    train_indexes = np.concatenate([train_indexes, val_indexes])
    directed_interactions_pairs_list = directed_interactions_pairs_list[train_indexes]
    train_set_genes = list(set([x for y in directed_interactions_pairs_list for x in y]))
    genes_ids_to_keep = set(x for id in train_set_genes for x in graph.neighbors(id))

    indexes_to_zeroise = [x for x in predictions.index if
                          (x[0] not in genes_ids_to_keep) or (x[1] not in genes_ids_to_keep)]
    predictions.drop(indexes_to_zeroise, inplace=True)
directed_interactions_list = list(directed_interactions.index)
directed_interactions_list = [x for x in directed_interactions if x in predictions.index]
predictions.drop(directed_interactions_list, inplace=True)
# pd.read_csv(complex_file, delimiter='\t')
all_edges = set()
with open(complex_file, 'r') as f:
    lines = f.readlines()
    complexes = []
    for line in lines[1:]:
        complex_genes = line.strip().split('\t')[18].split(';')
        translated_gene = list(translator.translate([x.strip(' ') for x in complex_genes], 'symbol', 'entrez_id').values())
        complex_edges = [tuple(sorted(x)) for x in combinations(translated_gene, 2)]
        all_edges.update(complex_edges)

# load consensus network
# add parameter for zeroising edges that are not adjacent to dataset edges
# check if edges in complexes probabilites are lower than edges not in complexes
filtered_edges = [tuple(sorted(x)) for x in list(all_edges) if x in predictions.index]
complex_edges_median = predictions.loc[filtered_edges].direction_prob.mean()
all_edges_mean = predictions.direction_prob.mean()
n_edges = len(predictions.index)//2
# edge_score = {tuple(sorted(x)): np.max([predictions.loc[x], predictions.loc[(x[1], x[0])]]) for x in list(predictions.index)[:n_edges]}

fold_enrichments = []
fold_enrichments_unoriented = []
p_value = []
left_unoriented = []
for threshold in quantile_thresholds:
    left_unoriented.append(threshold)
    consensus_predictions = set(predictions[predictions['direction_prob'] > predictions['direction_prob'].quantile(
        threshold)].index.to_list())

    complex_in_consesnus = [x for x in filtered_edges if x in consensus_predictions]
    fold_enrichments.append(1/((len(complex_in_consesnus) / len(filtered_edges)) / (len(consensus_predictions)/ len(predictions.index))))
    # fold_enrichments_unoriented.append((n_unoriented / len(filtered_edges)) / (len(unoriented_edge)/ len(predictions.index)))
    stats = hypergeom.sf(len(complex_in_consesnus), len(predictions.index), len(filtered_edges), len(consensus_predictions))
    p_value.append(np.minimum(stats, 1-stats))

results_dict = {}
args_dict = {'interaction_types':interaction_type,
             'quantile_thresholds': quantile_thresholds,
             'output_file_path':output_file_path,
             'prediction_types':prediction_types,
             'prediction_files':directions_predictions_files}
results_dict['args'] = args_dict

results_dict['results'] = {'left_unoriented':left_unoriented,
                           'fold_enrichment':fold_enrichments,
                           'pvalue':p_value}
with open(path.join(output_file_path, 'results'), 'w') as f:
    json.dump(results_dict, f, indent=4, separators=(',', ': '))
