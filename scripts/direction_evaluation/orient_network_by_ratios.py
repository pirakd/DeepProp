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

def get_edge_prob(df):
    df.apply(np.log)
    df.sum()
    return df

directions_predictions_files = {'breast': '08_05_2022__14_11_30',
                                'ovary':'08_05_2022__13_28_47',
                                'AML':'08_05_2022__13_27_08',
                                'colon':'08_05_2022__14_08_28',
                                'd2d_breast': '08_05_2022__14_27_16',
                                'd2d_AML': '08_05_2022__15_58_37',
                                'd2d_colon': '08_05_2022__14_26_35',
                                'd2d_ovary': '08_05_2022__14_25_49'}
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

by_prob = False
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
consensus_threshold = 0.8
threshold_type = 'ratio'
ratio_threshold = 1.01
prob_threshold = 0.7
prediction_types = ['ovary', 'AML', 'colon', 'breast', 'drug']
# prediction_types = ['drug']
# prediction_types = ['d2d_ovary', 'd2d_colon', 'd2d_breast', 'd2d_AML', ]
# prediction_types = ['d2d_ovary', 'd2d_colon', 'd2d_breast', 'd2d_AML', 'd2d_drug']
# prediction_types = ['d2d_drug']
# prediction_types = ['d2d_ovary']

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

n_interctions = len(list(directed_interactions.index))
# generating datasets
directed_interactions_set = set(directed_interactions.index)

predicted_edges = []
predictions = []
for n, name in enumerate(prediction_types):
    prediction_file_path = path.join(predction_folder, directions_predictions_files[name], 'directed_network')
    if by_prob:
        predictions.append(pd.read_csv(prediction_file_path, sep='\t', index_col=[0,1]))
    else:
        prediction = pd.read_csv(prediction_file_path, sep='\t', index_col=[0,1])
        predictions_dict = prediction[['direction_prob']].to_dict()['direction_prob']
        predicted_edges = {x: predictions_dict[x]/(predictions_dict[(x[1],x[0])] + 1e-12) for x in predictions_dict.keys()}
        prediction['direction_prob'].loc[predicted_edges.keys()] = list(predicted_edges.values())
        predictions.append(prediction)


predictions = pd.concat(predictions)
predictions['direction_prob'] = np.log(predictions['direction_prob']+ 1e-9)
predictions['direction_prob'] = predictions.groupby(level=[0,1])['direction_prob'].mean()
predictions = predictions.reset_index().drop_duplicates(keep='first').set_index(keys=['0', '1'])

predictions.to_csv(path.join(output_file_path, 'edge_probs_ratio'), sep='\t')