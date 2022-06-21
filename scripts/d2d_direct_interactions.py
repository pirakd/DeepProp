import pandas as pd
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import makedirs
from utils import read_data, load_model, log_results, get_time, get_root_path, train_test_split,\
    gen_propagation_scores, get_loss_function, redirect_output
import numpy as np
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import networkx as nx
from D2D import eval_D2D, generate_D2D_features_from_propagation_scores
import D2D
from presets import experiments_20
from sklearn.metrics import precision_recall_curve, auc
import copy
import json
import  pickle
from tqdm import tqdm

batch_size = 10000

n_exp = 0
edges_types = sorted(['KPI' , 'STKE', 'E3', 'EGFR'])
edges_types = sorted(['KEGG'])
# edges_types = sorted(['KPI'])
input_types = ['ovary', 'colon', 'AML', 'breast']
propagation_scores_files = ['{}_{}_{}'.format(x, '_'.join(edges_types),n_exp) for x in input_types]
propagation_scores_files_to_direct = ['{}_{}_direct'.format(x, '_'.join(edges_types)) for x in input_types]

root_path = get_root_path()
output_folder = 'output'
args = experiments_20

for i, input_type in enumerate(input_types):
    print('directing with {}'.format(input_type))
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))
    args['data']['load_prop_scores'] = True
    args['data']['save_prop_scores'] = False
    args['data']['prop_scores_filename'] = propagation_scores_files[i]
    args['train']['train_val_test_split'] = [0.8, 0, 0.2]
    args['data']['directed_interactions_filename'] = edges_types
    args['data']['sources_filename'] = sources_filenmae_dict[input_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[input_type]

    rng = np.random.RandomState(args['data']['random_seed'])

    # data read
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng, translate_genes=True)

    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list, calc_normalization_constants=False)

    train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'],
                                                                len(directed_interactions_pairs_list),
                                                                args['train']['train_val_test_split'],
                                                                random_state=rng,
                                                                directed_interactions=directed_interactions_pairs_list)

    train_indexes = np.sort(np.concatenate([train_indexes, val_indexes])).astype(int)
    sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
    terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]

    features, deconstructed_features = generate_D2D_features_from_propagation_scores(propagation_scores, pairs_indexes,
                                                                                     sources_indexes, terminals_indexes)

    d2d_results_dict, d2d_model = eval_D2D(features[train_indexes], features[test_indexes], source_types=directed_interactions_source_type[test_indexes])
    d2d_precision, d2d_recall, d2d_thresholds = precision_recall_curve(d2d_results_dict['overall']['labels'],
                                                                       d2d_results_dict['overall']['probs'][:, 1])
    d2d_auc = auc(d2d_recall, d2d_precision)
    rng = np.random.RandomState(args['data']['random_seed'])

    # load all prop scores
    args['data']['prop_scores_filename'] = propagation_scores_files_to_direct[i]

    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng, translate_genes=True)

    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    train_interactions_pairs_list = np.array(directed_interactions.index)
    train_set_genes = sorted(list(set([x for pair in train_interactions_pairs_list for x in pair])))
    genes_ids_to_keep = set(x for id in train_set_genes for x in graph.neighbors(id))
    directed_interactions_pairs_list  = np.array([(x[0], x[1]) for x in list(network.index) if (x[0] in genes_ids_to_keep and x[1] in genes_ids_to_keep)])
    genes_ids_to_keep = sorted(genes_ids_to_keep)
    directed_interactions_source_type = np.array(['_' for x in range(len(directed_interactions_pairs_list))])

    # here we have to propagate all genes
    propagation_scores, row_id_to_idx, col_id_to_idx, _ = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list, calc_normalization_constants=False)

    train_indexes, val_indexes, test_indexes = train_test_split('normal',len(directed_interactions_pairs_list), [0, 0, 1],
                                                                random_state=rng, directed_interactions=directed_interactions_pairs_list)

    sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
    terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
    pairs_indexes = np.array([(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list])

    n_batches = int((len(test_indexes)//batch_size)+1)
    all_predictions = []
    all_pair_ids = []
    for b in tqdm(range(n_batches), total=n_batches, desc='predicting scores for {}'.format(input_type)):
        if b != n_batches-1:
            batch_indexes = test_indexes[b*batch_size:(b+1)*batch_size]
        else:
            batch_indexes = test_indexes[b*batch_size:]
        batch_features, _ = generate_D2D_features_from_propagation_scores(propagation_scores, pairs_indexes[batch_indexes], sources_indexes, terminals_indexes)
        inverse_batch_features = 1 / batch_features
        combined_batch_features = np.concatenate([batch_features, inverse_batch_features])
        prediction = D2D.predict(d2d_model, combined_batch_features)
        all_predictions.append(prediction)
        all_pair_ids.append(np.concatenate([directed_interactions_pairs_list[batch_indexes,:], np.fliplr(directed_interactions_pairs_list[batch_indexes,:])]))

    all_predictions = np.vstack(all_predictions)
    all_pair_ids = np.vstack(all_pair_ids)
    directed_edge_prob = {tuple([id for id in tuple(all_pair_ids[i,:])]) : all_predictions[i,1] for i in range(all_predictions.shape[0])}


    network.pop('source')
    network_reverse = copy.deepcopy(network)
    reverse_indexes = np.array([[x[1], x[0]] for x in list(network.index)])
    network_reverse.index = pd.MultiIndex.from_arrays([reverse_indexes[:, 0], reverse_indexes[:, 1]])
    directed_network = pd.concat([network, network_reverse])
    directed_network['direction_prob'] = [directed_edge_prob.get((x[0],x[1]), 0) for x in list(directed_network.index)]
    directed_network.reset_index().to_csv(path.join(output_file_path, 'directed_network'), sep='\t', header=True, index=False)


    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))

    with open(path.join(output_file_path, 'model'), 'wb') as f:
        pickle.dump(d2d_model, f)