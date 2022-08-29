import pandas as pd
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))
from os import makedirs
from utils import read_data, load_model, log_results, get_time, get_root_path, train_test_split,\
    gen_propagation_scores, get_loss_function, redirect_output
import numpy as np
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import networkx as nx
from D2D import eval_D2D, generate_D2D_features_from_propagation_scores
import D2D
from presets import experiments_0
from sklearn.metrics import precision_recall_curve, auc
import copy
import json
import  pickle
from tqdm import tqdm
import argparse
d2d_models_names = {'drug':'23_06_2022__16_38_26',
              'AML':'23_06_2022__16_32_32',
              'breast':'23_06_2022__16_33_12',
              'colon':'23_06_2022__16_33_31',
              'ovary':'23_06_2022__16_33_46'}

batch_size = 10000
def run(sys_args):
    root_path = get_root_path()
    output_folder = 'output'

    print('directing with {}'.format(sys_args.experiments_type))
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    model_args_path = path.join(root_path, 'input', 'models', 'd2d', d2d_models_names[sys_args.experiments_type], 'args')
    model_path = path.join(root_path, 'input', 'models', 'd2d', d2d_models_names[sys_args.experiments_type], 'd2d_models')


    with open(model_path, 'rb') as f:
        model = pickle.load(f)['d2d']
    with open(model_args_path, 'r') as f:
        args = json.load(f)

    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename

    rng = np.random.RandomState(args['data']['random_seed'])
    # load all prop scores
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng, translate_genes=True)

    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    train_interactions_pairs_list = np.array(directed_interactions.index)
    train_set_genes = sorted(list(set([x for pair in train_interactions_pairs_list for x in pair])))
    genes_ids_to_keep = graph.nodes
    directed_interactions_pairs_list  = np.array([(x[0], x[1]) for x in list(network.index) if (x[0] in genes_ids_to_keep and x[1] in genes_ids_to_keep)])
    genes_ids_to_keep = sorted(genes_ids_to_keep)

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
        prediction = D2D.predict(model, combined_batch_features)
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

    args['model_path'] = model_path

    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    input_type = 'drug'
    n_exp = 0
    split = [0.99, 0, 0.01]

    interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])

    # interaction_type = 'yeast_KPI'
    device = 'cpu'
    model_name = 'drug_KEGG_5'
    # prop_scores_filename = 'yeast_KPI_direct'
    # prop_scores_filename = 'ovary_E3_EGFR_KPI_STKE_direct'
    prop_scores_filename = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str, help='name of experiment type(drug, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int, help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-m,', '--model_name', type=str, help='name of saved model folder in input/models', default=model_name)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores',  action='store_true', default=False, help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores',  action='store_true', default=False, help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split',  nargs=3, help='[train, val, test] sums to 1', default=split, type=float)
    parser.add_argument('-d', '--device', type=str, help='cpu or gpu number',  default=device)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', type=str, help='KPI/STKE',
                        default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)
    sys_args = parser.parse_args()
    sys_args.prop_scores_filename = sys_args.experiments_type + '_' + '_'.join(sys_args.directed_interactions_filename) + '_{}_direct'.format(sys_args.n_experiments)

    # sys_args.save_prop_scores =  True
    # sys_args.load_prop_scores =  True

    run(sys_args)
